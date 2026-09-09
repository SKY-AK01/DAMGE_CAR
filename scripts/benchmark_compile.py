#!/usr/bin/env python3
"""
benchmark_compile.py  --  Stage 5: torch.compile correctness & throughput check.

PRIMARY USE (local CPU dev):
    Compile-correctness only — does it compile cleanly, any graph breaks?
    Throughput numbers on CPU are NOT representative of GPU performance and
    are skipped by default when CUDA is not available.

SECONDARY USE (GPU VM):
    Full throughput benchmark with --throughput flag once on a real GPU.

Usage:
    # Local (CPU) — graph-break analysis only:
    python scripts/benchmark_compile.py --model both --dataset datasets/carparts-seg

    # GPU VM — add throughput timing:
    python scripts/benchmark_compile.py --model both --dataset /path/to/dataset --throughput

    # One model at a time:
    python scripts/benchmark_compile.py --model mask2former --dataset datasets/carparts-seg
    python scripts/benchmark_compile.py --model maskrcnn    --dataset datasets/carparts-seg
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = DEVICE == "cuda"


# ── Minimal datasets (no labels — forward-pass only) ──────────────────────

class M2FValDataset(Dataset):
    def __init__(self, images_dir, json_file, processor, limit=None):
        with open(json_file) as f:
            data = json.load(f)
        self.images_dir = Path(images_dir)
        self.processor  = processor
        self.image_list = [img for img in data["images"]
                           if (self.images_dir / img["file_name"]).exists()]
        if limit:
            self.image_list = self.image_list[:limit]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        info  = self.image_list[idx]
        image = PILImage.open(self.images_dir / info["file_name"]).convert("RGB")
        w, h  = image.size
        inp   = self.processor(images=image, return_tensors="pt")
        return {"pixel_values": inp["pixel_values"].squeeze(0), "orig_size": (h, w)}


def m2f_collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "orig_sizes":   [b["orig_size"] for b in batch],
    }


class MRCNNValDataset(Dataset):
    def __init__(self, images_dir, json_file, limit=None):
        import torchvision.transforms.functional as TF
        self._to_tensor = TF.to_tensor
        with open(json_file) as f:
            data = json.load(f)
        self.images_dir = Path(images_dir)
        self.image_list = [img for img in data["images"]
                           if (self.images_dir / img["file_name"]).exists()]
        if limit:
            self.image_list = self.image_list[:limit]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        info  = self.image_list[idx]
        image = PILImage.open(self.images_dir / info["file_name"]).convert("RGB")
        import torchvision.transforms.functional as TF
        return TF.to_tensor(image)


def mrcnn_collate(batch):
    return list(batch)   # Mask R-CNN eval forward takes a list of tensors


# ── Graph-break analysis ───────────────────────────────────────────────────

def analyse_graph_breaks(model, sample_batch, model_type, capture_scalars=False):
    """
    Use torch._dynamo.explain() to count graph breaks without compiling.
    Returns (n_graphs, n_breaks, reasons_list).
    All results are purely structural — valid on CPU and GPU alike.

    capture_scalars=True mirrors what we'd set in training to reduce breaks
    caused by Tensor.item() calls (e.g. in Mask2Former's deformable attention).
    Run twice — without and with — to see the real vs optimistic break count.
    """
    try:
        import torch._dynamo as dynamo

        # Suppress the onnxruntime/NumPy 2.x import error that fires during
        # dynamo's lazy backend registration on some VM setups.  It's harmless
        # for our purposes (we're not using the onnxrt backend) but it prints
        # a noisy traceback.
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*numpy.*",
            category=UserWarning,
        )

        prev = torch._dynamo.config.capture_scalar_outputs
        if capture_scalars:
            torch._dynamo.config.capture_scalar_outputs = True

        try:
            if model_type == "m2f":
                pv = sample_batch["pixel_values"].to(DEVICE)
                explanation = dynamo.explain(lambda x: model(pixel_values=x))(pv)
            else:
                imgs = [t.to(DEVICE) for t in sample_batch]
                explanation = dynamo.explain(lambda x: model(x))(imgs)
        finally:
            torch._dynamo.config.capture_scalar_outputs = prev
            torch._dynamo.reset()   # clear cached graphs between runs

        n_graphs = len(explanation.graphs)
        n_breaks = len(explanation.break_reasons)
        reasons  = [str(r) for r in explanation.break_reasons[:5]]
        return n_graphs, n_breaks, reasons

    except Exception as e:
        return None, None, [f"dynamo.explain failed: {e}"]


# ── Single-pass compile correctness check ─────────────────────────────────

def check_compile_correctness(model, sample_batch, model_type, label):
    """
    Calls torch.compile + runs exactly ONE forward pass to verify:
      - compile() itself doesn't raise
      - the compiled forward doesn't raise on real input
      - records wall-clock time (clearly labeled CPU-only if no CUDA)

    Returns (success: bool, elapsed_s: float | None, error_msg: str | None)
    """
    try:
        compiled = torch.compile(model, backend="inductor", fullgraph=False)
    except Exception as e:
        return False, None, f"torch.compile() call failed: {e}"

    # One eager pass for timing reference
    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            if model_type == "m2f":
                pv = sample_batch["pixel_values"].to(DEVICE)
                model(pixel_values=pv)
            else:
                imgs = [t.to(DEVICE) for t in sample_batch]
                model(imgs)
        except Exception as e:
            return False, None, f"Eager forward failed (pre-compile reference): {e}"
    eager_s = time.perf_counter() - t0

    # One compiled pass — first call includes compilation overhead
    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            if model_type == "m2f":
                pv = sample_batch["pixel_values"].to(DEVICE)
                compiled(pixel_values=pv)
            else:
                imgs = [t.to(DEVICE) for t in sample_batch]
                compiled(imgs)
        except Exception as e:
            return False, None, f"Compiled forward failed: {e}"
    compiled_s = time.perf_counter() - t0

    return True, {"eager_s": eager_s, "compiled_s": compiled_s}, None


# ── Throughput measurement (GPU VM only) ──────────────────────────────────

def measure_throughput(forward_fn, loader, n_warmup, n_timed, label):
    """Full timed benchmark — only called when --throughput flag is set."""
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= n_warmup + n_timed:
            break
    if len(batches) < 2:
        return None

    for b in batches[:n_warmup]:
        with torch.no_grad():
            forward_fn(b)

    if IS_CUDA:
        torch.cuda.synchronize()

    n_images = 0
    t0 = time.perf_counter()
    for b in batches[n_warmup:n_warmup + n_timed]:
        with torch.no_grad():
            forward_fn(b)
        if isinstance(b, dict):
            n_images += b["pixel_values"].shape[0]
        else:
            n_images += len(b)
    if IS_CUDA:
        torch.cuda.synchronize()

    return n_images / max(time.perf_counter() - t0, 1e-6)


# ── Per-model runners ──────────────────────────────────────────────────────

def run_mask2former(ds_path, do_throughput, warmup, timed, limit):
    print("\n" + "=" * 64)
    print("  MASK2FORMER  —  compile-correctness check")
    print("=" * 64)

    try:
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )
    except ImportError:
        print("  [SKIP] transformers not installed.")
        return None

    val_json = ds_path / "coco_val.json"
    val_imgs = ds_path / "images" / "val"
    if not val_json.exists():
        print(f"  [SKIP] {val_json} not found.")
        return None

    weights = PROJECT_ROOT / "archive" / "run_20260824_073035" / "mask2former" / "best_model"
    src = str(weights) if weights.exists() else "facebook/mask2former-swin-tiny-coco-instance"
    print(f"  Weights : {src}")

    processor = Mask2FormerImageProcessor.from_pretrained(src)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        src, ignore_mismatched_sizes=True
    ).to(DEVICE).eval()

    ds     = M2FValDataset(val_imgs, val_json, processor, limit=limit)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=m2f_collate, num_workers=0)
    sample = next(iter(loader))
    print(f"  Images  : {len(ds)} (limit={limit})  |  device: {DEVICE}")

    # ── 1. Graph-break analysis ────────────────────────────────────────────
    print("\n  [Step 1] Graph-break analysis (dynamo.explain) ...")
    n_graphs, n_breaks, reasons = analyse_graph_breaks(model, sample, "m2f")
    # Also check with capture_scalar_outputs=True — Mask2Former's Tensor.item()
    # breaks in multi_scale_deformable_attention can be resolved with that flag.
    n_graphs_cs, n_breaks_cs, _ = analyse_graph_breaks(
        model, sample, "m2f", capture_scalars=True)
    _print_breaks(n_graphs, n_breaks, reasons, n_graphs_cs, n_breaks_cs)

    # ── 2. Compile correctness (one forward pass) ─────────────────────────
    print("\n  [Step 2] Compile correctness — one forward pass each ...")
    ok, timing, err = check_compile_correctness(model, sample, "m2f", "M2F")
    _print_correctness(ok, timing, err)

    # ── 3. Throughput (GPU VM only) ───────────────────────────────────────
    eager_fps, compiled_fps = None, None
    if do_throughput:
        print("\n  [Step 3] Throughput benchmark ...")
        compiled = torch.compile(model, backend="inductor", fullgraph=False)
        eager_fps    = measure_throughput(
            lambda b: model(pixel_values=b["pixel_values"].to(DEVICE)),
            loader, warmup, timed, "M2F-eager")
        compiled_fps = measure_throughput(
            lambda b: compiled(pixel_values=b["pixel_values"].to(DEVICE)),
            loader, warmup, timed, "M2F-compiled")
        _print_throughput(eager_fps, compiled_fps)
    else:
        print("\n  [Step 3] Throughput — SKIPPED (CPU-only; re-run with --throughput on GPU VM)")

    verdict = _verdict(n_breaks_cs if n_breaks_cs is not None else n_breaks,
                       ok, eager_fps, compiled_fps, err)
    print(f"\n  Verdict: {verdict}")
    return {"model": "mask2former",
            "n_graphs": n_graphs, "n_breaks": n_breaks,
            "n_breaks_with_capture_scalars": n_breaks_cs,
            "compile_ok": ok, "compile_error": err,
            "eager_fps": eager_fps, "compiled_fps": compiled_fps,
            "verdict": verdict}


def run_maskrcnn(ds_path, do_throughput, warmup, timed, limit):
    print("\n" + "=" * 64)
    print("  MASK R-CNN  —  compile-correctness check")
    print("=" * 64)

    try:
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError:
        print("  [SKIP] torchvision not installed.")
        return None

    val_json = ds_path / "coco_val.json"
    val_imgs = ds_path / "images" / "val"
    if not val_json.exists():
        print(f"  [SKIP] {val_json} not found.")
        return None

    with open(val_json) as f:
        num_classes = len(json.load(f)["categories"]) + 1

    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)

    local_w = PROJECT_ROOT / "runs_comparison" / "maskrcnn" / "best_model.pt"
    if local_w.exists():
        model.load_state_dict(torch.load(local_w, map_location=DEVICE))
        print(f"  Weights : {local_w}")
    else:
        print("  Weights : ImageNet pretrained backbone (no local best_model.pt found)")

    model = model.to(DEVICE).eval()

    ds     = MRCNNValDataset(val_imgs, val_json, limit=limit)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=mrcnn_collate, num_workers=0)
    sample = next(iter(loader))
    print(f"  Images  : {len(ds)} (limit={limit})  |  device: {DEVICE}")

    # ── 1. Graph-break analysis ────────────────────────────────────────────
    print("\n  [Step 1] Graph-break analysis (dynamo.explain) ...")
    n_graphs, n_breaks, reasons = analyse_graph_breaks(model, sample, "maskrcnn")
    # Check with capture_scalar_outputs=True for Tensor.item() breaks in RoI heads.
    n_graphs_cs, n_breaks_cs, _ = analyse_graph_breaks(
        model, sample, "maskrcnn", capture_scalars=True)
    _print_breaks(n_graphs, n_breaks, reasons, n_graphs_cs, n_breaks_cs)

    # ── 2. Compile correctness (one forward pass) ─────────────────────────
    print("\n  [Step 2] Compile correctness — one forward pass each ...")
    ok, timing, err = check_compile_correctness(model, sample, "maskrcnn", "MRCNN")
    _print_correctness(ok, timing, err)

    # ── 3. Throughput (GPU VM only) ───────────────────────────────────────
    eager_fps, compiled_fps = None, None
    if do_throughput:
        print("\n  [Step 3] Throughput benchmark ...")
        compiled = torch.compile(model, backend="inductor", fullgraph=False)
        eager_fps    = measure_throughput(
            lambda b: model([t.to(DEVICE) for t in b]),
            loader, warmup, timed, "MRCNN-eager")
        compiled_fps = measure_throughput(
            lambda b: compiled([t.to(DEVICE) for t in b]),
            loader, warmup, timed, "MRCNN-compiled")
        _print_throughput(eager_fps, compiled_fps)
    else:
        print("\n  [Step 3] Throughput — SKIPPED (CPU-only; re-run with --throughput on GPU VM)")

    verdict = _verdict(n_breaks_cs if n_breaks_cs is not None else n_breaks,
                       ok, eager_fps, compiled_fps, err)
    print(f"\n  Verdict: {verdict}")
    return {"model": "maskrcnn",
            "n_graphs": n_graphs, "n_breaks": n_breaks,
            "n_breaks_with_capture_scalars": n_breaks_cs,
            "compile_ok": ok, "compile_error": err,
            "eager_fps": eager_fps, "compiled_fps": compiled_fps,
            "verdict": verdict}


# ── Shared print helpers ───────────────────────────────────────────────────

def _print_breaks(n_graphs, n_breaks, reasons, n_graphs_cs=None, n_breaks_cs=None):
    """Print graph-break summary. Optionally show capture_scalar_outputs=True result."""
    if n_graphs is None:
        print(f"  → dynamo.explain unavailable: {reasons[0]}")
        return
    print(f"  → Captured graphs : {n_graphs}")
    print(f"  → Graph breaks    : {n_breaks}  {'✓ none' if n_breaks == 0 else '⚠ see below'}")
    if n_breaks > 0:
        print("  → Break reasons (first 5):")
        for r in reasons:
            print(f"      • {str(r)[:220]}")
    if n_breaks_cs is not None and n_breaks_cs != n_breaks:
        delta = n_breaks - n_breaks_cs
        print(f"  → With capture_scalar_outputs=True: {n_breaks_cs} breaks "
              f"({delta} fewer — Tensor.item() breaks resolved by that flag)")


def _print_correctness(ok, timing, err):
    if not ok:
        print(f"  → FAILED: {err}")
        return
    print(f"  → Compiled forward: OK ✓")
    if timing and not IS_CUDA:
        print(f"  → Eager  wall time : {timing['eager_s']:.3f}s  "
              f"(CPU-only — NOT representative of GPU performance)")
        print(f"  → Compiled wall time: {timing['compiled_s']:.3f}s  "
              f"(includes JIT compilation overhead on first call)")
        print(f"  → NOTE: First compiled call includes the compilation itself; "
              f"steady-state GPU speedup must be measured on the VM.")
    elif timing:
        print(f"  → Eager  wall time : {timing['eager_s']:.3f}s")
        print(f"  → Compiled wall time: {timing['compiled_s']:.3f}s  (includes compile overhead)")


def _print_throughput(eager_fps, compiled_fps):
    if eager_fps is None or compiled_fps is None:
        print("  → Throughput measurement failed.")
        return
    delta = (compiled_fps - eager_fps) / eager_fps * 100 if eager_fps else 0
    sign  = "+" if delta >= 0 else ""
    print(f"  → Eager    : {eager_fps:.2f} img/s")
    print(f"  → Compiled : {compiled_fps:.2f} img/s  ({sign}{delta:.1f}%)")


def _verdict(n_breaks, compile_ok, eager_fps, compiled_fps, err):
    if err or not compile_ok:
        return "DO NOT ADD — compile failed; must fix before enabling"
    if n_breaks is None:
        return "DEFERRED — dynamo.explain unavailable; re-check on VM"
    if n_breaks > 10:
        return (f"DO NOT ADD locally — {n_breaks} graph breaks; "
                f"re-test with fullgraph=True on GPU to quantify real cost")
    if eager_fps is None:
        # Correctness-only mode (no throughput)
        if n_breaks == 0:
            return ("WIRE IN as --compile opt-in ✓  "
                    "— 0 graph breaks, compiled cleanly; "
                    "measure actual speedup on GPU VM before making default")
        return (f"WIRE IN as --compile opt-in (cautiously)  "
                f"— {n_breaks} break(s) detected; likely still useful on GPU "
                f"but verify speedup on VM before making default")
    # Full throughput mode
    ratio = compiled_fps / eager_fps if eager_fps else 0
    if ratio < 1.05:
        return f"DO NOT ADD — <5% gain on GPU ({ratio:.2f}×)"
    return f"ADD as default — {ratio:.2f}× speedup on GPU with {n_breaks} break(s)"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="both",
                        choices=["mask2former", "maskrcnn", "both"])
    parser.add_argument("--dataset",    default="datasets/carparts-seg")
    parser.add_argument("--throughput", action="store_true",
                        help="Run full timed throughput benchmark (GPU VM only). "
                             "Skipped by default on CPU builds.")
    parser.add_argument("--warmup",     type=int, default=3)
    parser.add_argument("--timed",      type=int, default=10)
    parser.add_argument("--limit",      type=int, default=20,
                        help="Max images to load (default 20 for fast correctness check)")
    args = parser.parse_args()

    do_throughput = args.throughput
    if do_throughput and not IS_CUDA:
        print("[WARN] --throughput requested but no CUDA device found.")
        print("       Throughput numbers on CPU are not representative of GPU performance.")
        print("       Continuing anyway — label results accordingly.")

    ds_path = Path(args.dataset)

    print(f"\n{'='*64}")
    print(f"  torch.compile  —  Stage 5  ({'correctness check' if not do_throughput else 'full benchmark'})")
    print(f"{'='*64}")
    print(f"  torch    : {torch.__version__}")
    print(f"  device   : {DEVICE}{'  (' + torch.cuda.get_device_name(0) + ')' if IS_CUDA else '  (CPU-only build)'}")
    print(f"  mode     : {'throughput + correctness' if do_throughput else 'correctness / graph-break analysis only'}")
    if not IS_CUDA:
        print(f"\n  ⚠  CPU-only environment detected.")
        print(f"     Graph-break results ARE valid and transfer to the GPU run.")
        print(f"     Wall-clock numbers are NOT representative of GPU performance.")
        print(f"     Re-run with --throughput on the GPU VM for speed decisions.")

    results = {}

    if args.model in ("mask2former", "both"):
        r = run_mask2former(ds_path, do_throughput, args.warmup, args.timed, args.limit)
        if r:
            results["mask2former"] = r

    if args.model in ("maskrcnn", "both"):
        r = run_maskrcnn(ds_path, do_throughput, args.warmup, args.timed, args.limit)
        if r:
            results["maskrcnn"] = r

    # ── Consolidated final report ─────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  STAGE 5 FINAL REPORT")
    print("=" * 64)
    for name, r in results.items():
        print(f"\n  {name.upper()}")
        print(f"    Graphs captured : {r['n_graphs'] if r['n_graphs'] is not None else 'N/A'}")
        print(f"    Graph breaks    : {r['n_breaks'] if r['n_breaks'] is not None else 'N/A'}")
        print(f"    Compile OK      : {'Yes ✓' if r['compile_ok'] else 'No ✗'}")
        if r['compile_error']:
            print(f"    Compile error   : {r['compile_error'][:200]}")
        if r['eager_fps']:
            print(f"    Eager img/s     : {r['eager_fps']:.2f}  (GPU VM only — meaningful)")
            print(f"    Compiled img/s  : {r['compiled_fps']:.2f}  (GPU VM only — meaningful)")
        else:
            print(f"    Throughput      : deferred to GPU VM run")
        print(f"    Verdict         : {r['verdict']}")

    print()
    print("  ── Pending items ─────────────────────────────────────────────")
    print("  ⚠  ACCURACY BASELINE: Stage 1 mAP (0.0229 mask mAP50) is invalid —")
    print("     checkpoint was evaluated against a dataset it wasn't trained on.")
    print("     True baseline requires a fresh training run on carparts-seg with")
    print("     all Stages 1-4 changes applied. Flag for first GPU VM run.")
    print()
    print("  ⚠  THROUGHPUT BASELINE: --throughput not run (CPU-only environment).")
    print("     Re-run on GPU VM: python scripts/benchmark_compile.py --model both \\")
    print("       --dataset <vm_dataset_path> --throughput")
    print()
    print("  ⚠  VM DETAILS NEEDED before GPU run:")
    print("     GPU model/VRAM, CUDA version, driver version,")
    print("     single vs multi-GPU, dataset path on VM.")


if __name__ == "__main__":
    main()
