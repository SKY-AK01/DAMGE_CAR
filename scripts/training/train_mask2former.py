#!/usr/bin/env python3
"""
train_mask2former.py -- Mask2Former (Swin Transformer) Instance Segmentation Training.

Fine-tunes HuggingFace Mask2Former (facebook/mask2former-swin-tiny-coco-instance)
on the COCO-formatted car-parts dataset.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# pycocotools: imported at module level so DataLoader workers (which use
# multiprocessing 'spawn' on Windows) don't re-import it on every sample.
from pycocotools import mask as mask_utils

from scripts.evaluation.unified_evaluator import UnifiedEvaluator
from scripts.evaluation.unified_logger import UnifiedLogger

try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
except ImportError:
    print("[ERROR] HuggingFace transformers is required for Mask2Former training. Install with: pip install transformers")
    sys.exit(1)


class COCOMask2FormerDataset(Dataset):
    def __init__(self, images_dir, json_file, processor):
        self.images_dir = Path(images_dir)
        with open(json_file, "r") as f:
            self.coco_data = json.load(f)
        
        self.processor = processor
        self.images_by_id = {img["id"]: img for img in self.coco_data.get("images", [])}

        self.anns_by_image = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.anns_by_image:
                self.anns_by_image[img_id] = []
            self.anns_by_image[img_id].append(ann)

        # Only include images that have at least one annotation.
        # Matches CocoMaskRCNNDataset behaviour and prevents the empty-image
        # phantom-instance fallback from polluting training batches.
        self.image_ids = [img_id for img_id in self.images_by_id
                          if img_id in self.anns_by_image]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_by_id[img_id]
        img_path = self.images_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        anns = self.anns_by_image.get(img_id, [])
        instance_masks = []
        class_labels = []

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
            cat_id = ann["category_id"]
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                rle  = mask_utils.merge(rles)
            elif isinstance(seg, dict):
                rle = seg
            else:
                continue
            m = mask_utils.decode(rle)
            if m.sum() > 0:
                instance_masks.append(m)
                class_labels.append(cat_id)

        # If all annotations decoded to zero area (degenerate polygons), we
        # should not inject a phantom instance — that would pollute the loss
        # with a fake cat_id=0 mask.  Instead we return empty tensors here;
        # the model's criterion handles N=0 gracefully.
        # Note: images with NO annotations at all are excluded in __init__,
        # so this branch only fires for images whose annotations are all
        # zero-area after decoding (an uncommon but valid COCO edge case).
        if len(instance_masks) == 0:
            pass  # mask_tensors / label_tensors stay empty; handled below

        # ── Build pixel_values / pixel_mask via processor (image only) ───────
        # We deliberately do NOT pass segmentation_maps to the processor because
        # its internal instance_id_to_semantic_id lookup compares np.uint8 keys
        # against our Python-int dict and raises KeyError on older transformers.
        # Instead we construct mask_labels and class_labels tensors manually,
        # which is exactly what the model's forward() / criterion expects.
        img_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].squeeze(0)          # (3, H', W')
        pixel_mask   = img_inputs.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.squeeze(0)                        # (H', W')

        # Processor may resize; get the processed spatial size for mask resizing
        _, proc_h, proc_w = pixel_values.shape

        # ── Build mask_labels (N, H', W') float32 and class_labels (N,) int64 ─
        mask_tensors  = []
        label_tensors = []
        for m, cat_id in zip(instance_masks, class_labels):
            # Resize binary mask to match processed image size using nearest interpolation
            m_pil     = PILImage.fromarray(m.astype(np.uint8) * 255).resize(
                            (proc_w, proc_h), PILImage.NEAREST)
            m_resized = (np.array(m_pil) > 128).astype(np.float32)
            mask_tensors.append(torch.from_numpy(m_resized))
            label_tensors.append(torch.tensor(cat_id, dtype=torch.long))

        # Handle the (rare) case where mask_tensors is empty — torch.stack([])
        # raises RuntimeError.  Use zero-shaped tensors instead so the model
        # criterion sees N=0 and skips this image's mask loss correctly.
        if mask_tensors:
            stacked_masks  = torch.stack(mask_tensors)   # (N, H', W')
            stacked_labels = torch.stack(label_tensors)  # (N,)
        else:
            stacked_masks  = torch.zeros((0, proc_h, proc_w), dtype=torch.float32)
            stacked_labels = torch.zeros((0,),                 dtype=torch.long)

        result = {
            "pixel_values": pixel_values,
            "mask_labels":  stacked_masks,
            "class_labels": stacked_labels,
            "image_id":     img_id,
            "orig_size":    (h, w),                           # actual image dims for post-processing
        }
        if pixel_mask is not None:
            result["pixel_mask"] = pixel_mask
        return result


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask   = torch.stack([b["pixel_mask"] for b in batch]) if "pixel_mask" in batch[0] else None

    # mask_labels: list of (N_i, H', W') tensors — N_i varies per image
    # class_labels: list of (N_i,) tensors
    # Keep as lists; the model's criterion handles variable N per image.
    mask_labels  = [b["mask_labels"]  for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    image_ids    = [b["image_id"]     for b in batch]
    # orig_size: list of (h, w) tuples — actual image dimensions before processor resize.
    # Used in the val loop so post_process_instance_segmentation rescales masks to the
    # correct resolution rather than the previously hardcoded (640, 640).
    orig_sizes   = [b["orig_size"]    for b in batch]

    res = {
        "pixel_values": pixel_values,
        "mask_labels":  mask_labels,
        "class_labels": class_labels,
        "image_ids":    image_ids,
        "orig_sizes":   orig_sizes,
    }
    if pixel_mask is not None:
        res["pixel_mask"] = pixel_mask
    return res


def main():
    parser = argparse.ArgumentParser(description="Mask2Former Instance Segmentation Training")
    parser.add_argument("--dataset", required=True, help="Dataset name or path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers. Default lowered 8→4 for Windows: 'spawn' "
                             "multiprocessing means each worker boots a fresh interpreter, so "
                             "8 workers consume ~3-4 GB RAM before a single batch is loaded. "
                             "4 workers with prefetch_factor=4 is more efficient on Windows.")
    parser.add_argument("--output_dir", default="runs_comparison/mask2former")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to train for quick capacity testing.")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Run validation + COCO eval every N epochs (default: 5). "
                             "Use 1 to validate every epoch.")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4). "
                             "Effective batch = batch * accum_steps. "
                             "With batch=2 and accum_steps=4 the optimizer sees an effective "
                             "batch of 8 without holding 8 images in VRAM simultaneously.")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Wrap the model with torch.compile(backend='inductor') before "
                             "training.  Benchmark (Stage 5) found 9 graph breaks in the "
                             "default config, all caused by Tensor.item() in "
                             "multi_scale_deformable_attention.  Setting "
                             "capture_scalar_outputs=True (done automatically here) resolves "
                             "all of them and gives inductor a single clean graph covering the "
                             "full Swin backbone + decoder.  Expected 10-30%% throughput gain "
                             "on GPU; not beneficial on CPU.  Disable if you hit compile "
                             "errors on a new torch/transformers version.")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'OK' if device == 'cuda' else 'WARNING'}] Using device: {device}")

    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        base = ds_path if (ds_path / "coco_train.json").exists() else ds_path / "combined_carparts"
        train_images = base / "images" / "train"
        train_json   = base / "coco_train.json"
        val_images   = base / "images" / "val"
        val_json     = base / "coco_val.json"
    else:
        train_images = PROJECT_ROOT / "datasets" / args.dataset / "images" / "train"
        train_json   = PROJECT_ROOT / "datasets" / args.dataset / "coco_train.json"
        val_images   = PROJECT_ROOT / "datasets" / args.dataset / "images" / "val"
        val_json     = PROJECT_ROOT / "datasets" / args.dataset / "coco_val.json"

    with open(train_json, "r") as f:
        coco_train_data = json.load(f)
    num_classes = len(coco_train_data["categories"])
    class_names = [c["name"] for c in coco_train_data["categories"]]

    model_id = "facebook/mask2former-swin-tiny-coco-instance"
    print(f"[*] Loading pretrained Mask2Former ({model_id})...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # ── torch.compile (optional, GPU-only) ───────────────────────────────────
    # Three dynamo issues specific to Mask2Former on this VM:
    #
    # 1. capture_scalar_outputs=True  — resolves 9 graph breaks in
    #    multi_scale_deformable_attention caused by Tensor.item() calls.
    #
    # 2. suppress_errors=True  — two environment-level failures cause dynamo
    #    to crash the process rather than falling back gracefully:
    #
    #    a) onnxruntime built against NumPy 1.x fails to import under NumPy 2.x
    #       with SystemError("__ARRAY_API not found") during dynamo's lazy
    #       backend registration.  This fires on the FIRST FORWARD PASS, not at
    #       torch.compile() call time, so try/except around torch.compile() does
    #       not help.  Permanent fix: pip install --upgrade onnxruntime-gpu
    #
    #    b) linear_sum_assignment (Mask2Former's Hungarian matcher) is a compiled
    #       C extension that dynamo partially inlines.  When it reaches the
    #       .numpy() call inside it, it fails with:
    #       "TorchRuntimeError: .numpy() is not supported for tensor subclasses"
    #       allow_in_graph() does NOT prevent inlining for partially-traceable
    #       C extensions — suppress_errors is the correct escape hatch here.
    #
    #    suppress_errors=True tells dynamo to silently fall back to eager for any
    #    subgraph that fails compilation.  The backbone and decoder still get
    #    compiled; only the criterion (loss) runs in eager.  This is a net win
    #    over no compile at all.
    #
    #    Once onnxruntime is upgraded (pip install --upgrade onnxruntime-gpu),
    #    the onnxrt crash disappears and suppress_errors becomes a belt-and-
    #    suspenders safety net rather than the primary workaround.
    if args.compile and device == "cuda":
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        # cache_size_limit: dynamo caches one compiled graph per unique input
        # signature.  The val loop switches grad_mode (train→eval), which looks
        # like a new signature.  Default limit of 8 fills up across train/val
        # transitions and dynamo falls back to eager permanently for that frame.
        # 64 gives enough headroom for all grad_mode / AMP / batch-size variants
        # without unbounded memory growth.
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model, backend="inductor")
        print("[OK] torch.compile enabled (inductor backend, capture_scalar_outputs=True, suppress_errors=True, cache_size_limit=64)")
        print("     Note: suppress_errors=True means dynamo falls back to eager for any")
        print("     subgraph that cannot be compiled (criterion/loss runs in eager mode).")
        print("     To resolve permanently: pip install --upgrade onnxruntime-gpu")
    elif args.compile:
        print("[WARNING] --compile requested but device is CPU — skipping torch.compile")

    # ── Stage-2 audit: log point-sampling config so every run is self-documenting ──
    # Per the Mask2Former paper, loss is computed on randomly sampled points rather
    # than the full mask, reducing training memory ~3×.  These three values control
    # that sampling.  If train_num_points were 0 or absent, point sampling would be
    # off and we'd be paying full-resolution mask loss cost.
    cfg = model.config
    _npts   = getattr(cfg, "train_num_points",        "MISSING")
    _over   = getattr(cfg, "oversample_ratio",         "MISSING")
    _imp    = getattr(cfg, "importance_sample_ratio",  "MISSING")
    print(f"\n[Mask2Former config — point-based mask loss sampling]")
    print(f"  train_num_points        = {_npts}   (official default: 12544)")
    print(f"  oversample_ratio        = {_over}  (official default: 3.0)")
    print(f"  importance_sample_ratio = {_imp}  (official default: 0.75)")
    if _npts == "MISSING" or _npts == 0:
        print(f"  [WARNING] Point sampling is OFF — full-resolution mask loss is active.")
        print(f"            Set model.config.train_num_points = 12544 before training.")
    else:
        print(f"  [OK] Point sampling active — expected ~3× memory reduction vs full-mask loss.")
    print()

    train_ds = COCOMask2FormerDataset(train_images, train_json, processor)
    val_ds   = COCOMask2FormerDataset(val_images, val_json, processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)

    # ── Param groups: backbone at 0.1× LR, head/decoder at full LR ─────────
    # The Swin backbone is pretrained; applying the same LR as the randomly-
    # initialised segmentation head destabilises learned features in early epochs.
    # Standard fine-tuning practice is backbone ~10× lower than the head.
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # pixel_level_module.encoder is the Swin backbone
        if "pixel_level_module.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = args.lr * 0.1   # e.g. 1e-5 when lr=1e-4
    head_lr     = args.lr         # e.g. 1e-4

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params,     "lr": head_lr,     "name": "head"},
        ],
        weight_decay=1e-4,
    )
    print(f"[OK] AdamW param groups: backbone lr={backbone_lr:.2e}  |  head lr={head_lr:.2e}")
    print(f"     backbone params: {len(backbone_params)}  |  head params: {len(head_params)}")

    # ── Cosine LR schedule with linear warmup ────────────────────────────────
    # Warmup over 10% of total optimizer steps prevents early gradient explosion
    # when the head is randomly initialised.  After warmup, cosine decay brings
    # LR smoothly to near-zero, avoiding the stall that a flat LR causes in
    # later epochs.
    #
    # Total optimizer steps = ceil(batches_per_epoch / accum_steps) * epochs
    batches_per_epoch  = len(train_loader)
    steps_per_epoch    = max(1, batches_per_epoch // args.accum_steps)
    total_steps        = steps_per_epoch * args.epochs
    warmup_steps       = max(1, int(total_steps * 0.10))

    def warmup_cosine_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    print(f"[OK] LR schedule: linear warmup ({warmup_steps} steps) → cosine decay ({total_steps} total steps)")
    print(f"     steps_per_epoch={steps_per_epoch}  |  accum_steps={args.accum_steps}  |  "
          f"effective batch={args.batch * args.accum_steps}")

    # AMP: mixed precision for faster training and lower VRAM usage
    # (mirrors train_maskrcnn.py which fixed NaN/overflow with float() casting)
    use_amp = device == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    target_out_dir = Path(args.output_dir).resolve()
    os.makedirs(target_out_dir, exist_ok=True)

    logger = UnifiedLogger(str(target_out_dir), "mask2former")
    logger.print_dataset_health(str(train_json), str(val_json))
    evaluator = UnifiedEvaluator(str(val_json), str(val_images), class_names, str(target_out_dir))

    print(f"[*] Starting Mask2Former Training for {args.epochs} epochs...")

    global_step = 0  # counts optimizer steps (not batch steps) for scheduler

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # reset at epoch start

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            mask_labels  = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )
                # Scale loss by 1/accum_steps so that gradients accumulated over
                # accum_steps micro-batches equal a true mean over the full batch.
                loss = outputs.loss / args.accum_steps

            scaler.scale(loss).backward()

            # Only step the optimizer every accum_steps batches (or at epoch end)
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % args.accum_steps == 0 or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # ── LR diagnostic: print per-group LRs at first two optimizer steps
                #    of epoch 1 so we can confirm warmup + param groups are wired correctly.
                if epoch == 1 and global_step <= 2:
                    lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
                    print(f"    [LR check] optimizer step {global_step}: "
                          + "  ".join(f"{k}={v:.3e}" for k, v in lrs.items()))

            # Accumulate unscaled loss for logging (multiply back by accum_steps)
            running_loss += loss.item() * args.accum_steps

        epoch_time = time.time() - start_time
        avg_train_loss = running_loss / max(len(train_loader), 1)

        # Evaluation
        val_metrics = None
        run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        if run_val:
            print(f"\n[Epoch {epoch}] Running validation (every {args.val_interval} epochs)...")
            model.eval()
            predictions_by_image = {}
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    # autocast in val matches the precision used during training,
                    # halving VRAM usage and speeding up val forward passes.
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(pixel_values=pixel_values)

                    # Use actual image dimensions (h, w) per sample so that
                    # post_process_instance_segmentation rescales prediction masks
                    # back to the true image resolution.  The previous hardcoded
                    # (640, 640) produced misaligned masks whenever an image was
                    # not exactly 640×640, silently corrupting mAP numbers.
                    results = processor.post_process_instance_segmentation(
                        outputs, target_sizes=batch["orig_sizes"]
                    )
                    
                    for img_id, res in zip(batch["image_ids"], results):
                        preds = []
                        masks = res["segmentation"].cpu().numpy()
                        labels = res["segments_info"]
                        for info in labels:
                            cid = info["label_id"]
                            score = info["score"]
                            m = (masks == info["id"])
                            if not m.any():
                                continue
                            ys, xs = np.where(m)
                            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                            preds.append({
                                "category_id": cid,
                                "score": float(score),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "segmentation": m
                            })
                        predictions_by_image[img_id] = preds

            val_metrics = evaluator.evaluate(epoch, predictions_by_image)
        else:
            print(f"\n[Epoch {epoch}] Skipping validation (next at epoch {epoch + (args.val_interval - epoch % args.val_interval)})")

        train_stats = {
            "train_loss": avg_train_loss,
            "val_loss": 0.0,
            "lr": optimizer.param_groups[1]["lr"],  # log head LR (index 1 = head group)
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "epoch_time_sec": epoch_time,
            "images_sec": len(train_ds) / max(epoch_time, 1e-3)
        }

        should_stop, is_best = logger.log_epoch(epoch, args.epochs, train_stats, val_metrics)
        
        weights_dir = target_out_dir / "weights"
        os.makedirs(weights_dir, exist_ok=True)
        last_dir = weights_dir / "last"
        model.save_pretrained(str(last_dir))
        processor.save_pretrained(str(last_dir))

        if is_best:
            best_dir = weights_dir / "best"
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))  # saves preprocessor_config.json needed for inference
            # Also legacy best_model location for backward compat
            legacy_best = target_out_dir / "best_model"
            model.save_pretrained(str(legacy_best))
            processor.save_pretrained(str(legacy_best))

        if should_stop:
            print("[!] Early stopping triggered.")
            break

    logger.print_final_summary()
    print(f"[OK] Mask2Former training completed!")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA Peak VRAM] {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
