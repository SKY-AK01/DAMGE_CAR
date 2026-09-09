#!/usr/bin/env python3
"""
verify_mask2former_dataset.py

Standalone verification of the manual mask_labels / class_labels construction
in COCOMask2FormerDataset.__getitem__.

Checks per sampled image:
  (a) instance count from __getitem__ matches the COCO annotation count
  (b) every class_label matches the category_id in the COCO annotation
  (c) each mask_label spatially aligns with the pycocotools-decoded ground-truth mask
      (IoU >= 0.95 required; overlay PNG saved for visual inspection)

Usage — spot-check 3 samples with overlays:
  python scripts/verify_mask2former_dataset.py \
      --dataset datasets/carparts-seg \
      --num_samples 3 \
      --output_dir verify_output

Usage — full-val sweep, checks (a) and (b) only, no GPU/overlays needed:
  python scripts/verify_mask2former_dataset.py \
      --dataset datasets/carparts-seg \
      --full_val

Exits 0 if all checks pass, 1 if any check fails.

────────────────────────────────────────────────────────────
KNOWN LIMITATION — small-object mask quantisation
────────────────────────────────────────────────────────────
Any GT instance whose bounding box is under ~20px in either dimension at
640px input will lose significant mask fidelity at Mask2Former's 384px
training resolution due to nearest-neighbour downscaling.

At the 0.6× scale factor (640→384), a 15px-wide feature maps to ~9px.
Nearest-neighbour resampling drops or duplicates entire pixel rows/columns,
so round-trip IoU for such masks can fall to ~0.84 even when the construction
logic is completely correct.

This is EXPECTED BEHAVIOUR, not a bug.  Flag it if a specific small-part
class (e.g. back_left_light, trim, badges) underperforms later — the cause
is quantisation at downscale, not annotation or label errors.

Verified instance:
  carparts-seg val, img_id=247, back_left_light:
    GT area=225px (15×29 bbox), proc_area=80px, round-trip IoU=0.8397
────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pycocotools import mask as mask_utils

# Import the dataset class directly from the training script so we test exactly
# the code that will run during training — not a copy.
from scripts.training.train_mask2former import COCOMask2FormerDataset

try:
    from transformers import Mask2FormerImageProcessor
except ImportError:
    print("[ERROR] transformers not installed.")
    sys.exit(1)


# ── ANSI colours for terminal output ────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
OK     = f"{GREEN}[OK]{RESET}"
FAIL   = f"{RED}[FAIL]{RESET}"
WARN   = f"{YELLOW}[WARN]{RESET}"


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary mask IoU. Both arrays are H×W bool/uint8."""
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = (a & b).sum()
    union = (a | b).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


def decode_coco_seg(seg, h: int, w: int) -> np.ndarray:
    """Decode COCO polygon or RLE segmentation to H×W uint8 mask."""
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle  = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        rle = seg
    else:
        raise ValueError(f"Unknown segmentation type: {type(seg)}")
    return mask_utils.decode(rle)


def save_overlay(image_path: Path, gt_masks: list, pred_masks: list,
                 cat_ids: list, cat_names: dict, out_path: Path):
    """
    Save a side-by-side overlay image:
      Left  = ground-truth masks (from pycocotools, one colour per instance)
      Right = dataset masks (from __getitem__, same colour scheme)
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    colours = [
        (255, 80,  80,  120),
        (80,  255, 80,  120),
        (80,  80,  255, 120),
        (255, 255, 80,  120),
        (255, 80,  255, 120),
        (80,  255, 255, 120),
        (255, 165, 0,   120),
        (128, 0,   128, 120),
    ]

    def overlay_masks(base_img, masks, resize_to=None):
        canvas = base_img.copy()
        for i, m in enumerate(masks):
            colour = colours[i % len(colours)]
            if resize_to is not None:
                m_pil = Image.fromarray((m * 255).astype(np.uint8)).resize(
                    resize_to, Image.NEAREST)
                m = (np.array(m_pil) > 128).astype(np.uint8)
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            colour_img = Image.new("RGBA", canvas.size, colour)
            mask_pil = Image.fromarray((m * 255).astype(np.uint8)).convert("L")
            if mask_pil.size != canvas.size:
                mask_pil = mask_pil.resize(canvas.size, Image.NEAREST)
            overlay.paste(colour_img, mask=mask_pil)
            canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        for i, m in enumerate(masks):
            ys, xs = np.where(m.astype(bool))
            if len(ys):
                cx, cy = int(xs.mean()), int(ys.mean())
                name = cat_names.get(cat_ids[i], str(cat_ids[i])) if cat_ids else str(i)
                draw.text((cx, cy), name, fill=(255, 255, 255, 255))
        return canvas

    gt_overlay   = overlay_masks(img, gt_masks)
    pred_overlay = overlay_masks(img, pred_masks, resize_to=(w, h))

    combined = Image.new("RGBA", (w * 2 + 10, h), (30, 30, 30, 255))
    combined.paste(gt_overlay,   (0, 0))
    combined.paste(pred_overlay, (w + 10, 0))

    draw = ImageDraw.Draw(combined)
    draw.text((4,  4), "GT (pycocotools)",    fill=(255, 255, 255, 255))
    draw.text((w + 14, 4), "Dataset __getitem__", fill=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.convert("RGB").save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Full-val sweep: checks (a) and (b) only, no IoU / no overlays
# Fast enough to run on CPU before every training session.
# ─────────────────────────────────────────────────────────────────────────────

def run_full_val(dataset: COCOMask2FormerDataset,
                 anns_by_image: dict,
                 categories: dict,
                 images_by_id: dict) -> bool:
    """
    Iterate every sample in the dataset.  For each one:
      (a) assert instance count matches non-zero-area COCO annotations
      (b) assert every class_label matches the corresponding category_id

    No IoU check, no overlay images — this is purely a logic/parsing sweep.
    Prints a one-line summary per failure; silent for passing images.
    Returns True if everything passed.
    """
    n_total    = len(dataset)
    n_fail_a   = 0
    n_fail_b   = 0
    failures   = []   # list of (img_id, check, detail) for the summary

    print(f"[*] Full-val sweep: {n_total} images, checks (a) and (b) only ...")

    for ds_idx in range(n_total):
        item        = dataset[ds_idx]
        img_id      = item["image_id"]
        class_labels = item["class_labels"].numpy()
        orig_h, orig_w = item["orig_size"]
        pred_count  = len(class_labels)

        raw_anns = anns_by_image.get(img_id, [])

        # Build ground-truth list in the same order the dataset would
        expected_cats = []
        for ann in raw_anns:
            try:
                m = decode_coco_seg(ann["segmentation"], orig_h, orig_w)
            except Exception as e:
                failures.append((img_id, "decode_error",
                                 f"ann_id={ann['id']} error={e}"))
                continue
            if m.sum() > 0:
                expected_cats.append(ann["category_id"])

        gt_count = len(expected_cats)

        # ── (a) count ────────────────────────────────────────────────────────
        if gt_count != pred_count:
            n_fail_a += 1
            failures.append((img_id, "a_count",
                             f"GT={gt_count} dataset={pred_count}"))

        # ── (b) labels ───────────────────────────────────────────────────────
        if gt_count == pred_count:
            for i, (exp, got) in enumerate(zip(expected_cats, class_labels)):
                if int(exp) != int(got):
                    n_fail_b += 1
                    cat_name = categories.get(int(got), "?")
                    failures.append((img_id, "b_label",
                                     f"instance {i}: expected cat_id={exp} "
                                     f"got={got} ({cat_name})"))

        # Progress dot every 50 images
        if (ds_idx + 1) % 50 == 0 or (ds_idx + 1) == n_total:
            print(f"  ... {ds_idx + 1}/{n_total}", flush=True)

    print()
    if not failures:
        print(f"{OK} Full-val sweep complete: all {n_total} images passed "
              f"checks (a) and (b).")
        return True
    else:
        print(f"{FAIL} Full-val sweep found {len(failures)} failure(s) "
              f"across {n_total} images:")
        for img_id, check, detail in failures:
            print(f"  img_id={img_id}  check={check}  {detail}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Spot-check: all three checks including IoU + overlays (original behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def verify_sample(dataset: COCOMask2FormerDataset,
                  dataset_idx: int,
                  coco_data: dict,
                  anns_by_image: dict,
                  categories: dict,
                  images_dir: Path,
                  images_by_id: dict,
                  output_dir: Path,
                  iou_threshold: float = 0.95) -> bool:
    """Run all three checks on one dataset sample. Returns True if all pass."""
    item = dataset[dataset_idx]
    img_id = item["image_id"]
    img_info = images_by_id[img_id]

    mask_labels  = item["mask_labels"].numpy()
    class_labels = item["class_labels"].numpy()
    orig_h, orig_w = item["orig_size"]

    image_path = images_dir / img_info["file_name"]

    gt_anns = [a for a in anns_by_image.get(img_id, [])
               if decode_coco_seg(a["segmentation"], orig_h, orig_w).sum() > 0]

    print(f"\n{'='*60}")
    print(f"Image idx={dataset_idx}  img_id={img_id}  file={img_info['file_name']}")
    print(f"  Original size: {orig_w}×{orig_h}")
    print(f"  Processed mask size: {mask_labels.shape[1]}×{mask_labels.shape[2]}")

    all_pass = True

    # ── (a) instance count ───────────────────────────────────────────────────
    gt_count   = len(gt_anns)
    pred_count = len(mask_labels)
    count_ok   = gt_count == pred_count
    print(f"\n  (a) Instance count: {OK if count_ok else FAIL}"
          f"  GT={gt_count}  dataset={pred_count}")
    if not count_ok:
        all_pass = False
        raw_anns = anns_by_image.get(img_id, [])
        print(f"      Raw COCO annotations: {len(raw_anns)}")
        for ann in raw_anns:
            m = decode_coco_seg(ann["segmentation"], orig_h, orig_w)
            print(f"      ann_id={ann['id']} cat_id={ann['category_id']} "
                  f"mask_sum={m.sum()} filtered={'yes' if m.sum()==0 else 'no'}")

    # ── (b) class labels ─────────────────────────────────────────────────────
    print(f"\n  (b) Class label mapping:")
    label_pass = True
    raw_anns_ordered = anns_by_image.get(img_id, [])
    expected_cats = []
    for ann in raw_anns_ordered:
        m = decode_coco_seg(ann["segmentation"], orig_h, orig_w)
        if m.sum() > 0:
            expected_cats.append(ann["category_id"])

    if len(expected_cats) != pred_count:
        print(f"      {FAIL} Cannot match labels — count mismatch")
        label_pass = False
        all_pass   = False
    else:
        for i, (exp_cat, pred_cat) in enumerate(zip(expected_cats, class_labels)):
            match    = int(exp_cat) == int(pred_cat)
            cat_name = categories.get(int(pred_cat), "?")
            sym      = OK if match else FAIL
            print(f"      instance {i}: expected cat_id={exp_cat}  "
                  f"got={pred_cat}  name={cat_name}  {sym}")
            if not match:
                label_pass = False
                all_pass   = False
    print(f"      → Labels: {OK if label_pass else FAIL}")

    # ── (c) spatial alignment ────────────────────────────────────────────────
    print(f"\n  (c) Spatial alignment (IoU threshold={iou_threshold}):")
    iou_pass = True

    if len(expected_cats) != pred_count:
        print(f"      {WARN} Skipping — count mismatch")
    else:
        gt_masks_ov   = []
        pred_masks_ov = []
        for i, ann in enumerate([a for a in raw_anns_ordered
                                   if decode_coco_seg(a["segmentation"],
                                                      orig_h, orig_w).sum() > 0]):
            gt_mask       = decode_coco_seg(ann["segmentation"], orig_h, orig_w)
            gt_masks_ov.append(gt_mask)

            pred_proc = mask_labels[i]
            pred_pil  = Image.fromarray((pred_proc * 255).astype(np.uint8)).resize(
                            (orig_w, orig_h), Image.NEAREST)
            pred_orig = (np.array(pred_pil) > 128).astype(np.uint8)
            pred_masks_ov.append(pred_orig)

            iou      = mask_iou(gt_mask, pred_orig)
            cat_name = categories.get(int(class_labels[i]), "?")
            sym      = OK if iou >= iou_threshold else FAIL
            print(f"      instance {i} ({cat_name}): IoU={iou:.4f}  {sym}")
            if iou < iou_threshold:
                iou_pass = False
                all_pass = False

        overlay_path = output_dir / f"overlay_idx{dataset_idx}_img{img_id}.png"
        try:
            save_overlay(image_path, gt_masks_ov, pred_masks_ov,
                         expected_cats, categories, overlay_path)
            print(f"\n      Overlay saved: {overlay_path}")
        except Exception as e:
            print(f"\n      {WARN} Overlay save failed: {e}")

    print(f"      → Spatial alignment: {OK if iou_pass else FAIL}")
    print(f"\n  Overall for this sample: {OK if all_pass else FAIL}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Verify Mask2Former dataset mask construction")
    parser.add_argument("--dataset", default="datasets/carparts-seg")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--image_ids", type=int, nargs="+", default=None,
                        help="Specific COCO image IDs to spot-check")
    parser.add_argument("--output_dir", default="verify_output")
    parser.add_argument("--iou_threshold", type=float, default=0.95)
    parser.add_argument("--model_id",
                        default="facebook/mask2former-swin-tiny-coco-instance")
    parser.add_argument("--full_val", action="store_true", default=False,
                        help="Run checks (a) and (b) across every image in the "
                             "split — no IoU, no overlays.  Fast CPU-only sweep.")
    args = parser.parse_args()

    ds_path    = PROJECT_ROOT / args.dataset
    json_file  = ds_path / f"coco_{args.split}.json"
    images_dir = ds_path / "images" / args.split
    output_dir = PROJECT_ROOT / args.output_dir

    if not json_file.exists():
        print(f"[ERROR] JSON not found: {json_file}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"[ERROR] Images dir not found: {images_dir}")
        sys.exit(1)

    print(f"[*] Loading processor: {args.model_id}")
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_id)

    print(f"[*] Building dataset from {json_file}")
    dataset = COCOMask2FormerDataset(images_dir, json_file, processor)
    print(f"    Dataset length: {len(dataset)} samples")

    with open(json_file) as f:
        coco_data = json.load(f)

    images_by_id  = {img["id"]: img for img in coco_data["images"]}
    categories    = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # ── Full-val mode ────────────────────────────────────────────────────────
    if args.full_val:
        passed = run_full_val(dataset, anns_by_image, categories, images_by_id)
        sys.exit(0 if passed else 1)

    # ── Spot-check mode (original behaviour) ────────────────────────────────
    if args.image_ids:
        id_to_idx = {dataset.image_ids[i]: i for i in range(len(dataset))}
        indices = []
        for coco_id in args.image_ids:
            if coco_id in id_to_idx:
                indices.append(id_to_idx[coco_id])
            else:
                print(f"{WARN} image_id={coco_id} not in dataset, skipping")
    else:
        ann_counts = []
        for i in range(len(dataset)):
            img_id = dataset.image_ids[i]
            ih = images_by_id[img_id]["height"]
            iw = images_by_id[img_id]["width"]
            c = len([a for a in anns_by_image.get(img_id, [])
                     if decode_coco_seg(a["segmentation"], ih, iw).sum() > 0])
            ann_counts.append((i, c))
        nonzero = sorted([(i, c) for i, c in ann_counts if c > 0], key=lambda x: x[1])
        if not nonzero:
            print("[ERROR] No annotated images found.")
            sys.exit(1)
        m = len(nonzero)
        n = args.num_samples
        if n == 1:
            indices = [nonzero[m // 2][0]]
        elif n == 2:
            indices = [nonzero[0][0], nonzero[-1][0]]
        else:
            step    = max(1, (m - 1) // (n - 1))
            indices = [nonzero[min(i * step, m - 1)][0] for i in range(n)]
            indices = list(dict.fromkeys(indices))[:n]

    print(f"\n[*] Spot-checking {len(indices)} sample(s):")
    for idx in indices:
        img_id = dataset.image_ids[idx]
        ih = images_by_id[img_id]["height"]
        iw = images_by_id[img_id]["width"]
        n_anns = len([a for a in anns_by_image.get(img_id, [])
                      if decode_coco_seg(a["segmentation"], ih, iw).sum() > 0])
        print(f"  dataset_idx={idx}  img_id={img_id}  non-zero-ann-count={n_anns}")

    results = [verify_sample(dataset, idx, coco_data, anns_by_image, categories,
                              images_dir, images_by_id, output_dir, args.iou_threshold)
               for idx in indices]

    print(f"\n{'='*60}")
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    if n_fail == 0:
        print(f"{OK} All {n_pass}/{len(results)} samples passed all checks.")
        print(f"    Overlays: {output_dir}/")
        sys.exit(0)
    else:
        print(f"{FAIL} {n_fail}/{len(results)} sample(s) failed.")
        print(f"    Overlays: {output_dir}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
