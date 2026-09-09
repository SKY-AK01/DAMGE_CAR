#!/usr/bin/env python3
"""
Diagnose the back_left_light mask failure on image 247 in carparts-seg val.
Measures pixel area, bounding box dimensions, and fill ratio of each annotation,
then computes what pixel area survives the 640->384->640 resize round-trip
to confirm the IoU drop is pure quantisation loss, not a logic error.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pycocotools import mask as mask_utils

COCO_JSON  = PROJECT_ROOT / "datasets/carparts-seg/coco_val.json"
IMAGES_DIR = PROJECT_ROOT / "datasets/carparts-seg/images/val"
PROC_SIZE  = (384, 384)   # Mask2FormerImageProcessor default for swin-tiny

with open(COCO_JSON) as f:
    coco = json.load(f)

imgs = {i["id"]: i for i in coco["images"]}
cats = {c["id"]: c["name"] for c in coco["categories"]}
anns_247 = [a for a in coco["annotations"] if a["image_id"] == 247]

img_info = imgs[247]
h, w = img_info["height"], img_info["width"]
print(f"Image 247: {w}x{h}  ->  processor resizes to {PROC_SIZE[0]}x{PROC_SIZE[1]}")
print(f"Scale factors:  x={PROC_SIZE[0]/w:.4f}  y={PROC_SIZE[1]/h:.4f}")
print()
print(f"{'category':25s}  {'gt_area':>8s}  {'bbox':>12s}  {'fill':>5s}  "
      f"{'proc_area':>9s}  {'rt_area':>8s}  {'rt_iou':>7s}")
print("-" * 90)

for ann in sorted(anns_247, key=lambda a: a["category_id"]):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle  = mask_utils.merge(rles)
    else:
        rle = seg
    gt = mask_utils.decode(rle).astype(bool)
    if not gt.any():
        continue

    ys, xs = np.where(gt)
    bx = xs.max() - xs.min() + 1
    by = ys.max() - ys.min() + 1
    fill = gt.sum() / (bx * by)

    # Simulate the __getitem__ resize: 640->384 (nearest), then back to 640
    gt_pil  = Image.fromarray(gt.astype(np.uint8) * 255)
    proc    = gt_pil.resize(PROC_SIZE, Image.NEAREST)
    rt      = proc.resize((w, h), Image.NEAREST)
    rt_mask = (np.array(rt) > 128).astype(bool)

    proc_area = (np.array(proc) > 128).sum()
    rt_area   = rt_mask.sum()
    intersection = (gt & rt_mask).sum()
    union        = (gt | rt_mask).sum()
    iou = intersection / union if union > 0 else 1.0

    name = cats[ann["category_id"]]
    print(f"{name:25s}  {gt.sum():8d}  {bx:5d}x{by:<5d}  {fill:5.2f}  "
          f"{proc_area:9d}  {rt_area:8d}  {iou:7.4f}")
