#!/usr/bin/env python3
"""
Diagnose the 12 images where the full-val verifier sees GT=0 but dataset=1.
For each: show what annotations exist, their decoded area, and what __getitem__
actually returns (class_labels + mask pixel sums at proc resolution).
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pycocotools import mask as mask_utils
from scripts.training.train_mask2former import COCOMask2FormerDataset

try:
    from transformers import Mask2FormerImageProcessor
except ImportError:
    print("[ERROR] transformers not installed")
    sys.exit(1)

COCO_JSON  = PROJECT_ROOT / "datasets/carparts-seg/coco_val.json"
IMAGES_DIR = PROJECT_ROOT / "datasets/carparts-seg/images/val"
PROBLEM_IDS = [4, 129, 158, 194, 197, 223, 244, 259, 297, 347, 376, 380]

with open(COCO_JSON) as f:
    coco = json.load(f)

imgs = {i["id"]: i for i in coco["images"]}
cats = {c["id"]: c["name"] for c in coco["categories"]}
anns_by_img = {}
for a in coco["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)

print("[*] Loading processor ...")
processor = Mask2FormerImageProcessor.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-instance")

print("[*] Building dataset ...")
dataset = COCOMask2FormerDataset(IMAGES_DIR, COCO_JSON, processor)
id_to_idx = {dataset.image_ids[i]: i for i in range(len(dataset))}

print()

for img_id in PROBLEM_IDS:
    info = imgs[img_id]
    h, w = info["height"], info["width"]
    anns = anns_by_img.get(img_id, [])

    print(f"{'='*60}")
    print(f"img_id={img_id}  size={w}x{h}  file={info['file_name']}")
    print(f"  COCO raw annotation count: {len(anns)}")

    # Decode each annotation at original resolution
    for ann in anns:
        seg = ann["segmentation"]
        try:
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                rle  = mask_utils.merge(rles)
                pts  = [len(s) // 2 for s in seg]
            else:
                rle = seg
                pts = "rle"
            m = mask_utils.decode(rle)
            area = int(m.sum())
        except Exception as e:
            area = f"ERROR: {e}"
            pts  = "?"
        cat_name = cats.get(ann["category_id"], "?")
        print(f"  ann_id={ann['id']}  cat={cat_name}  "
              f"area_at_orig_res={area}  pts_per_poly={pts}")

    # What does __getitem__ actually return for this image?
    if img_id in id_to_idx:
        item = dataset[id_to_idx[img_id]]
        ml   = item["mask_labels"]   # (N, H', W') tensor
        cl   = item["class_labels"]  # (N,) tensor
        print(f"  __getitem__ returned: {len(cl)} instance(s)")
        for i in range(len(cl)):
            proc_area = int((ml[i] > 0.5).sum().item())
            cat_name  = cats.get(int(cl[i].item()), "?")
            print(f"    instance {i}: cat_id={int(cl[i].item())}  "
                  f"name={cat_name}  proc_mask_sum={proc_area}")
    else:
        print(f"  img_id={img_id} NOT in dataset (no annotations at all?)")

    print()
