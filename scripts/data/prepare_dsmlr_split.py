"""
prepare_dsmlr_split.py
------------------------
The DSMLR Car-Parts-Segmentation dataset has no official train/val/test split
(it's just a folder of images + COCO-style annotation). This script splits it
80/10/10 randomly (seeded, so it's reproducible) and writes out three separate
COCO JSON files plus matching image folders, so it looks structurally the
same as the Ultralytics dataset.
"""

import os
import json
import random
import shutil
from pathlib import Path

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

DSMLR_ROOT = Path("datasets/dsmlr-carparts")
OUTPUT_ROOT = Path("datasets/dsmlr-carparts-split")


def find_coco_json(root):
    """DSMLR repo structure can vary by release; search for the annotation file."""
    candidates = list(root.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No COCO annotation json found under {root}. "
            "Check the repo structure -- it may have changed."
        )
    # Prefer files that look like a full annotation export (largest file, likely has 'annotations' key)
    for c in sorted(candidates, key=lambda p: p.stat().st_size, reverse=True):
        with open(c) as f:
            data = json.load(f)
        if "images" in data and "annotations" in data:
            return c, data
    raise FileNotFoundError("Found JSON files but none look like valid COCO annotations.")


def find_images_dir(root):
    candidates = [p for p in root.rglob("*") if p.is_dir() and
                  any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in p.glob("*"))]
    if not candidates:
        raise FileNotFoundError(f"No image folder found under {root}.")
    # pick the folder with the most images
    return max(candidates, key=lambda d: len(list(d.glob("*"))))


def split_coco(coco_data):
    random.seed(RANDOM_SEED)
    images = coco_data["images"][:]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    anns_by_image = {}
    for ann in coco_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    result = {}
    for split_name, split_images in splits.items():
        image_ids = {img["id"] for img in split_images}
        split_anns = []
        for img_id in image_ids:
            split_anns.extend(anns_by_image.get(img_id, []))
        result[split_name] = {
            "images": split_images,
            "annotations": split_anns,
            "categories": coco_data["categories"],
        }
    return result


def main():
    print("[*] Locating DSMLR annotation file and image folder ...")
    json_path, coco_data = find_coco_json(DSMLR_ROOT)
    images_dir = find_images_dir(DSMLR_ROOT)
    print(f"    Found annotations: {json_path}")
    print(f"    Found images dir : {images_dir}")

    total_images = len(coco_data["images"])
    print(f"[*] Total images: {total_images}")

    splits = split_coco(coco_data)

    for split_name, split_data in splits.items():
        img_out_dir = OUTPUT_ROOT / "images" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        ann_out_dir = OUTPUT_ROOT / "annotations"
        ann_out_dir.mkdir(parents=True, exist_ok=True)

        for img in split_data["images"]:
            src = images_dir / img["file_name"]
            dst = img_out_dir / img["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        ann_path = ann_out_dir / f"instances_{split_name}.json"
        with open(ann_path, "w") as f:
            json.dump(split_data, f)

        print(f"[OK] {split_name}: {len(split_data['images'])} images, "
              f"{len(split_data['annotations'])} annotations -> {ann_path}")

    print(f"\n[DONE] Split dataset ready at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
