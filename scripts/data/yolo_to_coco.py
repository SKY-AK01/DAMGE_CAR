"""
yolo_to_coco.py
------------------
Converts a YOLO-segmentation formatted dataset (images/ + labels/ with
normalized polygon .txt files, like Ultralytics carparts-seg) into COCO JSON
format, which Mask2Former and MaskDINO both expect for training.

Usage:
    python yolo_to_coco.py --dataset carparts-seg
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image

CARPARTS_SEG_CLASSES = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light",
    "back_light", "back_right_door", "back_right_light", "front_bumper", "front_door",
    "front_glass", "front_left_door", "front_left_light", "front_light", "front_right_door",
    "front_right_light", "hood", "left_mirror", "object", "right_mirror", "tailgate",
    "trunk", "wheel",
]


def yolo_polygon_to_coco(line, img_w, img_h):
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    # normalized x,y pairs -> pixel coords
    points = []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_w
        y = coords[i + 1] * img_h
        points.extend([x, y])

    xs = points[0::2]
    ys = points[1::2]
    x_min, y_min = min(xs), min(ys)
    width, height = max(xs) - x_min, max(ys) - y_min
    area = 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs))))

    return class_id, points, [x_min, y_min, width, height], area


def convert_split(images_dir, labels_dir, class_names, out_json_path):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }

    ann_id = 0
    image_files = sorted([f for f in images_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

    for img_id, img_path in enumerate(image_files):
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append({
            "id": img_id, "file_name": img_path.name, "width": w, "height": h,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                if not line.strip():
                    continue
                class_id, seg_points, bbox, area = yolo_polygon_to_coco(line, w, h)
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": class_id,
                    "segmentation": [seg_points], "bbox": bbox, "area": area, "iscrowd": 0,
                })
                ann_id += 1

    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f"[OK] {out_json_path} -- {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["carparts-seg", "dsmlr-carparts", "custom_carparts", "combined_carparts"])
    args = parser.parse_args()

    if args.dataset in ["carparts-seg", "custom_carparts", "combined_carparts"]:
        root = Path(f"datasets/{args.dataset}")
        class_names = CARPARTS_SEG_CLASSES
        for split in ["train", "val", "test"]:
            images_dir = root / "images" / split
            labels_dir = root / "labels" / split
            if not images_dir.exists():
                print(f"[skip] {images_dir} not found")
                continue
            out_path = root / f"coco_{split}.json"
            convert_split(images_dir, labels_dir, class_names, out_path)

    elif args.dataset == "dsmlr-carparts":
        # DSMLR is already COCO format after prepare_dsmlr_split.py -- nothing to convert.
        print("[i] dsmlr-carparts is already prepared as COCO JSON by prepare_dsmlr_split.py -- skipping.")


if __name__ == "__main__":
    main()
