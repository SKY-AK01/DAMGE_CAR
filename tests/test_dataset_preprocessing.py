import json
import os
from pathlib import Path

def validate_coco_json(json_path):
    assert os.path.exists(json_path), f"Missing {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 1. Validate top-level keys
    assert "categories" in coco, "Missing 'categories' key"
    assert "images" in coco, "Missing 'images' key"
    assert "annotations" in coco, "Missing 'annotations' key"

    # 2. Validate categories
    assert len(coco["categories"]) > 0, "No categories defined"
    cat_ids = set()
    for cat in coco["categories"]:
        assert "id" in cat and "name" in cat
        cat_ids.add(cat["id"])

    # 3. Validate images
    assert len(coco["images"]) > 0, "No images found"
    img_ids = set()
    for img in coco["images"]:
        assert "id" in img and "file_name" in img and "width" in img and "height" in img
        assert img["width"] > 0 and img["height"] > 0, f"Invalid dimensions: {img}"
        img_ids.add(img["id"])

    # 4. Validate annotations
    for ann in coco["annotations"]:
        assert "id" in ann and "image_id" in ann and "category_id" in ann
        assert ann["image_id"] in img_ids, f"Annotation refers to non-existent image {ann['image_id']}"
        assert ann["category_id"] in cat_ids, f"Annotation refers to non-existent category {ann['category_id']}"
        assert "bbox" in ann and len(ann["bbox"]) == 4
        assert "segmentation" in ann and len(ann["segmentation"]) > 0

    print(f"[OK] {json_path} passed all COCO schema validations ({len(coco['images'])} images, {len(coco['annotations'])} annotations)!")

if __name__ == "__main__":
    val_json = Path("datasets/carparts-seg/coco_val.json")
    if val_json.exists():
        validate_coco_json(val_json)
    else:
        print(f"[INFO] {val_json} not present, skipping test.")
