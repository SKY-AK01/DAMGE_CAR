import json
import os
import math
from pathlib import Path

def compare_coco_jsons(json1_path, json2_path):
    assert os.path.exists(json1_path), f"Missing {json1_path}"
    assert os.path.exists(json2_path), f"Missing {json2_path}"

    with open(json1_path, "r", encoding="utf-8") as f:
        coco1 = json.load(f)
    with open(json2_path, "r", encoding="utf-8") as f:
        coco2 = json.load(f)

    # 1. Compare Categories
    assert len(coco1["categories"]) == len(coco2["categories"]), "Categories count mismatch"
    for cat1, cat2 in zip(coco1["categories"], coco2["categories"]):
        assert cat1["id"] == cat2["id"]
        assert cat1["name"] == cat2["name"]

    # 2. Compare Images count & dimensions
    assert len(coco1["images"]) == len(coco2["images"]), "Images count mismatch"
    img_map1 = {img["file_name"]: img for img in coco1["images"]}
    img_map2 = {img["file_name"]: img for img in coco2["images"]}
    assert set(img_map1.keys()) == set(img_map2.keys()), "Image filenames mismatch"

    for name in img_map1:
        i1, i2 = img_map1[name], img_map2[name]
        assert i1["width"] == i2["width"], f"Width mismatch for {name}"
        assert i1["height"] == i2["height"], f"Height mismatch for {name}"

    # 3. Compare Annotations count
    assert len(coco1["annotations"]) == len(coco2["annotations"]), "Annotations count mismatch"

    print(f"[OK] {json1_path} and {json2_path} match structural specifications!")

if __name__ == "__main__":
    print("Test helper for COCO JSON equivalence verification ready.")
