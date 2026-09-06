#!/usr/bin/env python3
"""
remap_github_labels.py
-------------------------
Translates the class labels of downloaded GitHub datasets (COCO JSON or YOLO format)
to strictly adhere to the 23 canonical classes defined by RAW_DATASET.
"""

import os
import json
import argparse
from pathlib import Path
import yaml

# Hardcoded canonical mapping from prepare_raw_dataset.py
NAMES = {
    "back_bumper": 0, "back_door": 1, "back_glass": 2, "back_left_door": 3,
    "back_left_light": 4, "back_light": 5, "back_right_door": 6, "back_right_light": 7,
    "front_bumper": 8, "front_door": 9, "front_glass": 10, "front_left_door": 11,
    "front_left_light": 12, "front_light": 13, "front_right_door": 14, "front_right_light": 15,
    "hood": 16, "left_mirror": 17, "object": 18, "right_mirror": 19,
    "tailgate": 20, "trunk": 21, "wheel": 22
}

_ALIASES = {
    "front_bumper": "front_bumper",
    "rear_bumper": "back_bumper",
    "back_bumper": "back_bumper",
    "bonnet": "hood",
    "hood": "hood",
    "trunk_lid": "trunk",
    "trunk": "trunk",
    "tailgate": "tailgate",
    "front_door": "front_door",
    "rear_door": "back_door",
    "back_door": "back_door",
    "front_left_door": "front_left_door",
    "front_right_door": "front_right_door",
    "back_left_door": "back_left_door",
    "rear_left_door": "back_left_door",
    "back_right_door": "back_right_door",
    "rear_right_door": "back_right_door",
    "headlamp": "front_light",
    "front_light": "front_light",
    "headlight": "front_light",
    "front_left_light": "front_left_light",
    "front_right_light": "front_right_light",
    "taillight": "back_light",
    "rear_light": "back_light",
    "back_light": "back_light",
    "back_left_light": "back_left_light",
    "rear_left_light": "back_left_light",
    "back_right_light": "back_right_light",
    "rear_right_light": "back_right_light",
    "orvm": "left_mirror",
    "left_mirror": "left_mirror",
    "right_mirror": "right_mirror",
    "windscreen": "front_glass",
    "front_glass": "front_glass",
    "windshield": "front_glass",
    "rear_windscreen": "back_glass",
    "back_glass": "back_glass",
    "rear_windshield": "back_glass",
    "wheel": "wheel",
    "indicator": "object",
    "fog_lamp": "object",
    "grille": "object",
    "fuel_lid": "object",
    "number_plate": "object",
    "roof": "object",
    "door_glass": "object",
    "front_fender": "object",
    "rear_quarter_panel": "object",
    "side_skirt_/_rocker_panel": "object",
    "side_skirt": "object",
    "rocker_panel": "object",
}

def _normalise(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")

def get_canonical_id(raw_label: str) -> int:
    norm = _normalise(raw_label)
    canonical = _ALIASES.get(norm)
    if canonical and canonical in NAMES:
        return NAMES[canonical]
    return -1

def remap_coco_json(json_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    old_categories = data.get("categories", [])
    old_id_to_new_id = {}
    
    # Map old IDs to canonical IDs
    for cat in old_categories:
        old_id = cat["id"]
        raw_name = cat["name"]
        new_id = get_canonical_id(raw_name)
        if new_id != -1:
            old_id_to_new_id[old_id] = new_id
        else:
            print(f"[WARN] COCO: Could not map category '{raw_name}' in {json_path.name}")
            
    # Remap annotations
    new_annotations = []
    for ann in data.get("annotations", []):
        old_cat_id = ann.get("category_id")
        if old_cat_id in old_id_to_new_id:
            ann["category_id"] = old_id_to_new_id[old_cat_id]
            new_annotations.append(ann)
            
    data["annotations"] = new_annotations
    
    # Replace categories array with canonical classes
    data["categories"] = [{"id": v, "name": k} for k, v in sorted(NAMES.items(), key=lambda x: x[1])]
    
    with open(json_path, 'w') as f:
        json.dump(data, f)
        
    print(f"[OK] Remapped COCO JSON: {json_path}")

def remap_yolo_dataset(dataset_dir: Path):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"[WARN] No data.yaml found in {dataset_dir}, skipping YOLO remap.")
        return
        
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        
    old_names = data_yaml.get("names", {})
    if isinstance(old_names, list):
        old_names = {i: name for i, name in enumerate(old_names)}
        
    old_id_to_new_id = {}
    for old_id_str, raw_name in old_names.items():
        old_id = int(old_id_str)
        new_id = get_canonical_id(raw_name)
        if new_id != -1:
            old_id_to_new_id[old_id] = new_id
        else:
            print(f"[WARN] YOLO: Could not map category '{raw_name}' in {yaml_path}")
            
    # Process text files
    labels_dir = dataset_dir / "labels"
    if labels_dir.exists():
        for split in ["train", "val", "test"]:
            split_dir = labels_dir / split
            if not split_dir.exists():
                continue
            for txt_file in split_dir.glob("*.txt"):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    if old_id in old_id_to_new_id:
                        new_id = old_id_to_new_id[old_id]
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts))
                        
                with open(txt_file, 'w') as f:
                    f.write("\n".join(new_lines) + "\n")
                    
    # Update data.yaml
    data_yaml["names"] = {v: k for k, v in sorted(NAMES.items(), key=lambda x: x[1])}
    data_yaml["nc"] = len(NAMES)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
        
    print(f"[OK] Remapped YOLO dataset: {dataset_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["coco", "yolo"], required=True)
    parser.add_argument("--path", required=True, help="Path to COCO JSON or YOLO dataset root")
    args = parser.parse_args()
    
    path = Path(args.path)
    if args.format == "coco":
        remap_coco_json(path)
    else:
        remap_yolo_dataset(path)

if __name__ == "__main__":
    main()
