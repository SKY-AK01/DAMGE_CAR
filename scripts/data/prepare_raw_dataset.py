#!/usr/bin/env python3
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Canonical class ids (carparts-seg taxonomy)
NAMES = {
    "back_bumper": 0, "back_door": 1, "back_glass": 2, "back_left_door": 3,
    "back_left_light": 4, "back_light": 5, "back_right_door": 6, "back_right_light": 7,
    "front_bumper": 8, "front_door": 9, "front_glass": 10, "front_left_door": 11,
    "front_left_light": 12, "front_light": 13, "front_right_door": 14, "front_right_light": 15,
    "hood": 16, "left_mirror": 17, "object": 18, "right_mirror": 19,
    "tailgate": 20, "trunk": 21, "wheel": 22
}

# Map XML label names (Title Case / spaces / synonyms) to NAMES keys.
# The XML uses names like "Front Bumper", "Bonnet", "ORVM", etc.
# Normalise: lowercase + replace spaces/hyphens with underscores, then alias.
_ALIASES = {
    # Normalised XML name           : NAMES key
    "front_bumper":                   "front_bumper",
    "rear_bumper":                    "back_bumper",
    "back_bumper":                    "back_bumper",
    "bonnet":                         "hood",
    "hood":                           "hood",
    "trunk_lid":                      "trunk",
    "trunk":                          "trunk",
    "tailgate":                       "tailgate",
    "front_door":                     "front_door",
    "rear_door":                      "back_door",
    "back_door":                      "back_door",
    "front_left_door":                "front_left_door",
    "front_right_door":               "front_right_door",
    "back_left_door":                 "back_left_door",
    "rear_left_door":                 "back_left_door",
    "back_right_door":                "back_right_door",
    "rear_right_door":                "back_right_door",
    "headlamp":                       "front_light",
    "front_light":                    "front_light",
    "headlight":                      "front_light",
    "front_left_light":               "front_left_light",
    "front_right_light":              "front_right_light",
    "taillight":                      "back_light",
    "rear_light":                     "back_light",
    "back_light":                     "back_light",
    "back_left_light":                "back_left_light",
    "rear_left_light":                "back_left_light",
    "back_right_light":               "back_right_light",
    "rear_right_light":               "back_right_light",
    "orvm":                           "left_mirror",   # ORVM = wing mirror; position attr distinguishes L/R
    "left_mirror":                    "left_mirror",
    "right_mirror":                   "right_mirror",
    "windscreen":                     "front_glass",
    "front_glass":                    "front_glass",
    "windshield":                     "front_glass",
    "rear_windscreen":                "back_glass",
    "back_glass":                     "back_glass",
    "rear_windshield":                "back_glass",
    "wheel":                          "wheel",
    "indicator":                      "object",
    "fog_lamp":                       "object",
    "grille":                         "object",
    "fuel_lid":                       "object",
    "number_plate":                   "object",
    "roof":                           "object",
    "door_glass":                     "object",
    "front_fender":                   "object",
    "rear_quarter_panel":             "object",
    "side_skirt_/_rocker_panel":      "object",
    "side_skirt":                     "object",
    "rocker_panel":                   "object",
}


def _normalise(label: str) -> str:
    """Lowercase, strip, replace spaces/hyphens with underscores."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_label(raw_label: str, position_attr: str = "") -> int | None:
    """
    Map a raw XML label (e.g. 'Front Bumper', 'ORVM') plus an optional
    'Position' attribute value ('Left' / 'Right' / 'Front' / 'Rear') to
    a NAMES class id.  Returns None if the label is unknown.
    """
    norm = _normalise(raw_label)
    pos  = _normalise(position_attr)   # e.g. "left", "right", "front", "rear"

    # Special-case ORVM (wing mirror): use Position to distinguish L/R
    if norm == "orvm":
        if pos == "right":
            return NAMES["right_mirror"]
        return NAMES["left_mirror"]     # default to left if position missing

    # Special-case generic Door: use Position to map to the four door classes
    if norm in ("front_door", "rear_door", "back_door"):
        side_map = {
            ("front", "left"):  "front_left_door",
            ("front", "right"): "front_right_door",
            ("rear",  "left"):  "back_left_door",
            ("rear",  "right"): "back_right_door",
            ("back",  "left"):  "back_left_door",
            ("back",  "right"): "back_right_door",
        }
        # Try to split pos into (front/rear, left/right) if it contains both
        parts = pos.split("_")
        for fb in ("front", "rear", "back"):
            for lr in ("left", "right"):
                if fb in parts and lr in parts:
                    key = side_map.get((fb, lr))
                    if key:
                        return NAMES[key]
        # Fall back to the direct alias
        canonical = _ALIASES.get(norm)
        return NAMES[canonical] if canonical and canonical in NAMES else None

    canonical = _ALIASES.get(norm)
    if canonical is None:
        return None
    return NAMES.get(canonical)


def parse_cvat_xml(xml_path, out_labels_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    skipped_labels = set()

    for image in root.findall("image"):
        img_name = image.get("name")
        width    = float(image.get("width",  1))
        height   = float(image.get("height", 1))

        txt_name = Path(img_name).stem + ".txt"
        txt_path = out_labels_dir / txt_name

        polygons = image.findall("polygon")
        if not polygons:
            continue

        lines = []
        for poly in polygons:
            raw_label = poly.get("label", "")

            # Extract optional Position attribute to help disambiguate labels
            position_val = ""
            for attr in poly.findall("attribute"):
                if attr.get("name", "").lower() == "position":
                    position_val = attr.text or ""
                    break

            class_id = _resolve_label(raw_label, position_val)
            if class_id is None:
                skipped_labels.add(raw_label)
                continue

            points = poly.get("points", "").split(";")
            norm_pts = []
            for pt in points:
                try:
                    x, y = map(float, pt.split(","))
                except ValueError:
                    continue
                norm_pts.append(f"{x / width:.6f}")
                norm_pts.append(f"{y / height:.6f}")

            if len(norm_pts) >= 6:   # need at least 3 points
                lines.append(f"{class_id} " + " ".join(norm_pts))

        if lines:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")

    if skipped_labels:
        print(f"  [WARN] Unrecognised labels skipped: {sorted(skipped_labels)}")


def process(raw_dir="RAW_DATASET", out_dir="datasets/custom_carparts"):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    
    if not raw_path.exists():
        print(f"[SKIP] {raw_dir} does not exist.")
        return
        
    xml_path = raw_path / "XML"
    images_path = raw_path / "IMAGES"
    
    if not xml_path.exists() or not images_path.exists():
        print("[ERROR] RAW_DATASET must contain XML/ and IMAGES/ folders.")
        return

    out_images = out_path / "images"
    out_labels = out_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # 1. Parse XML to create labels
    for xml_file in xml_path.glob("*.xml"):
        print(f"[*] Parsing {xml_file.name} to YOLO format...")
        parse_cvat_xml(xml_file, out_labels)

    # 2. Match images with labels and split
    images = list(images_path.glob("*.*"))
    labeled_images = []
    for img in images:
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        txt_path = out_labels / (img.stem + ".txt")
        if txt_path.exists():
            labeled_images.append(img)
            
    if not labeled_images:
        print("[WARN] No valid labeled images found.")
        return

    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * 0.8)
    train_imgs = labeled_images[:split_idx]
    val_imgs = labeled_images[split_idx:]

    for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        split_img_dir = out_images / split
        split_lbl_dir = out_labels / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img in split_imgs:
            shutil.copy(img, split_img_dir / img.name)
            txt_path = out_labels / (img.stem + ".txt")
            if txt_path.exists():
                shutil.copy(txt_path, split_lbl_dir / txt_path.name)
                try:
                    txt_path.unlink()
                except OSError:
                    pass

    # Cleanup temporary labels folder if empty
    for f in out_labels.glob("*.txt"):
        f.unlink()

    # 3. Create data.yaml
    yaml_content = f"""path: {out_path.resolve()}
train: images/train
val: images/val
test:  # no test set

names:
"""
    for name, idx in sorted(NAMES.items(), key=lambda x: x[1]):
        yaml_content += f"  {idx}: {name}\n"

    with open(out_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"[OK] RAW_DATASET processed: {len(train_imgs)} train, {len(val_imgs)} val")
    print(f"[OK] Saved to {out_dir}")

if __name__ == "__main__":
    process()
