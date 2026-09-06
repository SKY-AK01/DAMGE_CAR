"""
convert_to_target_taxonomy.py
--------------------------------
Converts a CVAT-format annotations.xml using the flat "carparts-seg" label
scheme (side baked into the label name, e.g. "front_left_door",
"back_right_light", no attributes) into your target 24-class taxonomy
(generic label names + a separate "Position" select attribute, e.g. label
"Front Door" + attribute Position="Front-Left").

Output XML structure matches your reference sample exactly:
    <polygon label="Front Door" ...>
      <attribute name="Position">Front-Left</attribute>
    </polygon>

Run this ANYTIME on any new source XML in the carparts-seg schema -- it's a
deterministic, repeatable conversion, not a one-off manual edit.

CLEAN MAPPINGS (side already known from the source label name):
    front_bumper -> Front Bumper (Front)
    back_bumper  -> Rear Bumper (Rear)
    hood         -> Bonnet
    front_left_door / front_right_door -> Front Door (Front-Left / Front-Right)
    back_left_door / back_right_door   -> Rear Door (Rear-Left / Rear-Right)
    front_left_light / front_right_light -> Headlamp (Left / Right)
    back_left_light / back_right_light   -> Taillight (Left / Right)
    left_mirror / right_mirror -> ORVM (Left / Right)
    front_glass  -> Windscreen (Front)
    back_glass   -> Rear Windscreen (Rear)
    tailgate, trunk -> Trunk Lid

AMBIGUOUS CASES (side not in the source label -- inferred by geometry):
    front_door, back_door, front_light, back_light:
        Front/Rear is already known from the name; Left/Right is inferred
        by comparing the polygon's centroid x-position to the image's
        horizontal midpoint.
    wheel (no side info at all):
        Left/Right inferred the same way (centroid x vs image midpoint).
        Front/Rear inferred by comparing the wheel's centroid x-distance to
        the average x of known "front" parts (front_bumper/front_glass/hood)
        vs known "rear" parts (back_bumper/back_glass/trunk/tailgate) in the
        SAME image -- whichever reference is closer wins. If neither
        reference exists in that image, the wheel is skipped and counted
        under "unmappable".

DROPPED (no sensible target mapping):
    object  (generic/noise label)

TARGET CLASSES NEVER PRESENT IN THIS SOURCE SCHEMA (so never produced by
this script, since the source data was never labeled with these):
    Front Fender, Rear Quarter Panel, Fog Lamp, Indicator, Fuel Lid,
    Side Skirt / Rocker Panel, Number Plate, Antenna, Roof, Grille, Pillar,
    Door Glass

Usage:
    python convert_to_target_taxonomy.py --input annotations.xml --output converted_annotations.xml
    python convert_to_target_taxonomy.py --input annotations.xml --output converted_annotations.xml --no-infer
        (skip geometry inference entirely; ambiguous polygons are just dropped)
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Full target label taxonomy -- matches the reference sample exactly
# (name, color, [(attribute_name, [values], default_value)] or [] if none)
# ----------------------------------------------------------------------------
TARGET_LABELS = [
    ("Front Bumper", "#FF6D00", [("Position", ["Front"], "Front")]),
    ("Rear Bumper", "#2979FF", [("Position", ["Rear"], "Rear")]),
    ("Trunk Lid", "#00BFA5", []),
    ("Bonnet", "#FFD600", []),
    ("Front Fender", "#D50000", [("Position", ["Left", "Right"], "Left")]),
    ("Front Door", "#1DE9B6", [("Position", ["Front-Left", "Front-Right"], "Front-Left")]),
    ("Rear Door", "#F57F17", [("Position", ["Rear-Left", "Rear-Right"], "Rear-Left")]),
    ("Rear Quarter Panel", "#E040FB", [("Position", ["Left", "Right"], "Left")]),
    ("Headlamp", "#00E5FF", [("Position", ["Left", "Right"], "Left")]),
    ("Taillight", "#69F0AE", [("Position", ["Left", "Right"], "Left")]),
    ("Indicator", "#FF4081", [("Position", ["Left", "Right"], "Left")]),
    ("Fog Lamp", "#76FF03", [("Position", ["Left", "Right"], "Left")]),
    ("ORVM", "#FFAB40", [("Position", ["Left", "Right"], "Left")]),
    ("Windscreen", "#FF6D00", [("Position", ["Front"], "Front")]),
    ("Rear Windscreen", "#2979FF", [("Position", ["Rear"], "Rear")]),
    ("Door Glass", "#00BFA5", [("Position", ["Front-Left", "Front-Right", "Rear-Left", "Rear-Right"], "Front-Left")]),
    ("Roof", "#FFD600", []),
    ("Wheel", "#D50000", [("Position", ["Front-Left", "Front-Right", "Rear-Left", "Rear-Right"], "Front-Left")]),
    ("Grille", "#1DE9B6", []),
    ("Fuel Lid", "#F57F17", [("Position", ["Left", "Right"], "Left")]),
    ("Side Skirt / Rocker Panel", "#E040FB", [("Position", ["Left", "Right"], "Left")]),
    ("Number Plate", "#00E5FF", [("Position", ["Front", "Rear"], "Front")]),
    ("Antenna", "#69F0AE", []),
    ("Wheel Cap", "#FF4081", [("Position", ["Front-Left", "Front-Right", "Rear-Left", "Rear-Right"], "Front-Left")]),
    ("Pillar", "#76FF03", [("Position", ["Left Side", "Right Side"], "Left Side")]),
]

# Clean, unambiguous mappings: source label -> (target label, position value or None)
CLEAN_MAPPING = {
    "front_bumper": ("Front Bumper", "Front"),
    "back_bumper": ("Rear Bumper", "Rear"),
    "hood": ("Bonnet", None),
    "front_left_door": ("Front Door", "Front-Left"),
    "front_right_door": ("Front Door", "Front-Right"),
    "back_left_door": ("Rear Door", "Rear-Left"),
    "back_right_door": ("Rear Door", "Rear-Right"),
    "front_left_light": ("Headlamp", "Left"),
    "front_right_light": ("Headlamp", "Right"),
    "back_left_light": ("Taillight", "Left"),
    "back_right_light": ("Taillight", "Right"),
    "left_mirror": ("ORVM", "Left"),
    "right_mirror": ("ORVM", "Right"),
    "front_glass": ("Windscreen", "Front"),
    "back_glass": ("Rear Windscreen", "Rear"),
    "tailgate": ("Trunk Lid", None),
    "trunk": ("Trunk Lid", None),
}

# Ambiguous: front/rear known from name, left/right needs geometry inference
AMBIGUOUS_SIDE_MAPPING = {
    "front_door": "Front Door",     # position becomes Front-Left / Front-Right
    "back_door": "Rear Door",       # position becomes Rear-Left / Rear-Right
    "front_light": "Headlamp",      # position becomes Left / Right
    "back_light": "Taillight",      # position becomes Left / Right
}

FRONT_REFERENCE_LABELS = {"front_bumper", "front_glass", "hood", "front_light",
                           "front_left_light", "front_right_light",
                           "front_door", "front_left_door", "front_right_door"}
REAR_REFERENCE_LABELS = {"back_bumper", "back_glass", "tailgate", "trunk", "back_light",
                          "back_left_light", "back_right_light",
                          "back_door", "back_left_door", "back_right_door"}

DROPPED_LABELS = {"object"}


def polygon_centroid_x(points_str):
    pts = [tuple(map(float, p.split(","))) for p in points_str.split(";")]
    return sum(x for x, y in pts) / len(pts)


def infer_left_right(centroid_x, image_width):
    return "Left" if centroid_x < image_width / 2 else "Right"


def build_labels_xml(labels_el):
    for name, color, attrs in TARGET_LABELS:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = name
        ET.SubElement(label_el, "color").text = color
        ET.SubElement(label_el, "type").text = "polygon"
        attributes_el = ET.SubElement(label_el, "attributes")
        for attr_name, values, default in attrs:
            attr_el = ET.SubElement(attributes_el, "attribute")
            ET.SubElement(attr_el, "name").text = attr_name
            ET.SubElement(attr_el, "mutable").text = "False"
            ET.SubElement(attr_el, "input_type").text = "select"
            ET.SubElement(attr_el, "default_value").text = default
            ET.SubElement(attr_el, "values").text = "\n".join(values)


def convert(input_path, output_path, infer_geometry=True):
    tree = ET.parse(input_path)
    src_root = tree.getroot()

    out_root = ET.Element("annotations")
    ET.SubElement(out_root, "version").text = "1.1"
    meta = ET.SubElement(out_root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "converted_to_target_taxonomy"
    ET.SubElement(task, "mode").text = "annotation"
    labels_el = ET.SubElement(task, "labels")
    build_labels_xml(labels_el)
    ET.SubElement(meta, "dumped").text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    stats = {"converted": 0, "inferred": 0, "dropped_no_mapping": 0, "dropped_no_reference": 0}

    for src_image in src_root.findall(".//image"):
        img_w = float(src_image.get("width"))
        img_h = float(src_image.get("height"))

        polygons = src_image.findall("polygon")

        # Pre-compute front/rear reference x-positions for this image (needed
        # for wheel front/rear inference)
        front_xs, rear_xs = [], []
        for poly in polygons:
            label = poly.get("label")
            if label in FRONT_REFERENCE_LABELS:
                front_xs.append(polygon_centroid_x(poly.get("points")))
            elif label in REAR_REFERENCE_LABELS:
                rear_xs.append(polygon_centroid_x(poly.get("points")))
        front_ref_x = sum(front_xs) / len(front_xs) if front_xs else None
        rear_ref_x = sum(rear_xs) / len(rear_xs) if rear_xs else None

        out_image = None  # created lazily only if we keep at least one polygon

        for poly in polygons:
            label = poly.get("label")
            points = poly.get("points")

            if label in DROPPED_LABELS:
                continue

            target_label, position = None, None

            if label in CLEAN_MAPPING:
                target_label, position = CLEAN_MAPPING[label]
                stats["converted"] += 1

            elif label == "wheel":
                if not infer_geometry:
                    stats["dropped_no_reference"] += 1
                    continue
                cx = polygon_centroid_x(points)
                lr = infer_left_right(cx, img_w)
                if front_ref_x is None and rear_ref_x is None:
                    stats["dropped_no_reference"] += 1
                    continue
                elif front_ref_x is None:
                    fb = "Rear"
                elif rear_ref_x is None:
                    fb = "Front"
                else:
                    fb = "Front" if abs(cx - front_ref_x) < abs(cx - rear_ref_x) else "Rear"
                target_label = "Wheel"
                position = f"{fb}-{lr}"
                stats["inferred"] += 1

            elif label in AMBIGUOUS_SIDE_MAPPING:
                if not infer_geometry:
                    stats["dropped_no_reference"] += 1
                    continue
                target_label = AMBIGUOUS_SIDE_MAPPING[label]
                cx = polygon_centroid_x(points)
                lr = infer_left_right(cx, img_w)
                if target_label == "Front Door":
                    position = f"Front-{lr}"
                elif target_label == "Rear Door":
                    position = f"Rear-{lr}"
                else:  # Headlamp / Taillight
                    position = lr
                stats["inferred"] += 1

            else:
                stats["dropped_no_mapping"] += 1
                continue

            if out_image is None:
                out_image = ET.SubElement(out_root, "image",
                                           id=src_image.get("id"), name=src_image.get("name"),
                                           width=src_image.get("width"), height=src_image.get("height"))

            poly_el = ET.SubElement(out_image, "polygon",
                                     label=target_label, source=poly.get("source", "converted"),
                                     occluded=poly.get("occluded", "0"),
                                     points=points, z_order=poly.get("z_order", "0"))
            if position is not None:
                attr_el = ET.SubElement(poly_el, "attribute", name="Position")
                attr_el.text = position

    xml_str = minidom.parseString(ET.tostring(out_root)).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"[OK] Converted XML saved -> {output_path}")
    print(f"\n[SUMMARY]")
    print(f"  Cleanly converted (side known from name): {stats['converted']}")
    print(f"  Inferred via geometry (side guessed):      {stats['inferred']}")
    print(f"  Dropped - no target mapping (e.g. 'object'): {stats['dropped_no_mapping']}")
    print(f"  Dropped - ambiguous, no reference to infer from: {stats['dropped_no_reference']}")
    print(f"\n  NOTE: geometry-based inference (left/right via image-center x-position,")
    print(f"  front/rear-wheel via distance to nearest known front/rear part) is a")
    print(f"  best-effort heuristic, not guaranteed correct -- spot-check a sample")
    print(f"  of the inferred annotations before trusting them for training.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source CVAT XML (carparts-seg flat schema)")
    parser.add_argument("--output", required=True, help="Output path for converted XML")
    parser.add_argument("--no-infer", action="store_true",
                         help="Disable geometry-based inference; ambiguous polygons are dropped instead")
    args = parser.parse_args()

    convert(args.input, args.output, infer_geometry=not args.no_infer)


if __name__ == "__main__":
    main()