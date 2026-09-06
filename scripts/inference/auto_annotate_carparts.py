"""
Car Parts Auto-Annotation Pipeline
-----------------------------------
Grounding DINO (text-prompted boxes)  ->  SAM 2 (box -> pixel polygon)
                                       ->  CVAT 1.1 XML (annotations.xml)

Folder layout produced:
    output/
        annotations.xml          <- always saved (CVAT format, matches your sample)
        sam2_annotated/          <- always saved (final polygons drawn on image)
        dino_visualized/         <- only saved if you answer "y" to the prompt

One-time setup (creates a venv, installs everything, downloads all weights):
    chmod +x setup.sh
    ./setup.sh
    source venv/bin/activate

Run:
    python auto_annotate_carparts.py --input ./input_images --output ./output

Note: this script also auto-checks for missing weight files on every run
(via weights_downloader.py) and auto-detects whether a GPU is available.
"""

import os
import cv2
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone

from scripts.utils.weights_downloader import ensure_all_weights

# ----------------------------------------------------------------------------
# CONFIG -- file names (auto-downloaded by setup.sh / weights_downloader.py)
# ----------------------------------------------------------------------------
GDINO_CONFIG_PATH   = "configs/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT    = "weights/groundingdino_swint_ogc.pth"
SAM2_CONFIG_PATH    = "sam2_hiera_l.yaml"
SAM2_CHECKPOINT     = "weights/sam2_hiera_large.pt"

BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

# ----------------------------------------------------------------------------
# LABEL TAXONOMY -- matches the CVAT labels block you uploaded.
# Each entry: CVAT label name -> (grounding-dino text prompt, position values or None)
# For parts with Left/Right or Front/Rear variants we run one DINO prompt per side
# so we know which side we detected, instead of guessing after the fact.
# ----------------------------------------------------------------------------
LABEL_TAXONOMY = {
    "Front Bumper":               {"prompts": {"Front": "front bumper of car"}},
    "Rear Bumper":                {"prompts": {"Rear": "rear bumper of car"}},
    "Trunk Lid":                  {"prompts": {None: "trunk lid . car boot"}},
    "Bonnet":                     {"prompts": {None: "car bonnet . hood"}},
    "Front Fender": {"prompts": {"Left": "left front fender", "Right": "right front fender"}},
    "Front Door":   {"prompts": {"Front-Left": "front left car door", "Front-Right": "front right car door"}},
    "Rear Door":    {"prompts": {"Rear-Left": "rear left car door", "Rear-Right": "rear right car door"}},
    "Rear Quarter Panel": {"prompts": {"Left": "left rear quarter panel", "Right": "right rear quarter panel"}},
    "Headlamp":     {"prompts": {"Left": "left headlamp", "Right": "right headlamp"}},
    "Taillight":    {"prompts": {"Left": "left taillight", "Right": "right taillight"}},
    "Indicator":    {"prompts": {"Left": "left turn indicator light", "Right": "right turn indicator light"}},
    "Fog Lamp":     {"prompts": {"Left": "left fog lamp", "Right": "right fog lamp"}},
    "ORVM":         {"prompts": {"Left": "left side mirror", "Right": "right side mirror"}},
    "Windscreen":         {"prompts": {"Front": "front windscreen . windshield"}},
    "Rear Windscreen":    {"prompts": {"Rear": "rear windscreen glass"}},
    "Door Glass": {"prompts": {
        "Front-Left": "front left door glass window", "Front-Right": "front right door glass window",
        "Rear-Left": "rear left door glass window", "Rear-Right": "rear right door glass window"}},
    "Roof":         {"prompts": {None: "car roof"}},
    "Wheel": {"prompts": {
        "Front-Left": "front left car wheel tyre", "Front-Right": "front right car wheel tyre",
        "Rear-Left": "rear left car wheel tyre", "Rear-Right": "rear right car wheel tyre"}},
    "Grille":       {"prompts": {None: "front grille of car"}},
    "Fuel Lid":     {"prompts": {"Left": "left fuel lid door", "Right": "right fuel lid door"}},
    "Side Skirt / Rocker Panel": {"prompts": {"Left": "left side skirt rocker panel", "Right": "right side skirt rocker panel"}},
    "Number Plate": {"prompts": {"Front": "front number plate", "Rear": "rear number plate"}},
    "Antenna":      {"prompts": {None: "car antenna"}},
    "Wheel Cap": {"prompts": {
        "Front-Left": "front left wheel cap hubcap", "Front-Right": "front right wheel cap hubcap",
        "Rear-Left": "rear left wheel cap hubcap", "Rear-Right": "rear right wheel cap hubcap"}},
    "Pillar": {"prompts": {"Left Side": "left side car pillar", "Right Side": "right side car pillar"}},
}


def build_label_meta_xml(labels_el):
    """Recreates the <labels> metadata block exactly like your sample file."""
    colors = ["#FF6D00", "#2979FF", "#00BFA5", "#FFD600", "#D50000", "#1DE9B6",
              "#F57F17", "#E040FB", "#00E5FF", "#69F0AE", "#FF4081", "#76FF03", "#FFAB40"]
    for i, (label_name, meta) in enumerate(LABEL_TAXONOMY.items()):
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = label_name
        ET.SubElement(label_el, "color").text = colors[i % len(colors)]
        ET.SubElement(label_el, "type").text = "polygon"
        attrs_el = ET.SubElement(label_el, "attributes")
        pos_values = [p for p in meta["prompts"].keys() if p is not None]
        if pos_values:
            attr_el = ET.SubElement(attrs_el, "attribute")
            ET.SubElement(attr_el, "name").text = "Position"
            ET.SubElement(attr_el, "mutable").text = "False"
            ET.SubElement(attr_el, "input_type").text = "select"
            ET.SubElement(attr_el, "default_value").text = pos_values[0]
            ET.SubElement(attr_el, "values").text = "\n".join(pos_values)


def mask_to_polygon(mask, epsilon_ratio=0.003):
    """Convert a binary SAM2 mask into a simplified polygon point list."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 30:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_ratio * peri, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    return pts


def points_to_cvat_str(pts):
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def run_pipeline(input_dir, output_dir, save_dino_visual):
    os.makedirs(output_dir, exist_ok=True)
    sam2_dir = os.path.join(output_dir, "sam2_annotated")
    os.makedirs(sam2_dir, exist_ok=True)
    dino_dir = os.path.join(output_dir, "dino_visualized")
    if save_dino_visual:
        os.makedirs(dino_dir, exist_ok=True)

    # ---- Lazy imports so the script can at least be inspected without the
    # heavy CV libs installed. Install instructions are in the file header. ----
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch

    # Auto-download any missing weight/config files before doing anything else
    print("[*] Checking model weights ...")
    ensure_all_weights(".")

    # Auto-detect GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)} -- running on GPU.")
    else:
        device = "cpu"
        print("[WARNING] No GPU detected -- running on CPU. This will be much slower.")

    print("[*] Loading Grounding DINO ...")
    gdino_model = load_model(GDINO_CONFIG_PATH, GDINO_CHECKPOINT)
    gdino_model = gdino_model.to(device)

    print("[*] Loading SAM 2 ...")
    sam2_model = build_sam2(SAM2_CONFIG_PATH, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Build the root XML matching your CVAT sample
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "auto_annotated_carparts"
    ET.SubElement(task, "mode").text = "annotation"
    labels_el = ET.SubElement(task, "labels")
    build_label_meta_xml(labels_el)
    ET.SubElement(meta, "dumped").text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    image_files = sorted([f for f in os.listdir(input_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for img_id, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        print(f"[{img_id+1}/{len(image_files)}] {fname}")

        image_source, image_tensor = load_image(img_path)
        h, w = image_source.shape[:2]

        image_el = ET.SubElement(root, "image", id=str(img_id), name=fname, width=str(w), height=str(h))

        sam2_predictor.set_image(image_source)
        dino_vis = image_source.copy()

        for label_name, meta_info in LABEL_TAXONOMY.items():
            for position, prompt in meta_info["prompts"].items():
                boxes, scores, phrases = predict(
                    model=gdino_model,
                    image=image_tensor,
                    caption=prompt,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )
                if len(boxes) == 0:
                    continue

                # boxes are cxcywh normalized -> convert to xyxy pixel
                boxes_xyxy = boxes.clone()
                boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
                boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
                boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
                boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

                # take the single highest-confidence box for this specific prompt
                best_idx = int(scores.argmax())
                box = boxes_xyxy[best_idx].cpu().numpy()

                if save_dino_visual:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(dino_vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(dino_vis, label_name, (x1, max(y1 - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

                masks, mask_scores, _ = sam2_predictor.predict(
                    box=box[None, :],
                    multimask_output=False,
                )
                mask = masks[0].astype(bool)

                pts = mask_to_polygon(mask)
                if pts is None:
                    continue

                poly_el = ET.SubElement(image_el, "polygon",
                                         label=label_name, source="auto",
                                         occluded="0",
                                         points=points_to_cvat_str(pts),
                                         z_order="0")
                if position is not None:
                    attr_el = ET.SubElement(poly_el, "attribute", name="Position")
                    attr_el.text = position

        if save_dino_visual:
            cv2.imwrite(os.path.join(dino_dir, fname), cv2.cvtColor(dino_vis, cv2.COLOR_RGB2BGR))

        # draw final polygons for visual QA
        final_vis = image_source.copy()
        for poly_el in image_el.findall("polygon"):
            pts_str = poly_el.get("points")
            pts = np.array([[float(v) for v in p.split(",")] for p in pts_str.split(";")], dtype=np.int32)
            cv2.polylines(final_vis, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(sam2_dir, fname), cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))

    # pretty-print and save XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    xml_path = os.path.join(output_dir, "annotations.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"\n[OK] annotations.xml saved -> {xml_path}")
    print(f"[OK] SAM2 annotated images  -> {sam2_dir}")
    if save_dino_visual:
        print(f"[OK] DINO visualized images -> {dino_dir}")
    else:
        print("[i] DINO visualized images were NOT saved (as requested).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing raw car images")
    parser.add_argument("--output", default="./output", help="Where to save results")
    args = parser.parse_args()

    answer = input("Save Grounding DINO annotated images? (y/n): ").strip().lower()
    save_dino = answer.startswith("y")

    run_pipeline(args.input, args.output, save_dino)
