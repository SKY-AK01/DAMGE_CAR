"""
sanitize_and_validate_dataset.py
----------------------------------
Sanitizes and validates dataset images and XML annotation entries before training:
1. Scans all images in the specified dataset directory (or RAW_DATASET / datasets/).
2. Attempts to open and verify each image using PIL and OpenCV.
3. Fixes minor issues (e.g. converting/saving truncated JPEG/PNG images if repairable).
4. If an image is unrepairable/corrupt, removes it to prevent training crashes and logs it clearly.
5. Validates XML files to ensure referenced image entries match valid images, removing references to missing/corrupt images.

Saves a detailed log file under logs/00_dataset_sanitization.log.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from PIL import Image, ImageFile
import cv2

# Allow PIL to load truncated images where possible
ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_valid_and_repair(img_path: Path) -> bool:
    """Checks if an image can be loaded and read cleanly. Attempts basic repair if needed."""
    try:
        # Step 1: PIL verification
        with Image.open(img_path) as img:
            img.verify()
        
        # Step 2: Re-open and test pixel load & decode with OpenCV
        with Image.open(img_path) as img:
            img_converted = img.convert("RGB")
            # Save back clean if truncated/minor headers missing
            img_converted.save(img_path)
            
        cv_img = cv2.imread(str(img_path))
        if cv_img is None or cv_img.size == 0:
            return False
        return True
    except Exception as e:
        # Attempt OpenCV recovery
        try:
            cv_img = cv2.imread(str(img_path))
            if cv_img is not None and cv_img.size > 0:
                cv2.imwrite(str(img_path), cv_img)
                return True
        except Exception:
            pass
        return False


def sanitize_directory(dataset_dir: Path, log_file) -> tuple[int, int, int]:
    valid_count = 0
    repaired_count = 0
    corrupt_count = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    if not image_files:
        log_file.write(f"[INFO] No image files found in {dataset_dir}\n")
        return (0, 0, 0)

    log_file.write(f"[*] Scanning {len(image_files)} image(s) in {dataset_dir} ...\n")

    for img_path in image_files:
        rel_path = img_path.relative_to(dataset_dir.parent if dataset_dir.parent else dataset_dir)
        if is_image_valid_and_repair(img_path):
            valid_count += 1
        else:
            corrupt_count += 1
            log_file.write(f"  [CORRUPT - REMOVED] {rel_path}\n")
            print(f"  [CORRUPT - REMOVED] {rel_path}")
            try:
                img_path.unlink()
            except Exception as err:
                log_file.write(f"    Failed to delete corrupt file {img_path}: {err}\n")

    # Sanitize XML annotations if present
    xml_files = list(dataset_dir.rglob("*.xml"))
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            removed_xml_entries = 0
            for img_elem in root.findall(".//image"):
                img_name = img_elem.get("name")
                # Check if referenced image exists
                matching_imgs = list(dataset_dir.rglob(img_name))
                if not matching_imgs:
                    root.remove(img_elem)
                    removed_xml_entries += 1
            if removed_xml_entries > 0:
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)
                log_file.write(f"  [XML CLEANED] Removed {removed_xml_entries} missing image entries from {xml_path.name}\n")
        except Exception as e:
            log_file.write(f"  [XML ERROR] Could not parse {xml_path}: {e}\n")

    return (valid_count, repaired_count, corrupt_count)


def main():
    parser = argparse.ArgumentParser(description="Sanitize and remove corrupt images/XML entries before training")
    parser.add_argument("--dataset-dir", default="RAW_DATASET", help="Directory containing images and XML files")
    parser.add_argument("--log-dir", default="logs", help="Directory to save sanitization log")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    target_dir = project_root / args.dataset_dir
    log_dir = project_root / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "dataset_sanitization.log"

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n==============================================\n")
        log_file.write(" Dataset Sanitization & Repair Run\n")
        log_file.write("==============================================\n")

        print("==============================================")
        print(" Dataset Sanitization & Corruption Cleaning")
        print("==============================================")

        if not target_dir.exists():
            msg = f"[SKIP] Target directory '{args.dataset_dir}' does not exist.\n"
            log_file.write(msg)
            print(msg)
            return

        valid, repaired, corrupt = sanitize_directory(target_dir, log_file)

        summary = (
            f"\n[SANITIZATION SUMMARY]\n"
            f"  Valid/Repaired Images: {valid}\n"
            f"  Corrupt Images Removed: {corrupt}\n"
            f"  Log File Saved: {log_path}\n"
            f"==============================================\n"
        )
        log_file.write(summary)
        print(summary)


if __name__ == "__main__":
    main()
