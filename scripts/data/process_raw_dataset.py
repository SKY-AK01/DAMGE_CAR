"""
process_raw_dataset.py

Expected RAW_DATASET structure:

RAW_DATASET/
|--- IMAGES/
|   |--- image1.jpg
|   |--- image2.jpg
|   |__- ...
|__- XML/
    |--- combined_annotations.xml
    |__- ...

The script:
1. Reads images from RAW_DATASET/IMAGES
2. Reads XML files from RAW_DATASET/XML
3. Moves them into datasets/custom_carparts/
4. Preserves RAW_DATASET
"""

import sys

# Ensure UTF-8 output on Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import shutil
import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def process_raw_dataset(
    raw_dir="RAW_DATASET",
    target_dir="datasets/custom_carparts",
):
    raw_path = Path(raw_dir)
    target_path = Path(target_dir)

    images_path = raw_path / "IMAGES"
    xml_path = raw_path / "XML"

    if not raw_path.exists():
        print(f"[SKIP] RAW_DATASET directory does not exist: {raw_path}")
        return

    print("==============================================")
    print(" Processing RAW_DATASET")
    print("==============================================")

    # ---------------------------------------------------------
    # Check directories
    # ---------------------------------------------------------

    if not images_path.exists():
        print(f"[ERROR] Images directory not found: {images_path}")
        return

    if not xml_path.exists():
        print(f"[ERROR] XML directory not found: {xml_path}")
        return

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------

    image_files = [
        f
        for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # ---------------------------------------------------------
    # Find XML files
    # ---------------------------------------------------------

    xml_files = [
        f
        for f in xml_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".xml"
    ]

    print(f"[+] Images found : {len(image_files)}")
    print(f"[+] XML files found : {len(xml_files)}")

    if not image_files and not xml_files:
        print("[INFO] Nothing to process.")
        return

    # ---------------------------------------------------------
    # Create target directories
    # ---------------------------------------------------------

    target_images = target_path / "images"
    target_images.mkdir(parents=True, exist_ok=True)

    target_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Move XML files
    # ---------------------------------------------------------

    for xml_file in xml_files:
        destination = target_path / xml_file.name

        print(
            f"[*] Moving XML: "
            f"{xml_file} -> {destination}"
        )

        shutil.move(
            str(xml_file),
            str(destination),
        )

    # ---------------------------------------------------------
    # Move images
    # ---------------------------------------------------------

    for image_file in image_files:
        destination = target_images / image_file.name

        print(
            f"[*] Moving image: "
            f"{image_file.name} -> {destination}"
        )

        shutil.move(
            str(image_file),
            str(destination),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("")
    print("==============================================")
    print("[OK] RAW_DATASET processing complete")
    print("==============================================")
    print(f"Images moved : {len(image_files)}")
    print(f"XML moved    : {len(xml_files)}")
    print(f"Output       : {target_path}")
    print("")
    print("RAW_DATASET directory preserved.")
    print("==============================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process RAW_DATASET with IMAGES/ and XML/ folders."
    )

    parser.add_argument(
        "--raw-dir",
        default="RAW_DATASET",
        help="Path to RAW_DATASET",
    )

    parser.add_argument(
        "--target-dir",
        default="datasets/custom_carparts",
        help="Output dataset directory",
    )

    args = parser.parse_args()

    process_raw_dataset(
        raw_dir=args.raw_dir,
        target_dir=args.target_dir,
    )