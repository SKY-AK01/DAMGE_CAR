#!/usr/bin/env python3
"""
Quick diagnostic: Check if combined_carparts has valid label files.
"""
from pathlib import Path

combined = Path("datasets/combined_carparts")

for split in ["train", "val"]:
    img_dir = combined / "images" / split
    lbl_dir = combined / "labels" / split
    
    if not img_dir.exists():
        print(f"[MISSING] {img_dir}")
        continue
    if not lbl_dir.exists():
        print(f"[MISSING] {lbl_dir}")
        continue
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    labels = list(lbl_dir.glob("*.txt"))
    
    print(f"\n[{split.upper()}]")
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")
    
    # Check a few label files
    non_empty = 0
    for lbl in labels[:5]:
        content = lbl.read_text().strip()
        lines = [l for l in content.split("\n") if l.strip()]
        if lines:
            non_empty += 1
            print(f"    {lbl.name}: {len(lines)} annotations")
            # Show first line format
            first_line = lines[0].strip().split()
            if len(first_line) >= 7:
                print(f"      Format: class_id={first_line[0]}, {len(first_line)-1} coords")
            else:
                print(f"      [WARN] Too few values: {first_line}")
    
    print(f"  Non-empty labels (sample): {non_empty}/5")
