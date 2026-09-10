#!/usr/bin/env python3
"""
fix_labels_dir.py - One-time fix for labels_yolo -> labels rename
==================================================================
The old setup_dataset_structure.py created 'labels_yolo/' instead of 'labels/'.
This script renames it to fix the issue without re-processing everything.

Usage:
    python fix_labels_dir.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
raw_dir = PROJECT_ROOT / "datasets" / "raw"
old_labels = raw_dir / "labels_yolo"
new_labels = raw_dir / "labels"

print("=" * 70)
print(" FIXING LABEL DIRECTORY NAME")
print("=" * 70)

if not old_labels.exists():
    print(f"[SKIP] {old_labels} does not exist")
    if new_labels.exists():
        print(f"[OK] {new_labels} already exists - nothing to fix!")
    else:
        print(f"[ERROR] Neither labels_yolo/ nor labels/ exists in datasets/raw/")
    print("=" * 70)
    exit(0)

if new_labels.exists():
    print(f"[WARN] Both {old_labels} and {new_labels} exist")
    print(f"       Removing old labels_yolo/ since labels/ is already present...")
    shutil.rmtree(old_labels)
    print("[OK] Removed labels_yolo/")
else:
    print(f"[*] Renaming {old_labels} -> {new_labels}")
    old_labels.rename(new_labels)
    print("[OK] Renamed successfully")

# Verify
train_labels = list((new_labels / "train").glob("*.txt")) if (new_labels / "train").exists() else []
val_labels = list((new_labels / "val").glob("*.txt")) if (new_labels / "val").exists() else []

print()
print("Verification:")
print(f"  train labels: {len(train_labels)}")
print(f"  val labels:   {len(val_labels)}")
print()
print("=" * 70)
print("[OK] Fix complete! Re-run combine_datasets.py to propagate labels.")
print("=" * 70)
