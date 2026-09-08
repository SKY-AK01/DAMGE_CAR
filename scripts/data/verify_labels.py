#!/usr/bin/env python3
"""
verify_labels.py
----------------
Scan a dataset directory for unreadable or corrupt image files.

Uses header-only reads via `imagesize` (no pixel decode) for speed,
with ThreadPoolExecutor for parallelism.

REPORT-ONLY: does NOT delete, move, or modify any files.

Usage:
    python scripts/data/verify_labels.py --dir datasets
    python scripts/data/verify_labels.py --dir datasets/combined_carparts
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import imagesize
    _HAS_IMAGESIZE = True
except ImportError:
    from PIL import Image
    _HAS_IMAGESIZE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _check_image(path: Path) -> bool:
    """Returns True if image header can be read successfully."""
    try:
        if _HAS_IMAGESIZE:
            w, h = imagesize.get(str(path))
            return w > 0 and h > 0
        else:
            with Image.open(path) as im:
                im.verify()
            return True
    except Exception:
        return False


def verify_labels(dataset_dir: Path, num_workers: int = 8) -> dict:
    """
    Scan dataset_dir recursively for corrupt images.
    Returns dict with keys: total, valid, corrupt (list of paths).
    """
    all_images = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    total = len(all_images)

    if total == 0:
        print(f"  [INFO] No image files found in {dataset_dir}")
        return {"total": 0, "valid": 0, "corrupt": []}

    print(f"  Scanning {total} images ({num_workers} workers)...")

    corrupt = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(_check_image, p): p for p in all_images}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            if not future.result():
                corrupt.append(path)

    corrupt.sort()  # deterministic output
    valid = total - len(corrupt)
    return {"total": total, "valid": valid, "corrupt": corrupt}


def main():
    parser = argparse.ArgumentParser(description="Scan dataset for corrupt images (report-only).")
    parser.add_argument("--dir", required=True, help="Dataset directory to scan")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset_dir = Path(args.dir).resolve()
    if not dataset_dir.exists():
        print(f"[ERROR] Directory not found: {dataset_dir}")
        return

    report = verify_labels(dataset_dir, args.workers)

    print()
    print(f"  Total images : {report['total']}")
    print(f"  Valid        : {report['valid']}")
    print(f"  Corrupt      : {len(report['corrupt'])}")

    if report["corrupt"]:
        print()
        print("  Corrupt files (not deleted — remove manually if needed):")
        for p in report["corrupt"]:
            print(f"    - {p}")

    if not _HAS_IMAGESIZE:
        print()
        print("  [NOTE] Used PIL for image checks (imagesize not installed).")
        print("         Install with: pip install imagesize  for faster header-only reads.")


if __name__ == "__main__":
    main()
