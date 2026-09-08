#!/usr/bin/env python3
"""
match_carparts_seg.py
----------------------
"Matcher" for the Ultralytics carparts-seg dataset.

carparts-seg already uses exactly the same 23-class YOLO taxonomy as raw/.
This script copies images and labels from datasets/external/carparts-seg/
into datasets/matched/carparts-seg/, verifying that all class IDs fall
within [0..22] and logging any out-of-range IDs found.

Output layout:
  datasets/matched/carparts-seg/images/{train,val,test}/
  datasets/matched/carparts-seg/labels/{train,val,test}/
  datasets/matched/carparts-seg/data.yaml

Safe to re-run: skips files that already exist.
"""

import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC  = PROJECT_ROOT / "datasets" / "external" / "carparts-seg"
DST  = PROJECT_ROOT / "datasets" / "matched"  / "carparts-seg"

VALID_IDS = set(range(23))   # 0..22

# 23-class taxonomy (for the data.yaml)
NAMES = {
    0:"back_bumper", 1:"back_door",      2:"back_glass",      3:"back_left_door",
    4:"back_left_light", 5:"back_light", 6:"back_right_door", 7:"back_right_light",
    8:"front_bumper",    9:"front_door", 10:"front_glass",    11:"front_left_door",
    12:"front_left_light",13:"front_light",14:"front_right_door",15:"front_right_light",
    16:"hood",17:"left_mirror",18:"object",19:"right_mirror",
    20:"tailgate",21:"trunk",22:"wheel",
}


def verify_label(lbl_path: Path):
    bad = []
    with open(lbl_path) as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                if cid not in VALID_IDS:
                    bad.append((i, cid))
            except ValueError:
                bad.append((i, parts[0]))
    return bad


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path):
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if not img_dst.exists():
        shutil.copy2(img_src, img_dst)
    if lbl_src.exists() and not lbl_dst.exists():
        shutil.copy2(lbl_src, lbl_dst)
    return None


def main():
    if not SRC.exists():
        print(f"[ERROR] Source not found: {SRC}")
        print("        Run setup_dataset_structure.py first.")
        return

    print("=" * 60)
    print("Matcher: carparts-seg  ->  matched/carparts-seg/")
    print("=" * 60)
    print("Taxonomy: carparts-seg uses same 23-class IDs as raw/ — direct copy.")

    jobs = []
    bad_labels = []
    counts = {}

    for split in ["train", "val", "test"]:
        img_src_dir = SRC / "images" / split
        lbl_src_dir = SRC / "labels" / split
        if not img_src_dir.exists():
            continue

        imgs = list(img_src_dir.glob("*.*"))
        counts[split] = len(imgs)

        for img in imgs:
            lbl_src = lbl_src_dir / (img.stem + ".txt")
            img_dst = DST / "images" / split / img.name
            lbl_dst = DST / "labels" / split / (img.stem + ".txt")
            jobs.append((img, lbl_src, img_dst, lbl_dst))

            # Verify label IDs while we're here
            if lbl_src.exists():
                bad = verify_label(lbl_src)
                if bad:
                    bad_labels.append((lbl_src, bad))

    if bad_labels:
        print(f"\n[WARN] {len(bad_labels)} label files have out-of-range class IDs:")
        for path, issues in bad_labels[:10]:
            print(f"  {path.name}: {issues}")
    else:
        print("[OK] All label class IDs verified within [0..22]")

    print(f"\nCopying {len(jobs)} image/label pairs ({sum(counts.values())} images across splits)...")
    errors = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(copy_pair, *job): job for job in jobs}
        for fut in as_completed(futures):
            err = fut.result()
            if err:
                errors.append(err)

    if errors:
        print(f"[WARN] {len(errors)} copy errors")
    
    # Write data.yaml
    names_block = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    yaml_content = (
        f"path: {DST.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n{names_block}\n"
    )
    with open(DST / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("\n[OK] matched/carparts-seg/ complete:")
    for split, n in counts.items():
        print(f"     {split}: {n} images")
    total = sum(counts.values())
    print(f"     TOTAL: {total}")


if __name__ == "__main__":
    main()
