#!/usr/bin/env python3
"""
combine_datasets.py
--------------------
Builds datasets/combined_carparts/ from:
  - datasets/raw/                      (deduplicated HITL annotations)
  - datasets/matched/carparts-seg/     (Ultralytics open-source, taxonomy-verified)
  - datasets/matched/dsmlr/            (DSMLR open-source, converted to 23-class YOLO)
  - any additional folder under datasets/matched/ dropped in the future

IMPORTANT: The output directory is FULLY WIPED and rebuilt on every run.
This prevents stale data from previous runs contaminating the combined set.

Output layout:
  datasets/combined_carparts/
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml                 (written fresh each run)

Filename convention: {source_name}_{original_filename}
  e.g.  raw_Coupe_000003.jpg
        matched_carparts-seg_car10_jpg.rf.01c1.jpg
        matched_dsmlr_train1.jpg

This prefix is permanent — it is the only way to trace a training image
back to its source after the combine step.

Per-source count is printed at the end of every run.
"""

import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS     = PROJECT_ROOT / "datasets"
OUT_DIR      = DATASETS / "combined_carparts"


def _copy_file(args):
    src, dst = args
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return None
    except Exception as e:
        return f"{src} -> {dst}: {e}"


def _wipe_output(out_path: Path):
    """
    Fully delete and recreate the output directory.
    Only touches datasets/combined_carparts/ — never touches raw/, external/, or matched/.
    """
    if out_path.exists():
        print(f"[*] Wiping stale output: {out_path} ...")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    print(f"[OK] Output directory reset: {out_path}")


def _collect_sources():
    """
    Returns list of (source_label, dir_path) for every source that exists on disk.
    Order: raw/ first, then matched/* in sorted order.
    """
    sources = []

    raw = DATASETS / "raw"
    if raw.exists():
        sources.append(("raw", raw))
    else:
        print(f"[WARN] datasets/raw/ not found — run setup_dataset_structure.py first.")

    matched_root = DATASETS / "matched"
    if matched_root.exists():
        for sub in sorted(matched_root.iterdir()):
            if sub.is_dir():
                sources.append((f"matched_{sub.name}", sub))

    return sources


def combine(out_dir: Path = OUT_DIR, num_workers: int = 8):
    _wipe_output(out_dir)

    sources = _collect_sources()
    if not sources:
        print("[ERROR] No source directories found. Run setup_dataset_structure.py and matcher scripts first.")
        return

    copy_jobs = []
    source_counts = {}   # source_label -> count
    yaml_names_lines = []

    for source_label, dp in sources:
        if not dp.exists():
            print(f"  [skip] {dp} not found")
            continue

        print(f"[*] Scanning {source_label} ({dp}) ...")
        count = 0

        for split in ["train", "val", "test"]:
            img_dir = dp / "images" / split
            lbl_dir = dp / "labels" / split

            if img_dir.exists():
                for f in sorted(img_dir.glob("*.*")):
                    if f.is_file():
                        dst = out_dir / "images" / split / f"{source_label}_{f.name}"
                        copy_jobs.append((f, dst))
                        count += 1

            if lbl_dir.exists():
                for f in sorted(lbl_dir.glob("*.txt")):
                    dst = out_dir / "labels" / split / f"{source_label}_{f.name}"
                    copy_jobs.append((f, dst))
            elif img_dir.exists():
                # Fallback: labels mixed into image dir (uncommon but safe)
                for f in sorted(img_dir.glob("*.txt")):
                    dst = out_dir / "labels" / split / f"{source_label}_{f.name}"
                    copy_jobs.append((f, dst))

        # Inherit names block from first source that has a YAML
        if not yaml_names_lines:
            yaml_files = list(dp.glob("*.yaml"))
            if yaml_files:
                with open(yaml_files[0]) as yf:
                    in_names = False
                    for line in yf:
                        if line.startswith("names:"):
                            in_names = True
                            yaml_names_lines.append(line)
                        elif in_names:
                            if line.startswith((" ", "\t")):
                                yaml_names_lines.append(line)
                            else:
                                in_names = False

        source_counts[source_label] = count

    # Parallel copy
    print(f"\n[*] Copying {len(copy_jobs)} files ({num_workers} workers) ...")
    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_job = {executor.submit(_copy_file, job): job for job in copy_jobs}
        for future in as_completed(future_to_job):
            err = future.result()
            if err:
                errors.append(err)

    if errors:
        print(f"[WARN] {len(errors)} copy error(s):")
        for e in errors[:10]:
            print(f"  {e}")

    # Write data.yaml
    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
    )
    if yaml_names_lines:
        yaml_content += "".join(yaml_names_lines)
    with open(out_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    # ── Per-source breakdown (always printed) ─────────────────────────────
    total = sum(source_counts.values())
    print()
    print("=" * 52)
    print("Dataset source breakdown:")
    max_label = max((len(lbl) for lbl in source_counts), default=10)
    for lbl, cnt in source_counts.items():
        print(f"  {lbl:<{max_label+2}}: {cnt:>5} images")
    print("  " + "-" * (max_label + 11))
    print(f"  {'TOTAL':<{max_label+2}}: {total:>5} images")
    print("=" * 52)

    # Per-split summary
    print()
    for split in ["train", "val", "test"]:
        img_dir = out_dir / "images" / split
        n = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        if n:
            print(f"  {split:<6}: {n} images")

    print(f"\n[OK] combined_carparts rebuilt at: {out_dir}")
    return source_counts


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Rebuild combined_carparts from raw/ + matched/* (always wipes output first)."
    )
    parser.add_argument("--out_dir", default=str(OUT_DIR),
                        help="Output directory (default: datasets/combined_carparts)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel copy workers (default: 8)")
    args = parser.parse_args()

    combine(out_dir=Path(args.out_dir), num_workers=args.workers)


if __name__ == "__main__":
    main()
