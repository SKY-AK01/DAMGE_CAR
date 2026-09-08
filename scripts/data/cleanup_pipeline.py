#!/usr/bin/env python3
"""
cleanup_pipeline.py
--------------------
Removes stale pipeline outputs before a fresh run.

NEVER touches:
  - datasets/raw/          <- hand-annotated ground truth, inviolable

Removes (after confirmation):
  - datasets/combined_carparts/   <- rebuilt every run anyway
  - datasets/matched/             <- rebuilt by matcher scripts
  - datasets/external/            <- re-downloaded/re-copied by pipeline
  - runs_comparison/              <- training run outputs (logs, weights, metrics)
  - logs/                         <- pipeline log files

Usage:
  python scripts/data/cleanup_pipeline.py          # interactive confirmation
  python scripts/data/cleanup_pipeline.py --yes    # non-interactive (CI/scripted use)
  python scripts/data/cleanup_pipeline.py --keep-runs  # keep training outputs, only wipe datasets
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS     = PROJECT_ROOT / "datasets"

# ── What will be deleted ─────────────────────────────────────────────────────
TARGETS = [
    (DATASETS / "combined_carparts", "Rebuilt every run — stale data risk"),
    (DATASETS / "matched",           "Rebuilt by matcher scripts from external/"),
    (DATASETS / "external",          "Re-downloaded/re-copied from original sources"),
    (PROJECT_ROOT / "runs_comparison", "Training run outputs (logs, weights, metrics CSVs)"),
    (PROJECT_ROOT / "logs",          "Pipeline log files and pipeline reports"),
]

# ── What is NEVER touched ────────────────────────────────────────────────────
PROTECTED = [
    DATASETS / "raw",
    DATASETS / "carparts-seg",        # legacy location — safe to keep
    DATASETS / "custom_carparts",     # legacy location — safe to keep
    DATASETS / "dsmlr-carparts",      # raw DSMLR git clone
    DATASETS / "dsmlr-carparts-split",# legacy split — safe to keep
    PROJECT_ROOT / "RAW_DATASET",
]


def sizeof_dir(path: Path) -> str:
    if not path.exists():
        return "(not present)"
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total < 1024:
            return f"{total} B"
        elif total < 1024 ** 2:
            return f"{total/1024:.1f} KB"
        elif total < 1024 ** 3:
            return f"{total/1024**2:.1f} MB"
        else:
            return f"{total/1024**3:.2f} GB"
    except Exception:
        return "(unknown size)"


def run_cleanup(targets, dry_run=False):
    deleted = []
    skipped = []
    for path, reason in targets:
        if not path.exists():
            skipped.append(str(path))
            continue
        # Safety: never delete protected paths
        for protected in PROTECTED:
            try:
                path.resolve().relative_to(protected.resolve())
                print(f"[SAFETY] Refusing to delete protected path: {path}")
                skipped.append(str(path))
                break
            except ValueError:
                pass
        else:
            if not dry_run:
                shutil.rmtree(path)
            deleted.append(str(path))
    return deleted, skipped


def main():
    parser = argparse.ArgumentParser(description="Clean up stale pipeline outputs (never touches datasets/raw/).")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip interactive confirmation and proceed immediately.")
    parser.add_argument("--keep-runs", action="store_true",
                        help="Keep runs_comparison/ training outputs; only wipe dataset artifacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting anything.")
    args = parser.parse_args()

    targets = TARGETS.copy()
    if args.keep_runs:
        targets = [(p, r) for p, r in targets if "runs_comparison" not in str(p)]

    # ── Print what will be deleted ────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Pipeline Cleanup — The following will be DELETED:")
    print("=" * 65)
    any_present = False
    for path, reason in targets:
        exists = path.exists()
        size   = sizeof_dir(path) if exists else "(not present)"
        flag   = "  DELETE" if exists else "  skip  "
        print(f"{flag}  {path.relative_to(PROJECT_ROOT)}  [{size}]")
        print(f"         Reason: {reason}")
        if exists:
            any_present = True

    print()
    print("=" * 65)
    print("The following are PROTECTED and will NEVER be touched:")
    print("=" * 65)
    for p in PROTECTED:
        exists = "exists" if p.exists() else "not present"
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        print(f"  SAFE  {rel}  ({exists})")

    if not any_present:
        print()
        print("[OK] Nothing to clean up — all target directories are already absent.")
        return

    if args.dry_run:
        print()
        print("[DRY RUN] No files were deleted.")
        return

    # ── Confirm ───────────────────────────────────────────────────────────────
    if not args.yes:
        print()
        answer = input("Proceed with deletion? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("[CANCELLED] Nothing deleted.")
            sys.exit(0)

    # ── Execute ───────────────────────────────────────────────────────────────
    print()
    deleted, skipped = run_cleanup(targets)
    for p in deleted:
        print(f"[DELETED] {p}")
    for p in skipped:
        print(f"[SKIP]    {p} (not present)")

    print()
    print(f"[OK] Cleanup complete. {len(deleted)} directories removed.")
    print("     datasets/raw/ and all protected paths were not touched.")


if __name__ == "__main__":
    main()
