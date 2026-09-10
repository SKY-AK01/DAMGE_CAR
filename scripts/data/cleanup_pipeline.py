#!/usr/bin/env python3
"""
cleanup_pipeline.py
--------------------
Removes stale pipeline outputs before a fresh run.

TWO-QUESTION interactive flow (matches orchestrator.py's interactive_cleanup_prompt):
  Q1 [y/N]   -- Delete run outputs, logs, and derived dataset artifacts?
                 (runs_comparison/, logs/, combined_carparts/, matched/, external/)
  Q2 [yes]   -- Delete datasets/raw/? Requires typing the exact word 'yes' to confirm.
                 Default: No. This is your hand-annotated ground truth — IRREPLACEABLE.

CLI flags (for automation / VM runs):
  --yes          Skip Q1 interactively and proceed with run/derived cleanup.
  --yes-raw      Skip Q2 interactively and also delete datasets/raw/ (DANGEROUS).
  --keep-runs    Exclude runs_comparison/ from Q1 targets (only wipe dataset artifacts).
  --dry-run      Show what would be deleted without actually deleting anything.

Usage:
  python scripts/data/cleanup_pipeline.py            # full interactive two-question flow
  python scripts/data/cleanup_pipeline.py --yes      # auto-confirm Q1, still ask Q2
  python scripts/data/cleanup_pipeline.py --yes --yes-raw   # non-interactive full wipe
  python scripts/data/cleanup_pipeline.py --keep-runs       # keep training outputs
  python scripts/data/cleanup_pipeline.py --dry-run         # preview only
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS     = PROJECT_ROOT / "datasets"

# ── Q1 targets (run outputs + derived dataset artifacts) ─────────────────────
Q1_TARGETS = [
    (DATASETS / "combined_carparts",   "Rebuilt every run — stale data risk"),
    (DATASETS / "matched",             "Rebuilt by matcher scripts from external/"),
    (DATASETS / "external",            "Re-downloaded/re-copied from original sources"),
    (DATASETS / "carparts-seg",        "Legacy external dataset — re-downloaded when needed"),
    (DATASETS / "custom_carparts",     "Legacy dataset — re-copied from RAW_DATASET when needed"),
    (DATASETS / "dsmlr-carparts",      "Raw DSMLR git clone — re-cloned when needed"),
    (DATASETS / "dsmlr-carparts-split","Legacy DSMLR split — regenerated when needed"),
    (PROJECT_ROOT / "runs_comparison", "Training run outputs (logs, weights, metrics CSVs)"),
    (PROJECT_ROOT / "logs",            "Pipeline log files and pipeline reports"),
    (PROJECT_ROOT / "archive",         "Old training run archives"),
]

# ── Always-protected paths (never asked about, never deleted) ─────────────────
ALWAYS_PROTECTED = [
    PROJECT_ROOT / "RAW_DATASET",      # Your hand-annotated source data — IRREPLACEABLE
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


def _delete_targets(targets, dry_run=False):
    """Delete a list of (path, reason) pairs. Returns (deleted, skipped) lists."""
    deleted = []
    skipped = []
    for path, _ in targets:
        if not path.exists():
            skipped.append(str(path))
            continue
        # Safety belt: refuse if path is inside an always-protected directory
        for protected in ALWAYS_PROTECTED:
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
    parser = argparse.ArgumentParser(
        description="Clean up stale pipeline outputs — two-question interactive flow."
    )
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-confirm Q1 (run outputs/derived data). Still asks Q2 for raw/.")
    parser.add_argument("--yes-raw", action="store_true",
                        help="Auto-confirm Q2 deletion of datasets/raw/ (DANGEROUS — irreplaceable data).")
    parser.add_argument("--keep-runs", action="store_true",
                        help="Exclude runs_comparison/ from Q1; only wipe dataset artifacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting anything.")
    args = parser.parse_args()

    q1_targets = Q1_TARGETS.copy()
    if args.keep_runs:
        q1_targets = [(p, r) for p, r in q1_targets if "runs_comparison" not in str(p)]

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print(" Pipeline Cleanup")
    print("=" * 68)

    # ══════════════════════════════════════════════════════════════════════════
    # Q1 — Run outputs, logs, derived dataset artifacts
    # ══════════════════════════════════════════════════════════════════════════
    present_q1 = [(p, r) for p, r in q1_targets if p.exists()]

    print()
    print("  [Q1] Delete old run outputs, logs, and ALL derived dataset artifacts?")
    print("       (runs_comparison/, logs/, archive/, datasets/* EXCEPT RAW_DATASET/)")
    print("       These are rebuilt/re-downloaded fresh every run. Default: No")
    print()
    if present_q1:
        print("       Will remove:")
        for path, _ in present_q1:
            size = sizeof_dir(path)
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = path
            print(f"         DELETE  {rel}  [{size}]")
    else:
        print("       (nothing to remove — all targets already absent)")
    print()

    do_q1 = False
    if args.yes:
        print("       [AUTO] --yes flag set — proceeding with deletion.")
        do_q1 = True
    elif args.dry_run:
        print("       [DRY RUN] Would delete the above (if present).")
        do_q1 = True   # allow dry-run path to show what would happen
    elif present_q1:
        try:
            ans = input("       Delete run outputs? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        do_q1 = ans in ("y", "yes")
        if not do_q1:
            print("       [SKIP] Keeping existing run outputs.")

    if do_q1 and not args.dry_run:
        for path, _ in present_q1:
            print(f"  [DELETE] {path}")
            shutil.rmtree(path, ignore_errors=True)
        if present_q1:
            print("  [OK] Run outputs cleaned.")

    # ══════════════════════════════════════════════════════════════════════════
    # Q2 — datasets/raw/  (requires exact word 'yes')
    # ══════════════════════════════════════════════════════════════════════════
    raw_dir  = DATASETS / "raw"
    raw_size = sizeof_dir(raw_dir)

    print()
    print("  [Q2] Delete datasets/raw/? (your hand-annotated ground truth — IRREPLACEABLE!)")
    print(f"       Current size: {raw_size}")
    print("       Default: No. You must type 'yes' (not just 'y') to confirm.")
    print()

    do_q2 = False
    if args.yes_raw:
        print("       [AUTO] --yes-raw flag set — proceeding with deletion.")
        do_q2 = True
    elif args.dry_run:
        print(f"       [DRY RUN] Would remove: datasets/raw/  [{raw_size}]")
    elif raw_dir.exists() and any(raw_dir.rglob("*.*")):
        print(f"       Will remove: datasets/raw/  [{raw_size}]")
        try:
            ans = input("       Type 'yes' to DELETE datasets/raw/ (or anything else to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans == "yes":
            do_q2 = True
        else:
            print("       [SKIP] Keeping datasets/raw/ (smart choice).")
    else:
        print("       datasets/raw/ is empty or absent — nothing to delete.")

    if do_q2 and not args.dry_run:
        print(f"  [DELETE] {raw_dir}")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("  [OK] datasets/raw/ removed.")

    # ── Footer ────────────────────────────────────────────────────────────────
    print()
    print("=" * 68)

    if args.dry_run:
        print("[DRY RUN] No files were actually deleted.")
        return

    total_deleted = (len(present_q1) if do_q1 else 0) + (1 if do_q2 and raw_dir.exists() else 0)
    if total_deleted == 0 and not do_q1 and not do_q2:
        print("[OK] Nothing deleted.")
    else:
        print(f"[OK] Cleanup complete.")


if __name__ == "__main__":
    main()
