#!/usr/bin/env python3
"""
clean_all.py - Complete pipeline cleanup (protects RAW_DATASET)
================================================================
Deletes ALL generated/downloaded files EXCEPT your hand-annotated RAW_DATASET.

Safe to run before every training run to ensure a clean state.

Usage:
    python clean_all.py              # Interactive (asks for confirmation)
    python clean_all.py --yes        # Non-interactive (auto-confirm)
    python clean_all.py --dry-run    # Show what would be deleted
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = PROJECT_ROOT / "datasets"

# What gets deleted (everything generated/downloaded)
CLEANUP_TARGETS = [
    PROJECT_ROOT / "runs_comparison",
    PROJECT_ROOT / "logs", 
    PROJECT_ROOT / "archive",
    DATASETS / "combined_carparts",
    DATASETS / "matched",
    DATASETS / "external",
    DATASETS / "raw",
    DATASETS / "carparts-seg",
    DATASETS / "custom_carparts",
    DATASETS / "dsmlr-carparts",
    DATASETS / "dsmlr-carparts-split",
]

# NEVER deleted (your source data)
PROTECTED = [
    PROJECT_ROOT / "RAW_DATASET",
]


def sizeof_dir(path: Path) -> str:
    """Calculate human-readable directory size."""
    if not path.exists():
        return "(not present)"
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total < 1024**2:
            return f"{total/1024:.1f} KB"
        elif total < 1024**3:
            return f"{total/1024**2:.1f} MB"
        else:
            return f"{total/1024**3:.2f} GB"
    except Exception:
        return "(unknown)"


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline cleanup (protects RAW_DATASET)"
    )
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-confirm deletion (non-interactive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print(" COMPLETE PIPELINE CLEANUP")
    print("=" * 70)
    print()
    print("  This will delete ALL generated/downloaded files:")
    print("    • Training outputs (runs_comparison/, archive/, logs/)")
    print("    • ALL dataset folders (datasets/* EXCEPT RAW_DATASET/)")
    print()
    print("  ✅ PROTECTED (will NOT be deleted):")
    for p in PROTECTED:
        if p.exists():
            size = sizeof_dir(p)
            print(f"      {p.name}/  [{size}]")
    print()
    
    existing = [p for p in CLEANUP_TARGETS if p.exists()]
    
    if not existing:
        print("  Nothing to delete — workspace is already clean!")
        print("=" * 70)
        return
    
    print("  ❌ WILL DELETE:")
    total_size = 0
    for p in existing:
        size_str = sizeof_dir(p)
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        print(f"      {rel}  [{size_str}]")
        # Estimate total (rough)
        if "MB" in size_str:
            total_size += float(size_str.split()[0])
        elif "GB" in size_str:
            total_size += float(size_str.split()[0]) * 1024
    
    print()
    print(f"  Total space to free: ~{total_size:.1f} MB")
    print("=" * 70)
    print()

    if args.dry_run:
        print("[DRY RUN] No files were actually deleted.")
        return

    # Confirm deletion
    if not args.yes:
        try:
            ans = input("  Proceed with cleanup? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[CANCELLED] No files deleted.")
            return
        
        if ans not in ("y", "yes"):
            print("[CANCELLED] No files deleted.")
            return

    # Delete
    print()
    deleted_count = 0
    for p in existing:
        print(f"  [DELETE] {p}")
        try:
            shutil.rmtree(p)
            deleted_count += 1
        except Exception as e:
            print(f"    [WARNING] Could not delete: {e}")
    
    print()
    print("=" * 70)
    print(f"[OK] Cleanup complete! Deleted {deleted_count} items (~{total_size:.1f} MB freed)")
    print("=" * 70)


if __name__ == "__main__":
    main()
