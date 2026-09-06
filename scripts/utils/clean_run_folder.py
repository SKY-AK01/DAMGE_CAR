#!/usr/bin/env python3
"""
clean_run_folder.py
-------------------
Archives the training pipeline outputs and artifacts into a compressed ZIP file,
stores it under the archive/ directory, and cleans up workspace folders so that
the codebase is fresh and ready for the next training run.

Usage:
    python scripts/utils/clean_run_folder.py
    python scripts/utils/clean_run_folder.py --archive-dir archive --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
import os
from datetime import datetime
from pathlib import Path


DEFAULT_TARGET_FOLDERS = [
    "datasets",
    "runs_comparison",
    "logs",
    "test_result",
    "test_result1",
]

# Kept separate from DEFAULT_TARGET_FOLDERS on purpose: this is your raw,
# unprocessed source data (not a derived/regenerable artifact like datasets/
# runs_comparison/logs), so it should never be silently swept up by default.
# Whether it gets included is decided interactively (or via --keep-raw-dataset
# / --remove-raw-dataset for non-interactive use) in main().
RAW_DATASET_FOLDER = "RAW_DATASET"

DEFAULT_TARGET_FILES = [
    "last_yolo_weights_path.txt",
]


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def confirm_raw_dataset_removal(raw_dataset_path: Path) -> bool:
    """Interactively asks whether RAW_DATASET should also be archived+removed.
    Returns False (keep it) on any non-'y' answer, including empty input --
    raw source data should be opt-in to delete, not opt-out."""
    if not raw_dataset_path.exists():
        print(f"[SKIP] RAW_DATASET not found at: {raw_dataset_path}")
        return False

    print()
    print(f"[?] RAW_DATASET folder found at: {raw_dataset_path}")
    print("    This is your raw, unprocessed source data (not a regenerable")
    print("    artifact like datasets/runs_comparison/logs).")
    answer = input("    Remove it too? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def clean_run_folder(
    target_dirs: list[str],
    target_files: list[str],
    archive_dir_str: str,
    archive_name: str | None = None,
    dry_run: bool = False,
) -> Path | None:
    project_root = Path.cwd().resolve()
    
    # If executed from inside scripts/utils, project_root should be the repo root
    if project_root.name == "utils" or project_root.name == "scripts":
        while project_root.name in ["utils", "scripts"]:
            project_root = project_root.parent

    archive_base = resolve_path(archive_dir_str, project_root)
    archive_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_stem = archive_name or f"run_archive_{timestamp}"
    zip_path = archive_base / f"{archive_stem}.zip"

    print("==============================================")
    print(" CAR Pipeline - Clean & Archive Run Folder")
    print("==============================================")
    print(f"Project root: {project_root}")
    print(f"Target Archive ZIP: {zip_path}")
    print("----------------------------------------------")

    items_to_archive: list[tuple[Path, str]] = []

    # Collect folder items
    for target in target_dirs:
        target_path = resolve_path(target, project_root)
        if not target_path.exists():
            print(f"[SKIP] Non-existent folder: {target_path.relative_to(project_root)}")
            continue
        if not target_path.is_dir():
            print(f"[SKIP] Non-directory path: {target_path.relative_to(project_root)}")
            continue

        for item in target_path.rglob("*"):
            if item.is_file():
                arcname = str(item.relative_to(project_root))
                items_to_archive.append((item, arcname))

    # Collect file items
    for target in target_files:
        target_path = resolve_path(target, project_root)
        if target_path.exists() and target_path.is_file():
            items_to_archive.append((target_path, str(target_path.relative_to(project_root))))

    if not items_to_archive:
        print("[INFO] No files found to archive. Workspace is already clean.")
        return None

    print(f"[+] Found {len(items_to_archive)} file(s) to archive.")

    if dry_run:
        print("\n[DRY RUN] Would create ZIP archive containing:")
        for item, arcname in items_to_archive[:10]:
            print(f"  - {arcname}")
        if len(items_to_archive) > 10:
            print(f"  ... and {len(items_to_archive) - 10} more files.")
        print("[DRY RUN] Would remove target folders and files after zip creation.")
        return zip_path

    # Create ZIP archive
    print(f"[*] Creating ZIP archive at: {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in items_to_archive:
            zipf.write(file_path, arcname)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Archive created successfully ({zip_size_mb:.2f} MB)")

    # Cleaning workspace
    print("[*] Cleaning up workspace ...")
    
    # 1. Delete files that were archived, but NEVER delete .zip files
    for file_path, arcname in items_to_archive:
        if file_path.exists() and file_path.is_file():
            if file_path.suffix.lower() == '.zip':
                print(f"  - Keeping zip file: {file_path.relative_to(project_root)}")
                continue
            file_path.unlink()

    # 2. Delete empty directories within the target folders
    for target in target_dirs:
        target_path = resolve_path(target, project_root)
        if target_path.exists() and target_path.is_dir():
            # Bottom-up directory traversal to remove empty subdirectories
            for dir_path, dir_names, file_names in os.walk(target_path, topdown=False):
                # Try to remove the directory if it's empty
                try:
                    Path(dir_path).rmdir()
                except OSError:
                    pass  # Directory not empty (e.g., contains a .zip file)
            
            # Ensure the top-level target directory exists for the next run
            target_path.mkdir(parents=True, exist_ok=True)

    for target in target_files:
        target_path = resolve_path(target, project_root)
        if target_path.exists() and target_path.is_file():
            if target_path.suffix.lower() != '.zip':
                print(f"  - Removing file: {target_path.relative_to(project_root)}")
                target_path.unlink()

    print("==============================================")
    print(f"[SUCCESS] Clean up complete! Fresh workspace ready for next run.")
    print(f"Archive saved at: {zip_path}")
    print("==============================================")

    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive and clear training pipeline folders for a fresh run")
    parser.add_argument(
        "--target-dirs",
        nargs="*",
        default=DEFAULT_TARGET_FOLDERS,
        help="Folders to archive and clear",
    )
    parser.add_argument(
        "--archive-dir",
        default="archive",
        help="Folder where archived ZIPs will be saved",
    )
    parser.add_argument("--archive-name", default=None, help="Optional custom archive name (without .zip)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without modifying disk")

    raw_group = parser.add_mutually_exclusive_group()
    raw_group.add_argument(
        "--remove-raw-dataset",
        action="store_true",
        help="Also archive+remove RAW_DATASET without asking (non-interactive use).",
    )
    raw_group.add_argument(
        "--keep-raw-dataset",
        action="store_true",
        help="Never touch RAW_DATASET, skip the prompt (non-interactive use).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        target_dirs = list(args.target_dirs)

        # Decide whether RAW_DATASET is included, without ever silently
        # sweeping it up as part of the plain default folder list.
        project_root_for_prompt = Path.cwd().resolve()
        if project_root_for_prompt.name in ("utils", "scripts"):
            while project_root_for_prompt.name in ("utils", "scripts"):
                project_root_for_prompt = project_root_for_prompt.parent
        raw_dataset_path = resolve_path(RAW_DATASET_FOLDER, project_root_for_prompt)

        if args.remove_raw_dataset:
            print(f"[INFO] --remove-raw-dataset passed: RAW_DATASET will be archived and removed.")
            target_dirs.append(RAW_DATASET_FOLDER)
        elif args.keep_raw_dataset:
            print(f"[INFO] --keep-raw-dataset passed: RAW_DATASET will NOT be touched.")
        elif args.dry_run:
            # Don't prompt on a dry run; just report what would happen.
            print(f"[DRY RUN] Would prompt whether to remove RAW_DATASET (skipped for dry-run).")
        else:
            if confirm_raw_dataset_removal(raw_dataset_path):
                target_dirs.append(RAW_DATASET_FOLDER)
                print("[OK] RAW_DATASET will be archived and removed.")
            else:
                print("[OK] RAW_DATASET will be kept as-is.")

        clean_run_folder(
            target_dirs,
            DEFAULT_TARGET_FILES,
            args.archive_dir,
            args.archive_name,
            args.dry_run,
        )
    except Exception as exc:
        print(f"[ERROR] Clean run failed: {exc}")
        raise SystemExit(1) from exc