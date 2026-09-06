#!/usr/bin/env python3
"""
code_transfer/upload.py
=======================
Uploads the CAR_AZURE project to Azure Blob Storage.

HOW IT WORKS:
  1. Creates a timestamped subfolder in the container: car_azure_transfers/car_azure_YYYYMMDD_HHMMSS/
  2. Compresses the project root into a tar.gz archive (excluding build artifacts & model checkpoints)
  3. Uploads the archive using parallel block-upload (Azure SDK chunk upload)
  4. Uploads a SHA256 checksum file alongside the archive for integrity verification on download
  5. Shows real-time progress (bytes transferred / total, speed in MB/s)

CREDENTIALS:
  Place your SAS token in  code_transfer/cloud_config.json  (already pre-filled).
  Or set environment variables: AZURE_STORAGE_ACCOUNT, AZURE_CONTAINER_NAME, AZURE_SAS_TOKEN

USAGE:
  python code_transfer/upload.py                  # Upload everything (with prompts)
  python code_transfer/upload.py --yes            # Non-interactive, use defaults
  python code_transfer/upload.py --dry-run        # Show what would be uploaded without doing it

DEPENDENCIES:
  pip install azure-storage-blob tqdm
"""

import os
import sys
import json
import time
import hashlib
import tarfile
import tempfile
import argparse
import threading
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── third-party ──────────────────────────────────────────────────────────────
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import AzureError
except ImportError:
    print("[ERROR] azure-storage-blob not installed. Run:  pip install azure-storage-blob tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] tqdm not installed. Run:  pip install tqdm")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "cloud_config.json"

# Files/dirs to ALWAYS exclude from the archive
DEFAULT_EXCLUDES = [
    ".git",
    ".git/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    "*.pyo",
    "venv",
    ".venv",
    "venv/**",
    ".venv/**",
    "node_modules",
    "node_modules/**",
    "target",           # Rust build artifacts
    "target/**",
    "*.pdb",
    "*.pdb",
    "*.exp",
    "*.lib",
    # Model checkpoints — excluded per user preference
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.bin",
    # Temp / IDE
    ".idea/**",
    ".vscode/**",
    "*.tmp",
    "*.log",
    # Other large local zips that already exist
    "combined_dataset_*.zip",
    "Job_all_v5_*.zip",
    "CAR_AZURE_backup_*.zip",
    "rustup-init.exe",
]

# Patterns that ARE included (datasets/ is included per user request)
# Model checkpoint exclusion is the only addition over default

def load_config() -> dict:
    """Load credentials from cloud_config.json, then override with env vars."""
    cfg = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8-sig") as f:
            raw = json.load(f)
        # Strip comment keys
        cfg = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Environment variable overrides
    cfg["storage_account"] = os.environ.get("AZURE_STORAGE_ACCOUNT", cfg.get("storage_account", ""))
    cfg["container_name"]  = os.environ.get("AZURE_CONTAINER_NAME",  cfg.get("container_name", ""))
    cfg["sas_token"]       = os.environ.get("AZURE_SAS_TOKEN",       cfg.get("sas_token", ""))
    cfg["blob_prefix"]     = os.environ.get("AZURE_BLOB_PREFIX",     cfg.get("blob_prefix", "car_azure_transfers"))

    if not cfg["storage_account"] or not cfg["container_name"] or not cfg["sas_token"]:
        print("[ERROR] Missing Azure credentials. Set them in code_transfer/cloud_config.json or via env vars:")
        print("  AZURE_STORAGE_ACCOUNT, AZURE_CONTAINER_NAME, AZURE_SAS_TOKEN")
        sys.exit(1)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Archive creation
# ─────────────────────────────────────────────────────────────────────────────

def is_excluded(path: Path, root: Path, excludes: list[str]) -> bool:
    """Return True if this path should be excluded from the archive."""
    rel = path.relative_to(root).as_posix()
    name = path.name
    for pattern in excludes:
        # Match against the name alone
        if fnmatch.fnmatch(name, pattern.lstrip("**/").rstrip("/**")):
            return True
        # Match against full relative path
        if fnmatch.fnmatch(rel, pattern):
            return True
        # Check if any parent component matches a directory exclude
        parts = rel.split("/")
        for pat in excludes:
            clean = pat.strip("**/")
            if clean in parts:
                return True
    return False


def create_archive(
    root: Path,
    excludes: list[str],
    archive_path: Path,
    verbose: bool = False,
) -> tuple[int, str]:
    """
    Create a tar.gz archive of root at archive_path.
    Returns (file_count, sha256_hex).
    """
    file_count = 0
    sha256 = hashlib.sha256()

    print(f"[*] Compressing project: {root}")
    print(f"[*] Archive destination: {archive_path}")

    with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
        for fpath in sorted(root.rglob("*")):
            if not fpath.is_file():
                continue
            # Skip the archive itself and anything inside code_transfer output
            if archive_path.parent in fpath.parents and fpath.suffix in (".gz", ".tar"):
                continue
            if is_excluded(fpath, root, excludes):
                if verbose:
                    print(f"  [skip] {fpath.relative_to(root)}")
                continue
            rel = fpath.relative_to(root)
            tar.add(fpath, arcname=str(rel))
            file_count += 1
            if verbose:
                print(f"  [add]  {rel}")

    # Compute checksum of the archive
    with open(archive_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            sha256.update(chunk)

    return file_count, sha256.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Azure upload
# ─────────────────────────────────────────────────────────────────────────────

class ProgressCallback:
    """Thread-safe progress reporter for Azure SDK upload callbacks."""

    def __init__(self, total_bytes: int, desc: str = "Uploading"):
        self.total = total_bytes
        self.bar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            colour="cyan",
            dynamic_ncols=True,
        )
        self._lock = threading.Lock()
        self._uploaded = 0
        self._start = time.monotonic()

    def __call__(self, current: int, total: int = 0, *args, **kwargs):
        with self._lock:
            delta = current - self._uploaded
            self._uploaded = current
            self.bar.update(delta)

    def close(self):
        elapsed = time.monotonic() - self._start
        speed = self.total / elapsed / 1024 / 1024 if elapsed > 0 else 0
        self.bar.close()
        print(f"[OK] Upload complete — {self.total / 1024 / 1024:.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)")


def upload_file_to_blob(
    client: BlobServiceClient,
    container: str,
    blob_name: str,
    local_path: Path,
    max_concurrency: int = 8,
    dry_run: bool = False,
):
    """Upload a single file to Azure Blob Storage with parallel chunks and progress."""
    file_size = local_path.stat().st_size
    print(f"\n[*] Uploading: {local_path.name}  ({file_size / 1024 / 1024:.1f} MB)")
    print(f"    → blob: {blob_name}")

    if dry_run:
        print("    [DRY RUN] skipping actual upload")
        return

    progress = ProgressCallback(file_size, desc=f"  {local_path.name}")

    blob_client = client.get_blob_client(container=container, blob=blob_name)
    content_settings = ContentSettings(content_type="application/gzip")

    with open(local_path, "rb") as f:
        blob_client.upload_blob(
            f,
            overwrite=True,
            max_concurrency=max_concurrency,   # parallel chunk connections
            content_settings=content_settings,
            progress_hook=progress,
        )
    progress.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload CAR_AZURE project to Azure Blob Storage")
    parser.add_argument("--yes",      "-y", action="store_true", help="Non-interactive, accept all defaults")
    parser.add_argument("--dry-run",        action="store_true", help="Show what would be done without uploading")
    parser.add_argument("--verbose",  "-v", action="store_true", help="List every file added to the archive")
    parser.add_argument("--concurrency", type=int, default=8,   help="Parallel connections for blob upload (default: 8)")
    args = parser.parse_args()

    cfg = load_config()

    # Timestamped subfolder
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_folder = f"{cfg['blob_prefix']}/car_azure_{ts}"
    archive_name = f"car_azure_{ts}.tar.gz"
    checksum_name = f"car_azure_{ts}.sha256"

    print("=" * 60)
    print("  CAR_AZURE Project Uploader")
    print("=" * 60)
    print(f"  Account   : {cfg['storage_account']}")
    print(f"  Container : {cfg['container_name']}")
    print(f"  Blob path : {run_folder}/{archive_name}")
    print(f"  Source    : {PROJECT_ROOT}")
    print(f"  Dry run   : {args.dry_run}")
    print("=" * 60)

    if not args.yes and not args.dry_run:
        ans = input("\nProceed? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    # Create archive in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / archive_name

        file_count, sha256hex = create_archive(
            root=PROJECT_ROOT,
            excludes=DEFAULT_EXCLUDES,
            archive_path=archive_path,
            verbose=args.verbose,
        )

        archive_size_mb = archive_path.stat().st_size / 1024 / 1024
        print(f"\n[OK] Archive created: {file_count} files, {archive_size_mb:.1f} MB compressed")
        print(f"[OK] SHA256: {sha256hex}")

        # Write checksum file
        checksum_path = Path(tmpdir) / checksum_name
        checksum_path.write_text(f"{sha256hex}  {archive_name}\n", encoding="utf-8")

        if not args.dry_run:
            # Connect to Azure
            account_url = f"https://{cfg['storage_account']}.blob.core.windows.net"
            sas = cfg["sas_token"]
            if not sas.startswith("?"):
                sas = "?" + sas
            blob_service = BlobServiceClient(account_url=account_url + sas)

            # Upload archive (parallel chunks)
            upload_file_to_blob(
                client=blob_service,
                container=cfg["container_name"],
                blob_name=f"{run_folder}/{archive_name}",
                local_path=archive_path,
                max_concurrency=args.concurrency,
                dry_run=False,
            )

            # Upload checksum
            print(f"\n[*] Uploading checksum file...")
            blob_client = blob_service.get_blob_client(
                container=cfg["container_name"],
                blob=f"{run_folder}/{checksum_name}",
            )
            blob_client.upload_blob(checksum_path.read_bytes(), overwrite=True)
            print(f"[OK] Checksum uploaded: {checksum_name}")

        print(f"\n{'='*60}")
        print(f"  Upload finished!")
        print(f"  Run folder : {run_folder}")
        print(f"  Archive    : {archive_name}")
        print(f"  SHA256     : {sha256hex}")
        print(f"  To download on VM, run:")
        print(f"    python code_transfer/download.py")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
