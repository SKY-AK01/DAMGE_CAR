#!/usr/bin/env python3
"""
code_transfer/download.py
=========================
Downloads the latest CAR_AZURE project snapshot from Azure Blob Storage.

HOW IT WORKS:
  1. Lists all timestamped subfolders in the container under the blob_prefix
  2. Automatically selects the MOST RECENT snapshot (by folder timestamp)
  3. Downloads the tar.gz archive using parallel chunked download
  4. Verifies SHA256 checksum BEFORE extracting (aborts if corrupt)
  5. Extracts the archive into the target directory (default: project root or cwd)
  6. Shows real-time download progress (bytes / speed)

CREDENTIALS:
  Same as upload.py — reads from code_transfer/cloud_config.json or env vars:
  AZURE_STORAGE_ACCOUNT, AZURE_CONTAINER_NAME, AZURE_SAS_TOKEN

USAGE:
  python code_transfer/download.py                        # Download latest, extract here
  python code_transfer/download.py --target /opt/car_azure  # Extract to specific directory
  python code_transfer/download.py --list                 # List available snapshots only
  python code_transfer/download.py --snapshot car_azure_20260905_120000  # Pick a specific snapshot

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── third-party ──────────────────────────────────────────────────────────────
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    from azure.core.exceptions import AzureError, ResourceNotFoundError
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


def load_config() -> dict:
    """Load credentials from cloud_config.json, then override with env vars."""
    cfg = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8-sig") as f:
            raw = json.load(f)
        cfg = {k: v for k, v in raw.items() if not k.startswith("_")}

    cfg["storage_account"] = os.environ.get("AZURE_STORAGE_ACCOUNT", cfg.get("storage_account", ""))
    cfg["container_name"]  = os.environ.get("AZURE_CONTAINER_NAME",  cfg.get("container_name", ""))
    cfg["sas_token"]       = os.environ.get("AZURE_SAS_TOKEN",       cfg.get("sas_token", ""))
    cfg["blob_prefix"]     = os.environ.get("AZURE_BLOB_PREFIX",     cfg.get("blob_prefix", "car_azure_transfers"))

    if not cfg["storage_account"] or not cfg["container_name"] or not cfg["sas_token"]:
        print("[ERROR] Missing Azure credentials. Set them in code_transfer/cloud_config.json or via env vars:")
        print("  AZURE_STORAGE_ACCOUNT, AZURE_CONTAINER_NAME, AZURE_SAS_TOKEN")
        sys.exit(1)

    return cfg


def get_blob_service(cfg: dict) -> BlobServiceClient:
    account_url = f"https://{cfg['storage_account']}.blob.core.windows.net"
    sas = cfg["sas_token"]
    if not sas.startswith("?"):
        sas = "?" + sas
    return BlobServiceClient(account_url=account_url + sas)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot discovery
# ─────────────────────────────────────────────────────────────────────────────

def list_snapshots(container_client: ContainerClient, prefix: str) -> list[dict]:
    """
    List all uploaded snapshots (timestamped subfolders) sorted newest first.
    Returns list of dicts with keys: name, archive_blob, checksum_blob, timestamp_str
    """
    snapshots = {}

    # List all blobs under the prefix
    for blob in container_client.list_blobs(name_starts_with=prefix + "/"):
        blob_name: str = blob.name
        parts = blob_name.split("/")
        if len(parts) < 3:
            continue
        # e.g. car_azure_transfers/car_azure_20260905_120000/car_azure_20260905_120000.tar.gz
        folder = parts[1]  # car_azure_YYYYMMDD_HHMMSS
        if not folder.startswith("car_azure_"):
            continue
        ts_str = folder.replace("car_azure_", "")  # YYYYMMDD_HHMMSS

        if folder not in snapshots:
            snapshots[folder] = {"name": folder, "timestamp_str": ts_str, "archive_blob": None, "checksum_blob": None}

        if blob_name.endswith(".tar.gz"):
            snapshots[folder]["archive_blob"] = blob_name
            snapshots[folder]["size_bytes"] = blob.size
        elif blob_name.endswith(".sha256"):
            snapshots[folder]["checksum_blob"] = blob_name

    # Sort by timestamp descending (newest first)
    result = sorted(
        [s for s in snapshots.values() if s["archive_blob"]],
        key=lambda x: x["timestamp_str"],
        reverse=True,
    )
    return result


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse YYYYMMDD_HHMMSS to datetime."""
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Checksum verification
# ─────────────────────────────────────────────────────────────────────────────

def compute_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(archive_path: Path, expected_hash: str, archive_name: str) -> bool:
    """Verify archive integrity against expected SHA256. Returns True if OK."""
    print(f"\n[*] Verifying integrity of {archive_path.name}...")
    actual = compute_sha256(archive_path)
    if actual == expected_hash:
        print(f"[OK] Checksum verified: {actual}")
        return True
    else:
        print(f"[FAIL] Checksum MISMATCH!")
        print(f"  Expected : {expected_hash}")
        print(f"  Got      : {actual}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Download with progress
# ─────────────────────────────────────────────────────────────────────────────

def download_blob(
    blob_service: BlobServiceClient,
    container: str,
    blob_name: str,
    dest_path: Path,
    max_concurrency: int = 8,
    total_size: Optional[int] = None,
):
    """Download a blob to dest_path with parallel chunks and progress bar."""
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    if total_size is None:
        props = blob_client.get_blob_properties()
        total_size = props.size

    print(f"\n[*] Downloading: {blob_name.split('/')[-1]}  ({total_size / 1024 / 1024:.1f} MB)")

    _progress_lock = threading.Lock()
    _downloaded = [0]
    _start = [time.monotonic()]

    bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  {dest_path.name}",
        colour="green",
        dynamic_ncols=True,
    )

    with open(dest_path, "wb") as f:
        download = blob_client.download_blob(max_concurrency=max_concurrency)
        for chunk in download.chunks():
            f.write(chunk)
            bar.update(len(chunk))

    bar.close()
    elapsed = time.monotonic() - _start[0]
    speed = total_size / elapsed / 1024 / 1024 if elapsed > 0 else 0
    print(f"[OK] Download complete — {total_size / 1024 / 1024:.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_archive(archive_path: Path, target_dir: Path):
    """Extract the tar.gz archive into target_dir with progress."""
    print(f"\n[*] Extracting archive into: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="  Extracting", unit="files", colour="yellow", dynamic_ncols=True) as bar:
            for member in members:
                tar.extract(member, path=target_dir, filter="data")
                bar.update(1)

    print(f"[OK] Extracted {len(members)} files into {target_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download latest CAR_AZURE snapshot from Azure Blob Storage")
    parser.add_argument("--list",       "-l", action="store_true", help="List available snapshots and exit")
    parser.add_argument("--snapshot",   "-s", default=None,        help="Specific snapshot folder name (default: latest)")
    parser.add_argument("--target",     "-t", default=None,        help="Directory to extract into (default: project root)")
    parser.add_argument("--concurrency",      type=int, default=8, help="Parallel connections for download (default: 8)")
    parser.add_argument("--no-verify",        action="store_true", help="Skip SHA256 checksum verification")
    parser.add_argument("--yes",        "-y", action="store_true", help="Non-interactive")
    args = parser.parse_args()

    cfg = load_config()
    blob_service = get_blob_service(cfg)
    container_client = blob_service.get_container_client(cfg["container_name"])

    print("=" * 60)
    print("  CAR_AZURE Project Downloader")
    print("=" * 60)
    print(f"  Account   : {cfg['storage_account']}")
    print(f"  Container : {cfg['container_name']}")
    print(f"  Prefix    : {cfg['blob_prefix']}")
    print("=" * 60)

    # Discover snapshots
    print(f"\n[*] Scanning for available snapshots...")
    snapshots = list_snapshots(container_client, cfg["blob_prefix"])

    if not snapshots:
        print("[ERROR] No snapshots found in container. Upload first with:  python code_transfer/upload.py")
        sys.exit(1)

    print(f"[OK] Found {len(snapshots)} snapshot(s):\n")
    for i, s in enumerate(snapshots):
        dt = parse_timestamp(s["timestamp_str"])
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else s["timestamp_str"]
        size_mb = s.get("size_bytes", 0) / 1024 / 1024
        marker = " <- LATEST" if i == 0 else ""
        print(f"  [{i+1}] {s['name']}  ({size_mb:.0f} MB)  {dt_str}{marker}")

    if args.list:
        sys.exit(0)

    # Select snapshot
    if args.snapshot:
        matching = [s for s in snapshots if s["name"] == args.snapshot or s["timestamp_str"] == args.snapshot]
        if not matching:
            print(f"[ERROR] Snapshot '{args.snapshot}' not found.")
            sys.exit(1)
        selected = matching[0]
    else:
        selected = snapshots[0]  # Latest

    print(f"\n[*] Selected snapshot: {selected['name']}")

    # Determine target directory
    if args.target:
        target_dir = Path(args.target).resolve()
    else:
        target_dir = PROJECT_ROOT

    print(f"[*] Extract target  : {target_dir}")

    if not args.yes:
        ans = input("\nProceed? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        archive_local = tmpdir_path / selected["archive_blob"].split("/")[-1]

        # Download archive
        download_blob(
            blob_service=blob_service,
            container=cfg["container_name"],
            blob_name=selected["archive_blob"],
            dest_path=archive_local,
            max_concurrency=args.concurrency,
            total_size=selected.get("size_bytes"),
        )

        # Verify checksum
        if not args.no_verify and selected["checksum_blob"]:
            checksum_local = tmpdir_path / selected["checksum_blob"].split("/")[-1]
            checksum_blob_client = blob_service.get_blob_client(
                container=cfg["container_name"],
                blob=selected["checksum_blob"],
            )
            checksum_text = checksum_blob_client.download_blob().readall().decode("utf-8").strip()
            expected_hash = checksum_text.split()[0]
            archive_ok = verify_checksum(archive_local, expected_hash, archive_local.name)
            if not archive_ok:
                print("[ERROR] Integrity check failed. The download may be corrupt. Aborting extraction.")
                sys.exit(1)
        elif args.no_verify:
            print("[WARN] Skipping checksum verification (--no-verify).")
        else:
            print("[WARN] No checksum file found for this snapshot — skipping verification.")

        # Extract
        extract_archive(archive_local, target_dir)

    print(f"\n{'='*60}")
    print(f"  Download & extraction complete!")
    print(f"  Snapshot  : {selected['name']}")
    print(f"  Extracted : {target_dir}")
    print(f"{'='*60}\n")

    print("[NEXT STEPS]")
    print("  1. Build the Rust dataloader:  python build_rust_dataloader.py")
    print("  2. Install dependencies:        pip install -r requirements.txt")
    print("  3. Start training:              python orchestrator.py")
    print()


if __name__ == "__main__":
    main()
