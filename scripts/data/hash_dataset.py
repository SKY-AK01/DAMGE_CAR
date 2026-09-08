#!/usr/bin/env python3
"""
hash_dataset.py
---------------
Compute blake3 hashes for all files in a dataset directory and write
a manifest file.

Output format (one line per file, sorted by path):
  <64-hex-char-hash>  <relative/path/to/file>

Usage:
    python scripts/data/hash_dataset.py --dir datasets/combined_carparts
    python scripts/data/hash_dataset.py --dir datasets --out checksums.blake3
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import blake3 as _blake3
    _HAS_BLAKE3 = True
except ImportError:
    import hashlib
    _HAS_BLAKE3 = False


def _hash_file(path: Path) -> tuple[Path, str] | None:
    """Hash one file. Returns (path, hex_digest) or None on error."""
    try:
        if _HAS_BLAKE3:
            hasher = _blake3.blake3()
        else:
            hasher = hashlib.sha256()

        with open(path, "rb") as f:
            while chunk := f.read(65536):  # 64 KB buffer
                hasher.update(chunk)
        return path, hasher.hexdigest()
    except Exception as e:
        print(f"[WARN] Cannot hash {path}: {e}")
        return None


def hash_dataset(dataset_dir: Path, out_manifest: Path, num_workers: int = 8):
    all_files = sorted(p for p in dataset_dir.rglob("*") if p.is_file())

    if not all_files:
        print(f"[WARN] No files found in {dataset_dir}")
        return

    algo = "blake3" if _HAS_BLAKE3 else "sha256"
    print(f"  Hashing {len(all_files)} files ({algo}, {num_workers} workers)...")

    hash_lines = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(_hash_file, p): p for p in all_files}
        for future in as_completed(future_to_path):
            result = future.result()
            if result:
                path, hex_digest = result
                rel = path.relative_to(dataset_dir)
                hash_lines.append((str(rel), hex_digest))

    # Sort deterministically
    hash_lines.sort(key=lambda x: x[0])

    manifest_content = "\n".join(f"{hex_digest}  {rel}" for rel, hex_digest in hash_lines) + "\n"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(manifest_content, encoding="utf-8")

    print(f"  [OK] Hashed {len(hash_lines)} files -> {out_manifest}")
    if not _HAS_BLAKE3:
        print("  [NOTE] Used sha256 (blake3 not installed). Install with: pip install blake3")


def main():
    parser = argparse.ArgumentParser(description="Hash all files in a dataset directory.")
    parser.add_argument("--dir", required=True, help="Dataset directory to hash")
    parser.add_argument("--out", default=None,
                        help="Output manifest path (default: <dir>/checksums.blake3)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset_dir = Path(args.dir).resolve()
    out_manifest = Path(args.out) if args.out else dataset_dir / "checksums.blake3"

    if not dataset_dir.exists():
        print(f"[ERROR] Directory not found: {dataset_dir}")
        return

    hash_dataset(dataset_dir, out_manifest, args.workers)


if __name__ == "__main__":
    main()
