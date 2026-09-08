"""
weights_downloader.py
----------------------
Downloads all model weights and config files needed by the pipeline.
Safe to run multiple times -- skips any file that already exists on disk.

Can be run standalone:
    python weights_downloader.py

Or imported and called from the main script before it loads the models.
"""

import os
import requests
from tqdm import tqdm

# filename -> download URL
FILES_TO_DOWNLOAD = {
    # Grounding DINO
    "configs/GroundingDINO_SwinT_OGC.py":
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth":
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",

    # SAM 2 (Hiera-Large variant -- best accuracy; swap for hiera_small/tiny for speed)
    "sam2_hiera_l.yaml":
        "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_l.yaml",
    "weights/sam2_hiera_large.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}


def download_file(url, dest_path):
    """Stream-download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def ensure_all_weights(target_dir="."):
    """Downloads any missing file in FILES_TO_DOWNLOAD into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"[OK] {filename} already exists, skipping.")
            continue
        print(f"[*] Downloading {filename} ...")
        try:
            download_file(url, dest_path)
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            print(f"        You may need to download it manually from: {url}")
    print("[DONE] Weight check/download complete.")


if __name__ == "__main__":
    ensure_all_weights(".")
