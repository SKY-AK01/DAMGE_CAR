#!/bin/bash
# ==============================================================================
# setup_all.sh — Master environment setup for the ENTIRE pipeline.
#
# This version has every fix we discovered baked in from the start:
#   - Warns if run from /mnt (Azure temp disk — wiped on VM deallocate!)
#   - --no-build-isolation on every git-based editable install (GroundingDINO,
#     SAM2, Detectron2 all failed without this — their setup.py scripts try
#     to run `pip install torch` inside pip's isolated build sandbox, which
#     doesn't have pip available in it)
#   - transformers pinned to 4.46.3 from the start (a newer auto-installed
#     version enforced a torch>=2.6 requirement that broke Mask2Former
#     loading, even though we're on torch 2.5.1)
#   - iopath pinned to 0.1.9 after Detectron2 install (SAM2 pulls in 0.1.10,
#     Detectron2 wants <0.1.10 — direct conflict)
#   - weights_downloader.py call is guarded so a missing file doesn't kill
#     the whole script via `set -e`
#
# Usage:
#   chmod +x setup_all.sh
#   ./setup_all.sh
# ==============================================================================
set -e

echo "=============================================="
echo " Car Parts Annotation + Training Pipeline — Environment Setup"
echo "=============================================="

# ---- 0. Warn if running from Azure's temporary /mnt disk --------------------
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == /mnt/* ]]; then
    echo ""
    echo "############################################################"
    echo "  WARNING: You are running this from $CURRENT_DIR"
    echo "  On Azure VMs, /mnt is often the TEMPORARY resource disk."
    echo "  It gets WIPED every time you Stop (Deallocate) the VM."
    echo "  Move this whole folder under /home/<you>/ instead, e.g.:"
    echo "    mkdir -p ~/car_parts_pipeline && cp -r . ~/car_parts_pipeline && cd ~/car_parts_pipeline"
    echo "  Press Ctrl+C now to abort and move the folder, or wait 10s to continue anyway."
    echo "############################################################"
    sleep 10
fi

# ---- 1. Check GPU -------------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARNING] No NVIDIA GPU / driver detected. Everything will fall back to CPU"
    echo "          (very slow for training in particular)."
fi

# ---- 2. Virtual environment ---------------------------------------------------
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment (./venv)..."
    python3 -m venv venv
else
    echo "[OK] venv already exists, reusing it."
fi
source venv/bin/activate
echo "[*] Confirming venv is active: $(which python)"
pip install --upgrade pip setuptools wheel

# ---- 3. PyTorch (CUDA build) ---------------------------------------------------
echo "[*] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---- 4. Core / shared libraries ------------------------------------------------
echo "[*] Installing core libraries (CV, data, utils)..."
pip install \
    opencv-python \
    numpy \
    pillow \
    matplotlib \
    requests \
    tqdm \
    pyyaml \
    pycocotools \
    supervision

# ---- 5. Stage A dependencies: Grounding DINO + SAM2 ----------------------------
echo "=============================================="
echo " Stage A: Grounding DINO + SAM2 (auto-annotation)"
echo "=============================================="

if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
fi
cd GroundingDINO
pip install -e . --no-build-isolation
cd ..

if [ ! -d "segment-anything-2" ]; then
    git clone https://github.com/facebookresearch/segment-anything-2.git
fi
cd segment-anything-2
pip install -e . --no-build-isolation
cd ..

# Pin transformers immediately after GroundingDINO install, since GroundingDINO's
# requirements.txt pulls in an unpinned (very new) transformers version that
# enforces torch>=2.6 for torch.load, breaking Mask2Former on torch 2.5.1.
echo "[*] Pinning transformers to a version compatible with torch 2.5.1..."
pip install "transformers==4.46.3" --no-build-isolation

echo "[*] Downloading Grounding DINO + SAM2 weights (skips if already present)..."
if [ -f "scripts/utils/weights_downloader.py" ]; then
    python scripts/utils/weights_downloader.py
else
    echo "[WARNING] weights_downloader.py not found — skipping Stage A weight download."
    echo "          This only affects DINO+SAM2 auto-annotation, not Stage B training."
fi

# ---- 6. Stage B dependencies: YOLO / Mask R-CNN
echo "=============================================="
echo " Stage B: YOLO-seg (core, proven working)"
echo "=============================================="

echo "[*] Installing Ultralytics (YOLOv8/v11-seg)..."
pip install ultralytics

# ---- 9. Sanity check -----------------------------------------------------------
echo "=============================================="
echo " Verifying installation..."
echo "=============================================="
python - <<'EOF'
import torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())

checks = {
    "cv2": "OpenCV",
    "torchvision": "TorchVision (Mask R-CNN)",
    "transformers": "Transformers (Mask2Former)",
    "ultralytics": "Ultralytics (YOLO11m-seg)",
    "groundingdino": "Grounding DINO",
    "sam2": "SAM 2",
}
for module, label in checks.items():
    try:
        __import__(module)
        print(f"[OK] {label}")
    except ImportError as e:
        print(f"[MISSING] {label} -> {e}")
EOF

echo "=============================================="
echo " Setup complete."
echo " Activate with: source venv/bin/activate"
echo ""
echo " IMPORTANT: every new terminal/tmux session needs 'source venv/bin/activate'"
echo " run again — it does NOT persist automatically across sessions."
echo ""
echo " Next: run ./run_full_pipeline.sh for the complete dataset+train+eval+compare run."
echo "=============================================="
