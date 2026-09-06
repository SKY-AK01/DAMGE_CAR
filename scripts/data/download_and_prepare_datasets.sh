#!/bin/bash
# ==============================================================================
# download_and_prepare_datasets.sh
# Prompts user to select dataset source:
# 1) Local RAW_DATASET only
# 2) Download from online only (Ultralytics + DSMLR)
# 3) Both (Combines them into one dataset)
# ==============================================================================
set -e

echo ""
echo "======================================================================"
echo " Select Dataset Source:"
echo "======================================================================"
echo "  1) Local RAW_DATASET only"
echo "  2) Download from online only (Ultralytics + DSMLR)"
echo "  3) Both (Downloads and combines with RAW_DATASET)"
echo "======================================================================"
read -p "Enter choice [1-3]: " DATA_CHOICE

mkdir -p datasets

if [ "$DATA_CHOICE" == "1" ] || [ "$DATA_CHOICE" == "3" ]; then
    echo "=============================================="
    echo " Processing RAW_DATASET"
    echo "=============================================="
    # Prepare YOLO labels and splits
    python scripts/data/prepare_raw_dataset.py
    
    # Also run the old logic to move XMLs just in case other scripts rely on it
    python scripts/data/process_raw_dataset.py
    
    # Run yolo_to_coco so mask2former works on custom_carparts too
    python scripts/data/yolo_to_coco.py --dataset custom_carparts || echo "[WARN] COCO conversion failed for custom_carparts"
fi

if [ "$DATA_CHOICE" == "2" ] || [ "$DATA_CHOICE" == "3" ]; then
    cd datasets
    echo "=============================================="
    echo " Downloading Ultralytics Carparts-seg (133 MB)"
    echo "=============================================="
    if [ ! -d "carparts-seg" ]; then
        echo "[*] Starting download: carparts-seg.zip (133 MB) — progress shown below"
        wget --progress=dot:mega -O carparts-seg.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip 2>&1
        echo "[*] Extracting carparts-seg.zip..."
        unzip -o carparts-seg.zip -d carparts-seg 2>&1 | tail -5
        rm carparts-seg.zip
        echo "[OK] carparts-seg ready -> ./datasets/carparts-seg"
    else
        echo "[OK] carparts-seg already exists, skipping download."
    fi

    echo "=============================================="
    echo " Downloading DSMLR Car-Parts-Segmentation"
    echo "=============================================="
    if [ ! -d "dsmlr-carparts" ]; then
        echo "[*] Cloning DSMLR repo (~24 MB)..."
        GIT_TERMINAL_PROMPT=0 git clone --progress https://github.com/dsmlr/Car-Parts-Segmentation.git dsmlr-carparts 2>&1
        echo "[OK] DSMLR repo cloned -> ./datasets/dsmlr-carparts"
    else
        echo "[OK] dsmlr-carparts already exists, skipping clone."
    fi
    cd ..

    echo "=============================================="
    echo " Auto-splitting DSMLR into train/val/test"
    echo "=============================================="
    python scripts/data/prepare_dsmlr_split.py

    echo "=============================================="
    echo " Remapping downloaded datasets to canonical labels"
    echo "=============================================="
    python scripts/data/remap_github_labels.py --format yolo --path datasets/carparts-seg
    
    # Since DSMLR is split into COCO format by prepare_dsmlr_split.py, we remap the output JSONs
    python scripts/data/remap_github_labels.py --format coco --path datasets/dsmlr-carparts-split/annotations/instances_train.json
    python scripts/data/remap_github_labels.py --format coco --path datasets/dsmlr-carparts-split/annotations/instances_val.json
    python scripts/data/remap_github_labels.py --format coco --path datasets/dsmlr-carparts-split/annotations/instances_test.json

    echo "=============================================="
    echo " Converting downloaded datasets to COCO format"
    echo "=============================================="
    python scripts/data/yolo_to_coco.py --dataset carparts-seg
    python scripts/data/yolo_to_coco.py --dataset dsmlr-carparts
fi

if [ "$DATA_CHOICE" == "3" ]; then
    echo "=============================================="
    echo " Combining Datasets (RAW_DATASET + carparts-seg)"
    echo "=============================================="
    python scripts/data/combine_datasets.py
    
    # Convert combined dataset to COCO for Mask2Former
    python scripts/data/yolo_to_coco.py --dataset combined_carparts || echo "[WARN] COCO conversion failed for combined_carparts"
fi

# Save the choice so the main pipeline knows which dataset to train on
echo "$DATA_CHOICE" > datasets/.dataset_choice.txt

echo "=============================================="
echo " All datasets prepared."
echo "=============================================="
