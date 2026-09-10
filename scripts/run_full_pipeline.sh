#!/bin/bash
# ==============================================================================
# run_full_pipeline.sh — Interactive Pipeline Orchestrator
#
# Prompts the user to select the pipeline task:
#   1) Annotate Images (Grounding DINO + SAM 2 auto-annotation)
#   2) Train Model (Prepare datasets + Train YOLO / Mask R-CNN / Mask2Former + Evaluate)
#   3) Test on Trained Model (Inference using trained models)
#   4) Run Full Pipeline (Dataset Prep + Training + Evaluation + Comparison)
#   5) Exit
# ==============================================================================
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="runs_comparison/run_${TIMESTAMP}/logs"
mkdir -p "$LOG_DIR"

# Warn if on Azure's temp disk
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == /mnt/* ]]; then
    echo "############################################################"
    echo "  WARNING: Running from $CURRENT_DIR"
    echo "  If this is Azure's temporary resource disk, it will be WIPED"
    echo "  when you Stop (Deallocate) the VM. Move to /home first!"
    echo "############################################################"
    sleep 3
fi

# Confirm venv is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "[ERROR] No virtual environment active. Run: source venv/bin/activate"
    echo "        (This must be re-run in every new terminal/tmux session.)"
    exit 1
fi
echo "[OK] venv active: $VIRTUAL_ENV"

run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local log_file="${LOG_DIR}/${step_num}_${step_name}.log"

    echo ""
    echo "=============================================="
    echo " STEP ${step_num}: ${step_name}"
    echo " Log: ${log_file}"
    echo "=============================================="

    if "$@" 2>&1 | tee "$log_file"; then
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "[FAILED] Step ${step_num} (${step_name}) — see ${log_file}"
            exit 1
        fi
        echo "[OK] Step ${step_num} (${step_name}) complete."
    else
        echo "[FAILED] Step ${step_num} (${step_name}) — see ${log_file}"
        exit 1
    fi
}

display_menu() {
    echo ""
    echo "======================================================================"
    echo "                    Pipeline Flow — Task Selection                    "
    echo "======================================================================"
    echo " Select the task you want to perform:"
    echo ""
    echo "  1) Annotate Images       - Create auto-annotations for dataset (DINO + SAM2)"
    echo "  2) Train Model           - Prepare dataset & train models (YOLO / Mask R-CNN / Mask2Former)"
    echo "  3) Test on Trained Model - Run inference on test images using trained models"
    echo "  4) Run Full Pipeline     - Execute Dataset Prep + Training + Evaluation"
    echo "  5) Exit"
    echo "======================================================================"
}

ask_hparams() {
    local model_label="$1"
    local default_epochs="$2"
    local default_batch="$3"
    local default_workers="$4"

    echo ""
    echo "---- Hyperparameters for ${model_label} ----"
    read -p "  Epochs  [default: ${default_epochs}]: " HP_EPOCHS
    HP_EPOCHS=${HP_EPOCHS:-$default_epochs}
    read -p "  Batch size [default: ${default_batch}]: " HP_BATCH
    HP_BATCH=${HP_BATCH:-$default_batch}
    read -p "  Dataloader workers [default: ${default_workers}]: " HP_WORKERS
    HP_WORKERS=${HP_WORKERS:-$default_workers}
    echo "  -> ${model_label}: epochs=${HP_EPOCHS} batch=${HP_BATCH} workers=${HP_WORKERS}"
}

collect_models() {
    echo ""
    echo "======================================================================"
    echo " Select models to train"
    echo "======================================================================"
    echo "  1) YOLOv11m-seg only"
    echo "  2) Mask R-CNN only"
    echo "  3) Mask2Former only"
    echo "  4) ALL 3 Models (YOLO11m + Mask R-CNN + Mask2Former)"
    echo "======================================================================"

    read -p "Enter choice [1-4] (default 4): " MODEL_CHOICE
    MODEL_CHOICE=${MODEL_CHOICE:-4}

    ENABLE_YOLO=0
    ENABLE_MASKRCNN=0
    ENABLE_MASK2FORMER=0

    case "$MODEL_CHOICE" in
        1) ENABLE_YOLO=1 ;;
        2) ENABLE_MASKRCNN=1 ;;
        3) ENABLE_MASK2FORMER=1 ;;
        4) ENABLE_YOLO=1; ENABLE_MASKRCNN=1; ENABLE_MASK2FORMER=1 ;;
        *) echo "[ERROR] Invalid choice"; exit 1 ;;
    esac
}

collect_training_hparams() {
    echo ""
    echo "======================================================================"
    echo " Set training hyperparameters (press Enter to accept each default)"
    echo "======================================================================"

    if [ "$ENABLE_YOLO" -eq 1 ]; then
        ask_hparams "YOLOv11m-seg"  50 -1 8
        YOLO_EPOCHS=$HP_EPOCHS; YOLO_BATCH=$HP_BATCH; YOLO_WORKERS=$HP_WORKERS
    fi

    if [ "$ENABLE_MASKRCNN" -eq 1 ]; then
        ask_hparams "Mask R-CNN"    20 2  4
        MRCNN_EPOCHS=$HP_EPOCHS; MRCNN_BATCH=$HP_BATCH; MRCNN_WORKERS=$HP_WORKERS
    fi

    if [ "$ENABLE_MASK2FORMER" -eq 1 ]; then
        ask_hparams "Mask2Former"   20 2  4
        M2F_EPOCHS=$HP_EPOCHS; M2F_BATCH=$HP_BATCH; M2F_WORKERS=$HP_WORKERS
    fi
}

if [ -n "$1" ]; then
    CHOICE="$1"
else
    display_menu
    read -p " Enter choice [1-5]: " CHOICE
fi

BASE_OUT_DIR="runs_comparison/run_${TIMESTAMP}"

case "$CHOICE" in
    1)
        echo ""
        echo "[TASK] Starting Image Auto-Annotation..."
        read -p " Enter input images directory [default: ./RAW_DATASET/IMAGES]: " INPUT_DIR
        INPUT_DIR=${INPUT_DIR:-./RAW_DATASET/IMAGES}
        read -p " Enter output directory [default: ./datasets/auto_annotated]: " OUTPUT_DIR
        OUTPUT_DIR=${OUTPUT_DIR:-./datasets/auto_annotated}

        run_step "01" "auto_annotate" python scripts/inference/auto_annotate_carparts.py --input "$INPUT_DIR" --output "$OUTPUT_DIR"
        ;;

    2)
        echo ""
        echo "[TASK] Starting Dataset Prep & Model Training..."

        collect_models
        collect_training_hparams

        DATASET_NAME="combined_carparts"

        # Model Training
        if [ "$ENABLE_YOLO" -eq 1 ]; then
            run_step "02" "train_yolo" \
                python scripts/training/train_yolo_seg.py \
                --model yolo11m-seg \
                --epochs "$YOLO_EPOCHS" \
                --batch "$YOLO_BATCH" \
                --workers "$YOLO_WORKERS" \
                --dataset "$DATASET_NAME" \
                --output_dir "${BASE_OUT_DIR}/yolo11m-seg"
        fi

        if [ "$ENABLE_MASKRCNN" -eq 1 ]; then
            run_step "03" "train_maskrcnn" \
                python scripts/training/train_maskrcnn.py \
                --dataset "$DATASET_NAME" \
                --epochs "$MRCNN_EPOCHS" \
                --batch "$MRCNN_BATCH" \
                --num_workers "$MRCNN_WORKERS" \
                --output_dir "${BASE_OUT_DIR}/maskrcnn"
        fi

        if [ "$ENABLE_MASK2FORMER" -eq 1 ]; then
            run_step "04" "train_mask2former" \
                python scripts/training/train_mask2former.py \
                --dataset "$DATASET_NAME" \
                --epochs "$M2F_EPOCHS" \
                --batch "$M2F_BATCH" \
                --num_workers "$M2F_WORKERS" \
                --output_dir "${BASE_OUT_DIR}/mask2former"
        fi

        run_step "05" "export_excel" python scripts/evaluation/export_excel_report.py --run_dir "$BASE_OUT_DIR"
        ;;

    3)
        echo ""
        echo "[TASK] Starting Inference on Test Images..."
        read -p " Enter input test images directory [default: ./test]: " TEST_INPUT
        TEST_INPUT=${TEST_INPUT:-./test}
        read -p " Enter output directory [default: ./test_result]: " TEST_OUTPUT
        TEST_OUTPUT=${TEST_OUTPUT:-./test_result}

        run_step "01" "inference" python scripts/inference/infer_both_models.py --input "$TEST_INPUT" --output "$TEST_OUTPUT"
        ;;

    4)
        echo ""
        echo "[TASK] Executing Full Pipeline (Train -> Eval -> Compare)..."

        ENABLE_YOLO=1; ENABLE_MASKRCNN=1; ENABLE_MASK2FORMER=1
        collect_training_hparams
        DATASET_NAME="combined_carparts"

        run_step "02" "train_yolo" python scripts/training/train_yolo_seg.py --model yolo11m-seg --epochs "$YOLO_EPOCHS" --batch "$YOLO_BATCH" --workers "$YOLO_WORKERS" --dataset "$DATASET_NAME" --output_dir "${BASE_OUT_DIR}/yolo11m-seg"
        run_step "03" "train_maskrcnn" python scripts/training/train_maskrcnn.py --dataset "$DATASET_NAME" --epochs "$MRCNN_EPOCHS" --batch "$MRCNN_BATCH" --num_workers "$MRCNN_WORKERS" --output_dir "${BASE_OUT_DIR}/maskrcnn"
        run_step "04" "train_mask2former" python scripts/training/train_mask2former.py --dataset "$DATASET_NAME" --epochs "$M2F_EPOCHS" --batch "$M2F_BATCH" --num_workers "$M2F_WORKERS" --output_dir "${BASE_OUT_DIR}/mask2former"

        run_step "05" "compare_models" python scripts/evaluation/compare_models.py --run_dir "$BASE_OUT_DIR" --out_dir "$BASE_OUT_DIR" --non_interactive
        ;;

    5)
        echo "Exiting."
        exit 0
        ;;

    *)
        echo "[ERROR] Invalid option selected."
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo " TASK COMPLETE"
echo " All logs saved under: $LOG_DIR"
echo "=============================================="
