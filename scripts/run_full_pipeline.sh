#!/bin/bash
# ==============================================================================
# run_full_pipeline.sh — Interactive Pipeline Orchestrator
#
# Prompts the user to select the pipeline task:
#   1) Annotate Images (Grounding DINO + SAM 2 auto-annotation)
#   2) Train Model (Prepare datasets + Train YOLO / Mask2Former / Mask R-CNN + Evaluate)
#   3) Test on Trained Model (Inference using trained models)
# ==============================================================================
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/run_${TIMESTAMP}"
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
    echo "  2) Train Model           - Prepare dataset & train models (YOLO / Mask R-CNN)"
    echo "  3) Test on Trained Model - Run inference on test images using trained models"
    echo "  4) Run Full Pipeline     - Execute Dataset Prep + Training + Evaluation"
    echo "  5) Exit"
    echo "======================================================================"
}

# ------------------------------------------------------------------------
# ask_hparams <model_label> <default_epochs> <default_batch> <default_workers>
# Prompts once per model for epochs/batch/workers, falling back to the
# given defaults on empty input (just hit Enter to accept the default).
# Sets globals: HP_EPOCHS, HP_BATCH, HP_WORKERS
# so you can lower/raise these per model/per run without editing any
# script — handy if a run OOMs, or if you want to push batch size up
# after checking `nvidia-smi` shows headroom.
# ------------------------------------------------------------------------
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

# Collects hyperparams for all 3 models up front (asked once before any
# training starts, so you don't have to babysit the terminal mid-pipeline).
# Defaults here match the values already tuned per-model in the training
# scripts themselves (see the GPU UTILIZATION FIX comments in each file).


collect_models() {
    echo ""
    echo "======================================================================"
    echo " Select models to train"
    echo "======================================================================"
    echo "  1) YOLO only"
    echo "  2) Mask R-CNN only"
    echo "  3) Fast R-CNN only"
    echo "  4) YOLO + Mask R-CNN"
    echo "  5) YOLO + Fast R-CNN"
    echo "  6) Mask R-CNN + Fast R-CNN"
    echo "  7) All models"
    echo "======================================================================"

    read -p "Enter choice [1-7]: " MODEL_CHOICE

    ENABLE_YOLO=0
    ENABLE_MASKRCNN=0
    ENABLE_FASTRCNN=0

    case "$MODEL_CHOICE" in
        1) ENABLE_YOLO=1 ;;
        2) ENABLE_MASKRCNN=1 ;;
        3) ENABLE_FASTRCNN=1 ;;
        4) ENABLE_YOLO=1; ENABLE_MASKRCNN=1 ;;
        5) ENABLE_YOLO=1; ENABLE_FASTRCNN=1 ;;
        6) ENABLE_MASKRCNN=1; ENABLE_FASTRCNN=1 ;;
        7) ENABLE_YOLO=1; ENABLE_MASKRCNN=1; ENABLE_FASTRCNN=1 ;;
        *) echo "[ERROR] Invalid choice"; exit 1 ;;
    esac
}


collect_training_hparams() {
    echo ""
    echo "======================================================================"
    echo " Set training hyperparameters (press Enter to accept each default)"
    echo "======================================================================"

    ask_hparams "YOLOv11m-seg"  1 -1 16
    YOLO_EPOCHS=$HP_EPOCHS; YOLO_BATCH=$HP_BATCH; YOLO_WORKERS=$HP_WORKERS

    ask_hparams "Mask R-CNN"    1 2  8
    MRCNN_EPOCHS=$HP_EPOCHS; MRCNN_BATCH=$HP_BATCH; MRCNN_WORKERS=$HP_WORKERS

    ask_hparams "Fast R-CNN"    1 4  8
    FASTRCNN_EPOCHS=$HP_EPOCHS; FASTRCNN_BATCH=$HP_BATCH; FASTRCNN_WORKERS=$HP_WORKERS

    echo ""
    echo "---- Summary ----"
    echo "  YOLOv11m-seg : epochs=${YOLO_EPOCHS}  batch=${YOLO_BATCH}  workers=${YOLO_WORKERS}"
    echo "  Mask R-CNN   : epochs=${MRCNN_EPOCHS} batch=${MRCNN_BATCH} workers=${MRCNN_WORKERS}"
    echo "  Fast R-CNN   : epochs=${FASTRCNN_EPOCHS} batch=${FASTRCNN_BATCH} workers=${FASTRCNN_WORKERS}"
    echo "------------------"
}

if [ -n "$1" ]; then
    CHOICE="$1"
else
    display_menu
    read -p " Enter choice [1-5]: " CHOICE
fi

case "$CHOICE" in
    1)
        echo ""
        echo "[TASK] Starting Image Auto-Annotation..."
        read -p " Enter input images directory [default: ./RAW_DATASET]: " INPUT_DIR
        INPUT_DIR=${INPUT_DIR:-./RAW_DATASET}
        read -p " Enter output directory [default: ./datasets/auto_annotated]: " OUTPUT_DIR
        OUTPUT_DIR=${OUTPUT_DIR:-./datasets/auto_annotated}

        run_step "01" "auto_annotate" python scripts/inference/auto_annotate_carparts.py --input "$INPUT_DIR" --output "$OUTPUT_DIR"
        ;;

    2)
        echo ""
        echo "[TASK] Starting Dataset Prep & Model Training..."

        
	collect_models
	collect_training_hparams

        # Step A: Dataset Prep
        run_step "01" "dataset_prep" bash scripts/data/download_and_prepare_datasets.sh
        
        # Read dataset choice AFTER prep has run
        if [ -f "datasets/.dataset_choice.txt" ]; then
            DATA_CHOICE=$(cat datasets/.dataset_choice.txt)
            if [ "$DATA_CHOICE" == "1" ]; then
                DATASET_NAME="custom_carparts"
            elif [ "$DATA_CHOICE" == "3" ]; then
                DATASET_NAME="combined_carparts"
            else
                DATASET_NAME="carparts-seg"
            fi
        else
            DATASET_NAME="carparts-seg"
        fi

        # Step B: Model Training
        # Select models to train
        if [ "$ENABLE_YOLO" -eq 1 ]; then
            run_step "02" "train_yolo" \
                python scripts/training/train_yolo_seg.py \
                --model yolo11m-seg \
                --epochs "$YOLO_EPOCHS" \
                --batch "$YOLO_BATCH" \
                --workers "$YOLO_WORKERS" \
                --dataset "$DATASET_NAME"
        fi

        if [ "$ENABLE_MASKRCNN" -eq 1 ]; then
            run_step "03b" "train_maskrcnn" \
                python scripts/training/train_maskrcnn.py \
                --dataset "$DATASET_NAME" \
                --epochs "$MRCNN_EPOCHS" \
                --batch "$MRCNN_BATCH" \
                --num_workers "$MRCNN_WORKERS"
        fi

        if [ "$ENABLE_FASTRCNN" -eq 1 ]; then
            run_step "05" "train_fastrcnn" \
                python scripts/training/train_fastrcnn.py \
                --dataset "$DATASET_NAME" \
                --epochs "$FASTRCNN_EPOCHS" \
                --batch "$FASTRCNN_BATCH" \
                --num_workers "$FASTRCNN_WORKERS"
        fi

        # Step C: Evaluation
        if [ "$ENABLE_YOLO" -eq 1 ]; then
            YOLO_WEIGHTS=$(cat last_yolo_weights_path.txt 2>/dev/null || echo "runs_comparison/yolo11m-seg_carparts-seg/weights/best.pt")
            run_step "06a" "evaluate_yolo" python scripts/evaluation/evaluate_confusion_matrix.py --model yolo --weights "$YOLO_WEIGHTS" --dataset "$DATASET_NAME"
        fi
        if [ "$ENABLE_MASKRCNN" -eq 1 ]; then
            run_step "06e" "evaluate_maskrcnn" python scripts/evaluation/evaluate_confusion_matrix.py --model maskrcnn --weights runs_comparison/maskrcnn/best_model.pt --dataset "$DATASET_NAME"
        fi
        if [ "$ENABLE_FASTRCNN" -eq 1 ]; then
            run_step "06f" "evaluate_fastrcnn" python scripts/evaluation/evaluate_confusion_matrix.py --model fastrcnn --weights runs_comparison/fastrcnn/best_model.pt --dataset "$DATASET_NAME"
        fi
        run_step "07" "compare_all_models" python scripts/evaluation/compare_all_models.py --dataset "$DATASET_NAME"
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
        echo "[TASK] Executing Full Pipeline (Dataset Prep -> Train -> Eval)..."

        collect_training_hparams

        run_step "01" "dataset_prep" bash scripts/data/download_and_prepare_datasets.sh

        # Read dataset choice AFTER prep has run
        if [ -f "datasets/.dataset_choice.txt" ]; then
            DATA_CHOICE=$(cat datasets/.dataset_choice.txt)
            if [ "$DATA_CHOICE" == "1" ]; then
                DATASET_NAME="custom_carparts"
            elif [ "$DATA_CHOICE" == "3" ]; then
                DATASET_NAME="combined_carparts"
            else
                DATASET_NAME="carparts-seg"
            fi
        else
            DATASET_NAME="carparts-seg"
        fi
        run_step "02" "train_yolo" python scripts/training/train_yolo_seg.py --model yolo11m-seg --epochs "$YOLO_EPOCHS" --batch "$YOLO_BATCH" --workers "$YOLO_WORKERS" --dataset "$DATASET_NAME"
        run_step "03b" "train_maskrcnn" python scripts/training/train_maskrcnn.py --dataset "$DATASET_NAME" --epochs "$MRCNN_EPOCHS" --batch "$MRCNN_BATCH" --num_workers "$MRCNN_WORKERS"
        run_step "05" "train_fastrcnn" python scripts/training/train_fastrcnn.py --dataset "$DATASET_NAME" --epochs "$FASTRCNN_EPOCHS" --batch "$FASTRCNN_BATCH" --num_workers "$FASTRCNN_WORKERS"

        YOLO_WEIGHTS=$(cat last_yolo_weights_path.txt 2>/dev/null || echo "runs_comparison/yolo11m-seg_carparts-seg/weights/best.pt")
        run_step "06a" "evaluate_yolo" python scripts/evaluation/evaluate_confusion_matrix.py --model yolo --weights "$YOLO_WEIGHTS" --dataset "$DATASET_NAME"
        run_step "06e" "evaluate_maskrcnn" python scripts/evaluation/evaluate_confusion_matrix.py --model maskrcnn --weights runs_comparison/maskrcnn/best_model.pt --dataset "$DATASET_NAME"
        run_step "06f" "evaluate_fastrcnn" python scripts/evaluation/evaluate_confusion_matrix.py --model fastrcnn --weights runs_comparison/fastrcnn/best_model.pt --dataset "$DATASET_NAME"

        run_step "07" "compare_all_models" python scripts/evaluation/compare_all_models.py --dataset "$DATASET_NAME"
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
