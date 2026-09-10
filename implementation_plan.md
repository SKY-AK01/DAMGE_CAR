# Pipeline Cleanup & Standardization Plan: YOLO11m, Mask R-CNN & Mask2Former

## Overview
This plan cleans up the pipeline, eliminates clutter, removes dead/obsolete scripts, and standardizes file saving and logging across **YOLO11m-seg**, **Mask R-CNN**, and **Mask2Former**. It ensures that whether a model is run individually (standalone script) or jointly (via `orchestrator.py` or `run_full_pipeline.sh`), the resulting directory structure, best model weights, metrics, and logs are **identical, predictable, and self-contained**.

> [!IMPORTANT]
> **No training will be executed locally.** Per your instructions, all changes and model training scripts will be cross-checked and validated via dry runs / argument verification / syntax inspection, ready to be copied to and executed on your Azure VM.

---

## User Review Required
> [!NOTE]
> 1. **Orchestrator Menu Simplification**: The main menu in `orchestrator.py` is consolidated from 13 cluttered options down to 9 clean, logical options (merging redundant train/pipeline options and Azure options).
> 2. **Removal of Per-Run Deletion Prompt**: The disruptive `interactive_cleanup_prompt()` that popped up before *every* training session asking if you want to delete `runs_comparison` or `datasets/raw` is removed from normal training flows and kept strictly in the dedicated Cleanup option (Option 9) and via CLI flag.
> 3. **Dual-Stream Logging (`train.log`)**: Every model's `UnifiedLogger` will now automatically tee all terminal stdout/stderr to `<output_dir>/train.log`. This guarantees that running a model individually produces the exact same `train.log` as running through the orchestrator.
> 4. **Obsolete Scripts Removed**:
>    - `scripts/data/register_carparts_dataset.py` (MaskDINO leftover)
>    - `scripts/evaluation/compare_all_models.py` (superseded by `compare_models.py`)
>    - `scripts/setup_and_clean.py` (superseded by `cleanup_pipeline.py`)
>    - `scripts/diagnose_small_mask.py` and `scripts/diagnose_zero_area.py` (one-off debug files)

---

## Standardized Output Directory Layout

Every model run—whether launched standalone or via orchestrator—will output to a designated `<output_dir>` with the **exact same structure**:

```
<output_dir>/                               # e.g., runs_comparison/yolo11m-seg OR runs_comparison/run_<timestamp>/yolo11m-seg
├── weights/
│   ├── best.pt                             # Best model checkpoint (or best/ dir for Mask2Former)
│   └── last.pt                             # Last model checkpoint (or last/ dir for Mask2Former)
├── best_model.pt                           # Direct root-level alias to best weights (or best_model/ dir)
├── last_model.pt                           # Direct root-level alias to last weights
├── train.log                               # Complete console output & training log (created in both single & joint runs)
├── metrics.csv                             # Standardized epoch-by-epoch table from UnifiedLogger
├── best_metrics.json                       # Summary of best epoch and all metrics (box & mask mAP, precision, recall)
├── metrics_epoch_*.json                    # Per-epoch detailed metrics
└── eval/                                   # Validation / test evaluation curves & prediction samples
```

When running jointly via `orchestrator.py` or `run_full_pipeline.sh`, the root `runs_comparison/run_<timestamp>/` contains:
```
runs_comparison/run_<timestamp>/
├── yolo11m-seg/                            # Standardized YOLO folder (as above)
├── maskrcnn/                               # Standardized Mask R-CNN folder (as above)
├── mask2former/                            # Standardized Mask2Former folder (as above)
├── model_comparison_report.xlsx            # Multi-sheet Excel workbook (Summary + per-model tabs)
├── comparison_plots/                       # High-res comparison plots (loss, mAP50, VRAM, speed)
├── comparison_metrics.json                 # Side-by-side JSON comparison
├── comparison_summary.txt                  # Plaintext summary table
└── logs/                                   # Orchestrator-level pipeline execution logs
    ├── orchestrator.log
    ├── 01_dataset_prep.log
    └── run_config.json
```

---

## Proposed Changes

### Component 1: Unified Logging & Checkpointing
#### [MODIFY] [scripts/evaluation/unified_logger.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/evaluation/unified_logger.py)
- Add dual-stream console + file logger: All output is automatically mirrored into `<output_dir>/train.log`.
- Ensure flush after every epoch so logs are written immediately in real-time even on VM crashes or disconnects.
- Keep schema consistency for CSV columns (`map50_mask`, `map50_box`, `precision_mask`, `recall_mask`, `train_loss`, `val_loss`, `gpu_mem_gb`, `epoch_time_sec`).

---

### Component 2: Core Model Training Scripts (Cross-Check & Fixes)

#### [MODIFY] [scripts/training/train_yolo_seg.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/training/train_yolo_seg.py)
- **Path Portability**: Update `make_yaml()` to always write the resolved absolute posix path dynamically, so moving the dataset between Windows (`C:\...`) and Linux VM (`/home/...`) works seamlessly.
- **Output Standardization**:
  - Accept `--output_dir` and output directly to `<output_dir>/` without nested redundant directories.
  - Save best weights to `<output_dir>/weights/best.pt` and create root alias `<output_dir>/best_model.pt`.
  - Save last weights to `<output_dir>/weights/last.pt` and `<output_dir>/last_model.pt`.
- **Epoch 1 Safety**: Safeguard `trainer.best` in the evaluation callback (fallback to `trainer.last` if `trainer.best` is not yet created).
- **Cleanup**: Stop writing `last_yolo_weights_path.txt` in the repository root (write inside `<output_dir>/` if needed).

#### [MODIFY] [scripts/training/train_maskrcnn.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/training/train_maskrcnn.py)
- Default `--output_dir` to `runs_comparison/maskrcnn`.
- Standardize checkpoint savings:
  - Save to `<output_dir>/weights/best.pt` and `<output_dir>/best_model.pt`.
  - Save to `<output_dir>/weights/last.pt` and `<output_dir>/last_model.pt`.
- Ensure `train.log` is captured automatically via `UnifiedLogger`.

#### [MODIFY] [scripts/training/train_mask2former.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/training/train_mask2former.py)
- Default `--output_dir` to `runs_comparison/mask2former`.
- Standardize checkpoint savings:
  - Save HuggingFace model and processor config to `<output_dir>/weights/best/` AND `<output_dir>/best_model/`.
  - Save last checkpoint to `<output_dir>/weights/last/` AND `<output_dir>/last_model/`.
- Ensure `train.log` is captured automatically via `UnifiedLogger`.

---

### Component 3: Orchestrator & Shell Pipeline Simplification

#### [MODIFY] [orchestrator.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/orchestrator.py)
- **Streamline Main Menu** to 9 clear options:
  1. `Train Models Locally` (Select YOLO11m, Mask R-CNN, Mask2Former, or All 3 -> Dataset Prep -> Train -> Evaluate -> Comparison Report)
  2. `Compare Trained Models` (Run `compare_models.py` on existing runs -> plots + multi-sheet Excel)
  3. `Test on Trained Models (Inference)` (Run `infer_both_models.py` across YOLO, Mask R-CNN, Mask2Former)
  4. `Prepare Dataset` (Download missing external data, match taxonomy, build `combined_carparts`, generate COCO JSONs, verify labels)
  5. `Dataset Analytics` (Image counts, category breakdown, annotation statistics)
  6. `Preprocess Utilities` (Submenu: `yolo_to_coco`, `combine`, `verify_labels`, `hash_dataset`)
  7. `GPU Capacity Check` (Run batch & worker sweep for the 3 target models)
  8. `Train on Azure ML` (Submit GPU training job to Azure ML with optional 3-epoch dry run flag)
  9. `Cleanup Pipeline` (Interactively wipe stale run outputs and derived artifacts)
  10. `Exit`
- **Eliminate Startup Deletion Prompt**: Remove `interactive_cleanup_prompt()` call inside `ensure_and_prepare_datasets()`. It should only run when the user explicitly chooses Option 9 or sets CLI flags.
- **Pass Consistent `--output_dir`**: Always pass `--output_dir <base_out_dir>/<model_name>` so joint runs and standalone runs use the exact same directory semantics.

#### [MODIFY] [scripts/run_full_pipeline.sh](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/run_full_pipeline.sh)
- Clean menu and steps for the 3 target models (`yolo11m-seg`, `maskrcnn`, `mask2former`).
- Step 2 and Step 4 consistently use `compare_models.py` for uniform multi-sheet Excel and plot generation.

---

### Component 4: Evaluation, Inference & Cleanup

#### [MODIFY] [scripts/evaluation/evaluate_confusion_matrix.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/evaluation/evaluate_confusion_matrix.py)
- Remove `get_predictions_maskdino()` and `detectron2` dependency completely.
- Add `--out_dir` argument (defaulting to `runs_comparison/confusion_matrices`, but configurable so the orchestrator can store confusion matrices inside `<run_dir>/confusion_matrices`).

#### [MODIFY] [scripts/evaluation/compare_models.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/evaluation/compare_models.py)
- Confirm auto-discovery handles both flat directories (`runs_comparison/yolo11m-seg`, `runs_comparison/maskrcnn`, `runs_comparison/mask2former`) and timestamped directories (`runs_comparison/run_<timestamp>/<model_name>`).
- Ensure generated Excel workbook puts the **Summary** tab as the very first sheet.

#### [MODIFY] [scripts/inference/infer_both_models.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/inference/infer_both_models.py)
- Standardize weight discovery to check `weights/best.pt`, `best_model.pt`, and `weights/best` across both flat and timestamped run structures.

#### [MODIFY] [scripts/training/azure_train.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/training/azure_train.py)
- Clean environment description string (remove mentions of Fast R-CNN and SAM2).

#### [MODIFY] [scripts/training/capacity_check.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/training/capacity_check.py)
- Clean docstrings and comments to reference only the 3 active models.

#### [MODIFY] [scripts/setup_all.sh](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/setup_all.sh)
- Remove `MaskDINO` references from comments and warnings.
- Add `transformers` and `torchvision` to the post-installation verification check.

#### [DELETE] Obsolete and Dead Files:
- [scripts/data/register_carparts_dataset.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/data/register_carparts_dataset.py)
- [scripts/evaluation/compare_all_models.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/evaluation/compare_all_models.py)
- [scripts/setup_and_clean.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/setup_and_clean.py)
- [scripts/diagnose_small_mask.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/diagnose_small_mask.py)
- [scripts/diagnose_zero_area.py](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/scripts/diagnose_zero_area.py)

---

## Verification Plan

### Automated / Non-Modifying Script Checks (DO NOT RUN FULL TRAINING)
1. **Python Syntax & Import Verification**:
   - Run compilation checks on all modified scripts to ensure zero syntax errors:
     ```powershell
     python -m py_compile orchestrator.py
     python -m py_compile scripts/training/train_yolo_seg.py
     python -m py_compile scripts/training/train_maskrcnn.py
     python -m py_compile scripts/training/train_mask2former.py
     python -m py_compile scripts/training/azure_train.py
     python -m py_compile scripts/training/capacity_check.py
     python -m py_compile scripts/evaluation/unified_logger.py
     python -m py_compile scripts/evaluation/evaluate_confusion_matrix.py
     python -m py_compile scripts/evaluation/compare_models.py
     python -m py_compile scripts/inference/infer_both_models.py
     ```
2. **CLI Argument Parsing & Help Flag Check**:
   - Run `--help` on each training, evaluation, and orchestrator script to ensure arguments parse without exceptions:
     ```powershell
     python scripts/training/train_yolo_seg.py --help
     python scripts/training/train_maskrcnn.py --help
     python scripts/training/train_mask2former.py --help
     python scripts/evaluation/compare_models.py --help
     python scripts/inference/infer_both_models.py --help
     python orchestrator.py --help
     ```
3. **Dead Code & Reference Audit**:
   - Grep across the codebase to ensure zero remaining references to `fastrcnn` (except torchvision's RoI head `FastRCNNPredictor`), `maskdino`, `sam2_seg`, `segformer`, `oneformer`, or deleted scripts.
4. **Code Structure & Readiness Confirmation**:
   - Verify that all three model training entry points, dataset loaders, loss functions, learning rate schedules, and checkpoint saving paths are properly aligned for the Linux GPU VM.
