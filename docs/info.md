# Car Parts Segmentation Pipeline Documentation

## 1. Overview
This codebase is a machine learning pipeline and orchestration system designed to train, evaluate, and compare object detection and instance segmentation models for car parts. It currently supports **YOLO11m-seg**, **Mask R-CNN**, and **Fast R-CNN**, with training submitted to an Azure ML attached GPU VM. Dataset preparation and pipeline orchestration run locally via a Rust-based interactive CLI (`orchestrator/`). It is designed for ML engineers or researchers working on automotive damage assessment.

## 2. Tech Stack
- **Core Languages**: Python (ML and data logic) and Rust (interactive orchestrator CLI). Bash (pipeline runner glue).
- **ML Frameworks**: 
  - PyTorch (core deep learning backend)
  - Ultralytics (for YOLOv11m-seg)
  - `torchvision` (for Mask R-CNN)
- **Zero-Shot Annotation Tools**: Grounding DINO + SAM 2 (cloned via git for Stage A auto-annotation).
- **Azure ML SDK**: `azure-ai-ml`, `azure-identity` for submitting training jobs to Azure.
- **Utilities**: `rust_xlsxwriter` (Rust), `opencv-python`, `pycocotools`, `supervision`.

## 3. Folder/File Structure
- `orchestrator/`: Rust source code (`src/main.rs`) and configuration (`Cargo.toml`) for an interactive CLI wrapper that manages pipeline steps, logs them cleanly, and outputs an Excel summary report.
- `scripts/`: Python and Bash scripts holding the actual logic.
  - `data/`: Dataset acquisition and preprocessing (`download_and_prepare_datasets.sh`, `yolo_to_coco.py`, etc.).
  - `training/`: Training scripts per model architecture (`train_yolo_seg.py`, `train_maskrcnn.py`, `train_fastrcnn.py`, `azure_train.py`).
  - `inference/`: Scripts to run models on test images (`infer_both_models.py`) and zero-shot labeling (`auto_annotate_carparts.py`).
  - `evaluation/`: Generates confusion matrices (`evaluate_confusion_matrix.py`) and cross-model comparison tables (`compare_all_models.py`).
  - `utils/`: Helpers like `weights_downloader.py`.
- `setup_all.sh`: The master installation script that sets up the `venv`, installs PIP dependencies, clones repos, and downloads weights.
- `run_full_pipeline.sh`: A Bash alternative to the Rust orchestrator that prompts the user and runs the chosen steps end-to-end.
- `PROJECT_LOG.md` / `README.md`: Historical experimental logs and project instructions.

## 4. How It Works
1. **Entry Point**: The user runs either `./run_full_pipeline.sh` or the compiled Rust `orchestrator`. Both provide an interactive menu to select tasks (Annotate, Train, Test, or Full Pipeline).
2. **Data Preparation**: If training, `scripts/data/download_and_prepare_datasets.sh` fetches external data (like DSMLR or carparts-seg) and optionally merges it with local data in `RAW_DATASET/`. Python scripts convert formats (e.g. YOLO to COCO) and create dataset splits.
3. **Training**: The orchestrator triggers training scripts (e.g. `scripts/training/train_yolo_seg.py`) in sequence based on user selection. It streams output to the terminal and to dedicated log files in `logs/run_<timestamp>/`. Model weights are saved in `runs_comparison/run_<timestamp>/<model_name>/`.
4. **Evaluation**: Once training completes, `evaluate_confusion_matrix.py` is invoked for each trained model to test on a validation set and compute confusion matrices (specifically tracking left/right mixups). 
5. **Output**: Finally, `compare_all_models.py` builds a comparative text table, and the Rust orchestrator aggregates the JSON metrics into a formatted Excel report (`car_part_training_results.xlsx`).

## 5. Configuration
- **Hyperparameters**: Requested interactively via standard input (Epochs, Batch size, Dataloader workers) and passed to the Python scripts via CLI arguments.
- **Environment Variables**: 
  - None currently active. Previously `ENABLE_MASKDINO=1` and `ENABLE_ONEFORMER=1` were opt-in flags — those models have been removed.
- **Hardcoded Settings/Paths**: 
  - The Rust orchestrator assumes the project root is `..` relative to its execution directory.
  - Default input/output paths (e.g. `RAW_DATASET/IMAGES`, `./datasets/auto_annotated`, `./test`) are defined directly in `run_full_pipeline.sh` and `main.rs`.
  - Checkpoint paths for evaluation fallback to specific absolute/relative strings if dynamically saved paths (`last_yolo_weights_path.txt`) are missing.

## 6. Setup & Usage
**Prerequisites**: A Linux environment with an NVIDIA GPU and driver installed. 

**Steps**:
1. Place the project folder in a persistent directory (e.g., `~/car_parts_pipeline`). 
   > **Note**: Do not run from `/mnt` on Azure VMs, as the script will warn you that this temp drive wipes upon deallocation.
2. Run `./setup_all.sh` to initialize the Python virtual environment (`venv`), install PIP dependencies, clone SAM2/GroundingDINO, and fetch starting weights.
3. Activate the environment: 
   ```bash
   source venv/bin/activate
   ```
   *(This must be done in every new terminal session, as the scripts actively check for `$VIRTUAL_ENV`)*.
4. Run the pipeline:
   ```bash
   ./run_full_pipeline.sh
   ```
   *(Or navigate to `orchestrator/` and use `cargo run` if Rust is installed).*
5. Follow the interactive menu to annotate data, train models, or evaluate.

## 7. Data Flow / I/O
- **Inputs**: 
  - Local raw images (default: `RAW_DATASET/IMAGES`).
  - Online datasets (downloaded on the fly).
- **Processing**: Datasets are formatted into standard COCO JSON or YOLO structure in the `datasets/` directory.
- **Outputs**: 
  - Per-step log files in `logs/run_<timestamp>/`.
  - Trained model weights (e.g. `.pt`, HuggingFace format) in `runs_comparison/run_<timestamp>/`.
  - JSON metric files containing training loss and validation mAP50.
  - Excel summary reports (`car_part_training_results.xlsx`).
  - Annotated test images and CVAT XML formats (if running inference).

## 8. Known Issues / Gotchas
- **Azure `/mnt` Wipe Risk**: Mentioned above. Running from `/mnt` is dangerous.
- **Venv Enforcement**: Pipeline scripts will immediately exit if `VIRTUAL_ENV` is not detected.
- **Dependency Conflicts**: `transformers` is strictly pinned to `4.46.3` because newer versions introduce `torch.load` security restrictions that break Mask2Former loading. `iopath` is pinned to `0.1.9` due to conflicts between Detectron2 and SAM2.
- **Background Pixel Masks**: Background pixels in semantic maps are handled as `255` instead of `0` in `convert_segmentation_map_to_binary_masks` to avoid `KeyError: 0`.

## 9. Extending It
- **Adding a new model architecture**:
  1. Create a new training script (e.g., `scripts/training/train_newmodel.py`) accepting standard args (`--dataset`, `--epochs`, `--batch`, `--output_dir`).
  2. Implement inference and metrics saving logic similar to the existing ones (emitting JSON metric files per epoch).
  3. Add the model to the interactive menus in `orchestrator/src/main.rs` (under `ModelSelection`) and `scripts/run_full_pipeline.sh`.
  4. Add its evaluation call to the list of `eval_jobs` in `main.rs` and the `case` blocks in `run_full_pipeline.sh`.
- **Adding new data sources**: Modify `scripts/data/download_and_prepare_datasets.sh` to fetch or organize the data, and update dataset combiners like `combine_datasets.py` or `yolo_to_coco.py` to handle custom taxonomies.
