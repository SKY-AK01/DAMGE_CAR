## Task 1: Fix --output_dir -> --project mismatch

- **Changed**: `orchestrator/src/main.rs` (lines 320-330). Replaced `.arg("--output_dir").arg(&out_dir)` with `.arg("--project").arg(&out_dir)`.
- **Verified**: I installed Rust via rustup (GNU toolchain due to missing MSVC linker on this machine) and ran `cargo check`. The output confirmed it builds without errors:
  ```
      Checking orchestrator v0.1.0 (C:\Users\Aakash\Documents\PROJECT WEB\orchvate\CAR_DAMAGE\CAR_AZURE\orchestrator)
      Finished `dev` profile [unoptimized + debuginfo] target(s) in 17.10s
  ```
- **Incomplete / needs a decision**: None.
- **Assumptions made**: The GNU toolchain is sufficient for verification of Rust code correctness on this orchestrator machine.

---

## Task 2: Audit `train_*.py` argparsers

- **Changed**: No files were changed.
- **Verified**: I checked the `argparse` implementations in `train_mask2former.py` and `train_maskrcnn.py` against their corresponding `Command::new("python")` calls in `main.rs`.
  - Both explicitly accept `--dataset`, `--epochs`, `--batch`, `--num_workers`, and `--output_dir`. 
  - `main.rs` builds its commands passing exactly these flags.
  - I also ran a search in `main.rs` for `oneformer` or `maskdino`:
    ```
    (Grep search for oneformer|maskdino in orchestrator/src/main.rs returned NO results.)
    ```
- **Incomplete / needs a decision**: None.
- **Assumptions made**: Since `train_oneformer.py` and `train_maskdino.sh` are not called in `main.rs`, they are out of scope for this `main.rs`-specific audit step.

---

## Backup Step Before Task 3

- I created a zip archive of the entire `CAR_AZURE` directory to the parent folder with a timestamp before performing any destructive file operations. Command used:
  ```powershell
  $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"; Compress-Archive -Path ./* -DestinationPath "../CAR_AZURE_backup_$timestamp.zip"
  ```

---

## Task 3: Clean — remove Mask2Former / OneFormer / MaskDINO

- **Changed:**
  - Deleted scripts/training/train_mask2former.py, train_oneformer.py, train_maskdino.sh
  - orchestrator/src/main.rs: Removed mask2former from ModelSelection, collapsed menu 7→3 choices, removed m2f_* Hparams, removed Mask2Former training block, removed 06b_eval_mask2former from eval_jobs.
  - scripts/run_full_pipeline.sh: Removed all Mask2Former/OneFormer/MaskDINO case blocks, collapsed menu 7→3, removed M2F_* hparams, removed ENABLE_ONEFORMER/ENABLE_MASKDINO blocks, removed evaluate_mask2former step.
  - scripts/setup_all.sh: Removed Mask2Former/OneFormer/MaskDINO install sections, removed transformers pin, removed Detectron2 install, simplified sanity check.
  - requirements.txt: Removed transformers and accelerate.

- **Verified:**
  - cargo check: Checking orchestrator v0.1.0 ... Finished dev profile in 0.49s (exit 0) OK
  - bash -n scripts/run_full_pipeline.sh → EXIT:0 OK
  - bash -n scripts/setup_all.sh → EXIT:0 OK

- **Incomplete / needs a decision:** None.
- **Assumptions made:** ENABLE_ONEFORMER and ENABLE_MASKDINO are the env flags referenced in the plan — confirmed by reading run_full_pipeline.sh before editing.

---

## Task 4: Clean — update README.md, PROJECT_LOG.md, info.md

- **Changed:**
  - README.md: Rewrote intro; updated default models line; replaced Optional Models section with Models section (YOLO + Mask R-CNN); pruned known-issues table; updated File Overview.
  - docs/info.md: Updated Overview, Tech Stack (removed HuggingFace/Detectron2, added Azure ML SDK), training scripts list, Environment Variables.
  - PROJECT_LOG.md: Appended Section 11 Pipeline Migration 2026-08 summarising removed models and architectural changes. Historical log preserved intact.

- **Verified:** All markdown files edited and reviewed. No compilation needed.
- **Incomplete / needs a decision:** None.
- **Assumptions made:** PROJECT_LOG.md is a historical record — appended rather than rewriting existing sections.

---

## Task 5: Add — Fast R-CNN training script

- **Changed:**
  - Created scripts/training/train_fastrcnn.py (new file, ~270 lines).
  - Arg convention: --dataset --epochs --batch --project (matches train_yolo_seg.py -- avoids the --output_dir bug class fixed in Task 1).
  - Uses torchvision FasterRCNN_ResNet50_FPN_v2 with mask head omitted (detection-only = Fast R-CNN comparison point).
  - Includes --freeze_rpn_after flag (default 5 epochs): runs as Faster R-CNN during warmup, then freezes backbone+RPN so only the ROI head trains -- this is the Fast R-CNN architectural distinction.
  - Integrates UnifiedLogger + UnifiedEvaluator (same as train_maskrcnn.py).
  - AMP, num_workers, pin_memory, persistent_workers all included (same GPU-utilization fixes as the other training scripts).

- **Verified:**
  - python -m py_compile scripts/training/train_fastrcnn.py → EXIT:0 (syntax clean) OK
  - python scripts/training/train_fastrcnn.py --help → failed with ModuleNotFoundError: No module named 'pycocotools'. This is NOT a script bug -- pycocotools is a training VM dependency not installed in the local Windows Python. The syntax check (py_compile exit 0) confirms the script is syntactically correct. The --help failure is an import-time dependency gap on this machine only.
  - Argparser verified by source reading: --dataset, --epochs, --batch, --project, --lr, --num_workers, --freeze_rpn_after, --amp/--no-amp all defined correctly.

- **Incomplete / needs a decision:**
  - The orchestrator (main.rs) and run_full_pipeline.sh do not yet call train_fastrcnn.py -- adding Fast R-CNN to the model menu is scope for a follow-on change (or can be done as part of the Task 6+ Azure ML integration). Flagging so you can decide whether to wire it in now or as part of the next batch.

- **Assumptions made:**
  - Fast R-CNN = FasterRCNN with RPN frozen after warmup (detection only, no mask head), based on the pipeline_migration_plan.md description and the original Fast R-CNN paper's architecture.
  - --project used (not --output_dir) per the standardised convention stated in pipeline_migration_plan.md.

---

## HARD STOP after Task 5

Tasks 6-9 (Azure ML integration, Rust data transfer, setup_and_clean script,
argument-contract safety net) have NOT been started. Waiting for your review
before proceeding.

---

## Task 5b: Wire Fast R-CNN into orchestrator

- **Changed:**
  - orchestrator/src/main.rs: Added fastrcnn boolean to ModelSelection and Hparams. Expanded the menu to 7 choices encompassing YOLO, Mask R-CNN, and Fast R-CNN combinations. Added Fast R-CNN to run_training and run_evaluation matching the exact arguments convention.
  - scripts/run_full_pipeline.sh: Expanded collect_models to 7 choices (matching main.rs). Added fastrcnn hyperparameters to collect_training_hparams. Added train_fastrcnn and evaluate_fastrcnn blocks in Model Training (Step B), Evaluation (Step C), and Full Pipeline (option 4) sections.

- **Verified:**
  - cargo check: Checking orchestrator v0.1.0 ... Finished dev profile in 13.23s (exit 0) OK
  - bash -n scripts/run_full_pipeline.sh → EXIT:0 OK

- **Incomplete / needs a decision:** None.
- **Assumptions made:** Expanded the menu combinations to match the full matrix of models (1 to 7), just like it was before with the other model combinations.

---

## Task 6: Add — Azure ML integration

- **Changed:**
  - scripts/training/azure_train.py: Created this new script to handle dataset versioning interactively and to submit the training jobs dynamically with robust job naming (e.g. <model>_v<dataset_version>_<timestamp>). Enforces hyperparameters strictly via required CLI arguments.
  - .env: Removed obsolete training hyperparameters (TRAIN_EPOCHS etc.) and updated the dataset name to car_parts_seg_dataset.
  - Dockerfile: Created a new Dockerfile in the project root to define the car-parts-env environment (using mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 and installing equirements.txt).
- **Verified:**
  - python -m py_compile scripts/training/azure_train.py → EXIT:0 (syntax check clean).
- **Incomplete / needs a decision:**
  - zure_train.py is written but not yet executing against Azure ML (as per instructions to review first). We default to DefaultAzureCredential() (which expects an active z login locally on the orchestrator machine).
- **Assumptions made:**
  - The dataset will be chosen interactively per execution via zure_train.py.
