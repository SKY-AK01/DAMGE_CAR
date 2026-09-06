# Pipeline Migration Plan — Fix / Clean / Add

Based on `info.md` (current architecture) and `run_analysis.md` (known bug), adapted for
the new project scope: **YOLO + Mask R-CNN + Fast R-CNN only, Rust-driven data transfer,
training moved to Azure ML.**

---

## 🔧 Fix

- **`orchestrator/src/main.rs` (~line 327)** — YOLO training call passes `--output_dir`,
  but `train_yolo_seg.py`'s argparser only defines `--project`. Change:
  ```rust
  .arg("--output_dir").arg(&out_dir)
  →
  .arg("--project").arg(&out_dir)
  ```
  Recompile with `cargo build --release` after.

- **Audit every model's CLI contract**, not just YOLO. Same class of bug (Rust passes a
  flag the Python script doesn't define) can exist for Mask R-CNN / Fast R-CNN too —
  diff each `train_*.py` argparser against its corresponding `Command::new("python")...arg()`
  chain in `main.rs` before relying on the orchestrator.

- **Checkpoint fallback paths** (`last_yolo_weights_path.txt` etc.) — confirm these still
  resolve correctly once models are removed and paths/folder names change.

---

## 🧹 Clean

- **Remove unused model training scripts**: `train_mask2former.py`, `train_oneformer.py`,
  MaskDINO training/inference code.
- **Remove from `orchestrator/src/main.rs`**: `ModelSelection` enum entries for
  Mask2Former / OneFormer / MaskDINO, and their entries in the `eval_jobs` list.
- **Remove from `run_full_pipeline.sh`**: matching `case` blocks for those models.
- **Remove env flags**: `ENABLE_MASKDINO=1`, `ENABLE_ONEFORMER=1` (no longer relevant).
- **Drop dependencies only needed for removed models**:
  - `Detectron2` (MaskDINO only) — also removes the `nvcc`/CUDA-Toolkit-on-VM requirement
    that was a known gotcha.
  - Re-check whether `transformers==4.46.3` pin is still needed (it was pinned for
    Mask2Former/OneFormer) — HuggingFace Transformers likely no longer required at all
    once those two are gone.
  - Re-check `iopath==0.1.9` pin — was for Detectron2/SAM2 conflict; keep only if SAM2
    stays for zero-shot annotation.
- **Update docs**: `README.md`, `PROJECT_LOG.md`, and `info.md` itself still describe the
  5-model pipeline — rewrite to reflect YOLO / Mask R-CNN / Fast R-CNN only.
- **Local training output paths**: since training moves to Azure ML, decide whether
  `runs_comparison/run_<timestamp>/` stays as a local mirror/download target or gets
  replaced entirely by Azure job output paths.

---

## ➕ Add

- **Fast R-CNN training script** — not present in this codebase today (only Mask R-CNN
  via `torchvision` exists here); port/adapt a `train_fastrcnn.py` (the other project's
  license-plate pipeline already has this — reuse its argparser pattern, standardized on
  `--dataset --epochs --batch --project` to match the YOLO convention and avoid repeating
  the Fix above).

- **Azure ML training integration** (mirrors the other project's `PIPELINE_GUIDE.md` setup):
  - `scripts/upload_dataset.py` — uploads `dataset/` as a versioned Azure ML Data Asset.
  - `scripts/azure_train.py` — authenticates via `DefaultAzureCredential`, resolves the
    attached GPU VM compute, builds/reuses the Docker environment, submits the training
    job, enforces `max_concurrent_runs=1`, optionally streams logs.
  - `.env` entries: `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME`,
    `AZURE_COMPUTE_NAME`, `AZURE_DATASET_NAME`, `AZURE_DATASET_VERSION`.
  - Orchestrator step: add "Submit to Azure ML" as a pipeline step (replacing or sitting
    alongside local training) in both `main.rs` and `run_full_pipeline.sh`.

- **Rust-based data transfer layer** — move dataset upload out of Python into the Rust
  orchestrator for speed + safety:
  - Multithreaded blob upload (equivalent to the `upload_to_blob.py` logic, ported to Rust
    using the `azure_storage_blobs` crate or shelling out to `az storage blob upload-batch`).
  - Add integrity checks (file count / checksum verification pre- and post-upload) so a
    partial or failed transfer is caught before a training job is submitted against it.
  - Timestamped destination folders per run (same pattern as the current Python uploader)
    so nothing overwrites a prior dataset version.
  - Exclude known junk dirs (`venv/`, `target/`, `__pycache__/`, etc.) at the Rust layer too.

- **Argument-contract safety net** — given the recurring Rust↔Python CLI mismatch bug,
  add either a shared JSON/TOML schema both sides read, or a small integration test that
  runs each `train_*.py --help` and asserts the orchestrator's argument list is a subset —
  catches this bug class before a training run fails.

- **`setup_and_clean` script** (Rust, alongside the orchestrator, or a Python step run
  first) — combines first-time setup with ongoing housekeeping so raw data and archives
  don't rot into the same mess as the model list did:
  - **Setup**: on first run, creates the expected folder skeleton and prints exactly where
    raw data belongs — e.g. two explicit, separate subfolders such as
    `RAW_DATASET/open_source/` and `RAW_DATASET/user_data/`, so the origin of every image
    is always traceable and the two sources are never dumped into one undifferentiated pile.
  - **Clean-as-it-archives**: before/after each run, moves finished run output
    (`runs_comparison/run_<timestamp>/`, old logs) into `archive/` rather than deleting or
    leaving it in place, and prunes stale intermediates (augmentation cache, temp files) in
    the same pass — so working folders stay bounded run over run.
  - **Source presence check**: fail loudly, before training starts, if either
    `open_source/` or `user_data/` is missing or empty — don't silently proceed on one
    source when the pipeline is meant to combine both.
  - **Combined-run check**: verify the dataset-merge step (`combine_datasets.py`) actually
    ran on *both* sources for this run — record source counts pre-merge and assert
    `merged_count == open_source_count + user_data_count`, so a partial or skipped merge
    doesn't go unnoticed.
  - **Label conversion check**: immediately after merge (before training is allowed to
    start), verify every converted label file has a matching image, class indices are in
    range, and coordinates are correctly normalized — surfacing failures here rather than
    only relying on `verify_labels.py` later in the pipeline.

- **Rust for CPU/IO-bound pipeline steps** (Task 10 — not blocking, pick up once Tasks 1-9
  are stable) — actual GPU training won't speed up from Rust, but the prep work around it
  is often the real bottleneck:
  - **Label conversion** (`yolo_to_coco.py`) and **`combine_datasets.py`** — parallel
    parse/merge with `serde_json` + `rayon` instead of Python's single-threaded `json`
    loop; matters most once open-source + user-data merging is routine (Task 8).
  - **`augment_to_target.py`** — image augmentation is embarrassingly parallel; Rust's
    `image` crate + `rayon` across all CPU cores vs. single-threaded Python/OpenCV.
  - **`verify_labels.py` / `split_dataset.py` / `check_balance.py`** — same
    parallelization win, lower priority unless dataset size makes them a real bottleneck.
  - **DataLoader feed during training** — if the GPU is idling on CPU-bound decode/augment,
    a Rust preprocessing step (via `PyO3` inside `Dataset.__getitem__`) is the one case
    that actually improves training wall-clock time, not just prep time. Worth profiling
    first (e.g. `nvidia-smi` GPU utilization during a training run) before investing here.
  - **Integrity checks**: `blake3` hashing for dataset upload verification (ties into the
    Task 7 data-transfer layer) — much faster than Python's `hashlib` for large datasets.
