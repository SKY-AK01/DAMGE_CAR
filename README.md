# Car Parts Segmentation — Model Comparison Pipeline

Compares object detection/segmentation architectures on a car-parts dataset.
Currently supports **YOLO11m-seg**, **Mask R-CNN**, **Fast R-CNN**, **Mask2Former**, and more,
with training submitted to Azure ML on an attached GPU VM.
Dataset preparation and pipeline orchestration run via `python orchestrator.py`.

---

## ⚠️ IMPORTANT: Where to put this folder

**Do NOT run this from `/mnt` on an Azure VM.** On many Azure VM sizes,
`/mnt` is the *temporary* resource disk — it gets **wiped every time you
Stop (Deallocate) the VM**. This happened once already and cost a full
re-run of everything. Put this folder under your home directory instead:

```bash
mkdir -p ~/car_parts_pipeline
cd ~/car_parts_pipeline
# unzip this package here
```

`setup_all.sh` and `run_full_pipeline.sh` will both print a loud warning
(and pause 5-10s) if they detect they're running from `/mnt`, but they
won't stop you — it's your call if you know your VM's `/mnt` is persistent.

---

## Quick start

```bash
cd ~/CAR_TRAIN1
pip install -r requirements.txt
source venv/bin/activate            # do this in EVERY new terminal/tmux session!
python orchestrator.py              # interactive menu — dataset prep, train, evaluate
```

Default run trains and compares: **YOLOv11m-seg**, **Mask R-CNN**, and **Fast R-CNN**.
Takes a few hours total depending on your GPU. Run inside `tmux` so it
survives disconnects:

```bash
tmux new -s train
source venv/bin/activate
./run_full_pipeline.sh
# detach: Ctrl+B then D
# reattach later: tmux attach -t train
```

## Logs

Every run creates a timestamped folder under `logs/run_<timestamp>/` with
one clean log file per step (`01_dataset_prep.log`, `02_train_yolo.log`,
etc.) — both saved to disk AND streamed live to your terminal. No more
digging through one giant scrolling wall of mixed output. The script stops
immediately at the first real failure (not five confusing cascade errors
later), and tells you exactly which log file to check.

---

## Models

### YOLO11m-seg — default
Fast, NMS-free, single-stage segmentation. No exotic dependencies (just `ultralytics`).

### Mask R-CNN
Classic two-stage detector. No exotic dependencies (just `torchvision`, already
installed with `torch`).

### Fast R-CNN
Alternative classic object detector, supported directly via `torchvision`. Serves as an additional baseline.

---

## Every environment fix already baked into this package

These are all things that broke on the first real run and are now fixed
by default — documented here so you know why the scripts do what they do:

| Problem | Fix applied |
|---|---|
| `undefined symbol: ncclCommResume` | Root cause was never actually being inside the venv (`pip`/`python` resolved to a broken user-level install). Every script now checks `$VIRTUAL_ENV` and warns/fails loudly rather than silently using the wrong Python. |
| `pip install -e .` failing with `No module named pip` (GroundingDINO/SAM2) | Their `setup.py` scripts try to run `pip install torch` inside pip's isolated build sandbox, which has no pip in it. Fixed with `--no-build-isolation` on every git-based editable install. |
| YOLO `--output_dir` flag mismatch | Orchestrator was passing `--output_dir` but `train_yolo_seg.py` expects `--project`. Fixed. |
| YOLO weights landing in a confusing nested path (`runs/segment/runs_comparison/...`) | Fixed by using an **absolute** `project` path and explicitly passing `project`/`name` to both `model.train()` and the follow-up `model.val()` call, so nothing falls back to Ultralytics' own default location. The final weights path is also printed and saved to `last_yolo_weights_path.txt` so you never have to guess it again. |
| `weights_downloader.py` missing crashing the entire `setup_all.sh` via `set -e` | The call is now guarded with a file-existence check — a missing file just prints a warning and continues (it only affects Stage A annotation, not Stage B training). |
| Copy/paste truncating long terminal commands | Everything is now in proper `.sh` files instead of one-line pasted commands. |

---

## File overview

**Setup**
- `setup_all.sh` — one-time environment setup (venv, all deps, weight downloads)
- `requirements.txt` — plain pip list if you want to manage the venv yourself
- `weights_downloader.py` — downloads Grounding DINO + SAM2 weights (Stage A only)

**Dataset**
- `download_and_prepare_datasets.sh` — downloads + splits + converts both datasets
- `prepare_dsmlr_split.py` — auto 80/10/10 split for DSMLR (no official split)
- `yolo_to_coco.py` — converts YOLO-seg polygon format to COCO JSON
- `register_carparts_dataset.py` — registers datasets with Detectron2 (MaskDINO only)

**Training**
- `train_yolo_seg.py` — YOLOv11m-seg (Ultralytics)
- `train_maskrcnn.py` — Mask R-CNN (torchvision)
- `train_fastrcnn.py` — Fast R-CNN (torchvision)
- `azure_train.py` — Submits training jobs to Azure ML

**Evaluation**
- `evaluate_confusion_matrix.py` — runs one model on the test set, builds its confusion matrix
- `compare_all_models.py` — final side-by-side table across all models you ran

**Using your trained model on new images**
- `infer_both_models.py` — run YOLO + Mask R-CNN on any folder of images, saves annotated images + CVAT XML per model
- `auto_annotate_carparts.py` — Stage A: zero-shot Grounding DINO + SAM2 auto-labeling (no training needed — use to bootstrap a dataset for parts outside the taxonomy)

**Orchestration**
- `run_full_pipeline.sh` — runs everything above in the right order with clean logging

**Documentation**
- `PROJECT_LOG.md` — full narrative history: original goal, reasoning, every issue encountered and how it was diagnosed, and the final experimental results
- `README.md` — this file
