# Orchvate ML Pipeline — Complete Flow Guide

This document explains exactly how the pipeline works end-to-end: from raw data on your local machine, through Azure ML, to a trained model on the GPU VM.

---

## Overview

```
Your Machine (Windows)
│
│  python run_pipeline.py
│
├─── Data Preparation (local)
│       XML annotations → YOLO labels
│       Augmentation → Dataset split → Verification
│
├─── Upload to Azure ML
│       dataset/ folder → Azure Blob Storage (Data Asset)
│
└─── Submit Training Job
        Azure ML SDK → Job Queue → GPU VM (gpu-vm-clean)
                                        │
                                        ├── Docker container spins up
                                        ├── Dataset downloaded from Blob
                                        ├── Training runs (YOLO / R-CNN / RT-DETR)
                                        └── best.pt saved to /mnt/runs/train/
```

---

## Prerequisites

### Local Machine
```bash
pip install azure-ai-ml azure-identity ultralytics python-dotenv torch torchvision
az login   # authenticate Azure CLI once
```

### Azure Resources (already configured)
| Resource | Value |
|---|---|
| Subscription | `e2d35239-5448-45ae-b156-d3309a9052a9` |
| Resource Group | `Aakash_ML` |
| Workspace | `Aakash_ML` |
| Compute | `gpu-vm-clean` (attached VM, australiaeast) |
| Dataset Asset | `licence-plate-dataset` |
| Environment | `yolo26-licence-plate-env:12` |

---

## Setting Up Azure ML for a New Project

If you are reusing this pipeline for a different project or a different Azure account, follow these steps to configure everything from scratch.

### Step 1 — Create an Azure ML Workspace

1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for **Azure Machine Learning** → click **Create**
3. Fill in:
   - **Subscription** — your Azure subscription
   - **Resource Group** — create new or use existing (e.g. `MyProject_ML`)
   - **Workspace name** — e.g. `MyProject_ML`
   - **Region** — choose closest to your GPU VM (e.g. `australiaeast`)
4. Click **Review + Create**

### Step 2 — Get Your Credentials

You need three values from Azure Portal:

**Subscription ID**
- Azure Portal → search **Subscriptions** → copy the ID

**Resource Group**
- The resource group you created above (e.g. `MyProject_ML`)

**Workspace Name**
- The workspace name you created above (e.g. `MyProject_ML`)

Update these in `scripts/azure_train.py` and `scripts/upload_dataset.py`:
```python
SUBSCRIPTION_ID = "your-subscription-id"
RESOURCE_GROUP  = "your-resource-group"
WORKSPACE_NAME  = "your-workspace-name"
```

Or set them as environment variables in `.env`:
```
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
```

### Step 3 — Attach Your GPU VM as Compute

This pipeline uses an **attached VM** (not a managed cluster). Your VM must already exist (Azure VM, on-prem, or any Linux machine with GPU).

1. In Azure ML Studio → **Compute** → **Attached compute** → **+ New**
2. Select **SSH** → fill in:
   - **Compute name** — e.g. `gpu-vm-clean`
   - **Public IP or FQDN** — your VM's public IP
   - **SSH port** — 22
   - **Username** — VM username (e.g. `orchvate`)
   - **SSH private key** — paste your private key or password
3. Click **Attach**

Update the compute name in `scripts/azure_train.py`:
```python
COMPUTE_NAME = "your-compute-name"
```

Or in `.env`:
```
AZURE_COMPUTE_NAME=your-compute-name
```

### Step 4 — Prepare the GPU VM

SSH into your VM and install Docker + NVIDIA Container Toolkit:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU is visible
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
```

Also mount a data disk to `/mnt` if available (for storing training outputs):
```bash
# Check available disks
lsblk

# Format and mount (replace sdb with your disk)
sudo mkfs.ext4 /dev/sdb
sudo mount /dev/sdb /mnt
sudo chown -R $USER:$USER /mnt

# Make it persistent across reboots
echo '/dev/sdb /mnt ext4 defaults 0 0' | sudo tee -a /etc/fstab
```

### Step 5 — Build the Docker Environment

The training runs inside a Docker container. The environment is defined in `docker/Dockerfile`.

The first time you submit a job, Azure ML builds and registers the environment automatically. This takes 5-10 minutes. After that it's cached and reused.

If you want to pre-build it manually:
```bash
# In Azure ML Studio → Environments → + Create → From Dockerfile
# Or let azure_train.py handle it on first run
```

To use a different environment version, update in `scripts/azure_train.py`:
```python
env = ml_client.environments.get(env_name, version="12")  # change version
```

### Step 6 — Authenticate Locally

```bash
# Install Azure CLI
# Windows: https://aka.ms/installazurecliwindows
# Linux:   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Verify
az account show
```

The pipeline uses `DefaultAzureCredential` which automatically picks up your `az login` session. If that fails it opens a browser window for interactive login.

### Step 7 — Configure Dataset Name

Update the dataset asset name to match your project in `scripts/azure_train.py` and `scripts/upload_dataset.py`:

```python
DATASET_ASSET_NAME = "your-dataset-name"   # e.g. "my-detection-dataset"
```

Or in `.env`:
```
AZURE_DATASET_NAME=your-dataset-name
```

### Step 8 — Update data.yaml

Update `data.yaml` to match your classes:

```yaml
path: ./dataset
train: images/train
val:   images/val
test:  images/test

nc: 2   # number of classes

names:
  0: class_one
  1: class_two
```

### Complete .env Template

Create a `.env` file in the project root with all your credentials:

```env
# Azure ML
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
AZURE_COMPUTE_NAME=your-compute-name
AZURE_DATASET_NAME=your-dataset-name
AZURE_DATASET_VERSION=latest

# Training (optional overrides — pipeline will ask interactively)
TRAIN_DEVICE=0
```

> **Never commit `.env` to git.** It is already in `.gitignore`.

---

## Starting the Pipeline

```bash
python run_pipeline.py
```

The pipeline asks three things upfront before any steps run:

### 1. Model Selection
```
[1] YOLO26       — fast, NMS-free, recommended
[2] Fast R-CNN   — two-stage, RPN frozen
[3] Faster R-CNN — two-stage, end-to-end
[4] RT-DETR      — transformer-based
```
This choice determines which training script runs and which steps are shown.

- **YOLO / RT-DETR** → runs all 13 steps
- **R-CNN** → runs data prep steps + Azure upload + Azure training only (no local training step)

### 2. Global Config
- Path to raw images directory (default: `raw1/subset_images`)
- Path to labels directory (default: `dataset/labels/all`)

### 3. Hyperparameters
| Parameter | Default | Notes |
|---|---|---|
| Epochs | 60 | |
| Batch | 4 | Use 3 for R-CNN on 12GB GPU |
| Patience | 15 | Early stopping |
| LR (lr0) | 0.005 | Initial learning rate |
| LR final (lrf) | 0.001 | Final LR via cosine schedule |
| Warmup epochs | 3 | |
| Workers | 2 | DataLoader workers |
| Image size | 512 | px, square |
| Freeze N layers | 0 | YOLO/RT-DETR only — 0=all, 10=backbone frozen, 23=backbone+neck frozen |

---

## Pipeline Steps

### Step 1 — Pipeline Reset *(optional)*
**Script:** `scripts/reset_pipeline.py`

Archives current `dataset/`, `runs/`, and labels into `backups/backup_YYYYMMDD_HHMMSS/`. Run this when starting a completely fresh experiment. Safe to skip if continuing from a previous run.

---

### Step 2 — XML to YOLO Conversion
**Script:** `scripts/xml_to_yolo.py`

Reads CVAT XML annotation files from `work/` and converts them to YOLO format (`.txt` files with normalized bounding box coordinates). Output goes to `dataset/labels/all/`.

**Input:** `work/*.xml`  
**Output:** `dataset/labels/all/*.txt`

---

### Step 3 — Complexity-Aware Augmentation *(optional)*
**Script:** `scripts/augment_to_target.py`

Interactive augmentation to hit a target image count. Generates new images+labels using techniques like flipping, brightness shifts, mosaic, and rotation. Respects "complexity" tiers (simple/medium/complex plates) to keep class distribution balanced.

**Input:** Raw images + labels  
**Output:** Additional images and labels added to the pool

---

### Step 4 — Dataset Splitting
**Script:** `scripts/split_dataset.py`

Stratifies the full label pool into Train / Val / Test splits while preserving complexity distribution. Default split: 70% train, 15% val, 15% test.

**Output:**
```
dataset/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

---

### Step 5 — Label Verification
**Script:** `scripts/verify_labels.py`

Checks every label file for:
- Correct YOLO format (5 values per line: class x y w h)
- Values in range [0, 1]
- Matching image file exists
- No empty files

Removes or flags corrupt labels.

---

### Step 6 — Balance Check
**Script:** `scripts/check_balance.py`

Reports class distribution across train/val/test splits. Shows:
- Per-class image counts and percentages
- Imbalance ratio between classes
- Recommended `cls_pw` value for weighted loss

---

### Step 7 — Class Balancing *(optional)*
**Script:** `scripts/oversample.py`

Interactive. Choose to:
- Duplicate minority class samples (oversampling)
- Merge `painted_plate_number` into `licence_plate` (if ratio is too extreme)

---

### Step 8 — Generate Training Report
**Script:** `scripts/generate_training_report.py`

Generates `TRAINING_PLAN_REPORT.md` with full dataset stats, complexity breakdown, model config, and hyperparameter summary. Useful for tracking experiments.

---

### Step 9 — Select Best Pretrained Model *(optional)*
**Script:** `scripts/use_best_model.py`

**This is where you decide whether to fine-tune from `best_pretrained.pt`.**

```
>>> Action for Select Best Pretrained Model? (y=Run, n=Skip): y
Found: best_pretrained.pt
Do you want to use this model as the base for training? (y/n): y
```

- **y** → writes `TRAIN_MODEL=models/best_pretrained.pt` to `.env`. Training will fine-tune from this checkpoint.
- **n** → clears `TRAIN_MODEL` from `.env`. Training starts from ImageNet pretrained weights (default).

> **Note:** At the start of every pipeline run, `TRAIN_MODEL` is automatically cleared from `.env`. You must go through this step each run to use the pretrained model.

---

### Step 10 — Evaluate Best Model *(optional)*
**Script:** `scripts/evaluate_best.py`

Runs inference with `models/best_pretrained.pt` on the validation set and prints mAP50, mAP50-95, precision, recall. Useful to benchmark before deciding whether to fine-tune.

---

### Step 11 — Start Training (Local) *(YOLO/RT-DETR only)*
**Script:** `scripts/train.py`

Trains locally on your machine's GPU. Uses all hyperparameters set in Step 3.

- Detects GPU automatically — falls back to CPU if no CUDA
- Saves `best.pt` and `last.pt` to `runs/train/<run_name>/weights/`
- Logs per-epoch: `P`, `R`, `F1`, `mAP50`, `mAP50-95`, `lr`

> Skip this step if training on Azure ML.

---

### Step 12 — Upload Dataset to Azure
**Script:** `scripts/upload_dataset.py`

Uploads the local `dataset/` folder to Azure ML as a registered Data Asset (`licence-plate-dataset`).

**What happens:**
1. Authenticates via `az login` credentials (or browser fallback)
2. Uploads `dataset/images/` and `dataset/labels/` to Azure Blob Storage
3. Registers it as a versioned URI Folder asset in Azure ML workspace

**When to run:**
- First time setup
- After adding new images or re-splitting the dataset
- Skip if dataset hasn't changed since last upload

**Upload time:** ~10-20 min for ~4GB dataset. Subsequent training jobs reuse the same asset instantly.

---

### Step 13 — Start Training (Azure ML)
**Script:** `scripts/azure_train.py`

Submits a training job to Azure ML. This is the main cloud training step.

#### What happens on your local machine:

```
1. Authenticate to Azure ML workspace
2. Enforce max_concurrent_runs = 1 on compute (queue, don't run parallel)
3. Get or build Docker environment (yolo26-licence-plate-env:12)
4. Resolve latest dataset asset version
5. Ask job name (Base Training or Fine Tuning)
6. Submit command job to Azure ML
7. Optionally stream logs
```

#### What happens on the GPU VM (gpu-vm-clean):

```
1. Azure ML pulls the Docker image
2. Dataset downloads from Blob Storage to /tmp/dataset_<run_id>/
3. Command runs inside container:
   df -h
   pip install numpy<2.0.0
   cp dataset from /tmp → ./dataset/
   python3 -u scripts/train.py   (or train_rcnn.py)
4. Training runs, checkpoints saved to /mnt/runs/train/<run_name>/
5. Logs stream back to Azure ML Studio
```

#### Environment variables passed to the job:
| Variable | Purpose |
|---|---|
| `TRAIN_SCRIPT_CHOICE` | 1=YOLO, 2=FastRCNN, 3=FasterRCNN, 4=RTDETR |
| `TRAIN_MODEL` | Model weights path (empty = use default) |
| `ORCH_EPOCHS` | Number of training epochs |
| `ORCH_BATCH` | Batch size |
| `ORCH_PATIENCE` | Early stopping patience |
| `ORCH_LR` | Initial learning rate |
| `ORCH_LRF` | Final learning rate (cosine decay) |
| `ORCH_WARMUP_EPOCHS` | Warmup epochs |
| `ORCH_WORKERS` | DataLoader workers |
| `ORCH_IMGSZ` | Input image size (px) |
| `ORCH_FREEZE` | Freeze first N layers (YOLO/RT-DETR only) |

#### Job queuing:
If a job is already running, the new job queues automatically. `max_concurrent_runs=1` ensures they run sequentially, not in parallel.

#### Monitoring:
- Stream logs: answer `y` when asked after submission
- Azure ML Studio: `https://ml.azure.com` → Experiments → `licence-plate-detection`
- Detach at any time with `Ctrl+C` — job continues on VM

---

## After Training

### Download best.pt from Azure
```bash
az ml job download --name <job_name> --all --workspace-name Aakash_ML --resource-group Aakash_ML
```

Or from the VM directly (checkpoints are at `/mnt/runs/train/<run_name>/`):
```bash
scp orchvate@<vm_ip>:/mnt/runs/train/<run_name>/best.pt ./models/
```

### Export to ONNX
```bash
python scripts/export_onnx.py
```

### Evaluate
```bash
# YOLO / RT-DETR
python scripts/evaluate.py

# R-CNN
python scripts/evaluate_rcnn.py --weights runs/train/<run_name>/best.pt
```

---

## Model Output Locations

| Scenario | Path |
|---|---|
| Local training | `runs/train/<model>_<timestamp>/weights/best.pt` |
| Azure ML training | `/mnt/runs/train/<model>_<timestamp>/best.pt` (on VM) |
| Pretrained base | `models/best_pretrained.pt` |

---

## GPU VM Specs

| Spec | Value |
|---|---|
| Size | Standard NV18ads A10 v5 |
| vCPUs | 18 (AMD EPYC 74F3) |
| RAM | 220 GB |
| GPU | NVIDIA A10-12Q (12 GB VRAM) |
| Disk (root) | 124 GB |
| Disk (data) | 708 GB at `/mnt` |

Training outputs are written to `/mnt/runs/train/` to avoid filling the root disk.

---

## Recommended Configs Per Model

| Model | Epochs | Batch | LR | Workers | Notes |
|---|---|---|---|---|---|
| YOLO26 | 150 | 8 | 0.001 | 4 | Fast, ~2hr on A10-12Q |
| RT-DETR | 60 | 4 | 0.005 | 2 | ~3hr, better accuracy |
| Faster R-CNN | 60 | 3 | 0.005 | 2 | ~7hr, pin_memory disabled |

---

## Pipeline State

The pipeline tracks completed steps in `pipeline_state.json`. If a step was already completed, it asks before re-running. To reset all state:

```bash
del pipeline_state.json   # Windows
rm pipeline_state.json    # Linux/Mac
```

Or run Step 1 (Pipeline Reset) which handles this automatically.

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `CUDA device=0 invalid` | No GPU locally | Skip local training, use Azure |
| `CUDA out of memory` | Another job using GPU / batch too large | Kill other processes, reduce batch |
| `No space left on device (shm)` | Shared memory full | Reduce workers to 2, pin_memory disabled automatically on Azure |
| `best_pretrained.pt still loading` | Stale `TRAIN_MODEL` in `.env` | Cleared automatically at pipeline start |
| Two jobs running simultaneously | `max_concurrent_runs > 1` | Fixed automatically by `enforce_single_job()` |
