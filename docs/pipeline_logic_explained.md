# 🧠 Complete Pipeline Logic — Deep Dive

> All explanations are sourced directly from your actual code + real driver log (`70_driver_log (1).txt`).

---

## 🗺️ Big Picture — How Everything Connects

```
Your Machine (Windows)
        │
        ▼
[azure_train.py]  ← you run this locally
   │   ├─ 1. Authenticates with Azure ML
   │   ├─ 2. Looks up the registered dataset (already uploaded)
   │   ├─ 3. Picks model / script / job name
   │   └─ 4. Sends a "command job" to Azure cloud
                        │
                        ▼
            [Azure VM: gpu-vm-clean]
               │ (NVIDIA A10-8Q GPU)
               ├─ 5. Downloads dataset to /tmp/...
               ├─ 6. Copies dataset to ./dataset/
               ├─ 7. Installs numpy patch
               └─ 8. Runs train.py
                           │
                           ▼
                    [train.py]
                    ├─ Loads data.yaml
                    ├─ Loads YOLO26 model
                    ├─ Trains for N epochs
                    └─ Saves best.pt to /mnt/runs/train/
```

---

## 📁 PHASE 1 — Dataset Upload (`upload_dataset.py`)

> **Run this ONCE when you have new images. Never again unless the dataset changes.**

```python
my_data = Data(
    path=str(DATASET_DIR),          # Local: ./dataset/
    type=AssetTypes.URI_FOLDER,     # Tells Azure: "this is a folder of files"
    description="YOLO training dataset (images and labels)",
    name=DATASET_NAME,              # "licence-plate-dataset"
)
uploaded_data = ml_client.data.create_or_update(my_data)
```

### What this does step-by-step:
1. Scans your local `./dataset/` folder (images + YOLO label `.txt` files)
2. Uploads everything to **Azure Blob Storage** (inside your workspace's storage account)
3. **Registers** a named Data Asset called `licence-plate-dataset` with version `1`, `2`, etc.
4. That version number is what future jobs use to pull the correct snapshot

> [!IMPORTANT]
> `create_or_update` always creates a **new version** — it does NOT overwrite. So if you call it 3 times, you get versions 1, 2, 3. That's why `azure_train.py` has logic to find the "latest" version.

---

## 🚀 PHASE 2 — Job Submission (`azure_train.py`)

### Step 1 — Authentication (Lines 53–73)

```python
credential = DefaultAzureCredential()
client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
client.workspaces.get(WORKSPACE_NAME)  # ← "ping" to verify login works
```

- **`DefaultAzureCredential`** tries multiple auth methods in order:
  1. Environment variables (`AZURE_CLIENT_ID` etc.)
  2. Azure CLI token (`az login`) ← **this is what you use**
  3. Managed identity (VM/container)
  4. Visual Studio / VS Code token
- If all fail → falls back to **browser popup** (`InteractiveBrowserCredential`)
- `workspaces.get()` is just a test call — if it throws, auth failed

---

### Step 2 — Finding the Dataset (Lines 121–145)

```python
if DATASET_ASSET_VERSION == "latest":
    data_versions = list(ml_client.data.list(name=DATASET_ASSET_NAME))
    latest_asset = sorted(data_versions, key=lambda x: int(x.version))[-1]
    data_asset = ml_client.data.get(DATASET_ASSET_NAME, version=latest_asset.version)
```

- Azure ML SDK doesn't have a native `"latest"` keyword for data assets, so your code **manually sorts all versions by version number** and picks the highest
- `data_asset.path` gives the blob storage URL like `azureml://datastores/workspaceblobstore/paths/...`
- This path is then wrapped into an `Input` object:

```python
dataset_input = Input(
    type = AssetTypes.URI_FOLDER,
    path = data_asset.path,     # blob URL
    mode = "download"           # ← KEY DECISION: download vs mount
)
```

> [!NOTE]
> `mode = "download"` means Azure will **copy all files to local disk on the VM before your script starts**. Alternative is `"mount"` (FUSE filesystem, reads files on-demand from blob — slower for training).

---

### Step 3 — Model & Script Selection (Lines 147–163)

```python
model_choice = os.getenv("TRAIN_SCRIPT_CHOICE", "1")
# 1 → YOLO26     → scripts/train.py
# 2 → FastRCNN   → scripts/train_rcnn.py --model fast_rcnn
# 3 → FasterRCNN → scripts/train_rcnn.py --model faster_rcnn
# 4 → RT-DETR    → scripts/train.py (with different TRAIN_MODEL env var)
```

This allows switching the entire model architecture by just changing one env variable before running the script.

---

### Step 4 — Building the Shell Command (Lines 178–192)

This is the **most important part**. The full command that runs on the Azure VM is:

```bash
df -h &&
python3 -m pip install 'numpy<2.0.0' --force-reinstall --quiet &&
echo 'Copying dataset to local SSD...' &&
mkdir -p ./dataset &&
cp -r ${{inputs.dataset}}/. ./dataset/ &&
echo 'Copy complete.' &&
python3 -u scripts/train.py
```

Each part explained:

| Command | Why it's there |
|---|---|
| `df -h` | Print disk space — sanity check, visible in logs |
| `pip install 'numpy<2.0.0'` | YOLO26/Ultralytics has a bug with NumPy 2.x — forces downgrade |
| `mkdir -p ./dataset` | Creates the local `./dataset/` folder on the VM |
| `cp -r ${{inputs.dataset}}/. ./dataset/` | **The critical step** — copies all downloaded files to local folder |
| `python3 -u scripts/train.py` | `-u` = unbuffered stdout, so logs appear in real-time in Azure Studio |

#### What is `${{inputs.dataset}}`?
- This is **Azure ML's template syntax** — it gets replaced at runtime with the actual path where Azure downloaded the dataset
- Based on the real driver log: it resolved to `/tmp/dataset_orange_dog_rdzmg6gfnz_None`
- After `cp`, your files are at `./dataset/` which maps to `/azureml-run/dataset/`

---

### Step 5 — The `command()` Job Object (Lines 194–217)

```python
job = command(
    display_name    = "YOLO26_FineTune_0831_1800",
    experiment_name = "licence-plate-detection",
    code            = str(BASE_DIR),          # ← Snapshots your ENTIRE project
    command         = train_cmd,
    environment     = "yolo26-licence-plate-env:12",
    compute         = "gpu-vm-clean",
    inputs          = {"dataset": dataset_input},
    environment_variables = { ... },
)
```

Key detail: `code = str(BASE_DIR)` — Azure ML **zips and uploads your entire project folder** as a "snapshot". This is why `data.yaml`, `scripts/`, everything is available at `/azureml-run/` on the VM.

---

### Step 6 — Submitting (Line 223)

```python
returned_job = ml_client.jobs.create_or_update(job)
```

- Sends the job definition to Azure ML
- Azure queues it on `gpu-vm-clean`
- Returns a `Job` object with `.name` and `.studio_url` for tracking
- **Your local machine is done at this point** — the VM runs independently

---

## 🏋️ PHASE 3 — What Happens on the Azure VM

### From the Real Driver Log (`70_driver_log (1).txt`)

```
[06:51:45] Mode: 'mount'.   ← Azure internally mounted first (confusing naming)
[06:51:45] Mounting dataset to /tmp/dataset_orange_dog_rdzmg6gfnz_None
[06:51:50] Mounted dataset to /tmp/dataset_orange_dog_rdzmg6gfnz_None as folder.
[06:51:58] Command Working Directory=/azureml-run
[06:51:58] Starting Linux command: df -h && pip install numpy... && cp -r /tmp/dataset_... ./dataset/ && python3 scripts/train.py
```

> [!WARNING]
> The log says "mount" even though you set `mode="download"`. This is because the **older SDK runner** on the VM intercepts the Input and uses FUSE mount regardless. The `cp -r` command then reads through the mount and writes real copies to `./dataset/` — which is why the copy still solves the I/O problem.

### Disk Layout on VM
```
/azureml-run/           ← Working directory (your project snapshot)
├── data.yaml           ← references path: ./dataset
├── dataset/            ← Created by cp -r (real local files!)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── train.py
│   └── ...
└── runs/train/         ← Output (on Azure, redirected to /mnt/)
```

---

## 🧑‍💻 PHASE 4 — Training Script (`train.py`)

### Path Resolution (Lines 56–57)

```python
BASE_DIR  = Path(__file__).resolve().parent.parent
# On VM: /azureml-run/

DATA_YAML = BASE_DIR / "data.yaml"
# On VM: /azureml-run/data.yaml
```

`data.yaml` contains:
```yaml
path: ./dataset       # → /azureml-run/dataset (where you cp'd to!)
train: images/train
val:   images/val
```

This is why the paths **line up perfectly** — the `cp` put files exactly where `data.yaml` expects them.

---

### Output Redirect (Lines 60–66)

```python
def get_runs_dir() -> str:
    if os.getenv("AZUREML_RUN_ID"):       # ← Set automatically by Azure ML
        runs_dir = Path("/mnt/runs/train") # ← Redirects to 672GB data disk!
    else:
        runs_dir = BASE_DIR / "runs/train" # ← Local dev: ./runs/train
    runs_dir.mkdir(parents=True, exist_ok=True)
    return str(runs_dir)
```

- The VM's root filesystem (`/azureml-run`) is only ~57GB used out of 124GB
- `/mnt/` is the attached **data disk** (672GB) — much more space for saving model checkpoints
- `AZUREML_RUN_ID` is auto-set by Azure ML, so this detection is automatic

---

### Hyperparameter Cascade (Lines 90–161)

There are **3 layers** of hyperparameter priority:

```
Layer 1 (Lowest):  Hardcoded defaults in HYPERPARAMS dict
Layer 2 (Middle):  TRAIN_EPOCHS / TRAIN_BATCH env vars (legacy)
Layer 3 (Highest): ORCH_EPOCHS / ORCH_BATCH env vars (from azure_train.py)
                   via argparse → args.epochs overwrites HYPERPARAMS
```

```python
# argparse reads ORCH_* env vars as defaults
parser.add_argument("--epochs", type=int, default=int(os.getenv("ORCH_EPOCHS", 60)))

# Then overwrites the HYPERPARAMS dict
HYPERPARAMS["epochs"] = args.epochs
```

So if you set `ORCH_EPOCHS=100` in `.env`, it flows:
`.env` → `azure_train.py` sends it as env var → VM receives it → `train.py` reads it via argparse → overwrites default.

---

### Model Loading (Lines 185–190)

```python
if "rtdetr" in final_model.lower() or choice == "4":
    model = RTDETR(final_model)   # RT-DETR architecture
else:
    model = YOLO(final_model)     # YOLO26 architecture
```

- If `TRAIN_MODEL` is a `.pt` file (pretrained weights): loads and fine-tunes
- If `TRAIN_MODEL` is a `.yaml` file: trains from scratch using that architecture config
- From driver log: `yolo26s.pt` was downloaded from GitHub releases (19.5MB, 71.6MB/s)

---

### Epoch Callback (Lines 193–211)

```python
def on_fit_epoch_end(trainer):
    m = trainer.metrics
    # Extracts: box_loss, cls_loss, Precision, Recall, mAP50, mAP50-95
    f1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    print(f"\n[EPOCH {e:3d}/{total}] box={box:.4f}  cls={cls:.4f}  "
          f"P={P:.3f}  R={R:.3f}  F1={f1:.3f}  "
          f"mAP50={map50:.3f}  mAP50-95={map95:.3f}  lr={lr:.6f}")

model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
```

- Ultralytics' default epoch output is not always visible in Azure ML's log stream
- This **custom callback** forces a clean summary print after each epoch
- `flush=True` ensures it bypasses Python's output buffer (important for `-u` flag + Azure streaming)

---

## ⚙️ Environment — `yolo26-licence-plate-env:12`

```python
env = Environment(
    name    = "yolo26-licence-plate-env",
    version = "12",
    build   = BuildContext(
        path            = str(BASE_DIR),
        dockerfile_path = "docker/Dockerfile"
    ),
)
```

- **First time**: Azure builds the Docker image from your `Dockerfile` (5–10 min)
- **Subsequent runs**: Uses cached image from Azure Container Registry — instant
- Version `"12"` is hardcoded — you bump this manually when you change the Dockerfile
- If version `"12"` already exists → skips build entirely

---

## 🔄 Single-Job Enforcement (Lines 101–114)

```python
compute = ml_client.compute.get(COMPUTE_NAME)
if hasattr(compute, "max_concurrent_runs") and compute.max_concurrent_runs != 1:
    compute.max_concurrent_runs = 1
    ml_client.compute.begin_create_or_update(compute).result()
```

- Prevents two training jobs from fighting over the GPU simultaneously
- `.result()` waits for the update to complete before proceeding
- Wrapped in try/except because **attached VMs** may not expose this property (hence the `[WARN]` fallback)

---

## 📊 Complete Flow Summary

```
You run: python scripts/azure_train.py
│
├─ [AUTH]       DefaultAzureCredential → az login token
├─ [COMPUTE]    Verify gpu-vm-clean is reachable, set max_concurrent_runs=1
├─ [ENV]        Check if yolo26-licence-plate-env:12 exists (build if not)
├─ [DATASET]    List all versions of licence-plate-dataset → pick latest
├─ [INPUT]      Wrap blob URL as URI_FOLDER with mode=download
├─ [COMMAND]    Build shell string: df && pip && cp && python3 train.py
├─ [JOB]        command() → create_or_update() → job queued on Azure
│
└─ ON VM:
   ├─ [SETUP]   Azure mounts dataset to /tmp/dataset_<random>/
   ├─ [COPY]    cp -r /tmp/dataset_.../ ./dataset/  (your code does this)
   ├─ [TRAIN]   python3 scripts/train.py
   │   ├─ Reads /azureml-run/data.yaml
   │   ├─ Finds images at /azureml-run/dataset/images/...  ✅
   │   ├─ Loads yolo26s.pt (downloads from GitHub if needed)
   │   ├─ Trains N epochs with AMP, MuSGD optimizer
   │   ├─ Prints [EPOCH X/N] summary after each epoch  ← custom callback
   │   └─ Saves best.pt → /mnt/runs/train/<run_name>/weights/best.pt
   └─ [DONE]    Job completes, outputs uploaded to Azure ML experiment
```

---

## ❓ Key Design Decisions & Why

| Decision | Why |
|---|---|
| `mode="download"` on Input | Faster than blob mount for training (data read many times per epoch) |
| `cp -r ${{inputs.dataset}} ./dataset/` | Forces data from `/tmp/` FUSE mount to real local SSD |
| `python3 -u` flag | Unbuffered — logs appear instantly in Azure ML Studio stream |
| `/mnt/runs/train` on Azure | Saves to large data disk, not tiny root filesystem |
| `AZUREML_RUN_ID` detection | Same `train.py` works both locally and on Azure with zero changes |
| `numpy<2.0.0` reinstall | Ultralytics broke compatibility with NumPy 2.x — runtime patch |
| `save_period = -1` | Only saves `best.pt` (not every N epochs) — saves disk space |
| `workers=2` on Azure | YOLO's DataLoader had deadlock issues with high worker counts on this VM |
