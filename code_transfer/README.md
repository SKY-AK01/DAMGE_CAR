# code_transfer/

Fast project sync to/from **Azure Blob Storage** (`opencvatstorage` / `opencvatcontainer`).

---

## 1. Install dependencies (once, on both machines)

```bash
pip install -r code_transfer/requirements.txt
```

---

## 2. Credentials

Your SAS token is already stored in **`code_transfer/cloud_config.json`** (pre-filled).  
The file is gitignored — never commit it.

If the token expires, generate a new one:  
`Azure Portal → Storage Account → Shared Access Signature → Generate SAS`

Then update the `sas_token` field in `code_transfer/cloud_config.json`, or export:
```bash
export AZURE_STORAGE_ACCOUNT=opencvatstorage
export AZURE_CONTAINER_NAME=opencvatcontainer
export AZURE_SAS_TOKEN="sp=racwl&st=..."
```

---

## 3. Upload (local machine → cloud)

```bash
# Standard upload (asks for confirmation, shows progress)
python code_transfer/upload.py

# Non-interactive (good for scripts)
python code_transfer/upload.py --yes

# See what would be uploaded without doing it
python code_transfer/upload.py --dry-run

# Verbose (lists every file packed into the archive)
python code_transfer/upload.py --verbose

# Increase parallel connections (default: 8)
python code_transfer/upload.py --concurrency 16
```

### What gets included / excluded

| Included | Excluded |
|---|---|
| All Python scripts | `.git/`, `__pycache__/`, `*.pyc` |
| `datasets/` folder | `venv/`, `.venv/`, `node_modules/` |
| Pipeline configurations | `target/` (build artifacts), `*.pdb` |
| `code_transfer/` (these scripts) | `*.pt`, `*.pth`, `*.onnx` (model checkpoints) |
| `docs/`, `tests/`, configs | Large zip backups |

Each run creates a **new timestamped subfolder** in the container:
```
opencvatcontainer/car_azure_transfers/car_azure_20260905_143022/
  ├── car_azure_20260905_143022.tar.gz    (compressed archive)
  └── car_azure_20260905_143022.sha256    (SHA256 integrity file)
```

---

## 4. Download (VM → extract project)

```bash
# Download latest snapshot, extract into current directory (project root)
python code_transfer/download.py

# Extract into a specific path on the VM
python code_transfer/download.py --target /opt/car_azure

# List all available snapshots without downloading
python code_transfer/download.py --list

# Download a specific snapshot (not the latest)
python code_transfer/download.py --snapshot car_azure_20260905_143022

# Non-interactive + skip checksum (emergency use only)
python code_transfer/download.py --yes --no-verify
```

The script will:
1. Find the **most recent** snapshot automatically
2. Download with parallel chunks (fast)
3. **Verify SHA256 checksum** before extracting (aborts if corrupt)
4. Extract all files into the target directory

---

## 5. Folder structure

```
code_transfer/
├── upload.py           # Upload script
├── download.py         # Download script
├── requirements.txt    # pip dependencies (azure-storage-blob, tqdm)
├── cloud_config.json   # Credentials (gitignored)
└── README.md           # This file
```

---

## 6. After downloading on VM — next steps

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Launch the training orchestrator
python orchestrator.py
```

---

## 7. Rotating the SAS token

The current token expires **2026-10-10**. Before it expires:
1. Go to Azure Portal → `opencvatstorage` → Shared access signature
2. Set permissions: Read, Add, Create, Write, List
3. Set expiry date
4. Copy the SAS token string
5. Replace `sas_token` in `code_transfer/cloud_config.json`

