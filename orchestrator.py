#!/usr/bin/env python3
"""
orchestrator.py -- Interactive Pipeline Orchestrator in Python.

Provides an interactive CLI menu to run data prep, model training (local/Azure ML),
inference, evaluation, cleanup, and fast preprocessing without compilation dependencies.

CLI flags (for automation / VM runs):
  --yes-clean-runs   Skip the interactive prompt and DELETE runs/logs/combined/matched/external
  --yes-clean-raw    Skip the interactive prompt and DELETE datasets/raw/  (DANGEROUS -- irreplaceable data!)
                     Requires typing 'yes' interactively unless this flag is passed.
"""

import sys
import os
import argparse
import subprocess
import shutil
import datetime
import glob
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.resolve()

# ── CLI args (parsed once at module level so all functions can read them) ────
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument("--yes-clean-runs", action="store_true", default=False,
                          help="Non-interactively confirm deletion of runs/logs/combined/matched/external.")
_cli_parser.add_argument("--yes-clean-raw", action="store_true", default=False,
                          help="Non-interactively confirm deletion of datasets/raw/ (DANGEROUS).")
_CLI_ARGS, _CLI_UNKNOWN = _cli_parser.parse_known_args()

def prompt(msg):
    try:
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

def prompt_default(label, default):
    val = prompt(f"  {label} [default: {default}]: ")
    return val if val else str(default)

def prompt_int(label, default):
    val = prompt_default(label, default)
    try:
        return int(val)
    except ValueError:
        return int(default)

def display_main_menu():
    print()
    print("======================================================================")
    print("        Car-Parts Segmentation -- Interactive Python Orchestrator       ")
    print("======================================================================")
    print("  1) Annotate Images       -- Auto-annotate dataset (DINO + SAM2)")
    print("  2) Train Model Locally   -- Prepare dataset & train YOLO / Mask R-CNN / Fast R-CNN")
    print("  3) Test on Trained Model -- Run inference on test images")
    print("  4) Run Full Local        -- Dataset Prep -> Train -> Evaluate -> Excel Report")
    print("  5) Train on Azure ML     -- Auto-upload dataset & submit GPU job to Azure ML")
    print("  6) Setup & Clean         -- Folder setup and log archive")
    print("  7) Auto-Prepare Dataset  -- Download missing sources, match, combine, generate COCO JSON")
    print("  8) Preprocess Tools      -- Standalone tools (yolo_to_coco, combine, verify, blake3)")
    print("  9) Dataset Analytics     -- View image counts, total annotations, & per-label breakdown")
    print("  10) Benchmark            -- Compare preprocessing speed (yolo_to_coco + verify_labels)")
    print("  11) Quick Pipeline Check -- Fast 3-epoch dry run on Azure ML (All Models)")
    print("  12) GPU Capacity Check   -- Stepwise batch size & worker stress tuner")
    print("  13) Cleanup Pipeline     -- Wipe stale outputs interactively (safe, never deletes raw/)")
    print("  14) Exit")
    print("======================================================================")

class ModelSelection:
    def __init__(self, yolo=False, yolo11x=False, maskrcnn=False, fastrcnn=False, mask2former=False, sam2=False, maskdino=False, segformer=False):
        self.yolo = yolo
        self.yolo11x = yolo11x
        self.maskrcnn = maskrcnn
        self.fastrcnn = fastrcnn
        self.mask2former = mask2former
        self.sam2 = sam2
        self.maskdino = maskdino
        self.segformer = segformer

def collect_models():
    print()
    print("======================================================================")
    print(" Select Instance Segmentation Models to Train")
    print("======================================================================")
    print("  1) YOLOv11m-seg only")
    print("  2) YOLO11x-seg only (Extra-Large YOLO)")
    print("  3) Mask R-CNN only")
    print("  4) Mask2Former (Swin Transformer) only")
    print("  5) SAM2 Fine-Tuned only")
    print("  6) MaskDINO only")
    print("  7) SegFormer only")
    print("  8) ALL Instance Segmentation Models (Auto-Generates Excel Comparison)")
    print("======================================================================")
    choice = prompt("Enter choice [1-8] (default 8): ") or "8"
    mapping = {
        "1": ModelSelection(yolo=True),
        "2": ModelSelection(yolo11x=True),
        "3": ModelSelection(maskrcnn=True),
        "4": ModelSelection(mask2former=True),
        "5": ModelSelection(sam2=True),
        "6": ModelSelection(maskdino=True),
        "7": ModelSelection(segformer=True),
        "8": ModelSelection(yolo=True, yolo11x=True, maskrcnn=True, mask2former=True, sam2=True, maskdino=True, segformer=True),
    }
    return mapping.get(choice, ModelSelection(yolo=True, yolo11x=True, maskrcnn=True, mask2former=True, sam2=True, maskdino=True, segformer=True))

def load_gpu_recommendations():
    json_path = PROJECT_ROOT / "runs_comparison" / "gpu_capacity_recommendations.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("models", {})
        except Exception:
            pass
    return {}

class Hyperparams:
    def __init__(self, models=None):
        recs = load_gpu_recommendations()

        def get_default_batch(model_key, fallback):
            if model_key in recs and recs[model_key].get("batch", 0) > 0:
                return recs[model_key]["batch"]
            return fallback

        def get_default_workers(model_key, fallback):
            if model_key in recs and recs[model_key].get("workers", 0) > 0:
                return recs[model_key]["workers"]
            return fallback

        # Determine which model groups need prompting.
        # "yolo_group"   covers yolo + yolo11x (same hyperparams, different --model flag)
        # "m2f_group"    covers mask2former + sam2 + maskdino + segformer (all HuggingFace-style)
        need_yolo    = models is None or models.yolo or models.yolo11x
        need_maskrcnn = models is None or models.maskrcnn
        need_m2f     = models is None or models.mask2former or models.sam2 or models.maskdino or models.segformer

        print()
        print("======================================================================")
        if recs:
            print(" Set training hyperparameters (Auto-loaded defaults from latest GPU check)")
        else:
            print(" Set training hyperparameters (press Enter to accept defaults)")
        print("======================================================================")

        # ── YOLO ──────────────────────────────────────────────────────────────
        if need_yolo:
            yolo_default_b = get_default_batch("yolo11m-seg", 8)
            yolo_default_w = get_default_workers("yolo11m-seg", 8)
            print("\n---- YOLOv11m-seg ----")
            self.yolo_epochs      = prompt_int("Epochs", 50)
            self.yolo_batch       = prompt_int("Batch size (-1 = auto)", yolo_default_b)
            self.yolo_workers     = prompt_int("Dataloader workers", yolo_default_w)
            self.yolo_val_interval = prompt_int("Validate every N epochs", 5)
        else:
            self.yolo_epochs      = 50
            self.yolo_batch       = get_default_batch("yolo11m-seg", 8)
            self.yolo_workers     = get_default_workers("yolo11m-seg", 8)
            self.yolo_val_interval = 5

        # ── Mask R-CNN ────────────────────────────────────────────────────────
        if need_maskrcnn:
            mrcnn_default_b = get_default_batch("maskrcnn", 2)
            mrcnn_default_w = get_default_workers("maskrcnn", 4)
            print("\n---- Mask R-CNN ----")
            self.mrcnn_epochs       = prompt_int("Epochs", 20)
            self.mrcnn_batch        = prompt_int("Batch size", mrcnn_default_b)
            self.mrcnn_workers      = prompt_int("Dataloader workers", mrcnn_default_w)
            self.mrcnn_val_interval = prompt_int("Validate every N epochs", 5)
        else:
            self.mrcnn_epochs       = 20
            self.mrcnn_batch        = get_default_batch("maskrcnn", 2)
            self.mrcnn_workers      = get_default_workers("maskrcnn", 4)
            self.mrcnn_val_interval = 5

        # ── Fast R-CNN — always silent default (not in model selection menu) ──
        fastrcnn_default_b = get_default_batch("fastrcnn", 2)
        fastrcnn_default_w = get_default_workers("fastrcnn", 4)
        self.fastrcnn_epochs      = 20
        self.fastrcnn_batch       = fastrcnn_default_b
        self.fastrcnn_workers     = fastrcnn_default_w
        self.fastrcnn_val_interval = 5

        # ── Mask2Former / SAM2 / MaskDINO / SegFormer ────────────────────────
        if need_m2f:
            m2f_default_b = get_default_batch("mask2former", 2)
            m2f_default_w = get_default_workers("mask2former", 4)
            # Label the section after whichever model is actually selected
            if models and models.mask2former:
                label = "Mask2Former (Swin Transformer)"
            elif models and models.sam2:
                label = "SAM2 Fine-Tuned"
            elif models and models.maskdino:
                label = "MaskDINO"
            elif models and models.segformer:
                label = "SegFormer"
            else:
                label = "Mask2Former / SAM2 / MaskDINO / SegFormer"
            print(f"\n---- {label} ----")
            self.m2f_epochs       = prompt_int("Epochs", 20)
            self.m2f_batch        = prompt_int("Batch size", m2f_default_b)
            self.m2f_workers      = prompt_int("Dataloader workers", m2f_default_w)
            self.m2f_val_interval = prompt_int("Validate every N epochs", 5)
        else:
            self.m2f_epochs       = 20
            self.m2f_batch        = get_default_batch("mask2former", 2)
            self.m2f_workers      = get_default_workers("mask2former", 4)
            self.m2f_val_interval = 5

        # ── Summary: only show selected models ────────────────────────────────
        print("\n---- Summary ----")
        if need_yolo:
            print(f"  YOLOv11m-seg : epochs={self.yolo_epochs}  batch={self.yolo_batch}  workers={self.yolo_workers}  val_every={self.yolo_val_interval}")
        if need_maskrcnn:
            print(f"  Mask R-CNN   : epochs={self.mrcnn_epochs}  batch={self.mrcnn_batch}  workers={self.mrcnn_workers}  val_every={self.mrcnn_val_interval}")
        if need_m2f:
            print(f"  Mask2Former+ : epochs={self.m2f_epochs}  batch={self.m2f_batch}  workers={self.m2f_workers}  val_every={self.m2f_val_interval}")
        print("-----------------")

def run_cmd_and_log(cmd, log_path, label):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lf.write(f"=== {label} | {timestamp} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")

    print(f"\n[*] Running [{label}] ...")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    GPU_REC_MARKER = "[GPU_RECOMMENDATIONS_JSON]"
    captured_recs = None

    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            for line in proc.stdout:
                sys.stdout.write(f"[{label}] {line}")
                sys.stdout.flush()
                lf.write(line)
                # Capture GPU recommendations from streaming Azure ML logs
                if GPU_REC_MARKER in line:
                    try:
                        json_start = line.index(GPU_REC_MARKER) + len(GPU_REC_MARKER)
                        captured_recs = json.loads(line[json_start:].strip())
                    except Exception:
                        pass
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped watching [{label}]. Terminating background process...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print(f"[OK] Task [{label}] canceled. Returning to main menu.")
        return False

    success = proc.returncode == 0
    if success:
        # If GPU recommendations were embedded in the output, save them locally
        if captured_recs:
            local_rec_dir = PROJECT_ROOT / "runs_comparison"
            os.makedirs(local_rec_dir, exist_ok=True)
            local_rec_path = local_rec_dir / "gpu_capacity_recommendations.json"
            try:
                with open(local_rec_path, "w", encoding="utf-8") as f:
                    json.dump(captured_recs, f, indent=2)
                print(f"[OK] GPU recommendations captured from log & saved locally to:")
                print(f"     {local_rec_path}")
            except Exception as e:
                print(f"[WARN] Could not save recommendations locally: {e}")
        print(f"[OK] [{label}] completed successfully.")
        return True
    else:
        print(f"[FAILED] [{label}] failed with exit code {proc.returncode}. Log: {log_path}")
        return False

def _sizeof_dir(path: Path) -> str:
    """Return human-readable size string for a directory."""
    if not path.exists():
        return "(not present)"
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total < 1024**2:
            return f"{total/1024:.1f} KB"
        elif total < 1024**3:
            return f"{total/1024**2:.1f} MB"
        else:
            return f"{total/1024**3:.2f} GB"
    except Exception:
        return "(unknown)"


def interactive_cleanup_prompt():
    """
    Ask TWO separate yes/no questions before any pipeline work:
      Q1: Delete run outputs + logs + combined/matched/external  (default No)
      Q2: Delete datasets/raw/                                    (default No, requires 'yes' to confirm)

    CLI flags override interactive prompts:
      --yes-clean-runs  → auto-confirms Q1
      --yes-clean-raw   → auto-confirms Q2

    Returns (clean_runs: bool, clean_raw: bool).
    Nothing is deleted here — caller is responsible for deletion.
    """
    datasets_dir = PROJECT_ROOT / "datasets"

    # ── Targets for Q1 ──────────────────────────────────────────────────────
    runs_targets = [
        PROJECT_ROOT / "runs_comparison",
        PROJECT_ROOT / "logs",
        datasets_dir / "combined_carparts",
        datasets_dir / "matched",
        datasets_dir / "external",
    ]

    print()
    print("=" * 68)
    print(" Pipeline Startup — Cleanup Check")
    print("=" * 68)

    # ── Q1: Run outputs ──────────────────────────────────────────────────────
    present_runs = [p for p in runs_targets if p.exists()]
    print()
    print("  [Q1] Delete old run outputs, logs, and derived dataset artifacts?")
    print("       (runs_comparison/, logs/, combined_carparts/, matched/, external/)")
    print("       This will be rebuilt fresh. Default: No")
    print()
    if present_runs:
        print("       Will remove:")
        for p in present_runs:
            size = _sizeof_dir(p)
            try:
                rel = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = p
            print(f"         DELETE  {rel}  [{size}]")
    else:
        print("       (nothing to remove — all targets already absent)")
    print()

    clean_runs = False
    if _CLI_ARGS.yes_clean_runs:
        print("       [AUTO] --yes-clean-runs flag set — proceeding with deletion.")
        clean_runs = True
    elif present_runs:
        try:
            ans = input("       Delete run outputs? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        clean_runs = ans in ("y", "yes")
        if not clean_runs:
            print("       [SKIP] Keeping existing run outputs.")

    if clean_runs:
        for p in present_runs:
            print(f"  [DELETE] {p}")
            shutil.rmtree(p, ignore_errors=True)
        print("  [OK] Run outputs cleaned.")

    # ── Q2: datasets/raw/ ────────────────────────────────────────────────────
    raw_dir = datasets_dir / "raw"
    raw_size = _sizeof_dir(raw_dir)
    print()
    print("  [Q2] Delete datasets/raw/? (your hand-annotated ground truth — IRREPLACEABLE!)")
    print(f"       Current size: {raw_size}")
    print("       Default: No. You must type 'yes' (not just 'y') to confirm.")
    print()

    clean_raw = False
    if _CLI_ARGS.yes_clean_raw:
        print("       [AUTO] --yes-clean-raw flag set — proceeding with deletion.")
        clean_raw = True
    elif raw_dir.exists() and any(raw_dir.rglob("*.*")):
        print(f"       Will remove: datasets/raw/  [{raw_size}]")
        try:
            ans = input("       Type 'yes' to DELETE datasets/raw/ (or anything else to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans == "yes":
            clean_raw = True
        else:
            print("       [SKIP] Keeping datasets/raw/ (smart choice).")
    else:
        print("       datasets/raw/ is empty or absent — nothing to delete.")

    if clean_raw:
        print(f"  [DELETE] {raw_dir}")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("  [OK] datasets/raw/ removed.")

    print()
    print("=" * 68)
    return clean_runs, clean_raw


def ensure_and_prepare_datasets(logs_dir):
    """
    Full dataset preparation pipeline using the structured layout:

      datasets/raw/                    <- HITL ground-truth (deduplicated, never modified)
      datasets/external/               <- open-source datasets in original format
        carparts-seg/
        dsmlr/
      datasets/matched/                <- external datasets converted to 23-class YOLO taxonomy
        carparts-seg/
        dsmlr/
      datasets/combined_carparts/      <- final training set, rebuilt fresh every run

    Sequence (VM-ready):
      0. Interactive cleanup prompt (or --yes-clean-* flags)
      1. Migrate raw data from RAW_DATASET/  (prepare_raw_dataset + setup_dataset_structure)
      2. Download external datasets if missing  (carparts-seg from Ultralytics, DSMLR from GitHub)
      3. Run taxonomy matchers  (match_carparts_seg, match_dsmlr)
      4. Wipe + rebuild combined_carparts from raw/ + matched/*
      5. Regenerate COCO JSON for Mask R-CNN / Mask2Former
      6. Verify labels
      7. Generate Markdown report

    Adding a new external source in future:
      - Drop it into datasets/external/<name>/
      - Write scripts/data/match_<name>.py that outputs to datasets/matched/<name>/
      - Rerun this pipeline — combine_datasets.py auto-discovers matched/* directories.
    """
    # ── Step 0: Interactive cleanup prompt ────────────────────────────────────
    interactive_cleanup_prompt()

    datasets_dir = PROJECT_ROOT / "datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "01_smart_dataset_prep.log")

    # ── Step 1a: Convert RAW_DATASET XML → custom_carparts (YOLO) ────────────
    raw_images = PROJECT_ROOT / "RAW_DATASET" / "IMAGES"
    raw_xmls   = PROJECT_ROOT / "RAW_DATASET" / "XML"
    has_raw = (raw_images.exists() and any(raw_images.iterdir()) and
               raw_xmls.exists()   and any(raw_xmls.iterdir()))
    if has_raw:
        print("[*] Found RAW_DATASET — converting XML annotations to YOLO format...")
        run_cmd_and_log(["python", "scripts/data/prepare_raw_dataset.py"], log_path, "raw_prep")
    else:
        print("[INFO] RAW_DATASET/IMAGES or XML is empty — skipping XML conversion.")

    # ── Step 1b: Migrate custom_carparts → datasets/raw/ (dedup) ─────────────
    raw_dir = datasets_dir / "raw"
    if not raw_dir.exists() or not any((raw_dir / "images").rglob("*.*") if (raw_dir / "images").exists() else []):
        print("[*] Migrating deduplicated HITL data to datasets/raw/ ...")
        run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")
    else:
        print("[OK] datasets/raw/ already populated.")

    # ── Step 2a: Download carparts-seg if missing ─────────────────────────────
    # VM note: downloads to datasets/carparts-seg/ (legacy location), then
    # setup_dataset_structure.py migrates it to datasets/external/carparts-seg/.
    carparts_ext = datasets_dir / "external" / "carparts-seg"
    carparts_legacy = datasets_dir / "carparts-seg"
    if carparts_ext.exists() and any(carparts_ext.iterdir()):
        print("[OK] datasets/external/carparts-seg/ already populated.")
    elif carparts_legacy.exists() and any(carparts_legacy.iterdir()):
        # Legacy location exists — just migrate
        print("[*] Migrating carparts-seg from legacy location to external/carparts-seg/ ...")
        run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")
    else:
        # Fresh VM — must download
        print("\n[*] datasets/external/carparts-seg/ not found — downloading from Ultralytics (133 MB)...")
        zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
        zip_file = datasets_dir / "carparts-seg.zip"
        try:
            import urllib.request
            import zipfile as _zipfile

            def _reporthook(count, block_size, total_size):
                pct = min(100, int(count * block_size * 100 / total_size)) if total_size > 0 else 0
                if count % 50 == 0:
                    print(f"  Downloading... {pct}%", end="\r", flush=True)

            urllib.request.urlretrieve(zip_url, zip_file, reporthook=_reporthook)
            print()
            # Extract into datasets/carparts-seg/ (setup_dataset_structure expects this location)
            extract_target = datasets_dir / "carparts-seg"
            with _zipfile.ZipFile(zip_file, "r") as zf:
                # carparts-seg.zip may contain a top-level 'carparts-seg/' folder
                # or files directly — handle both by extracting to a temp dir
                members = zf.namelist()
                has_prefix = all(m.startswith("carparts-seg/") for m in members if m)
                if has_prefix:
                    # strip the prefix: extract to datasets/ so carparts-seg/ lands correctly
                    zf.extractall(datasets_dir)
                else:
                    extract_target.mkdir(parents=True, exist_ok=True)
                    zf.extractall(extract_target)
            zip_file.unlink(missing_ok=True)
            print("[OK] carparts-seg downloaded and extracted to datasets/carparts-seg/.")
            # Now migrate to external/
            run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")
        except Exception as e:
            print(f"[WARN] Failed to auto-download carparts-seg: {e}")
            print("       Please manually place carparts-seg/ in datasets/carparts-seg/ and re-run.")

    # ── Step 2b: Download / split DSMLR if missing ───────────────────────────
    # VM note: clones to datasets/dsmlr-carparts/, splits to datasets/dsmlr-carparts-split/,
    # then setup_dataset_structure.py migrates it to datasets/external/dsmlr/.
    dsmlr_ext = datasets_dir / "external" / "dsmlr"
    dsmlr_legacy_split = datasets_dir / "dsmlr-carparts-split"
    dsmlr_raw = datasets_dir / "dsmlr-carparts"
    if dsmlr_ext.exists() and any(dsmlr_ext.iterdir()):
        print("[OK] datasets/external/dsmlr/ already populated.")
    elif dsmlr_legacy_split.exists() and any(dsmlr_legacy_split.iterdir()):
        # Legacy split exists — just migrate
        print("[*] Migrating dsmlr from legacy location to external/dsmlr/ ...")
        run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")
    else:
        # Fresh VM — must clone and split
        if not dsmlr_raw.exists() or not any(dsmlr_raw.iterdir()):
            print("\n[*] datasets/external/dsmlr/ not found — cloning DSMLR Car-Parts-Segmentation (~24 MB)...")
            cmd = ["git", "clone", "--depth", "1",
                   "https://github.com/dsmlr/Car-Parts-Segmentation.git",
                   str(dsmlr_raw)]
            ok = run_cmd_and_log(cmd, log_path, "dsmlr_clone")
            if not ok:
                print("[WARN] Failed to clone DSMLR repo. Skipping DSMLR dataset.")
        if dsmlr_raw.exists() and any(dsmlr_raw.iterdir()):
            print("[*] Splitting DSMLR into train/val/test (80/10/10)...")
            run_cmd_and_log(["python", "scripts/data/prepare_dsmlr_split.py"], log_path, "dsmlr_split")
            print("[*] Migrating DSMLR split to external/dsmlr/ ...")
            run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")

    # ── Step 3: Run taxonomy matchers ────────────────────────────────────────
    matched_cs    = datasets_dir / "matched" / "carparts-seg"
    matched_dsmlr = datasets_dir / "matched" / "dsmlr"

    def _is_populated(d: Path) -> bool:
        img_dir = d / "images"
        return img_dir.exists() and any(img_dir.rglob("*.*"))

    if not _is_populated(matched_cs):
        if (datasets_dir / "external" / "carparts-seg").exists():
            print("\n[*] Running carparts-seg taxonomy matcher...")
            run_cmd_and_log(["python", "scripts/data/match_carparts_seg.py"], log_path, "match_carparts_seg")
        else:
            print("[WARN] external/carparts-seg/ missing — skipping matcher.")
    else:
        print("[OK] matched/carparts-seg/ already populated.")

    if not _is_populated(matched_dsmlr):
        if (datasets_dir / "external" / "dsmlr").exists():
            print("\n[*] Running DSMLR taxonomy matcher...")
            run_cmd_and_log(["python", "scripts/data/match_dsmlr.py"], log_path, "match_dsmlr")
        else:
            print("[WARN] external/dsmlr/ missing — skipping matcher.")
    else:
        print("[OK] matched/dsmlr/ already populated.")

    # ── Step 4: Rebuild combined_carparts (always wipe + rebuild) ────────────
    print("\n[*] Rebuilding combined_carparts from raw/ + matched/* ...")
    run_cmd_and_log(["python", "scripts/data/combine_datasets.py"], log_path, "combine_datasets")

    # ── Step 5: Regenerate COCO JSON ─────────────────────────────────────────
    print("\n[*] Regenerating COCO JSON for Mask R-CNN / Mask2Former...")
    run_cmd_and_log(["python", "scripts/data/yolo_to_coco.py", "--dataset", "combined_carparts"],
                    log_path, "yolo_to_coco")

    # ── Step 6: Verify labels ────────────────────────────────────────────────
    print("\n[*] Verifying labels (corrupt image scan)...")
    run_cmd_and_log(["python", "scripts/data/verify_labels.py",
                     "--dir", str(datasets_dir / "combined_carparts")],
                    log_path, "verify_labels")

    print("\n[OK] Dataset preparation complete. Training set: './datasets/combined_carparts'")
    return "combined_carparts"

def analyze_dataset(dataset_name="combined_carparts"):
    """
    Parses COCO JSON and YOLO annotations for a dataset split
    and prints a formatted breakdown of image count, annotation count,
    and annotations per label category.
    """
    ds_path = PROJECT_ROOT / "datasets" / dataset_name
    if not ds_path.exists():
        print(f"[ERROR] Dataset directory '{ds_path}' does not exist.")
        return

    print()
    print("======================================================================")
    print(f" Dataset Analytics Summary -- '{dataset_name}'")
    print("======================================================================")

    total_images = 0
    total_annotations = 0
    class_counts = {}

    for split in ["train", "val", "test"]:
        coco_path = ds_path / f"coco_{split}.json"
        if coco_path.exists():
            try:
                with open(coco_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                n_img = len(data.get("images", []))
                n_ann = len(data.get("annotations", []))
                total_images += n_img
                total_annotations += n_ann

                cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}

                for ann in data.get("annotations", []):
                    cat_id = ann.get("category_id")
                    cat_name = cat_map.get(cat_id, f"class_{cat_id}")
                    class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

                print(f"  Split [{split:<5}]: {n_img:>6} images | {n_ann:>7} annotations")
            except Exception as e:
                print(f"  [WARN] Failed to parse {coco_path.name}: {e}")
        else:
            img_dir = ds_path / "images" / split
            if img_dir.exists():
                n_img = len(list(img_dir.glob("*.*")))
                total_images += n_img
                print(f"  Split [{split:<5}]: {n_img:>6} images (YOLO raw format)")

    print("----------------------------------------------------------------------")
    print(f"  TOTAL     : {total_images:>6} images | {total_annotations:>7} annotations")
    print("======================================================================")

    if class_counts:
        print("\n Annotations Per Label Class:")
        print(" ---------------------------------------------------------------------")
        print(f"  {'Category Name':<28} | {'Annotations':<12} | {'Percentage':<10}")
        print(" ---------------------------------------------------------------------")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_classes:
            pct = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"  {name:<28} | {count:>12} | {pct:>9.2f}%")
        print(" ---------------------------------------------------------------------")


def run_dataset_prep(logs_dir):
    return ensure_and_prepare_datasets(logs_dir)


def _generate_report(run_config_path, logs_dir):
    """
    Generate Markdown pipeline report to:
      - logs/pipeline_reports/pipeline_report_<timestamp>.md  (archived)
      - logs/latest_pipeline_report.md                         (always latest)
    Called automatically after dataset preparation.
    """
    try:
        cmd = ["python", "scripts/data/generate_pipeline_report.py",
               "--run_config", run_config_path]
        log_path = os.path.join(logs_dir, "pipeline_report.log")
        run_cmd_and_log(cmd, log_path, "generate_pipeline_report")
    except Exception as e:
        print(f"[WARN] Failed to generate pipeline report: {e}")


def save_run_config(logs_dir, task_name, dataset, models=None, hparams=None, extra=None):
    """
    Write run_config.json into logs_dir so you always know what settings
    were used for any given run folder.

    Structure:
        run_config.json
        {
          "run_id":    "run_20260906_123456",
          "task":      "Train Model Locally",
          "started_at": "2026-09-06 12:34:56",
          "dataset":   "combined_carparts",
          "models_selected": ["maskrcnn", "yolo11m-seg"],
          "hyperparameters": {
            "yolo":     { "epochs": 50, "batch": 8,  "workers": 8  },
            "maskrcnn": { "epochs": 20, "batch": 2,  "workers": 4  },
            "fastrcnn": { "epochs": 20, "batch": 2,  "workers": 4  },
            "mask2former": { "epochs": 20, "batch": 2, "workers": 4 }
          },
          ...extra fields...
        }
    """
    run_id = os.path.basename(os.path.dirname(logs_dir))  # e.g. run_20260906_123456

    selected_models = []
    if models:
        if getattr(models, "yolo",        False): selected_models.append("yolo11m-seg")
        if getattr(models, "yolo11x",     False): selected_models.append("yolo11x-seg")
        if getattr(models, "maskrcnn",    False): selected_models.append("maskrcnn")
        if getattr(models, "fastrcnn",    False): selected_models.append("fastrcnn")
        if getattr(models, "mask2former", False): selected_models.append("mask2former")
        if getattr(models, "sam2",        False): selected_models.append("sam2")
        if getattr(models, "maskdino",    False): selected_models.append("maskdino")
        if getattr(models, "segformer",   False): selected_models.append("segformer")

    hp_dict = {}
    if hparams:
        hp_dict = {
            "yolo":       {"epochs": getattr(hparams, "yolo_epochs",      None),
                           "batch":  getattr(hparams, "yolo_batch",       None),
                           "workers":getattr(hparams, "yolo_workers",     None)},
            "maskrcnn":   {"epochs": getattr(hparams, "mrcnn_epochs",     None),
                           "batch":  getattr(hparams, "mrcnn_batch",      None),
                           "workers":getattr(hparams, "mrcnn_workers",    None)},
            "fastrcnn":   {"epochs": getattr(hparams, "fastrcnn_epochs",  None),
                           "batch":  getattr(hparams, "fastrcnn_batch",   None),
                           "workers":getattr(hparams, "fastrcnn_workers", None)},
            "mask2former":{"epochs": getattr(hparams, "m2f_epochs",       None),
                           "batch":  getattr(hparams, "m2f_batch",        None),
                           "workers":getattr(hparams, "m2f_workers",      None)},
        }
        # Remove model blocks that were never asked (all-None)
        hp_dict = {k: v for k, v in hp_dict.items()
                   if any(val is not None for val in v.values())}

    config = {
        "run_id":            run_id,
        "task":              task_name,
        "started_at":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":           dataset,
        "models_selected":   selected_models,
        "hyperparameters":   hp_dict,
    }
    if extra:
        config.update(extra)

    config_path = os.path.join(logs_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\n[OK] Run config saved -> {config_path}")
    return config_path


def check_azure_disk():
    cwd_str = str(PROJECT_ROOT)
    if cwd_str.startswith("/mnt/"):
        print("############################################################", file=sys.stderr)
        print(f"  WARNING: Running from {cwd_str}", file=sys.stderr)
        print("  If this is Azure's temporary resource disk, it will be WIPED", file=sys.stderr)
        print("  when you Stop (Deallocate) the VM. Move to /home first!", file=sys.stderr)
        print("############################################################", file=sys.stderr)
        import time
        time.sleep(3)


def run_evaluation(dataset, base_out_dir, logs_dir):
    print("\n========================================")
    print(" Running Evaluation")
    print("========================================")

    last_weights_file = PROJECT_ROOT / "last_yolo_weights_path.txt"
    if last_weights_file.exists():
        yolo_weights = last_weights_file.read_text(encoding="utf-8").strip()
    else:
        yolo_weights = str(Path(base_out_dir) / "yolo11m-seg" / "weights" / "best.pt")

    mrcnn_weights = str(Path(base_out_dir) / "maskrcnn" / "best_model.pt")
    fastrcnn_weights = str(Path(base_out_dir) / "fastrcnn" / "best_model.pt")

    eval_jobs = [
        ("06a_eval_yolo",     ["--model", "yolo",     "--weights", yolo_weights,     "--dataset", dataset]),
        ("06c_eval_maskrcnn", ["--model", "maskrcnn", "--weights", mrcnn_weights,    "--dataset", dataset]),
        ("06d_eval_fastrcnn", ["--model", "fastrcnn", "--weights", fastrcnn_weights, "--dataset", dataset]),
    ]

    for label, eval_args in eval_jobs:
        weights_path = Path(eval_args[3])
        if not weights_path.exists():
            print(f"  [skip] {label}: weights not found at {weights_path}")
            continue
        log_path = os.path.join(logs_dir, f"{label}.log")
        cmd = ["python", "scripts/evaluation/evaluate_confusion_matrix.py"] + eval_args
        print(f"[*] {label}...")
        run_cmd_and_log(cmd, log_path, label)

    # Compare all models
    log_path = os.path.join(logs_dir, "07_compare_all.log")
    cmd = ["python", "scripts/evaluation/compare_all_models.py", "--dataset", dataset]
    print("[*] Comparing all models...")
    run_cmd_and_log(cmd, log_path, "compare_all_models")


def generate_excel_report(base_out_dir):
    """
    Generates car_part_training_results.xlsx containing:
    - Summary sheet (Model, Best Epoch, Best Mask mAP50)
    - Per-model epoch detail sheets with losses and mAP metrics.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        print("[WARN] openpyxl is not installed. Skipping Excel report generation.")
        return

    base_path = Path(base_out_dir)
    if not base_path.exists():
        return

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default blank sheet

    bold_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    summary_rows = []
    model_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name != "logs"])

    for mdir in model_dirs:
        model_name = mdir.name
        metrics_csv = mdir / "metrics.csv"
        epoch_data = []

        if metrics_csv.exists():
            import csv
            try:
                with open(metrics_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        epoch_data.append(r)
            except Exception as e:
                print(f"[WARN] Could not read {metrics_csv}: {e}")
        else:
            for jf in sorted(mdir.glob("*.json")):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "epoch" in data:
                            epoch_data.append(data)
                except Exception:
                    pass

        if not epoch_data:
            continue

        sheet_title = f"{model_name[:24]}_Epochs".replace("-", "_")
        ws = wb.create_sheet(title=sheet_title)

        headers = ["Epoch", "Train Loss", "Val Loss", "Mask mAP50", "Box mAP50"]
        ws.append(headers)
        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.font = bold_font
            cell.fill = header_fill

        best_map = -1.0
        best_epoch = 1
        for row in epoch_data:
            ep = row.get("epoch", "")
            t_loss = row.get("train_loss") or (row.get("train_stats", {}).get("train_loss") if isinstance(row.get("train_stats"), dict) else "")
            v_loss = row.get("val_loss") or (row.get("train_stats", {}).get("val_loss") if isinstance(row.get("train_stats"), dict) else "")
            m_map = row.get("map50_mask") or row.get("best_mask_map50", "")
            b_map = row.get("map50_box", "")

            try:
                m_map_val = float(m_map)
                if m_map_val > best_map:
                    best_map = m_map_val
                    best_epoch = ep
            except (ValueError, TypeError):
                pass

            ws.append([ep, t_loss, v_loss, m_map, b_map])

        summary_rows.append((model_name, best_epoch, best_map if best_map >= 0 else "N/A"))

    summary_ws = wb.create_sheet(title="Summary", index=0)
    summary_headers = ["Model", "Best Epoch", "Best Mask mAP50"]
    summary_ws.append(summary_headers)
    for col_num in range(1, len(summary_headers) + 1):
        cell = summary_ws.cell(row=1, column=col_num)
        cell.font = bold_font
        cell.fill = header_fill

    summary_rows.sort(key=lambda x: (isinstance(x[2], (int, float)), x[2]), reverse=True)
    for row in summary_rows:
        summary_ws.append(list(row))

    excel_path = base_path / "car_part_training_results.xlsx"
    wb.save(excel_path)
    print(f"\n[OK] Excel report saved to {excel_path}")


def main():
    check_azure_disk()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = str(PROJECT_ROOT / "runs_comparison" / f"run_{timestamp}")
    logs_dir = os.path.join(base_out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(logs_dir, "orchestrator.log"), "w", encoding="utf-8") as f:
        f.write(f"Run started at {datetime.datetime.now()}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")

    display_main_menu()
    choice = prompt("Enter choice [1-14]: ")

    if choice == "1":
        print("\n[TASK] Auto-annotation (DINO + SAM2)")
        inp = prompt_default("Input images directory", "./RAW_DATASET/IMAGES")
        out = prompt_default("Output directory", "./datasets/auto_annotated")
        save_run_config(logs_dir, "Auto-annotation", dataset="N/A",
                        extra={"input_dir": inp, "output_dir": out})
        log_path = os.path.join(logs_dir, "01_annotation.log")
        cmd = ["python", "scripts/inference/auto_annotate_carparts.py", "--input", inp, "--output", out]
        run_cmd_and_log(cmd, log_path, "annotation")

    elif choice == "2":
        print("\n[TASK] Dataset Prep & Local Model Training")
        models = collect_models()
        hparams = Hyperparams(models)
        dataset = run_dataset_prep(logs_dir)
        run_config_path = save_run_config(logs_dir, "Train Model Locally", dataset, models, hparams)
        # Generate Markdown report after dataset prep
        _generate_report(run_config_path, logs_dir)
        print(f"\n[INFO] Using dataset: {dataset}")

        if models.yolo:
            log_path = os.path.join(logs_dir, "02_train_yolo.log")
            out_dir = os.path.join(base_out_dir, "yolo11m-seg")
            cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11m-seg",
                   "--dataset", dataset, "--epochs", str(hparams.yolo_epochs),
                   "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers),
                   "--val_interval", str(hparams.yolo_val_interval),
                   "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_yolo")

        if models.yolo11x:
            log_path = os.path.join(logs_dir, "02_train_yolo11x.log")
            out_dir = os.path.join(base_out_dir, "yolo11x-seg")
            cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11x-seg",
                   "--dataset", dataset, "--epochs", str(hparams.yolo_epochs),
                   "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers),
                   "--val_interval", str(hparams.yolo_val_interval),
                   "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_yolo11x")

        if models.maskrcnn:
            log_path = os.path.join(logs_dir, "02_train_maskrcnn.log")
            out_dir = os.path.join(base_out_dir, "maskrcnn")
            cmd = ["python", "scripts/training/train_maskrcnn.py", "--dataset", dataset,
                   "--epochs", str(hparams.mrcnn_epochs), "--batch", str(hparams.mrcnn_batch),
                   "--num_workers", str(hparams.mrcnn_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.mrcnn_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_maskrcnn")

        if models.fastrcnn:
            log_path = os.path.join(logs_dir, "02_train_fastrcnn.log")
            out_dir = os.path.join(base_out_dir, "fastrcnn")
            cmd = ["python", "scripts/training/train_fastrcnn.py", "--dataset", dataset,
                   "--epochs", str(hparams.fastrcnn_epochs), "--batch", str(hparams.fastrcnn_batch),
                   "--num_workers", str(hparams.fastrcnn_workers), "--project", out_dir,
                   "--val_interval", str(hparams.fastrcnn_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_fastrcnn")

        if models.mask2former:
            log_path = os.path.join(logs_dir, "02_train_mask2former.log")
            out_dir = os.path.join(base_out_dir, "mask2former")
            cmd = ["python", "scripts/training/train_mask2former.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_mask2former")

        if models.sam2:
            log_path = os.path.join(logs_dir, "02_train_sam2.log")
            out_dir = os.path.join(base_out_dir, "sam2")
            cmd = ["python", "scripts/training/train_sam2_seg.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_sam2")

        if models.maskdino:
            log_path = os.path.join(logs_dir, "02_train_maskdino.log")
            out_dir = os.path.join(base_out_dir, "maskdino")
            cmd = ["python", "scripts/training/train_maskdino.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_maskdino")

        if models.segformer:
            log_path = os.path.join(logs_dir, "02_train_segformer.log")
            out_dir = os.path.join(base_out_dir, "segformer")
            cmd = ["python", "scripts/training/train_segformer.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_segformer")

        print("\nAll training done. Generating Excel report...")
        generate_excel_report(base_out_dir)

    elif choice == "3":
        print("\n[TASK] Inference on Test Images")
        inp = prompt_default("Test images directory", "./test")
        out = prompt_default("Output directory", "./test_result")
        save_run_config(logs_dir, "Inference", dataset="N/A",
                        extra={"input_dir": inp, "output_dir": out})
        log_path = os.path.join(logs_dir, "01_inference.log")
        cmd = ["python", "scripts/inference/infer_both_models.py", "--input", inp, "--output", out]
        run_cmd_and_log(cmd, log_path, "inference")

    elif choice == "4":
        print("\n[TASK] Full Pipeline (Prep -> Train -> Evaluate)")
        models = collect_models()
        hparams = Hyperparams(models)
        dataset = run_dataset_prep(logs_dir)
        run_config_path = save_run_config(logs_dir, "Full Pipeline (Prep → Train → Evaluate)", dataset, models, hparams)
        _generate_report(run_config_path, logs_dir)
        print(f"\n[INFO] Using dataset: {dataset}")
        print("[*] Running local training...")
        if models.yolo:
            log_path = os.path.join(logs_dir, "02_train_yolo.log")
            out_dir = os.path.join(base_out_dir, "yolo11m-seg")
            cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11m-seg",
                   "--dataset", dataset, "--epochs", str(hparams.yolo_epochs),
                   "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers),
                   "--val_interval", str(hparams.yolo_val_interval),
                   "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_yolo")

        if models.maskrcnn:
            log_path = os.path.join(logs_dir, "02_train_maskrcnn.log")
            out_dir = os.path.join(base_out_dir, "maskrcnn")
            cmd = ["python", "scripts/training/train_maskrcnn.py", "--dataset", dataset,
                   "--epochs", str(hparams.mrcnn_epochs), "--batch", str(hparams.mrcnn_batch),
                   "--num_workers", str(hparams.mrcnn_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.mrcnn_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_maskrcnn")

        if models.fastrcnn:
            log_path = os.path.join(logs_dir, "02_train_fastrcnn.log")
            out_dir = os.path.join(base_out_dir, "fastrcnn")
            cmd = ["python", "scripts/training/train_fastrcnn.py", "--dataset", dataset,
                   "--epochs", str(hparams.fastrcnn_epochs), "--batch", str(hparams.fastrcnn_batch),
                   "--num_workers", str(hparams.fastrcnn_workers), "--project", out_dir,
                   "--val_interval", str(hparams.fastrcnn_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_fastrcnn")

        if models.mask2former:
            log_path = os.path.join(logs_dir, "02_train_mask2former.log")
            out_dir = os.path.join(base_out_dir, "mask2former")
            cmd = ["python", "scripts/training/train_mask2former.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_mask2former")

        if models.sam2:
            log_path = os.path.join(logs_dir, "02_train_sam2.log")
            out_dir = os.path.join(base_out_dir, "sam2")
            cmd = ["python", "scripts/training/train_sam2_seg.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_sam2")

        if models.maskdino:
            log_path = os.path.join(logs_dir, "02_train_maskdino.log")
            out_dir = os.path.join(base_out_dir, "maskdino")
            cmd = ["python", "scripts/training/train_maskdino.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_maskdino")

        if models.segformer:
            log_path = os.path.join(logs_dir, "02_train_segformer.log")
            out_dir = os.path.join(base_out_dir, "segformer")
            cmd = ["python", "scripts/training/train_segformer.py", "--dataset", dataset,
                   "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                   "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                   "--val_interval", str(hparams.m2f_val_interval)]
            run_cmd_and_log(cmd, log_path, "train_segformer")

        run_evaluation(dataset, base_out_dir, logs_dir)
        print("\nAll steps done. Generating Excel report...")
        generate_excel_report(base_out_dir)

    elif choice == "5":
        print("\n[TASK] Automated Azure ML Training")
        models = collect_models()
        hparams = Hyperparams(models)
        local_dir = prompt_default("Local dataset directory", "./datasets/combined_carparts")
        save_run_config(logs_dir, "Azure ML Training", dataset=local_dir, models=models, hparams=hparams,
                        extra={"mode": "azure"})

        all_selected = (models.yolo and models.yolo11x and models.maskrcnn and models.mask2former and models.sam2 and models.maskdino and models.segformer)
        if all_selected:
            log_path = os.path.join(logs_dir, "00_azure_train_all.log")
            cmd = ["python", "scripts/training/azure_train.py", "--model", "all",
                   "--local_dataset_dir", local_dir, "--epochs", str(hparams.yolo_epochs),
                   "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers), "--auto_upload"]
            run_cmd_and_log(cmd, log_path, "azure_train_all")
        else:
            if models.yolo:
                log_path = os.path.join(logs_dir, "00_azure_train_yolo.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "yolo11m-seg",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.yolo_epochs),
                       "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_yolo")

            if models.yolo11x:
                log_path = os.path.join(logs_dir, "00_azure_train_yolo11x.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "yolo11x-seg",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.yolo_epochs),
                       "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_yolo11x")

            if models.maskrcnn:
                log_path = os.path.join(logs_dir, "00_azure_train_maskrcnn.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "maskrcnn",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.mrcnn_epochs),
                       "--batch", str(hparams.mrcnn_batch), "--workers", str(hparams.mrcnn_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_maskrcnn")

            if models.mask2former:
                log_path = os.path.join(logs_dir, "00_azure_train_mask2former.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "mask2former",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.m2f_epochs),
                       "--batch", str(hparams.m2f_batch), "--workers", str(hparams.m2f_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_mask2former")

            if models.sam2:
                log_path = os.path.join(logs_dir, "00_azure_train_sam2.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "sam2",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.m2f_epochs),
                       "--batch", str(hparams.m2f_batch), "--workers", str(hparams.m2f_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_sam2")

            if models.maskdino:
                log_path = os.path.join(logs_dir, "00_azure_train_maskdino.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "maskdino",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.m2f_epochs),
                       "--batch", str(hparams.m2f_batch), "--workers", str(hparams.m2f_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_maskdino")

            if models.segformer:
                log_path = os.path.join(logs_dir, "00_azure_train_segformer.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "segformer",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.m2f_epochs),
                       "--batch", str(hparams.m2f_batch), "--workers", str(hparams.m2f_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_segformer")

    elif choice == "6":
        print("\n[TASK] Setup and Clean")
        log_path = os.path.join(logs_dir, "00_setup_clean.log")
        cmd = ["python", "scripts/setup_and_clean.py"]
        run_cmd_and_log(cmd, log_path, "setup_clean")

    elif choice == "7":
        print("\n[TASK] Smart Auto-Prepare Dataset (VM-Ready Full Pipeline)")
        combined_name = ensure_and_prepare_datasets(logs_dir)
        # Save a minimal run config so the report has something to read
        run_config_path = save_run_config(logs_dir, "Auto-Prepare Dataset", combined_name)
        # Generate Markdown pipeline report
        _generate_report(run_config_path, logs_dir)
        print(f"\n[OK] Dataset ready in './datasets/{combined_name}'!")
        analyze_dataset(combined_name)

    elif choice == "8":
        print("\n[TASK] Dataset Preprocessing Tools")
        print("  a) yolo_to_coco    -- Convert YOLO polygons -> COCO JSON (parallel)")
        print("  b) combine         -- Merge multiple YOLO datasets (parallel copy)")
        print("  c) verify_labels   -- Scan for corrupt images (report-only)")
        print("  d) hash_dataset    -- Write blake3 checksums manifest")
        sub = prompt("Enter sub-option [a-d]: ")
        if sub == "a":
            print("  Available datasets: carparts-seg | custom_carparts | combined_carparts")
            ds = prompt_default("Dataset", "carparts-seg")
            workers = prompt_default("Workers", "8")
            cmd = ["python", "scripts/data/yolo_to_coco.py", "--dataset", ds, "--num_workers", str(workers)]
            log_path = os.path.join(logs_dir, "preprocess_yolo_to_coco.log")
            run_cmd_and_log(cmd, log_path, "yolo_to_coco")
        elif sub == "b":
            dirs_raw = prompt_default("Dataset dirs (comma-separated)", "datasets/carparts-seg,datasets/custom_carparts")
            out = prompt_default("Output dir", "datasets/combined_carparts")
            workers = prompt_default("Workers", "8")
            cmd = ["python", "scripts/data/combine_datasets.py", "--datasets", dirs_raw, "--out_dir", out, "--workers", str(workers)]
            log_path = os.path.join(logs_dir, "preprocess_combine.log")
            run_cmd_and_log(cmd, log_path, "combine_datasets")
        elif sub == "c":
            dir_raw = prompt_default("Dataset directory to verify", "datasets")
            cmd = ["python", "scripts/data/verify_labels.py", "--dir", dir_raw]
            log_path = os.path.join(logs_dir, "preprocess_verify.log")
            run_cmd_and_log(cmd, log_path, "verify_labels")
        elif sub == "d":
            dir_raw = prompt_default("Directory to hash", "datasets")
            out_manifest = prompt_default("Output manifest path", f"{dir_raw}/checksums.blake3")
            workers = prompt_default("Workers", "8")
            cmd = ["python", "scripts/data/hash_dataset.py", "--dir", dir_raw, "--out", out_manifest, "--workers", str(workers)]
            log_path = os.path.join(logs_dir, "preprocess_hash.log")
            run_cmd_and_log(cmd, log_path, "hash_dataset")
        else:
            print(f"[ERROR] Unknown sub-option '{sub}'")

    elif choice == "9":
        print("\n[TASK] Dataset Analytics")
        ds = prompt_default("Dataset name (under ./datasets)", "combined_carparts")
        analyze_dataset(ds)

    elif choice == "10":
        print("\n[TASK] Benchmark Preprocessing")
        ds = prompt_default("Dataset to benchmark", "carparts-seg")
        print()
        print(f"[*] Benchmarking yolo_to_coco on '{ds}'...")
        t0 = datetime.datetime.now()
        subprocess.run(["python", "scripts/data/yolo_to_coco.py", "--dataset", ds], cwd=PROJECT_ROOT)
        t_coco = (datetime.datetime.now() - t0).total_seconds()

        print(f"\n[*] Benchmarking verify_labels on 'datasets/{ds}'...")
        t0 = datetime.datetime.now()
        subprocess.run(["python", "scripts/data/verify_labels.py",
                        "--dir", str(PROJECT_ROOT / "datasets" / ds)], cwd=PROJECT_ROOT)
        t_verify = (datetime.datetime.now() - t0).total_seconds()

        print("\n╔══════════════════╦══════════════╗")
        print("║ Step             ║ Time (s)     ║")
        print("╠══════════════════╬══════════════╣")
        print(f"║ yolo_to_coco     ║ {t_coco:>12.3f} ║")
        print(f"║ verify_labels    ║ {t_verify:>12.3f} ║")
        print("╚══════════════════╩══════════════╝")

    elif choice == "11":
        print("\n[TASK] Quick Pipeline Check -- Fast 3-Epoch Dry Run on Azure ML (All Models)")
        local_dir = prompt_default("Local dataset directory", "./datasets/combined_carparts")
        save_run_config(logs_dir, "Quick Pipeline Check (3-epoch dry run)", dataset=local_dir,
                        extra={"epochs": 3, "batch": "auto(-1)", "workers": 8, "mode": "azure_dryrun"})
        log_path = os.path.join(logs_dir, "00_azure_dryrun_all.log")
        print("[*] Submitting 3-epoch dry-run job for YOLO + Mask R-CNN + Mask2Former...")
        cmd = ["python", "scripts/training/azure_train.py", "--model", "all",
               "--local_dataset_dir", local_dir, "--epochs", "3",
               "--batch", "-1", "--workers", "8", "--auto_upload"]
        run_cmd_and_log(cmd, log_path, "azure_dryrun_all")

    elif choice == "12":
        print("\n[TASK] GPU Capacity & Batch Size / Worker Stress Tester")
        mode = prompt_default("Run mode (local / azure)", "local")
        ds = prompt_default("Dataset path", "./datasets/combined_carparts")
        save_run_config(logs_dir, "GPU Capacity Check", dataset=ds,
                        extra={"mode": mode})
        log_path = os.path.join(logs_dir, "00_capacity_check.log")
        cmd = ["python", "scripts/training/capacity_check.py", "--dataset", ds, "--mode", mode]
        run_cmd_and_log(cmd, log_path, "capacity_check")

    elif choice == "13":
        print("\n[TASK] Cleanup Pipeline — Wipe stale outputs (interactive)")
        log_path = os.path.join(logs_dir, "00_cleanup.log")
        cmd = ["python", "scripts/data/cleanup_pipeline.py"]
        run_cmd_and_log(cmd, log_path, "cleanup_pipeline")

    elif choice == "14":
        print("Exiting.")
        sys.exit(0)

    else:
        print("[ERROR] Invalid choice.")

    print("\n==============================================")
    print(" TASK COMPLETE")
    print(f" Outputs : {base_out_dir}")
    print(f" Logs    : {logs_dir}")
    print("==============================================")

if __name__ == "__main__":
    main()
