#!/usr/bin/env python3
"""
orchestrator.py -- Interactive Pipeline Orchestrator in Python.

Provides an interactive CLI menu to run data prep, model training (local/Azure ML),
inference, evaluation, cleanup, and fast preprocessing without compilation dependencies.
"""

import sys
import os
import subprocess
import datetime
import glob
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.resolve()

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
    print("  4) Run Full Local        -- Dataset Prep -> Train -> Evaluate")
    print("  5) Train on Azure ML     -- Auto-upload dataset & submit GPU job to Azure ML")
    print("  6) Setup & Clean         -- Folder setup and log archive")
    print("  7) Auto-Prepare Dataset  -- Check RAW/online, download if missing, convert & combine")
    print("  8) Dataset Analytics     -- View image counts, total annotations, & per-label breakdown")
    print("  9) Benchmark             -- Compare Rust vs Python preprocessing speed")
    print("  10) Quick Pipeline Check -- Fast 3-epoch dry run on Azure ML (All Models)")
    print("  11) GPU Capacity Check   -- Stepwise batch size & worker stress tuner")
    print("  12) Exit")
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

        print()
        print("======================================================================")
        if recs:
            print(" Set training hyperparameters (Auto-loaded defaults from latest GPU check)")
        else:
            print(" Set training hyperparameters (press Enter to accept defaults)")
        print("======================================================================")

        if models is None or getattr(models, "yolo", False):
            yolo_default_b = get_default_batch("yolo11m-seg", 8)
            yolo_default_w = get_default_workers("yolo11m-seg", 4)
            print("\n---- YOLOv11m-seg ----")
            self.yolo_epochs = prompt_int("Epochs", 50)
            self.yolo_batch = prompt_int("Batch size (-1 = auto)", yolo_default_b)
            self.yolo_workers = prompt_int("Dataloader workers", yolo_default_w)

        if models is None or getattr(models, "yolo11x", False):
            yolo11x_default_b = get_default_batch("yolo11x-seg", 8)
            yolo11x_default_w = get_default_workers("yolo11x-seg", 4)
            print("\n---- YOLO11x-seg (Extra Large) ----")
            self.yolo11x_epochs = prompt_int("Epochs", 50)
            self.yolo11x_batch = prompt_int("Batch size (-1 = auto)", yolo11x_default_b)
            self.yolo11x_workers = prompt_int("Dataloader workers", yolo11x_default_w)

        if models is None or getattr(models, "maskrcnn", False):
            mrcnn_default_b = get_default_batch("maskrcnn", 2)
            mrcnn_default_w = get_default_workers("maskrcnn", 4)
            print("\n---- Mask R-CNN ----")
            self.mrcnn_epochs = prompt_int("Epochs", 20)
            self.mrcnn_batch = prompt_int("Batch size", mrcnn_default_b)
            self.mrcnn_workers = prompt_int("Dataloader workers", mrcnn_default_w)

        if models is None or getattr(models, "fastrcnn", False):
            fastrcnn_default_b = get_default_batch("fastrcnn", 2)
            fastrcnn_default_w = get_default_workers("fastrcnn", 4)
            print("\n---- Fast R-CNN ----")
            self.fastrcnn_epochs = prompt_int("Epochs", 20)
            self.fastrcnn_batch = prompt_int("Batch size", fastrcnn_default_b)
            self.fastrcnn_workers = prompt_int("Dataloader workers", fastrcnn_default_w)

        if models is None or getattr(models, "mask2former", False):
            m2f_default_b = get_default_batch("mask2former", 2)
            m2f_default_w = get_default_workers("mask2former", 4)
            print("\n---- Mask2Former (Swin Transformer) ----")
            self.m2f_epochs = prompt_int("Epochs", 20)
            self.m2f_batch = prompt_int("Batch size", m2f_default_b)
            self.m2f_workers = prompt_int("Dataloader workers", m2f_default_w)

def run_cmd_and_log(cmd, log_path, label):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lf.write(f"=== {label} | {timestamp} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")

    print(f"\n[*] Running [{label}] ...")
    # Ensure python child processes stream output line-by-line without buffering
    if cmd and cmd[0] == "python" and (len(cmd) == 1 or cmd[1] != "-u"):
        cmd = ["python", "-u"] + cmd[1:]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env
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

def ensure_and_prepare_datasets(logs_dir):
    """
    Checks RAW_DATASET and online datasets. If online datasets are missing,
    automatically downloads them, converts custom RAW XMLs, remaps labels,
    and merges everything into datasets/combined_carparts.
    """
    datasets_dir = PROJECT_ROOT / "datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "01_smart_dataset_prep.log")

    # 1. Process custom RAW_DATASET if files exist
    raw_images = PROJECT_ROOT / "RAW_DATASET" / "IMAGES"
    raw_xmls   = PROJECT_ROOT / "RAW_DATASET" / "XML"
    has_raw = raw_images.exists() and any(raw_images.iterdir()) and raw_xmls.exists() and any(raw_xmls.iterdir())
    
    if has_raw:
        print("[*] Found custom RAW_DATASET files. Converting XML to YOLO format...")
        run_cmd_and_log(["python", "scripts/data/prepare_raw_dataset.py"], log_path, "raw_prep")
    else:
        print("[INFO] RAW_DATASET/IMAGES or XML is empty -- skipping custom raw prep.")

    # 2. Check and Download Ultralytics carparts-seg (133 MB)
    carparts_dir = datasets_dir / "carparts-seg"
    if not carparts_dir.exists() or not any(carparts_dir.iterdir()):
        print("\n[*] Downloading online dataset: Ultralytics carparts-seg (133 MB)...")
        zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
        zip_file = datasets_dir / "carparts-seg.zip"
        try:
            import urllib.request, zipfile
            print("    Downloading carparts-seg.zip...")
            urllib.request.urlretrieve(zip_url, zip_file)
            print("    Extracting to ./datasets/carparts-seg...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(carparts_dir)
            if zip_file.exists():
                os.remove(zip_file)
            print("[OK] Ultralytics carparts-seg downloaded and extracted!")
        except Exception as e:
            print(f"[WARN] Failed to download carparts-seg: {e}")
    else:
        print("[OK] Online dataset 'carparts-seg' already exists.")

    # 3. Check and Download DSMLR dataset
    dsmlr_dir = datasets_dir / "dsmlr-carparts"
    if not dsmlr_dir.exists() or not any(dsmlr_dir.iterdir()):
        print("\n[*] Cloning online dataset: DSMLR Car-Parts-Segmentation repo...")
        cmd = ["git", "clone", "https://github.com/dsmlr/Car-Parts-Segmentation.git", str(dsmlr_dir)]
        run_cmd_and_log(cmd, log_path, "dsmlr_clone")
    else:
        print("[OK] Online dataset 'dsmlr-carparts' already exists.")

    # 4. Auto-split DSMLR if present
    if dsmlr_dir.exists():
        print("\n[*] Preparing DSMLR splits...")
        run_cmd_and_log(["python", "scripts/data/prepare_dsmlr_split.py"], log_path, "dsmlr_split")

    # 5. Remap labels for canonical taxonomy
    print("\n[*] Remapping label taxonomies...")
    run_cmd_and_log(["python", "scripts/data/remap_github_labels.py", "--format", "yolo", "--path", "datasets/carparts-seg"], log_path, "remap_yolo")

    # 6. Combine Datasets into combined_carparts
    print("\n[*] Combining RAW_DATASET + Online datasets...")
    run_cmd_and_log(["python", "scripts/data/combine_datasets.py"], log_path, "combine_datasets")

    # 7. Convert to COCO JSON format
    print("\n[*] Converting combined dataset to COCO format...")
    run_cmd_and_log(["python", "scripts/data/yolo_to_coco.py", "--dataset", "combined_carparts"], log_path, "yolo_to_coco")

    print("\n[OK] Smart Auto-Preparation Complete! Combined dataset: './datasets/combined_carparts'")
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

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = str(PROJECT_ROOT / "runs_comparison" / f"run_{timestamp}")
    logs_dir = os.path.join(base_out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(logs_dir, "orchestrator.log"), "w", encoding="utf-8") as f:
        f.write(f"Run started at {datetime.datetime.now()}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")

    display_main_menu()
    choice = prompt("Enter choice [1-12]: ")

    if choice == "1":
        print("\n[TASK] Auto-annotation (DINO + SAM2)")
        inp = prompt_default("Input images directory", "./RAW_DATASET/IMAGES")
        out = prompt_default("Output directory", "./datasets/auto_annotated")
        log_path = os.path.join(logs_dir, "01_annotation.log")
        cmd = ["python", "scripts/inference/auto_annotate_carparts.py", "--input", inp, "--output", out]
        run_cmd_and_log(cmd, log_path, "annotation")

    elif choice == "2":
        print("\n[TASK] Dataset Prep & Local Model Training")
        models = collect_models()
        hparams = Hyperparams(models)
        dataset = run_dataset_prep(logs_dir)
        print(f"\n[INFO] Using dataset: {dataset}")
        
        if models.yolo:
            log_path = os.path.join(logs_dir, "02_train_yolo.log")
            out_dir = os.path.join(base_out_dir, "yolo11m-seg")
            cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11m-seg",
                   "--dataset", dataset, "--epochs", str(hparams.yolo_epochs),
                   "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers),
                   "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_yolo")

        if getattr(models, 'yolo11x', False):
            log_path = os.path.join(logs_dir, "02_train_yolo11x.log")
            out_dir = os.path.join(base_out_dir, "yolo11x-seg")
            cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11x-seg",
                   "--dataset", dataset, "--epochs", str(hparams.yolo11x_epochs),
                   "--batch", str(hparams.yolo11x_batch), "--workers", str(hparams.yolo11x_workers),
                   "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_yolo11x")

        if models.maskrcnn:
            log_path = os.path.join(logs_dir, "02_train_maskrcnn.log")
            out_dir = os.path.join(base_out_dir, "maskrcnn")
            cmd = ["python", "scripts/training/train_maskrcnn.py", "--dataset", dataset,
                   "--epochs", str(hparams.mrcnn_epochs), "--batch", str(hparams.mrcnn_batch),
                   "--num_workers", str(hparams.mrcnn_workers), "--output_dir", out_dir]
            run_cmd_and_log(cmd, log_path, "train_maskrcnn")

        if getattr(models, 'fastrcnn', False):
            log_path = os.path.join(logs_dir, "02_train_fastrcnn.log")
            out_dir = os.path.join(base_out_dir, "fastrcnn")
            cmd = ["python", "scripts/training/train_fastrcnn.py", "--dataset", dataset,
                   "--epochs", str(hparams.fastrcnn_epochs), "--batch", str(hparams.fastrcnn_batch),
                   "--num_workers", str(hparams.fastrcnn_workers), "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_fastrcnn")

    elif choice == "3":
        print("\n[TASK] Inference on Test Images")
        inp = prompt_default("Test images directory", "./test")
        out = prompt_default("Output directory", "./test_result")
        log_path = os.path.join(logs_dir, "01_inference.log")
        cmd = ["python", "scripts/inference/infer_both_models.py", "--input", inp, "--output", out]
        run_cmd_and_log(cmd, log_path, "inference")

    elif choice == "4":
        print("\n[TASK] Full Pipeline (Prep -> Train -> Evaluate)")
        models = collect_models()
        hparams = Hyperparams(models)
        dataset = run_dataset_prep(logs_dir)
        print(f"\n[INFO] Using dataset: {dataset}")
        print("[*] Running local training...")
        if models.yolo:
            log_path = os.path.join(logs_dir, "02_train_yolo.log")
            out_dir = os.path.join(base_out_dir, "yolo11m-seg")
            cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11m-seg",
                   "--dataset", dataset, "--epochs", str(hparams.yolo_epochs),
                   "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers),
                   "--project", out_dir]
            run_cmd_and_log(cmd, log_path, "train_yolo")

    elif choice == "5":
        print("\n[TASK] Automated Azure ML Training")
        models = collect_models()
        hparams = Hyperparams()
        local_dir = prompt_default("Local dataset directory", "./datasets/combined_carparts")

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
        print("\n[TASK] Smart Auto-Prepare Dataset")
        combined_name = ensure_and_prepare_datasets(logs_dir)
        print(f"\n[OK] Dataset ready in './datasets/{combined_name}'!")
        analyze_dataset(combined_name)

    elif choice == "8":
        print("\n[TASK] Dataset Analytics")
        ds = prompt_default("Dataset name (under ./datasets)", "combined_carparts")
        analyze_dataset(ds)

    elif choice == "9":
        print("\n[TASK] Benchmark Preprocessing")
        ds = prompt_default("Dataset to benchmark", "carparts-seg")
        print(f"[*] Running Python yolo_to_coco benchmark on '{ds}'...")
        start_time = datetime.datetime.now()
        subprocess.run(["python", "scripts/data/yolo_to_coco.py", "--dataset", ds])
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print(f"\n[BENCHMARK] Python yolo_to_coco completed in {elapsed:.3f} seconds.")

    elif choice == "10":
        print("\n[TASK] Quick Pipeline Check -- Fast 3-Epoch Dry Run on Azure ML (All Models)")
        local_dir = prompt_default("Local dataset directory", "./datasets/combined_carparts")
        log_path = os.path.join(logs_dir, "00_azure_dryrun_all.log")
        print("[*] Submitting 3-epoch dry-run job for YOLO + Mask R-CNN + Mask2Former...")
        cmd = ["python", "scripts/training/azure_train.py", "--model", "all",
               "--local_dataset_dir", local_dir, "--epochs", "3",
               "--batch", "-1", "--workers", "8", "--auto_upload"]
        run_cmd_and_log(cmd, log_path, "azure_dryrun_all")

    elif choice == "11":
        print("\n[TASK] GPU Capacity & Batch Size / Worker Stress Tester")
        mode = prompt_default("Run mode (local / azure)", "local")
        ds = prompt_default("Dataset path", "./datasets/combined_carparts")
        models_input = prompt_default("Models to test (all / yolo11m-seg / yolo11x-seg / maskrcnn / etc.)", "all")
        log_path = os.path.join(logs_dir, "00_capacity_check.log")
        cmd = ["python", "-u", "scripts/training/capacity_check.py", "--dataset", ds, "--mode", mode]
        if models_input.strip() and models_input.strip().lower() != "all":
            selected_models = [m.strip() for m in models_input.replace(",", " ").split() if m.strip()]
            if selected_models:
                cmd.extend(["--models"] + selected_models)
        run_cmd_and_log(cmd, log_path, "capacity_check")

    elif choice == "12":
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
