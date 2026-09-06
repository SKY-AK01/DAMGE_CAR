"""
train_yolo_seg.py
-------------------
Trains YOLOv8-seg or YOLOv11-seg on the carparts-seg dataset.
Auto-detects GPU. Auto-downloads the pretrained checkpoint (ultralytics
handles this internally on first run).

Usage:
    python train_yolo_seg.py --model yolo11m-seg --epochs 100 --dataset carparts-seg
    python train_yolo_seg.py --model yolo11m-seg --epochs 100 --dataset dsmlr-carparts

---------------------------------------------------------------------------
GPU UTILIZATION FIX (A10 12GB, was ~4.6GB used / ~49% util at batch=16):
---------------------------------------------------------------------------
Ultralytics already does real batched GPU training (unlike the HF scripts),
so this one just needed:
  1. --batch default raised 16 -> 24. You can also pass --batch -1 to let
     Ultralytics' autobatch pick a batch size targeting ~90% VRAM usage
     automatically instead of guessing a fixed number -- recommended if
     you're not sure how much headroom 512x512 vs other imgsz leaves.
  2. Explicit --workers (default 8) passed into model.train(), since the
     default of 8 wasn't being set explicitly here and dataloading can
     otherwise become the bottleneck between GPU steps.

---------------------------------------------------------------------------
ROUND 2 FIX (A10-12Q vGPU -- 15.8G/216G RAM used, only 2-3/18 CPU cores busy,
GPU util oscillating 5%-94% on a ~13s cycle matching epoch boundaries):
---------------------------------------------------------------------------
This is a vGPU (time-sliced) profile, so a flat 100% GPU util is not
achievable no matter what we change here -- the hypervisor rotates compute
across tenants. What IS fixable is the epoch-boundary dip and the huge
unused RAM/CPU headroom:
  3. --cache changed default to "ram": with 200GB+ system RAM free and a
     dataset this size, caching decoded images in RAM after epoch 1 removes
     repeated disk reads entirely.
  4. --workers default raised 8 -> 16, since CPU cores were sitting mostly
     idle -- cheap and removes any doubt dataloading contributes to the dip.
  5. --batch default raised 24 -> 32, since VRAM had ~4GB of headroom
     (8011MiB / 12288MiB used at batch=24). Still exposed as a flag, and
     --batch -1 (autobatch) remains the safest option if you change imgsz.
  6. plots=False during training (was implicitly True) -- Ultralytics
     regenerates matplotlib training plots at every epoch/validation
     boundary, which is CPU-bound work that stalls the pipeline right when
     GPU util was already dipping. Final plots are still generated once,
     explicitly, after training finishes via a dedicated model.val() plot
     pass -- see bottom of main().
"""

import argparse
import os
import json
import torch
import numpy as np
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(PROJECT_ROOT))
from scripts.evaluation.unified_logger import UnifiedLogger
from scripts.evaluation.unified_evaluator import UnifiedEvaluator


def make_yaml(dataset_name):
    """Generates or finds the data.yaml ultralytics needs, pointing at the right split."""
    ds_path = Path(dataset_name)
    if ds_path.is_dir():
        resolved_ds = ds_path.resolve()
        if (resolved_ds / "combined_carparts").exists():
            target_root = resolved_ds / "combined_carparts"
        else:
            target_root = resolved_ds

        yaml_content = f"""path: {target_root.as_posix()}
train: images/train
val: images/val

names:
  0: back_bumper
  1: back_door
  2: back_glass
  3: back_left_door
  4: back_left_light
  5: back_light
  6: back_right_door
  7: back_right_light
  8: front_bumper
  9: front_door
  10: front_glass
  11: front_left_door
  12: front_left_light
  13: front_light
  14: front_right_door
  15: front_right_light
  16: hood
  17: left_mirror
  18: object
  19: right_mirror
  20: tailgate
  21: trunk
  22: wheel
"""
        import tempfile
        yaml_file = Path(tempfile.gettempdir()) / "car_parts_data.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        return str(yaml_file)

    if dataset_name == "carparts-seg":
        return "datasets/carparts-seg/carparts-seg.yaml"
    elif dataset_name == "custom_carparts":
        return "datasets/custom_carparts/data.yaml"
    elif dataset_name == "combined_carparts":
        return "datasets/combined_carparts/data.yaml"
    else:
        return f"datasets/{dataset_name}/data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m-seg", help="yolov8n/s/m/l/x-seg or yolo11n/s/m/l/x-seg")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache", default="none", choices=["ram", "disk", "none"])  # none=fast when dataset is on local SSD
    parser.add_argument("--dataset", default="combined_carparts", help="Dataset name or path to dataset directory.")
    parser.add_argument("--project", default="runs_comparison")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to train for quick capacity testing.")
    args, _ = parser.parse_known_args()

    cache_arg = False if args.cache == "none" else args.cache

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] No GPU found -- training on CPU will be very slow.")

    data_yaml = make_yaml(args.dataset)

    # Use an ABSOLUTE project path. Ultralytics silently nests relative project
    # paths under its own default "runs/<task>/" directory in some versions,
    # which previously produced the confusing
    # "runs/segment/runs_comparison/..." path. Absolute paths avoid that entirely.
    project = os.path.abspath(args.project)
    # Use clean name -- never include dataset path (e.g. './dataset') in run_name
    # or YOLO saves its outputs INSIDE ./dataset/ and Azure ML re-uploads all 17k images!
    ds_label = Path(args.dataset).name if Path(args.dataset).is_dir() else args.dataset
    run_name = f"{args.model}_{ds_label}"

    # `model + .pt` auto-downloads the pretrained checkpoint from Ultralytics on first use
    model = YOLO(f"{args.model}.pt")

    # args.batch is a float so -1 (autobatch) and integer batch sizes both work
    # cleanly through argparse; Ultralytics accepts either.
    batch_arg = int(args.batch) if args.batch != -1 else -1

    # Setup custom logger and evaluator
    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        # Azure ML mounted path -- resolve COCO jsons directly from the mounted folder
        if (ds_path / "coco_train.json").exists():
            base = ds_path
        elif (ds_path / "combined_carparts" / "coco_train.json").exists():
            base = ds_path / "combined_carparts"
        else:
            base = ds_path
        train_json = str(base / "coco_train.json")
        val_json   = str(base / "coco_val.json")
        val_images = str(base / "images" / "val")
    elif args.dataset in ["carparts-seg", "custom_carparts", "combined_carparts"]:
        train_json = str(PROJECT_ROOT / "datasets" / args.dataset / "coco_train.json")
        val_json   = str(PROJECT_ROOT / "datasets" / args.dataset / "coco_val.json")
        val_images = str(PROJECT_ROOT / "datasets" / args.dataset / "images" / "val")
    else:
        train_json = str(PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "annotations" / "instances_train.json")
        val_json   = str(PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "annotations" / "instances_val.json")
        val_images = str(PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "images" / "val")


    with open(train_json) as f:
        class_names = [c["name"] for c in json.load(f)["categories"]]

    logger = UnifiedLogger(os.path.join(project, run_name), "yolo11m-seg")
    logger.print_dataset_health(train_json, val_json)
    evaluator = UnifiedEvaluator(val_json, val_images, class_names, os.path.join(project, run_name))

    def custom_eval_callback(trainer):
        epoch = trainer.epoch + 1
        run_unified = (epoch % 5 == 0) or (epoch == trainer.epochs)
        
        # Native YOLO stats
        tloss_raw = getattr(trainer, 'tloss', 0)
        if isinstance(tloss_raw, dict):
            train_loss_val = float(sum(v.item() if hasattr(v, 'item') else float(v) for v in tloss_raw.values()))
        elif hasattr(tloss_raw, 'mean'):
            train_loss_val = float(tloss_raw.mean())
        elif isinstance(tloss_raw, (list, tuple)):
            train_loss_val = float(sum(tloss_raw))
        else:
            try:
                train_loss_val = float(tloss_raw)
            except Exception:
                train_loss_val = 0.0

        train_stats = {
            "train_loss": train_loss_val,
            "val_loss": 0, # Not easily extracted from YOLO trainer
            "lr": trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0,
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "epoch_time_sec": trainer.epoch_time if hasattr(trainer, 'epoch_time') else 0,
            "images_sec": 0
        }

        val_metrics = None
        if run_unified:
            # Run inference manually for unified evaluator
            model_eval = YOLO(trainer.best) # Use current best weights
            results = model_eval.predict(source=val_images, conf=0.25, save=False, verbose=False)
            
            predictions_by_image = {}
            # Match image IDs using COCO val json
            with open(val_json) as f:
                coco_val = json.load(f)
            img_name_to_id = {img["file_name"]: img["id"] for img in coco_val["images"]}
            
            for r in results:
                img_name = Path(r.path).name
                img_id = img_name_to_id.get(img_name)
                if img_id is None: continue
                
                preds = []
                if r.boxes is not None and r.masks is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    scores = r.boxes.conf.cpu().numpy()
                    
                    # Original image shape handling
                    orig_h, orig_w = r.orig_shape
                    masks = r.masks.data.cpu().numpy() # shape (N, H, W)
                    
                    for box, cls, score, mask in zip(boxes, classes, scores, masks):
                        # Resize mask to original image size
                        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        preds.append({
                            "category_id": int(cls),
                            "score": float(score),
                            "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                            "segmentation": (mask_resized > 0.5)
                        })
                predictions_by_image[img_id] = preds
                
            val_metrics = evaluator.evaluate(epoch, predictions_by_image)
            
        should_stop, is_best = logger.log_epoch(epoch, trainer.epochs, train_stats, val_metrics)
        
        if run_unified and should_stop:
            print(f"\\n[!] Early stopping triggered by unified evaluator.")
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", custom_eval_callback)

    train_kwargs = {
        "data": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch_arg,
        "workers": args.workers,
        "cache": cache_arg,
        "device": device,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "plots": False,
        "val": True,
        "amp": True,    # re-enabled: numpy<2 pinned in requirements.txt fixes the cu118 compat crash
    }
    if args.max_batches:
        train_kwargs["time"] = 0.003 # ~10s max for fast micro-benchmark
        train_kwargs["val"] = False  # Skip per-epoch validation during capacity check

    model.train(**train_kwargs)
    logger.print_final_summary()

    best_weights_path = os.path.join(project, run_name, "weights", "best.pt")

    # Run final validation only during full training, skip during quick capacity testing
    if not args.max_batches:
        test_img_dir = ds_path / "images" / "test"
        eval_split = "test" if test_img_dir.exists() and any(test_img_dir.iterdir()) else "val"

        metrics = model.val(
            data=data_yaml,
            split=eval_split,
            project=project,
            name=f"{run_name}_{eval_split}_eval",
            exist_ok=True,
            plots=True,
        )
        print("\n[RESULTS] Test-set metrics:")
        print(metrics.results_dict)

    print(f"\n[OK] Trained weights saved at: {best_weights_path}")
    # Also write the path to a small text file so downstream scripts (or you)
    # don't have to guess/remember it.
    with open("last_yolo_weights_path.txt", "w") as f:
        f.write(best_weights_path + "\n")
    print("[OK] Path also saved to: last_yolo_weights_path.txt")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA Peak VRAM] {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()