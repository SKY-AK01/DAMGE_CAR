"""
train_maskrcnn.py
--------------------
OPTIONAL bonus model: Mask R-CNN via torchvision.
This is deliberately the "easy" alternative -- unlike OneFormer/MaskDINO, it
has NO exotic dependencies (just torchvision, already installed with torch),
no CUDA-compile step, and no custom data-pipeline quirks. It's a classic
two-stage detector with no cross-part attention, so it serves as a second
"no relationship-awareness" data point alongside YOLO, while Mask2Former
remains the "relationship-aware" comparison point.

Usage:
    python train_maskrcnn.py --dataset carparts-seg --epochs 10

---------------------------------------------------------------------------
GPU UTILIZATION FIX (A10-12Q vGPU, 15.8G/216G RAM used, only 2-3/18 CPU
cores busy, GPU util oscillating 5%-94%; unlike train_yolo_seg.py and
train_mask2former.py, this script had NO fixes applied yet):
---------------------------------------------------------------------------
  1. DataLoaders had no num_workers/pin_memory/persistent_workers at all,
     so the CPU-bound cv2.fillPoly mask rasterization in __getitem__ ran
     single-threaded on the main process and fully blocked the GPU between
     steps. Added num_workers/pin_memory/persistent_workers to both loaders.
  2. Added mixed precision (torch.cuda.amp autocast + GradScaler) to train
     and val loops -- halves compute per step, which matters more than usual
     on a vGPU profile where every step must fit inside a scheduled
     time-slice (a flat 100% util isn't achievable on -Q profiles no matter
     what we change, since the hypervisor time-slices compute across
     tenants -- this just gets more done inside each window we get).
  3. Default --batch bumped 4 -> 8, given ~4GB of unused VRAM headroom
     observed at 8011MiB/12288MiB used during the YOLO run. Mask R-CNN's
     two-stage RPN+ROI pipeline is heavier per-sample than YOLO, so this is
     a conservative bump -- watch VRAM on the first run before raising further.
  4. Default --num_workers added (8), exposed as a flag.
"""

import argparse
import json
import os
import numpy as np
import torch
import cv2
from pathlib import Path

import time
import sys
# Anchor to project root regardless of where the script is launched from.
# scripts/training/train_maskrcnn.py -> .parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from scripts.evaluation.unified_logger import UnifiedLogger
from scripts.evaluation.unified_evaluator import UnifiedEvaluator
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

# -- Rust DataLoader (optional speedup) ----------------------------------------
try:
    from scripts.training.rust_dataloader_bridge import build_rust_loader, RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False
    build_rust_loader = None


class CocoMaskRCNNDataset(Dataset):
    def __init__(self, images_dir, coco_json_path):
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        self.images_dir = Path(images_dir)
        self.images_by_id = {img["id"]: img for img in self.coco["images"]}
        self.anns_by_image = {}
        for ann in self.coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)
        self.image_ids = [i for i in self.images_by_id.keys() if i in self.anns_by_image]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_by_id[img_id]
        image = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")
        w, h = img_info["width"], img_info["height"]

        anns = self.anns_by_image[img_id]
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"] + 1)  # 0 is reserved for background in Mask R-CNN

            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        image_tensor = torchvision.transforms.functional.to_tensor(image)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
            }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name or path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2,
                         help="Raised from 4 -> 8 after confirming ~4GB VRAM headroom. "
                              "Mask R-CNN is heavier per-sample than YOLO, so raise "
                              "further only after watching nvidia-smi on a first run.")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=8,
                         help="DataLoader workers. Was hardcoded to 0 (default) before, "
                              "which fully blocked the GPU during mask rasterization.")
    parser.add_argument("--amp", action="store_true", default=True,
                         help="Use automatic mixed precision. On by default; pass "
                              "--no-amp to disable if you hit NaN losses.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--output_dir", default="runs_comparison/maskrcnn")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to train for quick capacity testing.")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'OK' if device == 'cuda' else 'WARNING'}] Using device: {device}")

    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        if (ds_path / "coco_train.json").exists():
            base = ds_path
        elif (ds_path / "combined_carparts" / "coco_train.json").exists():
            base = ds_path / "combined_carparts"
        else:
            base = ds_path
        train_images = base / "images" / "train"
        train_json   = base / "coco_train.json"
        val_images   = base / "images" / "val"
        val_json     = base / "coco_val.json"
    elif args.dataset in ["carparts-seg", "custom_carparts", "combined_carparts"]:
        train_images = PROJECT_ROOT / "datasets" / args.dataset / "images" / "train"
        train_json   = PROJECT_ROOT / "datasets" / args.dataset / "coco_train.json"
        val_images   = PROJECT_ROOT / "datasets" / args.dataset / "images" / "val"
        val_json     = PROJECT_ROOT / "datasets" / args.dataset / "coco_val.json"
    else:
        train_images = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "images" / "train"
        train_json   = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "annotations" / "instances_train.json"
        val_images   = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "images" / "val"
        val_json     = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "annotations" / "instances_val.json"


    with open(train_json) as f:
        coco_data = json.load(f)
        categories = coco_data["categories"]
        num_classes = len(categories) + 1  # +1 for background
        class_names = [c["name"] for c in categories]

    print(f"[*] Loading pretrained Mask R-CNN (auto-downloads from torchvision) ...")
    
    logger = UnifiedLogger(args.output_dir, "maskrcnn")
    logger.print_dataset_health(train_json, val_json)
    
    evaluator = UnifiedEvaluator(val_json, val_images, class_names, args.output_dir)

    model = build_model(num_classes).to(device)

    train_ds = CocoMaskRCNNDataset(train_images, train_json)
    val_ds = CocoMaskRCNNDataset(val_images, val_json)

    # -- DataLoader: use Rust-parallel loader when available ------------------
    # The Rust DataLoader (PyO3 + Rayon) decodes images in parallel across all
    # CPU cores without the Python GIL, keeping the GPU consistently fed.
    # Mask R-CNN requires torchvision-style dict targets, so we use the Rust
    # loader for image decode only (Python Dataset wraps the Rust-decoded batch).
    if RUST_AVAILABLE:
        print("[OK] Using Rust-parallel DataLoader (PyO3 + Rayon) for image decode")
        # num_workers=0: Rust already parallelises inside; extra workers would
        # cause redundant process forking with no benefit.
        train_loader = build_rust_loader(
            json_path=str(train_json),
            images_dir=str(train_images),
            img_size=640,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            augment=True,
            format="maskrcnn",
        ) or DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = build_rust_loader(
            json_path=str(val_json),
            images_dir=str(val_images),
            img_size=640,
            batch_size=args.batch,
            shuffle=False,
            num_workers=0,
            augment=False,
            format="maskrcnn",
        ) or DataLoader(
            val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
    else:
        print("[INFO] Rust DataLoader not found -- using Python DataLoader. "
              "Run `python build_rust_dataloader.py` to enable Rust-parallel decode.")
        # num_workers/pin_memory/persistent_workers: without these, the
        # CPU-bound cv2.fillPoly mask rasterization in __getitem__ ran
        # single-threaded and blocked the GPU between every step.
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # Mixed precision: halves compute per step on Ampere GPUs (A10 included).
    # Matters more than usual on a vGPU profile since every step needs to
    # fit inside a scheduled compute time-slice.
    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[OK] Mixed precision (AMP) enabled")

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        num_images = 0
        loss_components = {}
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            num_images += len(images)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                
                # track individual losses
                for k, v in loss_dict.items():
                    loss_components[k] = loss_components.get(k, 0) + v.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        lr_scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        
        epoch_time = time.time() - epoch_start_time
        images_sec = num_images / epoch_time

        # Validation (skip when running max_batches micro-benchmark for capacity check)
        val_loss = 0
        val_metrics = None
        if not args.max_batches:
            # Mask R-CNN's loss dict is only available in train() mode
            model.train()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_dict = model(images, targets)
                        val_loss += sum(loss_dict.values()).item()
            avg_val_loss = val_loss / len(val_loader)
            
            # Predictions for unified evaluator
            model.eval()
            predictions_by_image = {}
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    
                    # Fix: Process images one-by-one to avoid huge memory spike 
                    # during torchvision's paste_masks_in_image post-processing.
                    outputs = []
                    for img in images:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            out = model([img])[0]
                        # Move to CPU immediately to free GPU memory
                        outputs.append({k: v.cpu() for k, v in out.items()})

                    for t, o in zip(targets, outputs):
                        img_id = t["image_id"].item()
                        img_info = val_ds.images_by_id[img_id]
                        orig_w, orig_h = img_info["width"], img_info["height"]
                        
                        preds = []
                        boxes = o["boxes"].numpy()
                        labels = o["labels"].numpy()
                        scores = o["scores"].numpy()
                        masks = o["masks"].numpy()
                        for box, label, score, mask in zip(boxes, labels, scores, masks):
                            pred_h, pred_w = mask[0].shape
                            scale_x = orig_w / float(pred_w)
                            scale_y = orig_h / float(pred_h)
                            
                            m = (mask[0] > 0.5).astype(np.uint8)
                            if pred_h != orig_h or pred_w != orig_w:
                                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                            
                            x1 = float(box[0]) * scale_x
                            y1 = float(box[1]) * scale_y
                            w_box = float(box[2]-box[0]) * scale_x
                            h_box = float(box[3]-box[1]) * scale_y

                            preds.append({
                                "category_id": int(label) - 1, # undo +1 background
                                "score": float(score),
                                "bbox": [x1, y1, w_box, h_box],
                                "segmentation": m.astype(bool)
                            })
                        predictions_by_image[img_id] = preds
                        
            val_metrics = evaluator.evaluate(epoch, predictions_by_image)
        else:
            avg_val_loss = 0.0
        
        train_stats = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "epoch_time_sec": epoch_time,
            "images_sec": images_sec
        }
        for k, v in loss_components.items():
            train_stats[f"train_{k}"] = v / len(train_loader)
            
        should_stop, is_best = logger.log_epoch(epoch, args.epochs, train_stats, val_metrics)
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pt"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            
        if should_stop:
            print(f"\n[!] Early stopping triggered. No improvement for {logger.patience} epochs.")
            break

    logger.print_final_summary()

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA Peak VRAM] {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()