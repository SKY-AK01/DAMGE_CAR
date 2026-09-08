"""
train_fastrcnn.py
-------------------
Fast R-CNN variant: torchvision FasterRCNN backbone with the RPN frozen
after an optional warmup phase, so region proposals are fixed and only the
ROI head is trained. This is what the original Fast R-CNN paper describes --
the RPN provides proposals, but gradient does not flow back through it during
the main training phase.

This is the third model in the YOLO / Mask R-CNN / Fast R-CNN comparison.
No exotic dependencies -- just torchvision (already installed with torch).

Arg convention: --dataset --epochs --batch --project
(matches train_yolo_seg.py to avoid repeating the --output_dir vs --project
bug that was fixed in orchestrator/src/main.rs in Task 1).

Usage:
    python train_fastrcnn.py --dataset carparts-seg --epochs 20 --batch 4 --project runs_comparison/fastrcnn

---------------------------------------------------------------------------
Fast R-CNN vs Faster R-CNN distinction:
---------------------------------------------------------------------------
- Faster R-CNN: RPN and ROI head trained jointly, end-to-end.
- Fast R-CNN: proposals come from an external/fixed source; here we emulate
  this by freezing the RPN (backbone + rpn parameters) after a short warmup
  so that the ROI classifier/regressor is the only part being actively tuned.
  This is a meaningful architectural comparison point: does jointly learning
  proposals (Faster R-CNN) beat a fixed-proposal approach (Fast R-CNN) on
  this dataset?

The distinction is controlled by --freeze_rpn_after (default: 5 epochs).
Set to 0 to never freeze (= standard Faster R-CNN behaviour).
Set to -1 to freeze from epoch 1 (pure Fast R-CNN from the start).
---------------------------------------------------------------------------
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# Anchor to project root regardless of where the script is launched from.
# scripts/training/train_fastrcnn.py -> .parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from scripts.evaluation.unified_logger import UnifiedLogger
from scripts.evaluation.unified_evaluator import UnifiedEvaluator


class CocoDetectionDataset(Dataset):
    """COCO-format detection dataset (no mask head -- bounding boxes only).
    Mirrors CocoMaskRCNNDataset but omits the mask tensor since Fast R-CNN
    only trains the classification and regression heads."""

    def __init__(self, images_dir, coco_json_path):
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        self.images_dir = Path(images_dir)
        self.images_by_id = {img["id"]: img for img in self.coco["images"]}
        self.anns_by_image = {}
        for ann in self.coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)
        # Only keep images that have at least one annotation
        self.image_ids = [i for i in self.images_by_id if i in self.anns_by_image]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_by_id[img_id]
        image = Image.open(self.images_dir / img_info["file_name"]).convert("RGB")

        anns = self.anns_by_image[img_id]
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"] + 1)  # 0 reserved for background

        image_tensor = torchvision.transforms.functional.to_tensor(image)

        if len(boxes) == 0:
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
            }
        else:
            target = {
                "boxes":    torch.as_tensor(boxes, dtype=torch.float32),
                "labels":   torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
            }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    """FasterRCNN_ResNet50_FPN_v2 with the box predictor replaced for our
    class count. The mask head is not added -- this is a detection-only model
    (Fast R-CNN / Faster R-CNN without segmentation)."""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def freeze_rpn(model):
    """Freeze backbone + RPN parameters so only the ROI head is trained.
    This transitions from Faster R-CNN (joint) to Fast R-CNN (fixed proposals)."""
    for name, param in model.named_parameters():
        if "backbone" in name or "rpn" in name:
            param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"[OK] RPN frozen -- {frozen} parameters frozen, ROI head only training.")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN training on COCO-format car-parts dataset.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name or path to dataset directory.")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--batch",   type=int,   default=4,
                        help="Batch size. Fast R-CNN (detection only, no mask head) can run "
                             "a larger batch than Mask R-CNN at the same VRAM budget.")
    parser.add_argument("--lr",      type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers.")
    parser.add_argument("--project", default="runs_comparison/fastrcnn",
                        help="Output directory. Matches --project convention used by train_yolo_seg.py.")
    parser.add_argument("--freeze_rpn_after", type=int, default=5,
                        help="Freeze backbone+RPN after this many epochs (Fast R-CNN mode). "
                             "Set 0 to never freeze (= Faster R-CNN). Set -1 to freeze from epoch 1.")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (on by default).")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to train for quick capacity testing.")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Run validation + COCO eval every N epochs (default: 5). "
                             "Use 1 to validate every epoch.")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'OK' if device == 'cuda' else 'WARNING'}] Using device: {device}")

    # -- Dataset paths --------------------------------------------------------
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
    else:  # dsmlr-carparts
        train_images = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "images" / "train"
        train_json   = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "annotations" / "instances_train.json"
        val_images   = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "images" / "val"
        val_json     = PROJECT_ROOT / "datasets" / "dsmlr-carparts-split" / "annotations" / "instances_val.json"

    with open(train_json) as f:
        coco_data = json.load(f)
    categories   = coco_data["categories"]
    num_classes  = len(categories) + 1   # +1 for background
    class_names  = [c["name"] for c in categories]

    print(f"[*] Dataset: {args.dataset} | Classes: {num_classes - 1} ({class_names[:5]}...)")
    print(f"[*] Loading pretrained FasterRCNN-ResNet50-FPN-v2 (detection head only)...")

    os.makedirs(args.project, exist_ok=True)

    logger    = UnifiedLogger(args.project, "fastrcnn")
    logger.print_dataset_health(str(train_json), str(val_json))
    evaluator = UnifiedEvaluator(str(val_json), str(val_images), class_names, args.project)

    model = build_model(num_classes).to(device)

    # If freeze_rpn_after == -1, freeze immediately before any training
    if args.freeze_rpn_after == -1:
        freeze_rpn(model)

    train_ds = CocoDetectionDataset(train_images, train_json)
    val_ds   = CocoDetectionDataset(val_images,   val_json)

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

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    use_amp = args.amp and device == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[OK] Mixed precision (AMP) enabled")

    for epoch in range(1, args.epochs + 1):
        # Transition to Fast R-CNN mode (freeze RPN) after warmup
        if args.freeze_rpn_after > 0 and epoch == args.freeze_rpn_after + 1:
            freeze_rpn(model)
            # Rebuild optimizer with only unfrozen params
            params    = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, (args.epochs - args.freeze_rpn_after) // 3), gamma=0.1)
            print(f"[Epoch {epoch}] Switched to Fast R-CNN mode (RPN frozen).")

        epoch_start = time.time()
        model.train()
        total_loss    = 0.0
        loss_comps    = {}
        num_images    = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            num_images += len(images)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                for k, v in loss_dict.items():
                    loss_comps[k] = loss_comps.get(k, 0) + v.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        lr_scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        epoch_time     = time.time() - epoch_start
        images_sec     = num_images / epoch_time

        # ── Validation: only run every val_interval epochs (or on the last epoch) ──
        run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)

        avg_val_loss = 0.0
        val_metrics  = None

        if run_val:
            print(f"\n[Epoch {epoch}] Running validation (every {args.val_interval} epochs)...")

            # Validation loss (train mode — Faster R-CNN only returns loss_dict in train)
            model.train()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images  = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_dict = model(images, targets)
                        val_loss += sum(loss_dict.values()).item()
            avg_val_loss = val_loss / len(val_loader)

            # Predictions for unified evaluator (eval mode)
            model.eval()
            predictions_by_image = {}
            with torch.no_grad():
                for images, targets in val_loader:
                    images  = [img.to(device) for img in images]
                    outputs = model(images)
                    for t, o in zip(targets, outputs):
                        img_id = t["image_id"].item()
                        preds  = []
                        boxes  = o["boxes"].cpu().numpy()
                        labels = o["labels"].cpu().numpy()
                        scores = o["scores"].cpu().numpy()
                        for box, label, score in zip(boxes, labels, scores):
                            preds.append({
                                "category_id": int(label) - 1,  # undo +1 background shift
                                "score":       float(score),
                                "bbox":        [float(box[0]), float(box[1]),
                                                float(box[2] - box[0]), float(box[3] - box[1])],
                                "segmentation": None,  # detection-only model
                            })
                        predictions_by_image[img_id] = preds

            val_metrics = evaluator.evaluate(epoch, predictions_by_image)
        else:
            print(f"\n[Epoch {epoch}] Skipping validation (next at epoch {epoch + (args.val_interval - epoch % args.val_interval)})")

        train_stats = {
            "train_loss":    avg_train_loss,
            "val_loss":      avg_val_loss,
            "lr":            optimizer.param_groups[0]["lr"],
            "gpu_mem_gb":    torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "epoch_time_sec": epoch_time,
            "images_sec":    images_sec,
        }
        for k, v in loss_comps.items():
            train_stats[f"train_{k}"] = v / len(train_loader)

        should_stop, is_best = logger.log_epoch(epoch, args.epochs, train_stats, val_metrics)

        torch.save(model.state_dict(), os.path.join(args.project, "last.pt"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.project, "best_model.pt"))

        if should_stop:
            print(f"\n[!] Early stopping triggered. No improvement for {logger.patience} epochs.")
            break

    logger.print_final_summary()

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA Peak VRAM] {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
