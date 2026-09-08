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
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Run validation + COCO eval every N epochs (default: 5). "
                             "Use 1 to validate every epoch (original behaviour).")
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
        loss_components = {}
        num_images = 0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            num_images += len(images)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                # Cast all losses to float32 before summing to avoid the
                # float16 NaN/overflow in the mask head's BCE loss that occurs
                # with torch 2.0.1 + torchvision 0.15.2 under autocast.
                # binary_cross_entropy_with_logits on fp16 logits with large
                # magnitudes produces ±inf → NaN gradients → device-side assert
                # in fastrcnn_loss on the next forward pass.
                # Casting here is zero-cost (scalars) and prevents the overflow.
                loss = sum(v.float() for v in loss_dict.values())

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

        # ── Validation: only run every val_interval epochs (or on the last epoch) ──
        run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)

        avg_val_loss = 0.0
        val_metrics = None

        if run_val:
            print(f"\n[Epoch {epoch}] Running validation (every {args.val_interval} epochs)...")

            # Mask R-CNN's loss dict is only available in train() mode
            model.train()
            val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_dict = model(images, targets)
                        val_loss += sum(v.float() for v in loss_dict.values()).item()
            avg_val_loss = val_loss / len(val_loader)

            # Predictions for unified evaluator
            model.eval()
            predictions_by_image = {}
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    for t, o in zip(targets, outputs):
                        img_id = t["image_id"].item()
                        preds = []
                        boxes = o["boxes"].cpu().numpy()
                        labels = o["labels"].cpu().numpy()
                        scores = o["scores"].cpu().numpy()
                        masks = o["masks"].cpu().numpy()
                        for box, label, score, mask in zip(boxes, labels, scores, masks):
                            preds.append({
                                "category_id": int(label) - 1,  # undo +1 background
                                "score": float(score),
                                "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                                "segmentation": (mask[0] > 0.5)
                            })
                        predictions_by_image[img_id] = preds

            val_metrics = evaluator.evaluate(epoch, predictions_by_image)
        else:
            print(f"\n[Epoch {epoch}] Skipping validation (next at epoch {epoch + (args.val_interval - epoch % args.val_interval)})")
        
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