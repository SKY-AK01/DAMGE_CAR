#!/usr/bin/env python3
"""
train_mask2former.py -- Mask2Former (Swin Transformer) Instance Segmentation Training.

Fine-tunes HuggingFace Mask2Former (facebook/mask2former-swin-tiny-coco-instance)
on the COCO-formatted car-parts dataset.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.unified_evaluator import UnifiedEvaluator
from scripts.evaluation.unified_logger import UnifiedLogger

try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
except ImportError:
    print("[ERROR] HuggingFace transformers is required for Mask2Former training. Install with: pip install transformers")
    sys.exit(1)

# -- Rust DataLoader (optional speedup) ----------------------------------------
try:
    from scripts.training.rust_dataloader_bridge import build_rust_loader, RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False
    build_rust_loader = None


class COCOMask2FormerDataset(Dataset):
    def __init__(self, images_dir, json_file, processor):
        self.images_dir = Path(images_dir)
        with open(json_file, "r") as f:
            self.coco_data = json.load(f)
        
        self.processor = processor
        self.images_by_id = {img["id"]: img for img in self.coco_data.get("images", [])}
        self.image_ids = list(self.images_by_id.keys())
        
        self.anns_by_image = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.anns_by_image:
                self.anns_by_image[img_id] = []
            self.anns_by_image[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_by_id[img_id]
        img_path = self.images_dir / img_info["file_name"]
        
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        anns = self.anns_by_image.get(img_id, [])
        instance_masks = []
        class_labels = []
        
        from pycocotools import mask as mask_utils
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
            cat_id = ann["category_id"]
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                rle = mask_utils.merge(rles)
            elif isinstance(seg, dict):
                rle = seg
            else:
                continue
            
            m = mask_utils.decode(rle)
            if m.sum() > 0:
                instance_masks.append(m)
                class_labels.append(cat_id)

        if len(instance_masks) == 0:
            instance_masks = [np.zeros((h, w), dtype=np.uint8)]
            class_labels = [0]
            
        inputs = self.processor(
            images=image,
            segmentation_maps=instance_masks,
            class_labels=class_labels,
            return_tensors="pt"
        )
        # Squeeze batch dimension added by processor
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["image_id"] = img_id
        return inputs


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch]) if "pixel_mask" in batch[0] else None
    
    mask_labels = [b["mask_labels"] for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    
    res = {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "image_ids": image_ids
    }
    if pixel_mask is not None:
        res["pixel_mask"] = pixel_mask
    return res


def main():
    parser = argparse.ArgumentParser(description="Mask2Former Instance Segmentation Training")
    parser.add_argument("--dataset", required=True, help="Dataset name or path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="runs_comparison/mask2former")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to train for quick capacity testing.")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Run validation + COCO eval every N epochs (default: 5). "
                             "Use 1 to validate every epoch.")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'OK' if device == 'cuda' else 'WARNING'}] Using device: {device}")

    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        base = ds_path if (ds_path / "coco_train.json").exists() else ds_path / "combined_carparts"
        train_images = base / "images" / "train"
        train_json   = base / "coco_train.json"
        val_images   = base / "images" / "val"
        val_json     = base / "coco_val.json"
    else:
        train_images = PROJECT_ROOT / "datasets" / args.dataset / "images" / "train"
        train_json   = PROJECT_ROOT / "datasets" / args.dataset / "coco_train.json"
        val_images   = PROJECT_ROOT / "datasets" / args.dataset / "images" / "val"
        val_json     = PROJECT_ROOT / "datasets" / args.dataset / "coco_val.json"

    with open(train_json, "r") as f:
        coco_train_data = json.load(f)
    num_classes = len(coco_train_data["categories"])
    class_names = [c["name"] for c in coco_train_data["categories"]]

    model_id = "facebook/mask2former-swin-tiny-coco-instance"
    print(f"[*] Loading pretrained Mask2Former ({model_id})...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    train_ds = COCOMask2FormerDataset(train_images, train_json, processor)
    val_ds   = COCOMask2FormerDataset(val_images, val_json, processor)

    # -- DataLoader: use Rust-parallel loader when available ------------------
    if RUST_AVAILABLE:
        print("[OK] Using Rust-parallel DataLoader (PyO3 + Rayon) for image decode")
        train_loader = build_rust_loader(
            json_path=str(train_json),
            images_dir=str(train_images),
            img_size=640,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            augment=True,
            format="mask2former",
        ) or DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader = build_rust_loader(
            json_path=str(val_json),
            images_dir=str(val_images),
            img_size=640,
            batch_size=args.batch,
            shuffle=False,
            num_workers=0,
            augment=False,
            format="mask2former",
        ) or DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers)
    else:
        print("[INFO] Rust DataLoader not found -- using Python DataLoader.")
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                  collate_fn=collate_fn, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    run_name = f"mask2former_{ds_path.name if ds_path.is_dir() else args.dataset}"
    logger = UnifiedLogger(os.path.join(args.output_dir, run_name), "mask2former")
    logger.print_dataset_health(str(train_json), str(val_json))
    evaluator = UnifiedEvaluator(str(val_json), str(val_images), class_names, os.path.join(args.output_dir, run_name))

    print(f"[*] Starting Mask2Former Training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_train_loss = running_loss / max(len(train_loader), 1)

        # Evaluation
        val_metrics = None
        run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        if run_val:
            print(f"\n[Epoch {epoch}] Running validation (every {args.val_interval} epochs)...")
            model.eval()
            predictions_by_image = {}
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    outputs = model(pixel_values=pixel_values)
                    
                    results = processor.post_process_instance_segmentation(
                        outputs, target_sizes=[(640, 640)] * len(batch["image_ids"])
                    )
                    
                    for img_id, res in zip(batch["image_ids"], results):
                        preds = []
                        masks = res["segmentation"].cpu().numpy()
                        labels = res["segments_info"]
                        for info in labels:
                            cid = info["label_id"]
                            score = info["score"]
                            m = (masks == info["id"])
                            if not m.any():
                                continue
                            ys, xs = np.where(m)
                            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                            preds.append({
                                "category_id": cid,
                                "score": float(score),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "segmentation": m
                            })
                        predictions_by_image[img_id] = preds

            val_metrics = evaluator.evaluate(epoch, predictions_by_image)
        else:
            print(f"\n[Epoch {epoch}] Skipping validation (next at epoch {epoch + (args.val_interval - epoch % args.val_interval)})")

        train_stats = {
            "train_loss": avg_train_loss,
            "val_loss": 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "epoch_time_sec": epoch_time,
            "images_sec": len(train_ds) / max(epoch_time, 1e-3)
        }

        should_stop, is_best = logger.log_epoch(epoch, args.epochs, train_stats, val_metrics)
        if is_best:
            os.makedirs(os.path.join(args.output_dir, run_name, "weights"), exist_ok=True)
            model.save_pretrained(os.path.join(args.output_dir, run_name, "weights", "best"))

        if should_stop:
            print("[!] Early stopping triggered.")
            break

    logger.print_final_summary()
    print(f"[OK] Mask2Former training completed!")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA Peak VRAM] {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
