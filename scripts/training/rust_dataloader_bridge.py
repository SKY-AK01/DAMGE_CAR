#!/usr/bin/env python3
"""
rust_dataloader_bridge.py
=========================
Python-side bridge that wraps the Rust `CarPartsDataset` (rust_dataloader.pyd / .so)
into a proper high-performance, asynchronous PyTorch DataLoader.

The Rust extension handles:
  - Parallel JPEG decode via Rayon (across all CPU cores, releasing the Python GIL)
  - Polygon mask rasterisation (imageproc)
  - Random H-flip and brightness jitter augmentation
  - ImageNet normalisation

This bridge handles:
  - Asynchronous double-buffered prefetching via a background worker thread
  - Concurrent CPU decode and GPU training execution (eliminating GPU idle stalls)
  - Automatic bounding box derivation (masks_to_boxes_safe) for torchvision detection models
  - Zero-copy tensor conversion and pinned host memory buffers
  - Model-specific output formatting for torchvision Mask R-CNN and Hugging Face Mask2Former

Usage in training scripts:
--------------------------
    from scripts.training.rust_dataloader_bridge import build_rust_loader, RUST_AVAILABLE

    if RUST_AVAILABLE:
        train_loader = build_rust_loader(
            json_path=str(train_json),
            images_dir=str(train_images),
            img_size=640,
            batch_size=args.batch,
            shuffle=True,
            augment=True,
            format="maskrcnn",   # or "mask2former" / "auto"
            prefetch=True,
        )
"""

import sys
import math
import queue
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# Ensure UTF-8 output on Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -- Try importing the compiled Rust extension ----------------------------------
try:
    import rust_dataloader as _rust  # noqa: E402
    RUST_AVAILABLE = hasattr(_rust, "CarPartsDataset")
except (ImportError, AttributeError):
    RUST_AVAILABLE = False

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MAX_INSTANCES_PER_IMAGE = 64   # Pad mask tensor to this size for batching

# -----------------------------------------------------------------------------
# Bounding Box Calculation Helper
# -----------------------------------------------------------------------------

def masks_to_boxes_safe(masks: torch.Tensor, width: int = 640, height: int = 640) -> torch.Tensor:
    """
    Computes bounding boxes [N, 4] in (x1, y1, x2, y2) format from instance masks [N, H, W].
    Guarantees x2 > x1 and y2 > y1 for every box to prevent torchvision Mask R-CNN
    degenerate box errors.
    """
    if masks.numel() == 0 or masks.shape[0] == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    boxes = None
    try:
        from torchvision.ops import masks_to_boxes
        # masks_to_boxes expects boolean or binary tensor (N, H, W)
        boxes = masks_to_boxes(masks > 0.5)
    except Exception:
        pass

    if boxes is None:
        boxes_list = []
        for m in masks:
            pos = (m > 0.5).nonzero(as_tuple=False)
            if pos.shape[0] == 0:
                boxes_list.append([0.0, 0.0, 1.0, 1.0])
            else:
                y_min = float(pos[:, 0].min())
                y_max = float(pos[:, 0].max())
                x_min = float(pos[:, 1].min())
                x_max = float(pos[:, 1].max())
                boxes_list.append([x_min, y_min, x_max + 1.0, y_max + 1.0])
        boxes = torch.tensor(boxes_list, dtype=torch.float32)

    # Sanitize degenerate boxes (where width or height <= 0)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x2 = torch.max(x2, x1 + 1.0)
    y2 = torch.max(y2, y1 + 1.0)
    x1 = x1.clamp(0.0, float(width))
    y1 = y1.clamp(0.0, float(height))
    x2 = x2.clamp(0.0, float(width))
    y2 = y2.clamp(0.0, float(height))
    # Re-ensure positive area after clamping
    x2 = torch.max(x2, x1 + 1.0)
    y2 = torch.max(y2, y1 + 1.0)

    return torch.stack([x1, y1, x2, y2], dim=-1).to(dtype=torch.float32)


# -----------------------------------------------------------------------------
# Dual-Compatible Batch Container
# -----------------------------------------------------------------------------

class CompatibleBatch(dict):
    """
    A smart dictionary container that supports:
    1. Direct tuple unpacking: `images, targets = batch` (for torchvision Mask R-CNN)
    2. Dict key access: `batch["pixel_values"]`, `batch["mask_labels"]`, `batch["class_labels"]` (for Mask2Former)
    3. Legacy key access: `batch["masks"]`, `batch["labels"]`, `batch["image_ids"]`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tuple_repr = None

    def set_tuple_repr(self, tuple_repr: Tuple[Any, Any]):
        self._tuple_repr = tuple_repr

    def __iter__(self):
        # If iterated as a sequence (e.g. `images, targets = batch`), yield the 2-tuple items
        if self._tuple_repr is not None:
            return iter(self._tuple_repr)
        return super().__iter__()

    def __getitem__(self, key):
        if key == "mask_labels" and "mask_labels" not in self and "masks" in self:
            return self["masks"]
        if key == "class_labels" and "class_labels" not in self and "labels" in self:
            return self["labels"]
        if key == "image_id" and "image_id" not in self and "image_ids" in self:
            return self["image_ids"]
        return super().__getitem__(key)


# -----------------------------------------------------------------------------
# Asynchronous Double-Buffered Prefetch Engine
# -----------------------------------------------------------------------------

_DONE_SENTINEL = object()

class _ExceptionWrapper:
    def __init__(self, exc):
        self.exc = exc


class _PrefetchIter:
    def __init__(self, loader: "RustPrefetchDataLoader"):
        self.loader = loader
        self.queue = queue.Queue(maxsize=loader.prefetch_factor)
        self.stop_event = threading.Event()
        self.batches_yielded = 0
        self.total_batches = len(loader)

        indices = list(range(loader._num_samples))
        if loader.shuffle:
            import random
            random.shuffle(indices)

        self.batch_indices = [
            indices[i : i + loader.batch_size]
            for i in range(0, len(indices), loader.batch_size)
        ]

        self.worker = threading.Thread(
            target=self._worker_loop,
            name="RustDataLoaderPrefetchWorker",
            daemon=True
        )
        self.worker.start()

    def _worker_loop(self):
        try:
            for b_idx in self.batch_indices:
                if self.stop_event.is_set():
                    break
                # Rust parallel decode via Rayon (releases Python GIL inside `get_batch`)
                raw_items = self.loader._ds.get_batch(b_idx)
                batch = self.loader._collate_batch(raw_items)

                while not self.stop_event.is_set():
                    try:
                        self.queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.queue.put(_ExceptionWrapper(e))
        finally:
            while not self.stop_event.is_set():
                try:
                    self.queue.put(_DONE_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_yielded >= self.total_batches:
            self._cleanup()
            raise StopIteration

        item = self.queue.get()
        if item is _DONE_SENTINEL:
            self._cleanup()
            raise StopIteration
        if isinstance(item, _ExceptionWrapper):
            self._cleanup()
            raise item.exc

        self.batches_yielded += 1
        return item

    def _cleanup(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.worker.is_alive():
            self.worker.join(timeout=0.5)

    def __del__(self):
        self._cleanup()


class RustPrefetchDataLoader:
    """
    Asynchronous, double-buffered PyTorch DataLoader backed by the Rust native extension.
    Runs Rayon multi-threaded JPEG decode + rasterization in a background thread,
    pre-buffering batches so the GPU never sits idle waiting on data.
    """
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        img_size: int = 640,
        batch_size: int = 8,
        shuffle: bool = True,
        augment: bool = False,
        format: str = "auto",
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        max_instances: int = MAX_INSTANCES_PER_IMAGE,
    ):
        if not RUST_AVAILABLE:
            raise ImportError("rust_dataloader native extension not found.")
        self._ds = _rust.CarPartsDataset(json_path, images_dir, img_size, augment)
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.format = format.lower()
        self.prefetch_factor = max(1, prefetch_factor)
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.max_instances = max_instances
        self.num_classes = self._ds.num_classes()
        self.class_names = self._ds.class_names()
        self._num_samples = len(self._ds)
        self._num_batches = math.ceil(self._num_samples / batch_size) if self._num_samples > 0 else 0

        # Normalization constants (ImageNet stats)
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        return _PrefetchIter(self)

    def _collate_batch(self, raw_items: list):
        hw = self.img_size * self.img_size
        images_list = []
        masks_list = []
        labels_list = []
        ids_list = []
        n_inst_list = []

        for item in raw_items:
            if item is None:
                continue
            img_bytes, mask_bytes, lbl_list, img_id, n_inst = item

            # 1. Image tensor [3, H, W]
            img_np = np.frombuffer(img_bytes, dtype=np.float32).copy()
            img_t = torch.from_numpy(img_np).reshape(3, self.img_size, self.img_size)

            # 2. Masks tensor [N, H, W]
            if n_inst > 0:
                n = min(n_inst, self.max_instances)
                mask_np = np.frombuffer(mask_bytes, dtype=np.float32).copy()
                mask_np = mask_np[:n * hw]
                mask_t = torch.from_numpy(mask_np).reshape(n, self.img_size, self.img_size)
            else:
                n = 0
                mask_t = torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)

            # 3. Labels tensor [N]
            if lbl_list and n > 0:
                lbl_t = torch.tensor(lbl_list[:n], dtype=torch.long)
            else:
                lbl_t = torch.zeros((0,), dtype=torch.long)

            if self.pin_memory:
                img_t = img_t.pin_memory()
                mask_t = mask_t.pin_memory()
                lbl_t = lbl_t.pin_memory()

            images_list.append(img_t)
            masks_list.append(mask_t)
            labels_list.append(lbl_t)
            ids_list.append(int(img_id))
            n_inst_list.append(n)

        # Build torchvision Mask R-CNN representation: (images, targets)
        rcnn_images = []
        rcnn_targets = []
        for img_t, mask_t, lbl_t, img_id, n in zip(images_list, masks_list, labels_list, ids_list, n_inst_list):
            # Unnormalize from ImageNet stats to [0, 1] so torchvision's internal GeneralizedRCNNTransform operates cleanly
            unnorm_img = (img_t * self._std + self._mean).clamp(0.0, 1.0)
            if self.pin_memory:
                unnorm_img = unnorm_img.pin_memory()
            rcnn_images.append(unnorm_img)

            if n > 0:
                boxes = masks_to_boxes_safe(mask_t, width=self.img_size, height=self.img_size)
                # +1 offset for background class 0 in torchvision Mask R-CNN
                target_dict = {
                    "boxes": boxes,
                    "labels": lbl_t + 1,
                    "masks": (mask_t > 0.5).to(torch.uint8),
                    "image_id": torch.tensor([img_id], dtype=torch.int64),
                }
            else:
                target_dict = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8),
                    "image_id": torch.tensor([img_id], dtype=torch.int64),
                }
            if self.pin_memory:
                target_dict = {k: v.pin_memory() if hasattr(v, "pin_memory") else v for k, v in target_dict.items()}
            rcnn_targets.append(target_dict)

        if self.format == "maskrcnn":
            return (rcnn_images, rcnn_targets)

        # Build Mask2Former representation:
        m2f_masks = []
        m2f_labels = []
        for mask_t, lbl_t, n in zip(masks_list, labels_list, n_inst_list):
            if n > 0:
                m2f_masks.append((mask_t > 0.5).to(torch.float32))
                m2f_labels.append(lbl_t.to(torch.int64))
            else:
                m2f_masks.append(torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32))
                m2f_labels.append(torch.zeros((1,), dtype=torch.int64))

        stacked_pixels = torch.stack(images_list) if images_list else torch.zeros((0, 3, self.img_size, self.img_size))

        if self.format == "mask2former":
            return {
                "pixel_values": stacked_pixels,
                "mask_labels": m2f_masks,
                "class_labels": m2f_labels,
                "image_ids": ids_list,
            }

        # Format is "auto" or "raw": return CompatibleBatch supporting both
        compat = CompatibleBatch({
            "pixel_values": stacked_pixels,
            "mask_labels": m2f_masks,
            "class_labels": m2f_labels,
            "masks": m2f_masks,
            "labels": m2f_labels,
            "image_ids": ids_list,
            "image_id": ids_list,
            "num_instances": n_inst_list,
        })
        compat.set_tuple_repr((rcnn_images, rcnn_targets))
        return compat


# -----------------------------------------------------------------------------
# Legacy PyTorch Dataset wrappers (maintained for compatibility)
# -----------------------------------------------------------------------------

class RustCarPartsDataset(Dataset):
    """
    Wraps `rust_dataloader.CarPartsDataset` so it works with standard `torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        img_size: int = 640,
        augment: bool = False,
        max_instances: int = MAX_INSTANCES_PER_IMAGE,
    ):
        if not RUST_AVAILABLE:
            raise ImportError("rust_dataloader native extension not found.")
        self._ds = _rust.CarPartsDataset(json_path, images_dir, img_size, augment)
        self.img_size = img_size
        self.max_instances = max_instances
        self.num_classes = self._ds.num_classes()
        self.class_names = self._ds.class_names()

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img_bytes, mask_bytes, labels, image_id, n_inst = self._ds.get_item(idx)
        img_np = np.frombuffer(img_bytes, dtype=np.float32).copy()
        img_t = torch.from_numpy(img_np).reshape(3, self.img_size, self.img_size)

        hw = self.img_size * self.img_size
        mask_tensor = torch.zeros(self.max_instances, self.img_size, self.img_size, dtype=torch.float32)
        if n_inst > 0:
            n = min(n_inst, self.max_instances)
            mask_np = np.frombuffer(mask_bytes, dtype=np.float32).copy()
            mask_np = mask_np[:n * hw]
            mask_tensor[:n] = torch.from_numpy(mask_np).reshape(n, self.img_size, self.img_size)

        label_tensor = torch.full((self.max_instances,), -1, dtype=torch.long)
        if labels:
            n = min(len(labels), self.max_instances)
            label_tensor[:n] = torch.tensor(labels[:n], dtype=torch.long)

        num_valid = min(n_inst, self.max_instances)
        return {
            "pixel_values": img_t,
            "masks": mask_tensor,
            "mask_labels": mask_tensor[:num_valid],
            "labels": label_tensor,
            "class_labels": label_tensor[:num_valid],
            "num_instances": num_valid,
            "image_id": image_id,
        }


class RustBatchDataset(Dataset):
    """
    Legacy batch-fetch wrapper.
    """
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        img_size: int = 640,
        augment: bool = False,
        batch_size: int = 8,
        max_instances: int = MAX_INSTANCES_PER_IMAGE,
    ):
        if not RUST_AVAILABLE:
            raise ImportError("rust_dataloader native extension not found.")
        self._ds = _rust.CarPartsDataset(json_path, images_dir, img_size, augment)
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.num_classes = self._ds.num_classes()
        self.class_names = self._ds.class_names()
        self._n = len(self._ds)
        self._n_batches = math.ceil(self._n / batch_size)

    def __len__(self) -> int:
        return self._n_batches

    def __getitem__(self, batch_idx: int):
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self._n)
        indices = list(range(start, end))
        items = self._ds.get_batch(indices)

        imgs, masks, labels, ids, n_insts = [], [], [], [], []
        hw = self.img_size * self.img_size

        for item in items:
            if item is None:
                continue
            img_bytes, mask_bytes, lbl_list, img_id, n_inst = item
            img_np = np.frombuffer(img_bytes, dtype=np.float32).copy()
            img_t = torch.from_numpy(img_np).reshape(3, self.img_size, self.img_size)

            mask_tensor = torch.zeros(self.max_instances, self.img_size, self.img_size, dtype=torch.float32)
            if n_inst > 0:
                n = min(n_inst, self.max_instances)
                mn = np.frombuffer(mask_bytes, dtype=np.float32).copy()
                mn = mn[:n * hw]
                mask_tensor[:n] = torch.from_numpy(mn).reshape(n, self.img_size, self.img_size)

            lbl_tensor = torch.full((self.max_instances,), -1, dtype=torch.long)
            if lbl_list:
                n = min(len(lbl_list), self.max_instances)
                lbl_tensor[:n] = torch.tensor(lbl_list[:n], dtype=torch.long)

            imgs.append(img_t)
            masks.append(mask_tensor)
            labels.append(lbl_tensor)
            ids.append(img_id)
            n_insts.append(min(n_inst, self.max_instances))

        return {
            "pixel_values": torch.stack(imgs),
            "masks": torch.stack(masks),
            "labels": torch.stack(labels),
            "num_instances": n_insts,
            "image_ids": ids,
        }


# -----------------------------------------------------------------------------
# Convenience builder
# -----------------------------------------------------------------------------

def build_rust_loader(
    json_path: str,
    images_dir: str,
    img_size: int = 640,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = False,
    max_instances: int = MAX_INSTANCES_PER_IMAGE,
    use_batch_mode: bool = True,
    format: str = "auto",
    prefetch: bool = True,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> Optional[Any]:
    """
    Build an asynchronous, high-performance DataLoader backed by the Rust native extension.

    Returns None if rust_dataloader is not compiled (caller should fallback to Python).

    Args:
        json_path       -- Path to COCO-format annotation JSON.
        images_dir      -- Directory with image files.
        img_size        -- Resize images to img_size × img_size.
        batch_size      -- Number of samples per batch.
        shuffle         -- Shuffle samples each epoch.
        num_workers     -- Unused when prefetch=True (Rust manages its own Rayon worker pool).
        augment         -- Enable Rust-side random H-flip + brightness jitter.
        max_instances   -- Max polygon instances per image (padding limit).
        use_batch_mode  -- Legacy batch mode flag.
        format          -- Target output format:
                           'maskrcnn'   -> Tuple (images: List[Tensor], targets: List[Dict[boxes, labels, masks, image_id]])
                           'mask2former'-> Dict {pixel_values, mask_labels, class_labels, image_ids}
                           'auto'       -> CompatibleBatch (supports both tuple unpacking and dict keys)
        prefetch        -- Enable asynchronous background double-buffered queue (eliminates GPU idle).
        prefetch_factor -- Number of pre-decoded batches to hold in memory ahead of the GPU (default: 2).
        pin_memory      -- Pin memory for fast DMA transfers to GPU.
    """
    if not RUST_AVAILABLE:
        return None

    try:
        if prefetch:
            return RustPrefetchDataLoader(
                json_path=json_path,
                images_dir=images_dir,
                img_size=img_size,
                batch_size=batch_size,
                shuffle=shuffle,
                augment=augment,
                format=format,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                max_instances=max_instances,
            )
        elif use_batch_mode:
            ds = RustBatchDataset(json_path, images_dir, img_size, augment, batch_size, max_instances)
            return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                              collate_fn=lambda x: x[0])
        else:
            ds = RustCarPartsDataset(json_path, images_dir, img_size, augment, max_instances)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    except Exception as e:
        print(f"[INFO] Rust DataLoader initialization failed ({e}) -- using Python DataLoader.")
        return None
