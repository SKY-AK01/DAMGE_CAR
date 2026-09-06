"""
register_carparts_dataset.py
-------------------------------
Registers both carparts datasets (COCO-format) with Detectron2's dataset
catalog, using the naming convention MaskDINO's train_net.py expects:
    <dataset_name>_train, <dataset_name>_val, <dataset_name>_test
"""

from detectron2.data.datasets import register_coco_instances

DATASETS = {
    "carparts-seg": {
        "train": ("datasets/carparts-seg/images/train", "datasets/carparts-seg/coco_train.json"),
        "val":   ("datasets/carparts-seg/images/val",   "datasets/carparts-seg/coco_val.json"),
        "test":  ("datasets/carparts-seg/images/test",  "datasets/carparts-seg/coco_test.json"),
    },
    "dsmlr-carparts": {
        "train": ("datasets/dsmlr-carparts-split/images/train",
                  "datasets/dsmlr-carparts-split/annotations/instances_train.json"),
        "val":   ("datasets/dsmlr-carparts-split/images/val",
                  "datasets/dsmlr-carparts-split/annotations/instances_val.json"),
        "test":  ("datasets/dsmlr-carparts-split/images/test",
                  "datasets/dsmlr-carparts-split/annotations/instances_test.json"),
    },
}

for dataset_name, splits in DATASETS.items():
    for split_name, (images_dir, json_path) in splits.items():
        register_coco_instances(
            f"{dataset_name}_{split_name}",
            {},
            json_path,
            images_dir,
        )
        print(f"[OK] Registered: {dataset_name}_{split_name}")
