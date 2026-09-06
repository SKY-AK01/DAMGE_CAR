#!/usr/bin/env python3
import shutil
from pathlib import Path

def combine_yolo_datasets(dataset_dirs, out_dir="datasets/combined_carparts"):
    out_path = Path(out_dir)
    out_images_train = out_path / "images/train"
    out_images_val = out_path / "images/val"
    out_labels_train = out_path / "labels/train"
    out_labels_val = out_path / "labels/val"

    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    names = set()
    yaml_lines = []

    for ddir in dataset_dirs:
        dp = Path(ddir)
        if not dp.exists():
            continue
        
        print(f"[*] Combining {ddir}...")
        for split in ["train", "val"]:
            img_dir = dp / "images" / split
            lbl_dir = dp / "labels" / split
            if not img_dir.exists() and split == "train":
                # Some datasets like carparts-seg don't have labels/train, they might have labels together?
                # Actually YOLO format is images/train and labels/train
                pass

            # Copy images
            if img_dir.exists():
                for f in img_dir.glob("*.*"):
                    if f.is_file():
                        shutil.copy2(f, out_path / "images" / split / f"{dp.name}_{f.name}")
            
            # Copy labels
            # In carparts-seg, labels are in carparts-seg/labels/train? Usually yes.
            # Let's handle flat labels too if present
            labels_exist = False
            if lbl_dir.exists():
                labels_exist = True
                for f in lbl_dir.glob("*.txt"):
                    shutil.copy2(f, out_path / "labels" / split / f"{dp.name}_{f.name}")
            
            # Fallback if labels are mixed in images dir
            if not labels_exist and img_dir.exists():
                for f in img_dir.glob("*.txt"):
                    shutil.copy2(f, out_path / "labels" / split / f"{dp.name}_{f.name}")

        # Check for yaml to inherit names
        yaml_files = list(dp.glob("*.yaml"))
        if yaml_files and not yaml_lines:
            with open(yaml_files[0], "r") as yf:
                lines = yf.readlines()
                # find names block
                in_names = False
                for line in lines:
                    if line.startswith("names:"):
                        in_names = True
                        yaml_lines.append(line)
                    elif in_names:
                        if line.startswith(" ") or line.startswith("\t"):
                            yaml_lines.append(line)
                        else:
                            in_names = False

    # Create combined yaml
    yaml_content = f"path: {out_path.resolve()}\ntrain: images/train\nval: images/val\ntest: \n\n"
    if yaml_lines:
        yaml_content += "".join(yaml_lines)
    
    with open(out_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"[OK] Datasets combined into {out_dir}")

if __name__ == "__main__":
    datasets_to_combine = ["datasets/carparts-seg", "datasets/custom_carparts"]
    combine_yolo_datasets(datasets_to_combine)
