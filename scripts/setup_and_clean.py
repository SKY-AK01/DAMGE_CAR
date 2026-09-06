import os
import shutil
import datetime

def main():
    print("--- Setup and Clean ---")
    
    # 1. Folder Setup
    raw_dataset = "RAW_DATASET"
    images_dir = os.path.join(raw_dataset, "IMAGES")
    xml_dir = os.path.join(raw_dataset, "XML")
    open_source_dir = os.path.join(raw_dataset, "open_source")
    user_data_dir = os.path.join(raw_dataset, "user_data")
    
    for d in [images_dir, xml_dir, open_source_dir, user_data_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"[CREATED] {d}")
        else:
            print(f"[OK] {d} exists.")
            
    # Check if empty
    img_empty = len(os.listdir(images_dir)) == 0
    xml_empty = len(os.listdir(xml_dir)) == 0
    
    if img_empty or xml_empty:
        print("[WARNING] Put your raw image files (.jpg/.png) in RAW_DATASET/IMAGES and annotation files (.xml) in RAW_DATASET/XML.")
        
    # 2. Archive old runs
    runs_dir = "runs_comparison"
    archive_dir = "archive"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        
    if os.path.exists(runs_dir):
        for item in os.listdir(runs_dir):
            if item.startswith("run_"):
                src = os.path.join(runs_dir, item)
                dst = os.path.join(archive_dir, item)
                print(f"[*] Archiving {src} to {dst}...")
                shutil.move(src, dst)
    
    print("[OK] Setup and Clean complete.")

if __name__ == "__main__":
    main()
