import subprocess
import os

def test_cli_args():
    # We want to ensure that the arguments passed by the Rust orchestrator
    # are accepted by the python training scripts.
    
    # The orchestrator uses these arguments:
    # yolo: --model, --dataset, --epochs, --batch, --workers, --project
    # maskrcnn: --dataset, --epochs, --batch, --num_workers, --output_dir
    # fastrcnn: --dataset, --epochs, --batch, --num_workers, --project
    
    scripts = {
        "train_yolo_seg.py": ["--model", "yolo11m-seg", "--dataset", "test", "--epochs", "1", "--batch", "1", "--workers", "1", "--project", "test_proj"],
        "train_maskrcnn.py": ["--dataset", "test", "--epochs", "1", "--batch", "1", "--num_workers", "1", "--output_dir", "test_out"],
        "train_fastrcnn.py": ["--dataset", "test", "--epochs", "1", "--batch", "1", "--num_workers", "1", "--project", "test_proj"]
    }
    
    for script, args in scripts.items():
        script_path = os.path.join("scripts", "training", script)
        if not os.path.exists(script_path):
            continue
            
        # Call script with --help to ensure it parses successfully
        # Wait, --help exits with 0. 
        # Alternatively, we can just run the script with the args and a --help flag to test parsing.
        # But some scripts might start training if we don't pass --help. 
        # So we can pass --help and grep the output to ensure the flags are present in the help message.
        result = subprocess.run(["python", script_path, "--help"], capture_output=True, text=True)
        help_output = result.stdout
        
        for i in range(0, len(args), 2):
            flag = args[i]
            if flag not in help_output:
                print(f"[FAILED] {script} is missing argument {flag}")
                assert False, f"Missing {flag} in {script}"
                
    print("[OK] All CLI argument contracts match!")

if __name__ == "__main__":
    test_cli_args()
