# Run Analysis: YOLOv11m-seg Training Failure

## 1. Did the run succeed or fail?
**The run failed immediately** during the YOLO training step due to an unrecognized command-line argument.

## 2. Error Breakdown
The error occurred right at the start of the YOLO training script.

**Log Excerpt:**
```
[ERR] train_yolo_seg.py: error: unrecognized arguments: --output_dir /home/orchvate/CAR_NEW_Train/runs_comparison/run_20260824_073035/yolo11m-seg
```

- **Trigger:** The Rust orchestrator binary (`orchestrator/src/main.rs`) attempted to spawn the `train_yolo_seg.py` Python script and passed it an `--output_dir` argument.
- **Root Cause:** A code bug in the argument parsing handoff. The Python script `train_yolo_seg.py` uses `--project` to define its output location (as is standard for Ultralytics/YOLO), but the Rust orchestrator is mistakenly passing `--output_dir` instead.
- **Issue Type:** Code bug (mismatched CLI arguments).

## 3. Code Comparison
If we look at the argument parser inside `scripts/training/train_yolo_seg.py` (lines 96-114):

```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m-seg", ...)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=16, ...)
    parser.add_argument("--workers", type=int, default=16, ...)
    parser.add_argument("--cache", default="ram", ...)
    parser.add_argument("--dataset", default="carparts-seg", ...)
    parser.add_argument("--project", default="runs_comparison")
```
Notice that `--output_dir` is completely absent from the parser.

However, in the Rust orchestrator `orchestrator/src/main.rs` (lines 320-328), we see:
```rust
        let mut cmd = Command::new("python");
        cmd.arg(format!("{}/scripts/training/train_yolo_seg.py", project_root))
            .arg("--model").arg("yolo11m-seg")
            .arg("--dataset").arg(dataset)
            .arg("--epochs").arg(hp.yolo_epochs.to_string())
            .arg("--batch").arg(hp.yolo_batch.to_string())
            .arg("--workers").arg(hp.yolo_workers.to_string())
            .arg("--output_dir").arg(&out_dir)  // <--- THE BUG IS HERE
            .current_dir(project_root);
```

The Rust script expects the YOLO training script to accept `--output_dir` (which the Mask2Former and Mask R-CNN scripts likely do), but YOLO uses `--project`.

## 4. The Fix
To fix this, you must change how the Rust orchestrator calls the YOLO script so it uses the correct argument flag.

Open [`orchestrator/src/main.rs`](file:///c:/Users/Aakash/Documents/PROJECT%20WEB/orchvate/CAR_DAMAGE/CAR_AZURE/orchestrator/src/main.rs) and modify line 327:

**Change this:**
```rust
            .arg("--output_dir").arg(&out_dir)
```

**To this:**
```rust
            .arg("--project").arg(&out_dir)
```

After making this change, you will need to recompile the Rust orchestrator (by running `cargo build --release` inside the `orchestrator/` folder) for the fix to take effect.

*(Alternatively, you could add `--output_dir` to the `argparse` in `train_yolo_seg.py` and map it to `project`, but changing the caller in Rust is much cleaner.)*
