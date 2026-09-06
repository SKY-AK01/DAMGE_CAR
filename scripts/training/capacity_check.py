#!/usr/bin/env python3
"""
capacity_check.py -- GPU Capacity & Batch Size / Worker Stress Tester.

Tests candidate batch sizes and worker counts for YOLOv11m-seg, Mask R-CNN, Fast R-CNN, etc.,
to find the maximum stable batch size, peak VRAM usage, and optimal throughput (images/sec)
before encountering Out-Of-Memory (OOM) or system limits.
"""

import os
import sys
import time
import json
import re
import argparse
import subprocess
import threading
from pathlib import Path
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# Ensure UTF-8 output and unbuffered line-streaming
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

CUDA_PEAK_VRAM_PREFIX = "[CUDA Peak VRAM]"

def parse_args():
    parser = argparse.ArgumentParser(description="GPU Capacity and Batch Size Stress Tester")
    parser.add_argument("--dataset", default="./datasets/combined_carparts", help="Path to dataset directory")
    parser.add_argument("--models", nargs="+", default=["yolo11m-seg", "yolo11x-seg", "maskrcnn", "mask2former", "sam2", "maskdino", "segformer"],
                        choices=["yolo11m-seg", "yolo11x-seg", "maskrcnn", "mask2former", "sam2", "maskdino", "segformer"], help="Models to test")
    parser.add_argument("--batch-list", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64],
                        help="Batch sizes to test sequentially")
    parser.add_argument("--workers-list", nargs="+", type=int, default=[4, 8, 16],
                        help="Worker counts to test")
    parser.add_argument("--mode", choices=["local", "azure"], default="local",
                        help="Whether to run capacity check locally or generate Azure job configuration")
    parser.add_argument("--strategy", choices=["top_down", "bottom_up"], default="top_down",
                        help="Top-Down (descending max-first fast-exit, saves ~70%% GPU time/cost) or Bottom-Up (ascending).")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Run quick 10-batch micro-benchmarks for 20x faster capacity sweeps.")
    parser.add_argument("--full", dest="quick", action="store_false",
                        help="Run full 1-epoch sweeps instead of 10-batch micro-benchmarks.")
    return parser.parse_args()

def get_dataset_image_count(dataset_path):
    """
    Determines actual number of training images in dataset_path.
    Checks images/train, train, or parses coco_train.json.
    """
    path = Path(dataset_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    for cand in [path / "images" / "train", path / "train", path / "images"]:
        if cand.exists() and cand.is_dir():
            imgs = [f for f in cand.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
            if len(imgs) > 0:
                return len(imgs)

    coco_path = path / "coco_train.json"
    if coco_path.exists():
        try:
            with open(coco_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "images" in data and len(data["images"]) > 0:
                    return len(data["images"])
        except Exception:
            pass

    all_imgs = [f for f in path.glob("**/*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    return len(all_imgs) if len(all_imgs) > 0 else 100

class GPUMonitor:
    """
    Polls nvidia-smi in a background thread to track actual peak VRAM
    across subprocess boundaries as a secondary fallback.
    """
    def __init__(self, interval=0.2):
        self.interval = interval
        self.max_vram_mb = 0.0
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.max_vram_mb = 0.0
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                if res.returncode == 0 and res.stdout.strip():
                    val = float(res.stdout.strip().splitlines()[0])
                    if val > self.max_vram_mb:
                        self.max_vram_mb = val
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        return self.max_vram_mb / 1024.0  # Return GB

def is_oom_error(output_text):
    if not output_text:
        return False
    lower = output_text.lower()
    exact_oom_phrases = [
        "cuda out of memory",
        "outofmemoryerror",
        "out of memory",
        "std::bad_alloc",
        "cuda error: out of memory",
        "allocator returned null",
        "gpu memory is full",
        "torch_use_cuda_dsa",
        "device-side assert",
        "cuda error: an illegal memory access",
        "cuda error: unspecified launch failure"
    ]
    return any(phrase in lower for phrase in exact_oom_phrases)

def cleanup_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    time.sleep(1.5)

def extract_peak_vram_from_output(output_text, fallback_monitor_gb=0.0):
    """
    Parses PyTorch CUDA Peak VRAM logged directly by child training scripts.
    Handles variations of '[CUDA Peak VRAM] X.XX GB'.
    """
    if output_text:
        match = re.search(r"\[?CUDA\s*Peak\s*VRAM\]?:?\s*([\d\.]+)\s*(?:GB)?", output_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return fallback_monitor_gb

def build_training_cmd(model_name, dataset, batch, workers, quick=True):
    cmd = []
    if model_name in ["yolo11m-seg", "yolo11x-seg"]:
        cmd = ["python", "scripts/training/train_yolo_seg.py",
                "--model", model_name, "--dataset", dataset,
                "--epochs", "1", "--batch", str(batch), "--workers", str(workers),
                "--project", f"runs_comparison/capacity_test/{model_name}"]
    elif model_name == "maskrcnn":
        cmd = ["python", "scripts/training/train_maskrcnn.py",
                "--dataset", dataset, "--epochs", "1", "--batch", str(batch),
                "--num_workers", str(workers), "--output_dir", f"runs_comparison/capacity_test/{model_name}"]
    elif model_name == "mask2former":
        cmd = ["python", "scripts/training/train_mask2former.py",
                "--dataset", dataset, "--epochs", "1", "--batch", str(batch),
                "--num_workers", str(workers), "--output_dir", f"runs_comparison/capacity_test/{model_name}"]
    elif model_name == "sam2":
        cmd = ["python", "scripts/training/train_sam2_seg.py",
                "--dataset", dataset, "--epochs", "1", "--batch", str(batch),
                "--num_workers", str(workers), "--output_dir", f"runs_comparison/capacity_test/{model_name}"]
    elif model_name == "maskdino":
        cmd = ["python", "scripts/training/train_maskdino.py",
                "--dataset", dataset, "--epochs", "1", "--batch", str(batch),
                "--num_workers", str(workers), "--output_dir", f"runs_comparison/capacity_test/{model_name}"]
    elif model_name == "segformer":
        cmd = ["python", "scripts/training/train_segformer.py",
                "--dataset", dataset, "--epochs", "1", "--batch", str(batch),
                "--num_workers", str(workers), "--output_dir", f"runs_comparison/capacity_test/{model_name}"]
    
    if quick:
        cmd.extend(["--max_batches", "5"])
    return cmd

def test_model_capacity(model_name, dataset, batch_list, workers_list, quick=True, strategy="top_down"):
    mode_desc = f"Quick Micro-Benchmark | Strategy: {strategy.upper()}" if quick else f"Full Sweep | Strategy: {strategy.upper()}"
    print(f"\n======================================================================")
    print(f" [TEST] CAPACITY & STRESS TEST: {model_name.upper()} ({mode_desc})")
    print(f"======================================================================")
    
    import torch
    if not torch.cuda.is_available():
        print(" [SKIP] No CUDA GPU detected on this machine.")
        print("        Capacity check is GPU-only -- batch/VRAM sweep on CPU takes hours and")
        print("        yields no meaningful recommendations. Run this on Azure ML with a GPU node.")
        print("        Returning default safe values (batch=8, workers=4).")
        return {
            "model": model_name,
            "max_stable_batch": 0,
            "safe_prod_batch": 8,     # sensible default for GPU training
            "safe_prod_vram": 0.0,
            "optimal_workers": 4,
            "best_fps": 0.0,
            "est_epoch_min": 0.0,
            "results": [],
            "skipped": "no_gpu",
        }

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f" [OK] GPU: {device_name} ({total_vram:.2f} GB VRAM)")

    total_num_images = get_dataset_image_count(dataset)
    # Quick mode: 120s cap per batch size. Full mode: 10-min cap.
    timeout_sec = 120 if quick else 600

    results = []
    max_stable_batch = 0
    safe_prod_batch = 0
    safe_prod_vram = 0.0
    optimal_workers = 4
    best_fps = 0.0

    gpu_monitor = GPUMonitor()

    # Determine sweep search order
    if strategy == "top_down":
        sorted_batches = sorted(batch_list, reverse=True)
    else:
        sorted_batches = sorted(batch_list)

    # Phase 1: Batch Size Sweep
    print(f"\n--- Phase 1: Batch Size Sweep ({strategy.upper()}) ---")
    print(f" {'Batch Size':<12} | {'Status':<15} | {'Peak VRAM (GB)':<20} | {'Throughput (img/s)':<20}")
    print("-" * 75)

    last_failed_batch = None

    for batch in sorted_batches:
        cleanup_gpu()
        cmd = build_training_cmd(model_name, dataset, batch, workers=8, quick=quick)
        if not cmd:
            continue

        num_images_tested = min(5 * batch, total_num_images) if quick else total_num_images

        print(f" -> Testing batch size {batch} (workers=8)...", flush=True)
        gpu_monitor.start()
        start_t = time.time()
        
        output = ""
        timed_out = False
        # Force UTF-8 so YOLO's unicode progress bar chars (##) don't crash on Windows cp1252
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                timeout=timeout_sec,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            elapsed = time.time() - start_t
            output = res.stdout or ""
            exit_code = res.returncode
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_t
            raw = e.stdout or b""
            output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
            exit_code = -1
            timed_out = True

        monitor_vram_gb = gpu_monitor.stop()
        peak_vram_gb = extract_peak_vram_from_output(output, fallback_monitor_gb=monitor_vram_gb)

        if total_vram > 0 and peak_vram_gb > 0:
            pct = (peak_vram_gb / total_vram) * 100.0
            vram_str = f"{peak_vram_gb:.2f} GB ({pct:.1f}%)"
        elif peak_vram_gb > 0:
            vram_str = f"{peak_vram_gb:.2f} GB"
        else:
            vram_str = "N/A"

        if timed_out:
            last_failed_batch = batch
            status_str = "FAILED (Timeout)"
            print(f" {batch:<12} | {status_str:<15} | {'STALLED':<20} | {'0.00':<20}")
            print(f"   [!] Subprocess timed out after {timeout_sec}s (stalled/deadlocked).")
            results.append({"batch": batch, "status": status_str, "vram": "Timeout", "vram_gb": 0.0, "fps": 0})
            if strategy == "bottom_up":
                break
        elif exit_code == 0:
            fps = num_images_tested / elapsed if elapsed > 0 else 0
            print(f" {batch:<12} | {'PASSED':<15} | {vram_str:<20} | {fps:<20.2f}")
            results.append({"batch": batch, "status": "PASSED", "vram": vram_str, "vram_gb": peak_vram_gb, "fps": fps})
            
            if strategy == "top_down":
                max_stable_batch = batch
                if total_vram == 0 or (peak_vram_gb / total_vram) <= 0.85:
                    safe_prod_batch = batch
                    safe_prod_vram = peak_vram_gb
                
                # Fast Midpoint Refinement: if headroom is ample (>15%) and we had a higher failure, test the midpoint!
                if last_failed_batch and last_failed_batch > batch:
                    free_vram = total_vram - peak_vram_gb
                    mid_batch = (batch + last_failed_batch) // 2
                    if free_vram > 1.8 and mid_batch > batch:
                        print(f"   [*] Substantial headroom ({free_vram:.2f} GB free). Testing midpoint batch {mid_batch}...", flush=True)
                        cleanup_gpu()
                        mid_cmd = build_training_cmd(model_name, dataset, mid_batch, workers=8, quick=quick)
                        gpu_monitor.start()
                        m_start_t = time.time()
                        try:
                            m_res = subprocess.run(mid_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                   text=True, encoding="utf-8", errors="replace", timeout=timeout_sec,
                                                   cwd=str(PROJECT_ROOT), env=env)
                            m_elapsed = time.time() - m_start_t
                            m_vram = gpu_monitor.stop()
                            m_peak = extract_peak_vram_from_output(m_res.stdout or "", fallback_monitor_gb=m_vram)
                            if m_res.returncode == 0 and (total_vram == 0 or (m_peak / total_vram) <= 0.88):
                                m_pct = (m_peak / total_vram) * 100.0 if total_vram > 0 else 0
                                m_vram_str = f"{m_peak:.2f} GB ({m_pct:.1f}%)"
                                m_fps = min(5 * mid_batch, total_num_images) / m_elapsed if m_elapsed > 0 else 0
                                print(f" {mid_batch:<12} | {'PASSED':<15} | {m_vram_str:<20} | {m_fps:<20.2f}")
                                print(f"   [OK] Midpoint Refinement: Successfully upgraded to larger batch {mid_batch}!")
                                max_stable_batch = mid_batch
                                safe_prod_batch = mid_batch
                                safe_prod_vram = m_peak
                                results.append({"batch": mid_batch, "status": "PASSED", "vram": m_vram_str, "vram_gb": m_peak, "fps": m_fps})
                            else:
                                print(f"   [-] Midpoint batch {mid_batch} exceeded headroom. Keeping stable batch {batch}.")
                        except Exception:
                            print(f"   [-] Midpoint batch {mid_batch} failed. Keeping stable batch {batch}.")
                        cleanup_gpu()

                print(f"   [OK] Top-Down Fast Exit: Found largest stable batch size ({max_stable_batch}). Stopping search early!")
                break
            else:
                max_stable_batch = batch
                if total_vram == 0 or (peak_vram_gb / total_vram) <= 0.85:
                    safe_prod_batch = batch
                    safe_prod_vram = peak_vram_gb
                if fps > best_fps:
                    best_fps = fps
        else:
            last_failed_batch = batch
            err_lines = [line.strip() for line in output.splitlines() if line.strip()]
            # Print last 5 meaningful lines so we always see the real error
            last_lines = err_lines[-5:] if len(err_lines) >= 5 else err_lines
            if is_oom_error(output):
                status_str = "FAILED (OOM)"
                print(f" {batch:<12} | {status_str:<15} | {'EXCEEDED VRAM':<20} | {'0.00':<20}")
                for ln in last_lines:
                    print(f"   [OOM] {ln[:120]}")
                results.append({"batch": batch, "status": status_str, "vram": "OOM", "vram_gb": 0.0, "fps": 0})
                if strategy == "bottom_up":
                    print(f"   [!] Stopping batch sweep for {model_name} due to GPU Out-Of-Memory limit at batch size {batch}.")
                    break
            else:
                status_str = "FAILED (Error)"
                print(f" {batch:<12} | {status_str:<15} | {vram_str:<20} | {'0.00':<20}")
                for ln in last_lines:
                    print(f"   [ERR] {ln[:120]}")
                results.append({"batch": batch, "status": status_str, "vram": vram_str, "vram_gb": peak_vram_gb, "fps": 0})
                if strategy == "bottom_up":
                    break

    # Fix: Correct fallback for safe_prod_vram if all passing batches exceed 85% VRAM
    if safe_prod_batch == 0 and max_stable_batch > 0:
        safe_prod_batch = max_stable_batch
        for r in results:
            if r.get("batch") == max_stable_batch and r.get("status") == "PASSED":
                safe_prod_vram = r.get("vram_gb", 0.0)
                break

    # Phase 2: Worker count sweep on max stable batch
    if max_stable_batch > 0:
        print(f"\n--- Phase 2: Worker Count Sweep (Batch Size = {max_stable_batch}) ---")
        print(f" {'Workers':<12} | {'Status':<15} | {'Throughput (img/s)':<20}")
        print("-" * 55)
        for w in workers_list:
            cleanup_gpu()
            cmd = build_training_cmd(model_name, dataset, max_stable_batch, workers=w, quick=quick)
            if not cmd:
                continue

            num_images_tested = min(5 * max_stable_batch, total_num_images) if quick else total_num_images
            print(f" -> Testing workers={w} (batch={max_stable_batch})...", flush=True)
            start_t = time.time()
            
            # Force UTF-8 so YOLO's unicode progress bar chars don't crash on Windows cp1252
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                res = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=timeout_sec,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                )
                elapsed = time.time() - start_t
                if res.returncode == 0:
                    fps = num_images_tested / elapsed if elapsed > 0 else 0
                    print(f" {w:<12} | {'PASSED':<15} | {fps:<20.2f}")
                    if fps > best_fps:
                        best_fps = fps
                        optimal_workers = w
                    elif best_fps > 0 and fps < best_fps * 0.90:
                        # Worker saturation reached, skip higher worker counts to save time
                        print(f"   [OK] Worker saturation reached ({w} workers is slower than {optimal_workers} workers). Stopping early!")
                        break
                else:
                    print(f" {w:<12} | {'FAILED':<15} | {'0.00':<20}")
            except subprocess.TimeoutExpired:
                print(f" {w:<12} | {'FAILED (Timeout)':<15} | {'0.00':<20}")

    # Calculate epoch ETA specifically from throughput at safe_prod_batch
    prod_fps = 0.0
    for r in results:
        if r.get("batch") == safe_prod_batch and r.get("status") == "PASSED":
            prod_fps = r.get("fps", 0.0)
            break
    if prod_fps == 0.0:
        prod_fps = best_fps

    est_epoch_min = (total_num_images / prod_fps / 60.0) if prod_fps > 0 else 0.0
    headroom_gb = max(0.0, total_vram - safe_prod_vram) if total_vram > 0 else 0.0

    print(f"\n======================================================================")
    print(f" [RECOMMENDATION SUMMARY] FOR {model_name.upper()}:")
    print(f"   * Maximum Capacity Batch Size : {max_stable_batch}")
    print(f"   * Recommended Prod Batch Size : {safe_prod_batch} (Peak VRAM: {safe_prod_vram:.2f} GB | Free Headroom: {headroom_gb:.2f} GB)")
    if total_vram > 0 and safe_prod_vram > 0 and (safe_prod_vram / total_vram) > 0.85:
        print(f"   [!] WARNING: Recommended batch ({safe_prod_batch}) exceeds 85% VRAM safety margin ({safe_prod_vram:.2f}/{total_vram:.2f} GB).")
    print(f"   * Optimal Dataloader Workers  : {optimal_workers} workers")
    print(f"   * Production Throughput Rate  : {prod_fps:.2f} images/sec")
    if est_epoch_min > 0:
        print(f"   * Estimated Time per Epoch   : {est_epoch_min:.1f} minutes ({total_num_images} images)")
    print(f"======================================================================\n")

    return {
        "model": model_name,
        "max_stable_batch": max_stable_batch,
        "safe_prod_batch": safe_prod_batch,
        "safe_prod_vram": safe_prod_vram,
        "optimal_workers": optimal_workers,
        "best_fps": prod_fps,
        "est_epoch_min": est_epoch_min,
        "results": results
    }

def main():
    args = parse_args()

    if args.mode == "azure":
        print("\n[*] Submitting GPU Capacity Check job to Azure ML Compute...")
        azure_cmd = [
            "python", "scripts/training/azure_train.py",
            "--model", "capacity_check",
            "--local_dataset_dir", args.dataset,
            "--epochs", "1",
            "--batch", "8",
            "--workers", "8",
            "--auto_upload"
        ]
        subprocess.run(azure_cmd, check=True)
        return

    summary = []
    for model in args.models:
        res = test_model_capacity(model, args.dataset, args.batch_list, args.workers_list, quick=args.quick, strategy=args.strategy)
        summary.append(res)

    print("\n" + "="*88)
    print(" [SUMMARY] FINAL MODEL TRAINING OPTIMIZATION RECOMMENDATIONS")
    print("="*88)
    print(f" {'Model':<14} | {'Best Batch (Safe)':<18} | {'Best Workers':<14} | {'Throughput':<14} | {'Est. Time/Epoch':<15}")
    print("-" * 88)
    for s in summary:
        if s.get("skipped") == "no_gpu":
            print(f" {s['model'].upper():<14} | {'[SKIP-CPU]':<18} | {'[SKIP-CPU]':<14} | {'[SKIP-CPU]':<14} | {'[SKIP-CPU]':<15}")
        else:
            fps_str = f"{s['best_fps']:.2f} img/s"
            eta_str = f"{s['est_epoch_min']:.1f} min" if s.get('est_epoch_min', 0) > 0 else "N/A"
            print(f" {s['model'].upper():<14} | {s['safe_prod_batch']:<18} | {s['optimal_workers']:<14} | {fps_str:<14} | {eta_str:<15}")
    print("="*88)
    no_gpu_all = all(s.get("skipped") == "no_gpu" for s in summary)
    if no_gpu_all:
        print("\n[!] All models skipped -- no CUDA GPU on this machine.")
        print("    Run capacity check via Azure ML (Option 11 in orchestrator) to get real results.")
        return

    # Save recommendations to JSON for orchestrator auto-population
    save_dir = PROJECT_ROOT / "runs_comparison"
    os.makedirs(save_dir, exist_ok=True)
    json_path = save_dir / "gpu_capacity_recommendations.json"

    rec_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    for s in summary:
        rec_data["models"][s["model"]] = {
            "batch": s["safe_prod_batch"],
            "workers": s["optimal_workers"],
            "max_batch": s["max_stable_batch"],
            "fps": s["best_fps"],
            "vram_gb": s["safe_prod_vram"]
        }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec_data, f, indent=2)
        print(f"\n[OK] GPU recommendations saved to: {json_path}")
        print("     Orchestrator will auto-populate these values as training defaults!")
    except Exception as e:
        print(f"[WARN] Could not save GPU recommendations to disk: {e}")

    # Always print JSON to stdout with a special marker so orchestrator.py can
    # parse it from streaming logs regardless of whether it's local or Azure ML.
    print(f"[GPU_RECOMMENDATIONS_JSON] {json.dumps(rec_data)}")

if __name__ == "__main__":
    main()
