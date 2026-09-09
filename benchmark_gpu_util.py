"""
GPU utilization benchmark for the training pipeline.

Runs `nvidia-smi dmon` in the background while a short training run executes,
then reports average / min / max GPU utilization (sm%) over the run.

Usage:
    python benchmark_gpu_util.py --script scripts/training/train_maskrcnn.py \
        --args "--epochs 1 --limit-batches 100" \
        --label "Mask R-CNN (Python DataLoader, num_workers=8)"

Run this once per training script you want to check. Compare the average sm%
across runs — that's your honest "is the GPU actually being fed" number.
"""

import argparse
import csv
import io
import subprocess
import sys
import time
from pathlib import Path


def parse_dmon_output(raw_lines):
    """Parse `nvidia-smi dmon -s u` output into a list of sm% values."""
    sm_values = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # dmon -s u columns: gpu  sm  mem  enc  dec  ...  (sm is 2nd column)
        if len(parts) >= 2:
            try:
                sm_values.append(int(parts[1]))
            except ValueError:
                continue
    return sm_values


def run_benchmark(script, script_args, label, sample_interval=1):
    print(f"\n=== Benchmarking: {label} ===")
    print(f"Command: python {script} {script_args}")

    # Start nvidia-smi dmon in the background, sampling every `sample_interval` sec
    dmon_proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", str(sample_interval)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    start = time.perf_counter()
    train_cmd = [sys.executable, script] + script_args.split()
    train_proc = subprocess.run(train_cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    # Stop dmon and collect its output
    dmon_proc.terminate()
    try:
        dmon_out, _ = dmon_proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        dmon_proc.kill()
        dmon_out, _ = dmon_proc.communicate()

    sm_values = parse_dmon_output(dmon_out.splitlines())

    if train_proc.returncode != 0:
        print(f"  WARNING: training script exited with code {train_proc.returncode}")
        print(f"  stderr (last 500 chars): {train_proc.stderr[-500:]}")

    if not sm_values:
        print("  No GPU samples captured — check nvidia-smi is available and the run was long enough.")
        return None

    avg_sm = sum(sm_values) / len(sm_values)
    result = {
        "label": label,
        "elapsed_sec": round(elapsed, 1),
        "samples": len(sm_values),
        "avg_sm_pct": round(avg_sm, 1),
        "min_sm_pct": min(sm_values),
        "max_sm_pct": max(sm_values),
    }

    print(f"  Elapsed: {result['elapsed_sec']}s over {result['samples']} samples")
    print(f"  GPU util (sm%): avg={result['avg_sm_pct']}%  min={result['min_sm_pct']}%  max={result['max_sm_pct']}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU utilization during a training run.")
    parser.add_argument("--script", required=True, help="Path to training script")
    parser.add_argument("--args", default="", help="Args to pass to the training script, as one quoted string")
    parser.add_argument("--label", default=None, help="Label for this run in the report")
    parser.add_argument("--sample-interval", type=int, default=1, help="nvidia-smi sampling interval in seconds")
    args = parser.parse_args()

    if not Path(args.script).exists():
        print(f"Script not found: {args.script}")
        sys.exit(1)

    label = args.label or args.script
    result = run_benchmark(args.script, args.args, label, args.sample_interval)

    if result:
        report_path = Path("gpu_benchmark_results.csv")
        write_header = not report_path.exists()
        with open(report_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        print(f"\nAppended to {report_path.resolve()}")


if __name__ == "__main__":
    main()
