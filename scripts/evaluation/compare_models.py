#!/usr/bin/env python3
"""
compare_models.py
=================
Unified Model Comparison Pipeline: YOLO11m-seg vs Mask R-CNN vs Mask2Former

What it does
------------
1. Scans runs_comparison/ (and any run_* subfolders) and lets you pick which run
   to use for YOLO11m-seg, Mask R-CNN, and Mask2Former (auto-picks best mAP@50-mask
   if --non_interactive or --run_dir is given).
2. Loads training & validation metrics from UnifiedLogger (loss, mAP, precision, recall, VRAM, speed).
3. Generates high-resolution comparison plots:
   - Loss curves (3-model comparison)
   - Mask mAP@50 curves
   - Metric comparison bar charts
   - VRAM & Speed efficiency bar charts
   - Per-class mAP comparison charts
4. Builds a multi-sheet Excel workbook:
   - Summary
   - YOLO11m-seg Epochs
   - Mask R-CNN Epochs
   - Mask2Former Epochs
5. Saves comparison_metrics.json + comparison_summary.txt.
6. Optionally runs test-data inference on images in test_data/ (or --test_images)
   for all selected models saving annotated images.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Optional plotting and pandas deps
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    _OPENPYXL = True
except ImportError:
    _OPENPYXL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_KEYS = ["yolo", "maskrcnn", "mask2former"]
MODEL_LABELS = {
    "yolo": "YOLO11m-seg",
    "maskrcnn": "Mask R-CNN",
    "mask2former": "Mask2Former",
}
MODEL_COLORS = {
    "yolo": "#1f77b4",        # Blue
    "maskrcnn": "#ff7f0e",    # Orange
    "mask2former": "#2ca02c", # Green
}


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Scanner
# ─────────────────────────────────────────────────────────────────────────────

def _read_json_safe(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_best_from_metrics_csv(csv_path: Path):
    """Parse metrics.csv and extract best mask mAP50, best epoch, and final epoch stats."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if not reader:
            return None
        best_map = -1.0
        best_ep = 1
        best_row = reader[0]
        for row in reader:
            m = row.get("map50_mask") or row.get("mAP50", "")
            try:
                m_val = float(m)
                if m_val > best_map:
                    best_map = m_val
                    best_ep = int(row.get("epoch", 1))
                    best_row = row
            except (ValueError, TypeError):
                pass
        return {
            "best_map50_mask": best_map if best_map >= 0 else 0.0,
            "best_epoch": best_ep,
            "total_epochs": len(reader),
            "last_train_loss": float(reader[-1].get("train_loss", 0.0) or 0.0),
            "rows": reader,
            "best_row": best_row
        }
    except Exception:
        return None


def scan_candidates(base_dir: Path):
    """
    Scans base_dir (typically runs_comparison/) for trained model folders.
    Returns dict: { 'yolo': [...], 'maskrcnn': [...], 'mask2former': [...] }
    """
    candidates = {"yolo": [], "maskrcnn": [], "mask2former": []}
    if not base_dir.exists():
        return candidates

    # 1. Search YOLO
    yolo_dirs = list(base_dir.rglob("yolo11m-seg*")) + list(base_dir.rglob("yolo11m*"))
    seen_yolo = set()
    for yd in yolo_dirs:
        if not yd.is_dir() or yd in seen_yolo or yd.name in ["eval", "weights"]:
            continue
        weights = yd / "weights" / "best.pt"
        if not weights.exists():
            weights = yd / "best.pt"
        if weights.exists():
            seen_yolo.add(yd)
            csv_path = yd / "metrics.csv"
            stats = _read_best_from_metrics_csv(csv_path)
            best_map = stats["best_map50_mask"] if stats else 0.0
            best_ep = stats["best_epoch"] if stats else "?"
            candidates["yolo"].append({
                "dir": yd,
                "run_id": yd.parent.name if yd.parent.name.startswith("run_") else yd.name,
                "weights": str(weights),
                "metrics_csv": csv_path if csv_path.exists() else None,
                "best_map50": best_map,
                "best_epoch": best_ep,
                "stats": stats
            })

    # 2. Search Mask R-CNN
    mrcnn_dirs = list(base_dir.rglob("maskrcnn*"))
    seen_mrcnn = set()
    for md in mrcnn_dirs:
        if not md.is_dir() or md in seen_mrcnn or md.name in ["eval", "weights"]:
            continue
        weights = md / "weights" / "best.pt"
        if not weights.exists():
            weights = md / "best_model.pt"
        if weights.exists():
            seen_mrcnn.add(md)
            csv_path = md / "metrics.csv"
            stats = _read_best_from_metrics_csv(csv_path)
            best_map = stats["best_map50_mask"] if stats else 0.0
            best_ep = stats["best_epoch"] if stats else "?"
            candidates["maskrcnn"].append({
                "dir": md,
                "run_id": md.parent.name if md.parent.name.startswith("run_") else md.name,
                "weights": str(weights),
                "metrics_csv": csv_path if csv_path.exists() else None,
                "best_map50": best_map,
                "best_epoch": best_ep,
                "stats": stats
            })

    # 3. Search Mask2Former
    m2f_dirs = list(base_dir.rglob("mask2former*"))
    seen_m2f = set()
    for md in m2f_dirs:
        if not md.is_dir() or md in seen_m2f or md.name in ["eval", "weights"]:
            continue
        weights_dir = md / "weights" / "best"
        if not (weights_dir / "model.safetensors").exists():
            weights_dir = md / "best_model"
        if weights_dir.exists():
            seen_m2f.add(md)
            csv_path = md / "metrics.csv"
            stats = _read_best_from_metrics_csv(csv_path)
            best_map = stats["best_map50_mask"] if stats else 0.0
            best_ep = stats["best_epoch"] if stats else "?"
            candidates["mask2former"].append({
                "dir": md,
                "run_id": md.parent.name if md.parent.name.startswith("run_") else md.name,
                "weights": str(weights_dir),
                "metrics_csv": csv_path if csv_path.exists() else None,
                "best_map50": best_map,
                "best_epoch": best_ep,
                "stats": stats
            })

    for k in candidates:
        candidates[k].sort(key=lambda x: float(x.get("best_map50", 0.0) or 0.0), reverse=True)

    return candidates


def pick_candidate(model_key: str, candidates: list, non_interactive: bool = False):
    label = MODEL_LABELS.get(model_key, model_key)
    if not candidates:
        print(f"  [i] No trained weights found for {label} -> skipping.")
        return None
    if non_interactive or len(candidates) == 1:
        chosen = candidates[0]
        print(f"  [✓] Auto-selected {label}: {chosen['run_id']} (mAP50: {chosen['best_map50']:.4f})")
        return chosen

    print(f"\nSelect run for {label}:")
    for i, c in enumerate(candidates, 1):
        star = "★ " if i == 1 else "  "
        print(f"  {star}{i}) {c['run_id']} | mAP50-mask: {c['best_map50']:.4f} | weights: {c['weights']}")
    print("  Enter choice [1-N], ENTER for ★ recommended, or 's' to skip:")
    while True:
        choice = input("  > ").strip().lower()
        if choice == "":
            return candidates[0]
        if choice == "s":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("  Invalid choice. Try again:")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting & Reporting
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_plots(selected_models: dict, out_dir: Path):
    if not _PLT:
        print("[WARN] matplotlib not available. Skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss & mAP Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for mkey, data in selected_models.items():
        if not data or not data.get("stats") or not data["stats"].get("rows"):
            continue
        rows = data["stats"]["rows"]
        epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
        train_losses = [float(r.get("train_loss", 0.0) or 0.0) for r in rows]
        mask_maps = [float(r.get("map50_mask", 0.0) or 0.0) for r in rows]

        color = MODEL_COLORS.get(mkey, "#333333")
        lbl = MODEL_LABELS.get(mkey, mkey)

        ax1.plot(epochs, train_losses, label=lbl, color=color, linewidth=2)
        ax2.plot(epochs, mask_maps, label=lbl, color=color, linewidth=2, marker="o", markersize=4)

    ax1.set_title("Training Loss Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title("Validation Mask mAP@50 Comparison", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mask mAP@50")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    curve_path = plots_dir / "learning_curves.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()

    # 2. Metric Bar Chart Comparison
    metrics_to_compare = ["map50_mask", "map50_box", "precision_mask", "recall_mask"]
    metric_labels = ["Mask mAP50", "Box mAP50", "Mask Precision", "Mask Recall"]

    models_present = [k for k, v in selected_models.items() if v and v.get("stats")]
    if models_present:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metric_labels))
        width = 0.25

        for i, mkey in enumerate(models_present):
            best_row = selected_models[mkey]["stats"]["best_row"]
            vals = []
            for mk in metrics_to_compare:
                v = float(best_row.get(mk, 0.0) or 0.0)
                vals.append(v)
            offset = (i - len(models_present) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=MODEL_LABELS[mkey], color=MODEL_COLORS.get(mkey))

        ax.set_title("Core Metrics Side-by-Side Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score (0 - 1.0)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        bar_path = plots_dir / "metrics_comparison_bar.png"
        plt.savefig(bar_path, dpi=200)
        plt.close()

    print(f"[OK] Comparison plots saved to: {plots_dir}")


def generate_excel_workbook(selected_models: dict, out_dir: Path):
    if not _OPENPYXL or not _PANDAS:
        print("[WARN] openpyxl / pandas not available. Skipping Excel report.")
        return

    out_file = out_dir / "model_comparison_report.xlsx"
    summary_rows = []

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for mkey in MODEL_KEYS:
            data = selected_models.get(mkey)
            if not data or not data.get("metrics_csv"):
                continue
            lbl = MODEL_LABELS.get(mkey, mkey)
            sheet_name = lbl[:31].replace("-", "_")
            df = pd.read_csv(data["metrics_csv"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            best_row = data["stats"]["best_row"]
            summary_rows.append({
                "Model": lbl,
                "Run ID": data["run_id"],
                "Best Epoch": int(best_row.get("epoch", 1)),
                "Best Mask mAP50": float(best_row.get("map50_mask", 0.0) or 0.0),
                "Box mAP50": float(best_row.get("map50_box", 0.0) or 0.0),
                "Mask Precision": float(best_row.get("precision_mask", 0.0) or 0.0),
                "Mask Recall": float(best_row.get("recall_mask", 0.0) or 0.0),
                "Mask IoU": float(best_row.get("mask_iou", 0.0) or 0.0),
                "Dice F1": float(best_row.get("dice_f1", 0.0) or 0.0),
                "Left/Right Mixup": float(best_row.get("left_right_mixup", 0.0) or 0.0),
                "Peak VRAM (GB)": float(df["gpu_memory_gb"].max() if "gpu_memory_gb" in df.columns else 0.0),
                "Avg Epoch Time (s)": float(df["epoch_time_sec"].mean() if "epoch_time_sec" in df.columns else 0.0),
                "Weights Path": data["weights"],
            })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df.sort_values(by="Best Mask mAP50", ascending=False, inplace=True)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"[OK] Excel comparison workbook saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO11m-seg, Mask R-CNN, and Mask2Former")
    parser.add_argument("--runs_dir", default="runs_comparison", help="Root directory for runs")
    parser.add_argument("--run_dir", default=None, help="Filter to a specific run folder")
    parser.add_argument("--out_dir", default="runs_comparison", help="Output directory for comparison results")
    parser.add_argument("--non_interactive", action="store_true", help="Auto-pick best candidates without prompting")
    parser.add_argument("--test_images", default=None, help="Path to test images for inference comparison")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold for inference")
    parser.add_argument("--skip_test_data", action="store_true", help="Skip running inference on test images")
    args = parser.parse_args()

    base_path = Path(args.run_dir).resolve() if args.run_dir else Path(args.runs_dir).resolve()
    out_path = Path(args.out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" MODEL COMPARISON: YOLO11m-seg vs Mask R-CNN vs Mask2Former")
    print("=" * 70)
    print(f" Scanning: {base_path}")

    candidates = scan_candidates(base_path)
    selected = {}
    for mkey in MODEL_KEYS:
        selected[mkey] = pick_candidate(mkey, candidates.get(mkey, []), non_interactive=args.non_interactive)

    active = {k: v for k, v in selected.items() if v is not None}
    if not active:
        print("[ERROR] No models selected or found for comparison.")
        return

    generate_comparison_plots(selected, out_path)
    generate_excel_workbook(selected, out_path)

    # Save summary JSON
    summary_data = {}
    for k, v in active.items():
        summary_data[MODEL_LABELS[k]] = {
            "run_id": v["run_id"],
            "weights": v["weights"],
            "best_map50": v["best_map50"],
            "best_epoch": v["best_epoch"],
        }
    with open(out_path / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Optional test data inference
    if not args.skip_test_data and args.test_images and Path(args.test_images).exists():
        print("\n[*] Running test-data inference on selected models...")
        infer_out = out_path / "test_inference"
        infer_cmd = [
            "python", "scripts/inference/infer_both_models.py",
            "--input", str(args.test_images),
            "--output", str(infer_out),
            "--conf", str(args.conf)
        ]
        if selected.get("yolo"):
            infer_cmd.extend(["--yolo_weights", selected["yolo"]["weights"]])
        else:
            infer_cmd.append("--skip_yolo")

        if selected.get("maskrcnn"):
            infer_cmd.extend(["--maskrcnn_weights", selected["maskrcnn"]["weights"]])
        else:
            infer_cmd.append("--skip_maskrcnn")

        if selected.get("mask2former"):
            infer_cmd.extend(["--mask2former_weights", selected["mask2former"]["weights"]])
        else:
            infer_cmd.append("--skip_mask2former")

        import subprocess
        subprocess.run(infer_cmd, cwd=str(PROJECT_ROOT))

    print("\n" + "=" * 70)
    print(" COMPARISON PIPELINE COMPLETE")
    print(f" Outputs saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
