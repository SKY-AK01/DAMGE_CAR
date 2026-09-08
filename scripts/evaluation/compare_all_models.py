"""
compare_all_models.py
------------------------
Run this last, after evaluate_confusion_matrix.py has been run once per model.
Loads the saved .npy confusion matrices for yolo / mask2former / maskdino and
prints a single side-by-side summary table, focused on the front/rear and
left/right mixup question this whole experiment exists to answer.

Usage:
    python compare_all_models.py --dataset carparts-seg
"""

import argparse
import numpy as np
from pathlib import Path

CARPARTS_SEG_CLASSES = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light",
    "back_light", "back_right_door", "back_right_light", "front_bumper", "front_door",
    "front_glass", "front_left_door", "front_left_light", "front_light", "front_right_door",
    "front_right_light", "hood", "left_mirror", "object", "right_mirror", "tailgate",
    "trunk", "wheel",
]

WATCH_PAIRS = [
    ("front_left_door", "back_left_door"),
    ("front_right_door", "back_right_door"),
    ("front_left_door", "front_right_door"),
    ("back_left_door", "back_right_door"),
    ("front_left_light", "front_right_light"),
    ("back_left_light", "back_right_light"),
    ("left_mirror", "right_mirror"),
]

MODELS = ["yolo", "maskrcnn", "mask2former", "oneformer", "maskdino"]


def overall_accuracy(matrix):
    """Diagonal sum (correct matches) / total ground-truth instances (excluding the background row)."""
    n = matrix.shape[0] - 1  # exclude background row/col
    correct = np.trace(matrix[:n, :n])
    total = matrix[:n, :].sum()
    return correct / total if total > 0 else 0


def mixup_rate(matrix, class_names, a, b):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    if a not in class_to_idx or b not in class_to_idx:
        return None
    ia, ib = class_to_idx[a], class_to_idx[b]
    a_total = matrix[ia, :].sum()
    b_total = matrix[ib, :].sum()
    if a_total + b_total == 0:
        return None
    mixed = matrix[ia, ib] + matrix[ib, ia]
    return mixed / (a_total + b_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["carparts-seg", "dsmlr-carparts", "custom_carparts"])
    args = parser.parse_args()

    matrix_dir = Path("runs_comparison/confusion_matrices")
    class_names = CARPARTS_SEG_CLASSES

    matrices = {}
    for model in MODELS:
        path = matrix_dir / f"{model}_{args.dataset}_matrix.npy"
        if path.exists():
            matrices[model] = np.load(path)
        else:
            print(f"[skip] No results found for {model} -- run evaluate_confusion_matrix.py first.")

    if not matrices:
        print("[ERROR] No model results found. Run evaluate_confusion_matrix.py for each model first.")
        return

    print("\n" + "=" * 70)
    print(f" OVERALL ACCURACY -- {args.dataset}")
    print("=" * 70)
    for model, matrix in matrices.items():
        acc = overall_accuracy(matrix)
        print(f"  {model:15s}: {acc:.1%}")

    print("\n" + "=" * 70)
    print(f" FRONT/REAR & LEFT/RIGHT MIXUP RATE (lower = better)")
    print("=" * 70)
    header = f"{'Pair':40s}" + "".join(f"{m:>15s}" for m in matrices.keys())
    print(header)
    print("-" * len(header))
    for a, b in WATCH_PAIRS:
        row = f"{a + ' vs ' + b:40s}"
        for model, matrix in matrices.items():
            rate = mixup_rate(matrix, class_names, a, b)
            row += f"{(f'{rate:.1%}' if rate is not None else 'N/A'):>15s}"
        print(row)

    print("\n[i] Full confusion matrix images are in runs_comparison/confusion_matrices/*.png")


if __name__ == "__main__":
    main()
