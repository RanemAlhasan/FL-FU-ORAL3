#!/usr/bin/env python3
"""
Collect completed FL learning-curve runs and generate CSV + plots.

Example:
python3 scripts/learning_curve/collect_learning_curve.py \
  --study-tag lc_s42 \
  --settings fedbn_stdce fedavg_wrs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="logs/fl")
    parser.add_argument(
        "--dataset-root",
        default="dataset/oral3_learning_curve_seed42",
    )
    parser.add_argument("--study-tag", default="lc_s42")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["fedbn_stdce", "fedavg_wrs"],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/learning_curve",
    )
    return parser.parse_args()


def get_metric(data: dict, key: str) -> Optional[float]:
    final = data.get("final", {})
    if key in final:
        try:
            return float(final[key])
        except (TypeError, ValueError):
            pass

    scalars = data.get("scalars", {})
    values = scalars.get(key, [])
    if values:
        try:
            return float(values[-1]["value"])
        except (KeyError, TypeError, ValueError):
            pass

    if key == "eval/classification/macro_f1":
        c = data.get("classification", {}).get("eval/classification", {})
        if "macro_f1" in c:
            return float(c["macro_f1"])

    if key == "eval/classification/accuracy":
        c = data.get("classification", {}).get("eval/classification", {})
        if "accuracy" in c:
            return float(c["accuracy"])

    return None


def load_train_counts(dataset_root: Path) -> Dict[int, int]:
    path = dataset_root / "subset_summary.csv"
    counts: Dict[int, int] = {}
    if not path.exists():
        return counts

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["hospital"] == "__TOTAL__"
                and row["class_name"] == "__TOTAL__"
            ):
                counts[int(row["fraction_pct"])] = int(row["selected_count"])
    return counts


def newest_matching_run(
    logs_root: Path,
    study_tag: str,
    setting: str,
    pct: int,
) -> Optional[Path]:
    # job id can differ across submissions. If several completed runs exist,
    # use the newest metrics.json.
    prefix = f"{study_tag}_{setting}_p{pct}_seed"
    candidates = []
    for d in logs_root.glob(f"{prefix}*"):
        metrics = d / "metrics.json"
        if metrics.exists():
            candidates.append(metrics)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    args = parse_args()

    logs_root = Path(args.logs_root)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts = load_train_counts(dataset_root)
    percentages = [20, 40, 60, 80, 100]
    rows: List[dict] = []

    for setting in args.settings:
        for pct in percentages:
            metrics_path = newest_matching_run(
                logs_root,
                args.study_tag,
                setting,
                pct,
            )
            if metrics_path is None:
                print(f"WARNING: no completed metrics found for {setting} p{pct}")
                continue

            with metrics_path.open() as f:
                data = json.load(f)

            row = {
                "setting": setting,
                "fraction_pct": pct,
                "train_samples": train_counts.get(pct, ""),
                "run_id": data.get("run_id", metrics_path.parent.name),
                "overall_acc": get_metric(data, "eval/overall/acc"),
                "macro_f1": get_metric(
                    data,
                    "eval/classification/macro_f1",
                ),
                "weighted_f1": get_metric(
                    data,
                    "eval/classification/weighted_f1",
                ),
                "spain_acc": get_metric(
                    data,
                    "eval/per_hospital/Spain_Dataset/acc",
                ),
                "canada_acc": get_metric(
                    data,
                    "eval/per_hospital/Canada_Dataset/acc",
                ),
                "india_acc": get_metric(
                    data,
                    "eval/per_hospital/India_Dataset/acc",
                ),
                "metrics_path": str(metrics_path),
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No completed learning-curve runs were found.")

    csv_path = output_dir / f"{args.study_tag}_fl_learning_curve.csv"
    fields = [
        "setting",
        "fraction_pct",
        "train_samples",
        "run_id",
        "overall_acc",
        "macro_f1",
        "weighted_f1",
        "spain_acc",
        "canada_acc",
        "india_acc",
        "metrics_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; CSV was still created.")
        return

    def plot_metric(metric: str, ylabel: str, filename: str):
        plt.figure(figsize=(7.5, 5.0))
        for setting in args.settings:
            subset = [
                r for r in rows
                if r["setting"] == setting and r[metric] is not None
            ]
            subset.sort(key=lambda r: r["fraction_pct"])
            if not subset:
                continue
            x = [r["fraction_pct"] for r in subset]
            y = [100.0 * float(r[metric]) for r in subset]
            plt.plot(x, y, marker="o", label=setting)

        plt.xlabel("Training data used (%)")
        plt.ylabel(ylabel)
        plt.xticks([20, 40, 60, 80, 100])
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Wrote: {path}")

    plot_metric(
        "overall_acc",
        "Overall accuracy (%)",
        f"{args.study_tag}_overall_accuracy.png",
    )
    plot_metric(
        "macro_f1",
        "Macro-F1 (%)",
        f"{args.study_tag}_macro_f1.png",
    )

    print("\nLearning-curve table:")
    for row in sorted(rows, key=lambda r: (r["setting"], r["fraction_pct"])):
        acc = (
            f"{100 * row['overall_acc']:.2f}%"
            if row["overall_acc"] is not None
            else "NA"
        )
        f1 = (
            f"{100 * row['macro_f1']:.2f}%"
            if row["macro_f1"] is not None
            else "NA"
        )
        print(
            f"{row['setting']:14s} "
            f"p{row['fraction_pct']:3d} "
            f"n={str(row['train_samples']):>5s} "
            f"acc={acc:>8s} macro_f1={f1:>8s}"
        )


if __name__ == "__main__":
    main()
