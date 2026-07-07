#!/usr/bin/env python3
"""
Aggregates logs/<run_id>/metrics.json across multiple run_ids into a single
comparison table (printed + optionally saved as CSV), e.g. to compare:
    FedAvg vs FedProx vs FedBN vs FedMOON
    FL baseline vs FUSED vs Retrain
    domain_adaptation on vs off

Usage:
    python scripts/compare_runs.py --run_ids fl_fedavg_... fl_fedbn_... fu_fused-client_...
    python scripts/compare_runs.py --run_ids fl_fedavg_... fu_fused-client_... --out comparison.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Metrics we always try to surface if present, regardless of which phase
# produced them — pulled from metrics.json's "final" block (see
# src/utils/logger.py::set_final_metric) where available, else the last
# logged scalar value for that tag.
KEY_METRICS = [
    "eval/overall/acc",
    "eval/overall/loss",
    "eval/before_unlearning/overall/acc",
    "eval/after_unlearning/overall/acc",
    "eval/unlearning/RA",
    "eval/unlearning/FA",
    "eval/unlearning/ReA",
    "eval/unlearning/MIA_acc",
    "eval/unlearning/before_acc",
    "eval/unlearning/after_acc",
    "fu/adapter_density",
    "fu/adapter_trainable_params",
]


def load_metrics(logs_root: str, run_id: str) -> dict:
    path = os.path.join(logs_root, run_id, "metrics.json")
    if not os.path.exists(path):
        print(f"WARNING: no metrics.json found for run_id '{run_id}' at {path}; skipping.")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def last_scalar_value(metrics: dict, tag: str):
    scalars = metrics.get("scalars", {})
    if tag not in scalars or not scalars[tag]:
        return None
    return scalars[tag][-1]["value"]


def extract_row(metrics: dict, run_id: str) -> dict:
    row = {"run_id": run_id}
    final = metrics.get("final", {})
    for key in KEY_METRICS:
        row[key] = final.get(key, last_scalar_value(metrics, key))
    return row


def print_table(rows: list):
    if not rows:
        print("No data to display.")
        return
    columns = ["run_id"] + KEY_METRICS
    widths = {c: max(len(c), max((len(f"{r.get(c, '')}") for r in rows), default=0)) for c in columns}

    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(f"{r.get(c, ''):>{widths[c]}}" if r.get(c) is not None
                           else " " * widths[c] for c in columns)
        print(line)


def save_csv(rows: list, out_path: str):
    import csv
    columns = ["run_id"] + KEY_METRICS
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved comparison table to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare metrics across run_ids.")
    parser.add_argument("--run_ids", nargs="+", required=True)
    parser.add_argument("--logs_root", default="logs")
    parser.add_argument("--out", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for run_id in args.run_ids:
        metrics = load_metrics(args.logs_root, run_id)
        if metrics:
            rows.append(extract_row(metrics, run_id))

    print_table(rows)
    if args.out:
        save_csv(rows, args.out)


if __name__ == "__main__":
    main()
