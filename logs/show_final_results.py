#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python show_final_results.py <run_folder_or_metrics_json>")
    sys.exit(1)

input_path = Path(sys.argv[1])
path = input_path / "metrics.json" if input_path.is_dir() else input_path

if not path.exists():
    print(f"Error: metrics.json not found at: {path}")
    sys.exit(1)

with open(path, "r") as f:
    data = json.load(f)

run_id = data.get("run_id", path.parent.name)
final = data.get("final", {})
classification = data.get("classification", {}).get("eval/classification", {})
scalars = data.get("scalars", {})

def latest_scalar(key):
    values = scalars.get(key, [])
    if not values:
        return None
    return values[-1].get("value")

print("\n========== FINAL RESULTS ==========\n")
print(f"Run ID: {run_id}")
print(f"Samples: {classification.get('num_samples', 'N/A')}")

# Normal FedAvg/FedProx/FedMoon style
if final:
    print("\n--- Overall ---")
    metrics = [
        ("Accuracy", "eval/overall/acc"),
        ("Loss", "eval/overall/loss"),
        ("Macro Precision", "eval/classification/macro_precision"),
        ("Macro Recall", "eval/classification/macro_recall"),
        ("Macro F1", "eval/classification/macro_f1"),
        ("Weighted Precision", "eval/classification/weighted_precision"),
        ("Weighted Recall", "eval/classification/weighted_recall"),
        ("Weighted F1", "eval/classification/weighted_f1"),
    ]

    for name, key in metrics:
        value = final.get(key)
        if value is None:
            continue
        if "Loss" in name:
            print(f"{name:20s}: {value:.4f}")
        else:
            print(f"{name:20s}: {value:.4f}  ({value*100:.2f}%)")

    print("\n--- Per Hospital ---")
    for key, value in final.items():
        if key.startswith("eval/per_hospital/") and key.endswith("/acc"):
            hospital = key.split("/")[2]
            loss = final.get(f"eval/per_hospital/{hospital}/loss")
            if loss is not None:
                print(f"{hospital:20s} Acc: {value:.4f} ({value*100:.2f}%)   Loss: {loss:.4f}")
            else:
                print(f"{hospital:20s} Acc: {value:.4f} ({value*100:.2f}%)")

    print("\n--- Per Class ---")
    per_class = classification.get("per_class", {})
    for cls, m in per_class.items():
        print(
            f"{cls:25s} "
            f"Precision: {m.get('precision', 0):.4f}  "
            f"Recall: {m.get('recall', 0):.4f}  "
            f"F1: {m.get('f1', 0):.4f}  "
            f"Support: {m.get('support', 0)}"
        )

# FedBN style
elif latest_scalar("eval/fedbn_weighted_overall/acc") is not None:
    acc = latest_scalar("eval/fedbn_weighted_overall/acc")
    loss = latest_scalar("eval/fedbn_weighted_overall/loss")

    print("\n--- FedBN Weighted Overall ---")
    print(f"Accuracy            : {acc:.4f}  ({acc*100:.2f}%)")
    if loss is not None:
        print(f"Loss                : {loss:.4f}")

    print("\n--- FedBN Per Hospital ---")
    hospitals = ["Spain_Dataset", "Canada_Dataset", "India_Dataset"]
    for hospital in hospitals:
        h_acc = latest_scalar(f"eval/fedbn_per_hospital/{hospital}/acc")
        h_loss = latest_scalar(f"eval/fedbn_per_hospital/{hospital}/loss")
        if h_acc is not None:
            if h_loss is not None:
                print(f"{hospital:20s} Acc: {h_acc:.4f} ({h_acc*100:.2f}%)   Loss: {h_loss:.4f}")
            else:
                print(f"{hospital:20s} Acc: {h_acc:.4f} ({h_acc*100:.2f}%)")

    print("\nNote: FedBN results were saved under scalar keys, not under the normal 'final' section.")

else:
    print("\nNo final results found in either normal format or FedBN format.")

print("\n===================================\n")
