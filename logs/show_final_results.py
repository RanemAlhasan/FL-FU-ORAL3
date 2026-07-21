#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set


def usage() -> None:
    print(
        "Usage: python3 show_final_results.py "
        "<run_folder_or_metrics_json>"
    )


if len(sys.argv) != 2:
    usage()
    sys.exit(1)


input_path = Path(sys.argv[1])
path = input_path / "metrics.json" if input_path.is_dir() else input_path

if not path.exists():
    print(f"Error: metrics.json not found at: {path}")
    sys.exit(1)

try:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
except json.JSONDecodeError as exc:
    print(f"Error: invalid JSON in {path}")
    print(exc)
    sys.exit(1)
except OSError as exc:
    print(f"Error reading {path}: {exc}")
    sys.exit(1)


run_id = data.get("run_id", path.parent.name)
final: Dict[str, Any] = data.get("final", {}) or {}
classification_sections: Dict[str, Any] = (
    data.get("classification", {}) or {}
)
scalars: Dict[str, Any] = data.get("scalars", {}) or {}

printed_final_keys: Set[str] = set()


def latest_scalar(key: str) -> Optional[float]:
    values = scalars.get(key, [])

    if not values:
        return None

    last_value = values[-1]

    if isinstance(last_value, dict):
        return last_value.get("value")

    if isinstance(last_value, (int, float)):
        return float(last_value)

    return None


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def print_percentage_metric(
    name: str,
    value: Optional[float],
) -> None:
    if value is None:
        return

    print(
        f"{name:24s}: "
        f"{value:.4f}  ({value * 100:.2f}%)"
    )


def print_loss_metric(
    name: str,
    value: Optional[float],
) -> None:
    if value is None:
        return

    print(f"{name:24s}: {value:.4f}")


def print_confusion_matrix(matrix: Any) -> None:
    if not matrix:
        return

    print("\nConfusion Matrix:")

    for row in matrix:
        print("  " + "  ".join(f"{int(value):5d}" for value in row))


def print_per_class(per_class: Dict[str, Any]) -> None:
    if not per_class:
        print("No per-class results found.")
        return

    for class_name, metrics in per_class.items():
        print(
            f"{class_name:25s} "
            f"Precision: {metrics.get('precision', 0.0):.4f}  "
            f"Recall: {metrics.get('recall', 0.0):.4f}  "
            f"F1: {metrics.get('f1', 0.0):.4f}  "
            f"Support: {metrics.get('support', 0)}"
        )


def reconstruct_per_class_from_final() -> Dict[str, Dict[str, Any]]:
    reconstructed: Dict[str, Dict[str, Any]] = {}

    prefix = "eval/per_class/"

    for key, value in final.items():
        if not key.startswith(prefix):
            continue

        parts = key.split("/")

        if len(parts) != 5:
            continue

        class_name = parts[3]
        metric_name = parts[4]

        reconstructed.setdefault(class_name, {})[metric_name] = value
        printed_final_keys.add(key)

    return reconstructed


def print_classification_section(
    tag: str,
    title: str,
) -> None:
    section = classification_sections.get(tag)

    if not section:
        return

    print(f"\n--- {title} ---")

    num_samples = section.get("num_samples")
    if num_samples is not None:
        print(f"Samples                 : {num_samples}")

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted Precision", "weighted_precision"),
        ("Weighted Recall", "weighted_recall"),
        ("Weighted F1", "weighted_f1"),
    ]

    for display_name, metric_name in metric_names:
        value = section.get(metric_name)
        print_percentage_metric(display_name, value)

    print("\nPer Class:")
    print_per_class(section.get("per_class", {}))

    print_confusion_matrix(section.get("confusion_matrix"))


print("\n========== FINAL RESULTS ==========\n")
print(f"Run ID: {run_id}")
print(f"Metrics file: {path}")


# ============================================================
# Current normal format
# ============================================================

if final:
    overall_classification = classification_sections.get(
        "eval/classification",
        {},
    )

    samples = overall_classification.get("num_samples", "N/A")
    print(f"Samples: {samples}")

    print("\n--- Overall ---")

    overall_metrics = [
        ("Accuracy", "eval/overall/acc", False),
        ("Loss", "eval/overall/loss", True),
        (
            "Macro Precision",
            "eval/classification/macro_precision",
            False,
        ),
        (
            "Macro Recall",
            "eval/classification/macro_recall",
            False,
        ),
        (
            "Macro F1",
            "eval/classification/macro_f1",
            False,
        ),
        (
            "Weighted Precision",
            "eval/classification/weighted_precision",
            False,
        ),
        (
            "Weighted Recall",
            "eval/classification/weighted_recall",
            False,
        ),
        (
            "Weighted F1",
            "eval/classification/weighted_f1",
            False,
        ),
    ]

    for display_name, key, is_loss in overall_metrics:
        value = final.get(key)

        if value is None:
            continue

        printed_final_keys.add(key)

        if is_loss:
            print_loss_metric(display_name, value)
        else:
            print_percentage_metric(display_name, value)

    print("\n--- Per Hospital ---")

    hospital_names = sorted(
        {
            key.split("/")[2]
            for key in final
            if key.startswith("eval/per_hospital/")
            and key.endswith("/acc")
        }
    )

    if hospital_names:
        for hospital in hospital_names:
            acc_key = f"eval/per_hospital/{hospital}/acc"
            loss_key = f"eval/per_hospital/{hospital}/loss"

            accuracy = final.get(acc_key)
            loss = final.get(loss_key)

            printed_final_keys.add(acc_key)

            if loss is not None:
                printed_final_keys.add(loss_key)
                print(
                    f"{hospital:24s} "
                    f"Acc: {accuracy:.4f} "
                    f"({accuracy * 100:.2f}%)   "
                    f"Loss: {loss:.4f}"
                )
            else:
                print(
                    f"{hospital:24s} "
                    f"Acc: {accuracy:.4f} "
                    f"({accuracy * 100:.2f}%)"
                )
    else:
        print("No hospital-level results found.")

    print("\n--- Overall Per Class ---")

    overall_per_class = overall_classification.get("per_class", {})

    if not overall_per_class:
        overall_per_class = reconstruct_per_class_from_final()

    print_per_class(overall_per_class)

    if overall_classification:
        print_confusion_matrix(
            overall_classification.get("confusion_matrix")
        )

    # ========================================================
    # Unlearning summary
    # ========================================================

    unlearning_metrics = [
        ("Retain Accuracy (RA)", "eval/unlearning/RA"),
        ("Forget Accuracy (FA)", "eval/unlearning/FA"),
        ("Relearn Accuracy (ReA)", "eval/unlearning/ReA"),
        (
            "MIA Accuracy",
            "eval/unlearning/MIA_acc",
        ),
        (
            "Remember Macro F1",
            "eval/unlearning/remember_macro_f1",
        ),
        (
            "Forget Macro F1",
            "eval/unlearning/forget_macro_f1",
        ),
        (
            "Remember Weighted F1",
            "eval/unlearning/remember_weighted_f1",
        ),
        (
            "Forget Weighted F1",
            "eval/unlearning/forget_weighted_f1",
        ),
    ]

    available_unlearning_metrics = [
        item
        for item in unlearning_metrics
        if final.get(item[1]) is not None
    ]

    if available_unlearning_metrics:
        print("\n--- Unlearning Evaluation ---")

        for display_name, key in available_unlearning_metrics:
            value = final.get(key)
            printed_final_keys.add(key)
            print_percentage_metric(display_name, value)

    # Full remember-set classification results
    print_classification_section(
        tag="eval/unlearning/remember_classification",
        title="Remember-Set Classification",
    )

    # Mark duplicated remember classification final keys as printed
    for metric_name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ):
        key = (
            "eval/unlearning/remember_classification/"
            f"{metric_name}"
        )
        if key in final:
            printed_final_keys.add(key)

    # Full forget-set classification results
    print_classification_section(
        tag="eval/unlearning/forget_classification",
        title="Forget-Set Classification",
    )

    # Mark duplicated forget classification final keys as printed
    for metric_name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ):
        key = (
            "eval/unlearning/forget_classification/"
            f"{metric_name}"
        )
        if key in final:
            printed_final_keys.add(key)

    # ========================================================
    # Print any future/new final metrics not handled above
    # ========================================================

    remaining_final_metrics = {
        key: value
        for key, value in final.items()
        if key not in printed_final_keys
        and not key.startswith("eval/per_class/")
    }

    if remaining_final_metrics:
        print("\n--- Additional Saved Final Metrics ---")

        for key in sorted(remaining_final_metrics):
            value = remaining_final_metrics[key]

            if isinstance(value, float):
                print(
                    f"{key:55s}: "
                    f"{value:.4f} "
                    f"({value * 100:.2f}%)"
                )
            else:
                print(f"{key:55s}: {format_value(value)}")


# ============================================================
# Older FedBN scalar-only format
# ============================================================

elif latest_scalar("eval/fedbn_weighted_overall/acc") is not None:
    accuracy = latest_scalar(
        "eval/fedbn_weighted_overall/acc"
    )
    loss = latest_scalar(
        "eval/fedbn_weighted_overall/loss"
    )

    print("Samples: N/A")

    print("\n--- FedBN Weighted Overall ---")
    print_percentage_metric("Accuracy", accuracy)
    print_loss_metric("Loss", loss)

    print("\n--- FedBN Per Hospital ---")

    hospitals = [
        "Spain_Dataset",
        "Canada_Dataset",
        "India_Dataset",
    ]

    for hospital in hospitals:
        hospital_accuracy = latest_scalar(
            f"eval/fedbn_per_hospital/{hospital}/acc"
        )
        hospital_loss = latest_scalar(
            f"eval/fedbn_per_hospital/{hospital}/loss"
        )

        if hospital_accuracy is None:
            continue

        if hospital_loss is not None:
            print(
                f"{hospital:24s} "
                f"Acc: {hospital_accuracy:.4f} "
                f"({hospital_accuracy * 100:.2f}%)   "
                f"Loss: {hospital_loss:.4f}"
            )
        else:
            print(
                f"{hospital:24s} "
                f"Acc: {hospital_accuracy:.4f} "
                f"({hospital_accuracy * 100:.2f}%)"
            )

    print(
        "\nNote: This run uses the older FedBN scalar-only "
        "metrics format."
    )


else:
    print("Samples: N/A")
    print(
        "\nNo final results were found in either the current "
        "format or the older FedBN scalar format."
    )


print("\n===================================\n")