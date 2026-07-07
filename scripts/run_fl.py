#!/usr/bin/env python3
"""
Phase 1: Federated Learning training ONLY.

This script:
  1. Loads + resolves the YAML config, then applies any --set overrides.
  2. Indexes the dataset and builds per-hospital or simulated client partitions.
  3. Runs the Flower FL simulation for the configured algorithm.
  4. Evaluates the final model.
  5. Saves checkpoint(s), config snapshot, and metrics.

For FedBN/domain_adaptation=True, final evaluation is done using each
hospital's own BN-complete model:
  Spain test  -> Spain BN model
  Canada test -> Canada BN model
  India test  -> India BN model

This fixed version also saves FedBN classification outputs:
  - classification_report_eval_classification.json
  - confusion_matrix_eval_classification.csv
  - predictions.csv

and fills metrics.json:
  - final
  - classification
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import OralCancerDataset, index_dataset
from src.data.partition import build_client_partitions, partitions_to_datasets
from src.data.transforms import build_transforms
from src.fl.simulation import run_federated_learning
from src.utils.checkpoint import save_checkpoint
from src.utils.config import (
    apply_overrides,
    load_config,
    make_run_id,
    parse_set_overrides,
    resolve_run_dirs,
    save_config_snapshot,
)
from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: FL training only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_fl.py --config configs/fl_fedavg.yaml

  python scripts/run_fl.py --config configs/fl_fedbn.yaml \\
      --set global_epochs=10 \\
      --set batch_size=32

  python scripts/run_fl.py --config configs/fl_fedavg.yaml \\
      --set dataset_path=/data/oral_cancer \\
      --set device=cpu
""",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to FL config YAML.",
    )

    parser.add_argument(
        "--run_id",
        default=None,
        help="Override the auto-generated run_id.",
    )

    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a config value, e.g. --set global_epochs=10. "
            "Repeatable. Values are parsed as YAML scalars."
        ),
    )

    return parser.parse_args()


def get_class_names(config: Dict) -> List[str]:
    """
    Return class names in the expected label order.
    Falls back to the oral-cancer 3-class names if not provided in config.
    """
    if "class_names" in config and config["class_names"]:
        return list(config["class_names"])

    return [
        "Benign",
        "Potentially_Malignant",
        "Malignant",
    ]


def safe_class_name(class_names: List[str], class_id: int) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def build_classification_dict(
    *,
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[List[float]],
    class_names: List[str],
    tag: str = "eval/classification",
) -> Dict:
    """
    Build a classification dictionary similar to the logger's standard format.
    """
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}

    for class_id, class_name in enumerate(class_names):
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)

        class_report = report.get(class_name, {})

        per_class[class_name] = {
            "class_id": class_id,
            "precision": float(class_report.get("precision", 0.0)),
            "recall": float(class_report.get("recall", 0.0)),
            "f1": float(class_report.get("f1-score", 0.0)),
            "support": int(class_report.get("support", 0)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    accuracy = float(report.get("accuracy", 0.0))
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    return {
        "tag": tag,
        "num_samples": len(y_true),
        "accuracy": accuracy,
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1-score", 0.0)),
        "weighted_precision": float(weighted.get("precision", 0.0)),
        "weighted_recall": float(weighted.get("recall", 0.0)),
        "weighted_f1": float(weighted.get("f1-score", 0.0)),
        "labels": labels,
        "class_names": class_names,
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def save_classification_artifacts(
    *,
    log_dir: str,
    classification: Dict,
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[List[float]],
    hospitals: List[str],
    class_names: List[str],
):
    """
    Save the same important files produced by the normal evaluation path:
      - classification_report_eval_classification.json
      - confusion_matrix_eval_classification.csv
      - predictions.csv
    """
    os.makedirs(log_dir, exist_ok=True)

    report_path = os.path.join(
        log_dir,
        "classification_report_eval_classification.json",
    )
    confusion_path = os.path.join(
        log_dir,
        "confusion_matrix_eval_classification.csv",
    )
    predictions_path = os.path.join(log_dir, "predictions.csv")

    with open(report_path, "w") as f:
        json.dump(classification, f, indent=2)

    cm = classification["confusion_matrix"]

    with open(confusion_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"pred_{name}" for name in class_names])

        for class_name, row in zip(class_names, cm):
            writer.writerow([f"true_{class_name}"] + row)

    with open(predictions_path, "w", newline="") as f:
        fieldnames = [
            "index",
            "hospital",
            "y_true",
            "y_true_name",
            "y_pred",
            "y_pred_name",
            "correct",
        ] + [f"prob_{name}" for name in class_names]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (true_label, pred_label, probs, hospital) in enumerate(
            zip(y_true, y_pred, y_prob, hospitals)
        ):
            row = {
                "index": idx,
                "hospital": hospital,
                "y_true": true_label,
                "y_true_name": safe_class_name(class_names, true_label),
                "y_pred": pred_label,
                "y_pred_name": safe_class_name(class_names, pred_label),
                "correct": int(true_label == pred_label),
            }

            for class_id, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = (
                    float(probs[class_id]) if class_id < len(probs) else 0.0
                )

            writer.writerow(row)


def build_final_metrics_for_fedbn(
    *,
    overall_acc: float,
    overall_loss: float,
    per_hospital_results: Dict,
    classification: Dict,
) -> Dict[str, float]:
    """
    Build the normal 'final' section expected by show_final_results.py.
    """
    final = {
        "eval/overall/acc": float(overall_acc),
        "eval/overall/loss": float(overall_loss),
        "eval/classification/accuracy": float(classification["accuracy"]),
        "eval/classification/macro_precision": float(
            classification["macro_precision"]
        ),
        "eval/classification/macro_recall": float(
            classification["macro_recall"]
        ),
        "eval/classification/macro_f1": float(classification["macro_f1"]),
        "eval/classification/weighted_precision": float(
            classification["weighted_precision"]
        ),
        "eval/classification/weighted_recall": float(
            classification["weighted_recall"]
        ),
        "eval/classification/weighted_f1": float(
            classification["weighted_f1"]
        ),
    }

    for hospital, values in per_hospital_results.items():
        final[f"eval/per_hospital/{hospital}/acc"] = float(values["acc"])
        final[f"eval/per_hospital/{hospital}/loss"] = float(values["loss"])

        final[f"eval/fedbn_per_hospital/{hospital}/acc"] = float(values["acc"])
        final[f"eval/fedbn_per_hospital/{hospital}/loss"] = float(values["loss"])

    final["eval/fedbn_weighted_overall/acc"] = float(overall_acc)
    final["eval/fedbn_weighted_overall/loss"] = float(overall_loss)

    for class_name, metrics in classification["per_class"].items():
        final[f"eval/per_class/{class_name}/acc"] = float(metrics["recall"])
        final[f"eval/per_class/{class_name}/precision"] = float(
            metrics["precision"]
        )
        final[f"eval/per_class/{class_name}/recall"] = float(
            metrics["recall"]
        )
        final[f"eval/per_class/{class_name}/f1"] = float(metrics["f1"])

    return final


def inject_fedbn_results_into_logger(
    *,
    logger,
    final_metrics: Dict[str, float],
    classification: Dict,
):
    """
    Best-effort injection into the logger object before logger.close().
    This supports loggers that store metrics in logger.metrics.
    """
    if hasattr(logger, "metrics") and isinstance(logger.metrics, dict):
        logger.metrics.setdefault("final", {})
        logger.metrics["final"].update(final_metrics)

        logger.metrics.setdefault("classification", {})
        logger.metrics["classification"]["eval/classification"] = classification


def patch_metrics_json_after_close(
    *,
    log_dir: str,
    final_metrics: Dict[str, float],
    classification: Dict,
):
    """
    Guaranteed post-close patch.

    Some logger implementations only write metrics.json during close().
    This function reopens metrics.json after logger.close() and fills the
    final/classification sections for FedBN.
    """
    metrics_path = os.path.join(log_dir, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "run_id": os.path.basename(log_dir),
            "scalars": {},
            "final": {},
            "classification": {},
        }

    data.setdefault("final", {})
    data["final"].update(final_metrics)

    data.setdefault("classification", {})
    data["classification"]["eval/classification"] = classification

    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=2)


def log_fedbn_classification_scalars(
    *,
    logger,
    classification: Dict,
    step: int,
):
    """
    Log classification scalars to TensorBoard/metrics scalars.
    """
    logger.log_scalar(
        "eval/classification/accuracy",
        classification["accuracy"],
        step,
    )
    logger.log_scalar(
        "eval/classification/macro_precision",
        classification["macro_precision"],
        step,
    )
    logger.log_scalar(
        "eval/classification/macro_recall",
        classification["macro_recall"],
        step,
    )
    logger.log_scalar(
        "eval/classification/macro_f1",
        classification["macro_f1"],
        step,
    )
    logger.log_scalar(
        "eval/classification/weighted_precision",
        classification["weighted_precision"],
        step,
    )
    logger.log_scalar(
        "eval/classification/weighted_recall",
        classification["weighted_recall"],
        step,
    )
    logger.log_scalar(
        "eval/classification/weighted_f1",
        classification["weighted_f1"],
        step,
    )

    for class_name, metrics in classification["per_class"].items():
        logger.log_scalar(
            f"eval/per_class/{class_name}/precision",
            metrics["precision"],
            step,
        )
        logger.log_scalar(
            f"eval/per_class/{class_name}/recall",
            metrics["recall"],
            step,
        )
        logger.log_scalar(
            f"eval/per_class/{class_name}/f1",
            metrics["f1"],
            step,
        )
        logger.log_scalar(
            f"eval/per_class/{class_name}/acc",
            metrics["recall"],
            step,
        )


def evaluate_fedbn_per_hospital(
    *,
    per_hospital_models,
    test_samples,
    eval_transform,
    config,
    global_hospital_to_idx,
    device,
    logger,
    log_dir,
):
    """
    Correct FedBN evaluation.

    Each hospital test split must be evaluated using the model that contains
    that hospital's own BatchNorm statistics.

    This fixed version also collects all predictions and computes:
      - precision
      - recall
      - F1
      - macro/weighted metrics
      - confusion matrix
      - predictions.csv
    """
    logger.info("Running FedBN final evaluation with hospital-specific BN models.")

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    class_names = get_class_names(config)
    num_classes = config["num_classes"]

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    per_hospital_results = {}

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_y_prob: List[List[float]] = []
    all_hospitals: List[str] = []

    for hospital in config["hospitals"]:
        if hospital not in per_hospital_models:
            logger.info(
                f"[eval/fedbn] WARNING: no BN-complete model found for "
                f"hospital={hospital}; skipping."
            )
            continue

        hospital_samples = [s for s in test_samples if s.hospital == hospital]

        hospital_dataset = OralCancerDataset(
            hospital_samples,
            transform=eval_transform,
            load_metadata=config["load_metadata"],
            hospital_to_idx=global_hospital_to_idx,
        )

        hospital_loader = DataLoader(
            hospital_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
        )

        model = per_hospital_models[hospital].to(device)
        model.eval()

        hospital_correct = 0
        hospital_total = 0
        hospital_loss = 0.0

        with torch.no_grad():
            for batch in hospital_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                hospital_correct += (preds == labels).sum().item()
                hospital_total += labels.numel()
                hospital_loss += loss.item()

                batch_y_true = labels.detach().cpu().tolist()
                batch_y_pred = preds.detach().cpu().tolist()
                batch_y_prob = probs.detach().cpu().tolist()

                all_y_true.extend(int(x) for x in batch_y_true)
                all_y_pred.extend(int(x) for x in batch_y_pred)
                all_y_prob.extend(batch_y_prob)
                all_hospitals.extend([hospital] * len(batch_y_true))

        hospital_acc = hospital_correct / max(1, hospital_total)
        hospital_avg_loss = hospital_loss / max(1, hospital_total)

        per_hospital_results[hospital] = {
            "acc": hospital_acc,
            "loss": hospital_avg_loss,
            "n": hospital_total,
        }

        logger.info(
            f"[eval/fedbn_per_hospital] hospital={hospital} "
            f"acc={hospital_acc:.4f} loss={hospital_avg_loss:.4f} "
            f"n={hospital_total}"
        )

        logger.log_scalar(
            f"eval/fedbn_per_hospital/{hospital}/acc",
            hospital_acc,
            config["global_epochs"],
        )

        logger.log_scalar(
            f"eval/fedbn_per_hospital/{hospital}/loss",
            hospital_avg_loss,
            config["global_epochs"],
        )

        total_correct += hospital_correct
        total_samples += hospital_total
        total_loss += hospital_loss

    overall_acc = total_correct / max(1, total_samples)
    overall_loss = total_loss / max(1, total_samples)

    logger.info(
        f"[eval/fedbn_weighted_overall] overall_acc={overall_acc:.4f} "
        f"overall_loss={overall_loss:.4f} total_samples={total_samples}"
    )

    logger.log_scalar(
        "eval/fedbn_weighted_overall/acc",
        overall_acc,
        config["global_epochs"],
    )

    logger.log_scalar(
        "eval/fedbn_weighted_overall/loss",
        overall_loss,
        config["global_epochs"],
    )

    logger.log_scalar(
        "eval/overall/acc",
        overall_acc,
        config["global_epochs"],
    )

    logger.log_scalar(
        "eval/overall/loss",
        overall_loss,
        config["global_epochs"],
    )

    classification = build_classification_dict(
        y_true=all_y_true,
        y_pred=all_y_pred,
        y_prob=all_y_prob,
        class_names=class_names[:num_classes],
        tag="eval/classification",
    )

    log_fedbn_classification_scalars(
        logger=logger,
        classification=classification,
        step=config["global_epochs"],
    )

    save_classification_artifacts(
        log_dir=log_dir,
        classification=classification,
        y_true=all_y_true,
        y_pred=all_y_pred,
        y_prob=all_y_prob,
        hospitals=all_hospitals,
        class_names=class_names[:num_classes],
    )

    logger.info(
        "[eval/classification] "
        f"accuracy={classification['accuracy']:.4f} "
        f"macro_precision={classification['macro_precision']:.4f} "
        f"macro_recall={classification['macro_recall']:.4f} "
        f"macro_f1={classification['macro_f1']:.4f} "
        f"weighted_f1={classification['weighted_f1']:.4f}"
    )

    final_metrics = build_final_metrics_for_fedbn(
        overall_acc=overall_acc,
        overall_loss=overall_loss,
        per_hospital_results=per_hospital_results,
        classification=classification,
    )

    inject_fedbn_results_into_logger(
        logger=logger,
        final_metrics=final_metrics,
        classification=classification,
    )

    return {
        "overall_acc": overall_acc,
        "overall_loss": overall_loss,
        "total_samples": total_samples,
        "per_hospital": per_hospital_results,
        "classification": classification,
        "final_metrics": final_metrics,
    }


def main():
    args = parse_args()

    config = load_config(args.config)
    overrides = parse_set_overrides(args.set_overrides)
    config = apply_overrides(config, overrides)

    if overrides:
        print(f"Applied --set overrides: {overrides}")

    dataset_tag = "oralcancer"

    run_id = args.run_id or make_run_id(
        phase="fl",
        algorithm=config["algorithm"],
        backbone=config["model"],
        dataset_tag=dataset_tag,
    )

    dirs = resolve_run_dirs(
        run_id,
        config["logs_root"],
        config["checkpoints_root"],
        config["outputs_root"],
    )

    save_config_snapshot(
        config,
        os.path.join(dirs["log_dir"], "config.snapshot.yaml"),
    )

    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])

    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Run directories: {dirs}")

    device = (
        config["device"]
        if torch.cuda.is_available() and config["device"] == "cuda"
        else "cpu"
    )

    if device != config["device"]:
        logger.info(
            f"Requested device '{config['device']}' unavailable; falling back to CPU."
        )

    torch.manual_seed(config["seed"])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Indexing dataset...")

    train_samples = index_dataset(
        config["dataset_path"],
        "Train",
        config["hospitals"],
    )

    test_samples = index_dataset(
        config["dataset_path"],
        "Test",
        config["hospitals"],
    )

    logger.info(
        f"Indexed {len(train_samples)} train samples, "
        f"{len(test_samples)} test samples."
    )

    global_hospital_to_idx = {
        hospital: idx for idx, hospital in enumerate(config["hospitals"])
    }

    client_partitions = build_client_partitions(
        train_samples,
        config["hospitals"],
        config["client_split"],
        config.get("clients_per_hospital", 1),
        config["seed"],
    )

    logger.info(
        f"Built {len(client_partitions)} client partition(s) "
        f"(client_split={config['client_split']})."
    )

    for partition in client_partitions:
        logger.info(
            f"  client_id={partition.client_id} "
            f"hospital={partition.hospital} "
            f"n_samples={len(partition.samples)}"
        )

    train_transform = build_transforms(
        config["image_size"],
        train=True,
        augmentation=config["augmentation"],
    )

    eval_transform = build_transforms(
        config["image_size"],
        train=False,
    )

    train_datasets = partitions_to_datasets(
        client_partitions,
        train_transform,
        config["load_metadata"],
        global_hospital_to_idx,
    )

    val_partitions = build_client_partitions(
        test_samples,
        config["hospitals"],
        config["client_split"],
        config.get("clients_per_hospital", 1),
        config["seed"],
    )

    val_datasets = partitions_to_datasets(
        val_partitions,
        eval_transform,
        config["load_metadata"],
        global_hospital_to_idx,
    )

    global_test_dataset = OralCancerDataset(
        test_samples,
        transform=eval_transform,
        load_metadata=config["load_metadata"],
        hospital_to_idx=global_hospital_to_idx,
    )

    global_test_loader = DataLoader(
        global_test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # ------------------------------------------------------------------
    # FL training
    # ------------------------------------------------------------------
    trained_model, per_hospital_models = run_federated_learning(
        config=config,
        client_partitions=client_partitions,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        device=device,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    from src.eval.evaluator import run_standard_evaluation

    fedbn_eval_results: Optional[Dict] = None

    if config.get("domain_adaptation", False):
        fedbn_eval_results = evaluate_fedbn_per_hospital(
            per_hospital_models=per_hospital_models,
            test_samples=test_samples,
            eval_transform=eval_transform,
            config=config,
            global_hospital_to_idx=global_hospital_to_idx,
            device=device,
            logger=logger,
            log_dir=dirs["log_dir"],
        )
    else:
        run_standard_evaluation(
            trained_model,
            global_test_loader,
            device,
            config["num_classes"],
            logger,
            step=config["global_epochs"],
            tag_prefix="eval",
        )

    # ------------------------------------------------------------------
    # Save checkpoint(s)
    # ------------------------------------------------------------------
    checkpoint_path = save_checkpoint(
        trained_model,
        dirs["checkpoint_dir"],
        "best",
        extra={
            "config": config,
            "run_id": run_id,
        },
    )

    logger.info(f"Saved final global model checkpoint to {checkpoint_path}")

    if config.get("domain_adaptation", False):
        per_hospital_dir = os.path.join(
            dirs["checkpoint_dir"],
            "per_hospital",
        )

        os.makedirs(per_hospital_dir, exist_ok=True)

        for client_id, model in per_hospital_models.items():
            save_checkpoint(
                model,
                per_hospital_dir,
                client_id,
            )

        logger.info(
            f"Saved {len(per_hospital_models)} per-hospital BN-complete "
            f"checkpoints to {per_hospital_dir}"
        )

    logger.close()

    # Guarantee that FedBN metrics.json contains final/classification even if
    # the logger only writes metrics.json during close().
    if fedbn_eval_results is not None:
        patch_metrics_json_after_close(
            log_dir=dirs["log_dir"],
            final_metrics=fedbn_eval_results["final_metrics"],
            classification=fedbn_eval_results["classification"],
        )

    print(f"\nDone. run_id = {run_id}")
    print("Use this run_id as --source_run for scripts/run_fu.py.")


if __name__ == "__main__":
    main()