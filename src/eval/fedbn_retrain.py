"""
FedBN-specific evaluation for exact retraining.

Retained hospitals use their own personalized BatchNorm models.
The forgotten hospital uses a fallback model built only from retained
hospital BN states.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES
from src.eval.metrics import (
    compute_mia_accuracy,
    compute_relearn_accuracy,
)


def build_hospital_model_map(
    per_client_models: Dict[str, nn.Module],
    client_partitions,
    logger=None,
) -> Dict[str, nn.Module]:
    """
    Convert client_id -> model into hospital -> model.

    For simulated splits with multiple clients per hospital, the
    alphabetically first client is selected. This matches the existing
    representative-client behavior used by the Phase-1 FedBN evaluation.
    """
    hospital_to_client_ids: Dict[str, List[str]] = {}

    for partition in client_partitions:
        if partition.client_id not in per_client_models:
            continue

        hospital_to_client_ids.setdefault(
            partition.hospital,
            [],
        ).append(partition.client_id)

    hospital_models: Dict[str, nn.Module] = {}

    for hospital, client_ids in hospital_to_client_ids.items():
        selected_client_id = sorted(client_ids)[0]
        hospital_models[hospital] = per_client_models[selected_client_id]

        if logger is not None:
            logger.info(
                f"[fedbn/retrain] hospital={hospital}, "
                f"selected_client_id={selected_client_id}, "
                f"available_client_ids={sorted(client_ids)}"
            )

    return hospital_models


@torch.no_grad()
def evaluate_hospital_routed_models(
    hospital_models: Dict[str, nn.Module],
    fallback_model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    """
    Evaluate each sample using its hospital-specific FedBN model.

    When a hospital has no personalized model, use fallback_model.
    """
    fallback_model = fallback_model.to(device)
    fallback_model.eval()

    for model in hospital_models.values():
        model.to(device)
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    per_hospital_correct: Dict[str, int] = {}
    per_hospital_total: Dict[str, int] = {}
    per_hospital_loss: Dict[str, float] = {}

    y_true: List[int] = []
    y_pred: List[int] = []
    image_paths: List[str] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        hospitals = [str(value) for value in batch["hospital"]]

        logits = None

        for hospital in sorted(set(hospitals)):
            indices_list = [
                index
                for index, batch_hospital in enumerate(hospitals)
                if batch_hospital == hospital
            ]

            indices = torch.tensor(
                indices_list,
                dtype=torch.long,
                device=device,
            )

            selected_model = hospital_models.get(
                hospital,
                fallback_model,
            )

            hospital_logits = selected_model(
                images.index_select(0, indices)
            )

            if logits is None:
                logits = torch.empty(
                    labels.size(0),
                    hospital_logits.size(1),
                    dtype=hospital_logits.dtype,
                    device=device,
                )

            logits.index_copy_(
                0,
                indices,
                hospital_logits,
            )

        if logits is None:
            continue

        losses = F.cross_entropy(
            logits,
            labels,
            reduction="none",
        )

        predictions = logits.argmax(dim=1)

        total_loss += losses.sum().item()
        total_correct += (
            predictions == labels
        ).sum().item()
        total_samples += labels.size(0)

        for index, hospital in enumerate(hospitals):
            per_hospital_total[hospital] = (
                per_hospital_total.get(hospital, 0) + 1
            )

            per_hospital_correct[hospital] = (
                per_hospital_correct.get(hospital, 0)
                + int(predictions[index] == labels[index])
            )

            per_hospital_loss[hospital] = (
                per_hospital_loss.get(hospital, 0.0)
                + float(losses[index].item())
            )

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

        paths = batch.get("image_path")

        if paths is None:
            image_paths.extend([""] * labels.size(0))
        elif isinstance(paths, (list, tuple)):
            image_paths.extend([str(path) for path in paths])
        else:
            image_paths.extend([str(paths)] * labels.size(0))

    per_hospital = {
        hospital: {
            "acc": (
                per_hospital_correct[hospital]
                / max(1, per_hospital_total[hospital])
            ),
            "loss": (
                per_hospital_loss[hospital]
                / max(1, per_hospital_total[hospital])
            ),
            "n": per_hospital_total[hospital],
        }
        for hospital in per_hospital_total
    }

    return {
        "overall": {
            "acc": total_correct / max(1, total_samples),
            "loss": total_loss / max(1, total_samples),
            "n": total_samples,
        },
        "per_hospital": per_hospital,
        "y_true": y_true,
        "y_pred": y_pred,
        "image_paths": image_paths,
    }


def _log_global_results(
    results: Dict,
    num_classes: int,
    logger,
    step: int,
) -> Dict:
    overall = results["overall"]

    logger.log_scalar(
        "eval/overall/acc",
        overall["acc"],
        step,
    )
    logger.log_scalar(
        "eval/overall/loss",
        overall["loss"],
        step,
    )

    logger.set_final_metric(
        "eval/overall/acc",
        overall["acc"],
    )
    logger.set_final_metric(
        "eval/overall/loss",
        overall["loss"],
    )

    for hospital, values in results["per_hospital"].items():
        logger.log_scalar(
            f"eval/per_hospital/{hospital}/acc",
            values["acc"],
            step,
        )
        logger.log_scalar(
            f"eval/per_hospital/{hospital}/loss",
            values["loss"],
            step,
        )

        logger.set_final_metric(
            f"eval/per_hospital/{hospital}/acc",
            values["acc"],
        )
        logger.set_final_metric(
            f"eval/per_hospital/{hospital}/loss",
            values["loss"],
        )

    classification = logger.log_classification_results(
        tag="eval/classification",
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        image_paths=results["image_paths"],
        class_names=CLASS_NAMES[:num_classes],
        step=step,
    )

    for metric_name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ):
        logger.set_final_metric(
            f"eval/classification/{metric_name}",
            classification[metric_name],
        )

    for class_name, class_metrics in classification[
        "per_class"
    ].items():
        logger.set_final_metric(
            f"eval/per_class/{class_name}/precision",
            class_metrics["precision"],
        )
        logger.set_final_metric(
            f"eval/per_class/{class_name}/recall",
            class_metrics["recall"],
        )
        logger.set_final_metric(
            f"eval/per_class/{class_name}/f1",
            class_metrics["f1"],
        )
        logger.set_final_metric(
            f"eval/per_class/{class_name}/acc",
            class_metrics["recall"],
        )

    hospital_summary = ", ".join(
        f"{hospital}: {values['acc']:.4f}"
        for hospital, values in results["per_hospital"].items()
    )

    logger.info(
        f"[eval/fedbn_retrain] "
        f"overall_acc={overall['acc']:.4f} "
        f"overall_loss={overall['loss']:.4f} "
        f"macro_f1={classification['macro_f1']:.4f} "
        f"weighted_f1={classification['weighted_f1']:.4f} "
        f"per_hospital={{ {hospital_summary} }}"
    )

    return classification


def run_fedbn_retrain_evaluation(
    hospital_models: Dict[str, nn.Module],
    fallback_model: nn.Module,
    global_test_loader: DataLoader,
    remember_test_loader: DataLoader,
    forget_test_loader: DataLoader,
    device: str,
    num_classes: int,
    logger,
    step: int,
    relearn_steps: int = 50,
    relearn_lr: float = 1e-3,
    member_loader: Optional[DataLoader] = None,
    nonmember_loader: Optional[DataLoader] = None,
) -> Dict:
    """
    Run standard and unlearning evaluation for a FedBN retrain baseline.
    """
    logger.info(
        "[fedbn/retrain] Retained hospitals use personalized BN models. "
        "The forgotten hospital uses the retained-BN average model."
    )

    global_results = evaluate_hospital_routed_models(
        hospital_models=hospital_models,
        fallback_model=fallback_model,
        loader=global_test_loader,
        device=device,
    )

    global_classification = _log_global_results(
        results=global_results,
        num_classes=num_classes,
        logger=logger,
        step=step,
    )

    remember_results = evaluate_hospital_routed_models(
        hospital_models=hospital_models,
        fallback_model=fallback_model,
        loader=remember_test_loader,
        device=device,
    )

    forget_results = evaluate_hospital_routed_models(
        hospital_models={},
        fallback_model=fallback_model,
        loader=forget_test_loader,
        device=device,
    )

    remember_classification = logger.log_classification_results(
        tag="eval/unlearning/remember_classification",
        y_true=remember_results["y_true"],
        y_pred=remember_results["y_pred"],
        image_paths=remember_results["image_paths"],
        class_names=CLASS_NAMES[:num_classes],
        step=step,
    )

    forget_classification = logger.log_classification_results(
        tag="eval/unlearning/forget_classification",
        y_true=forget_results["y_true"],
        y_pred=forget_results["y_pred"],
        image_paths=forget_results["image_paths"],
        class_names=CLASS_NAMES[:num_classes],
        step=step,
    )

    ra = remember_results["overall"]["acc"]
    fa = forget_results["overall"]["acc"]

    rea = compute_relearn_accuracy(
        model=fallback_model,
        forget_loader=forget_test_loader,
        device=device,
        relearn_steps=relearn_steps,
        learning_rate=relearn_lr,
    )

    mia_member_loader = (
        member_loader
        if member_loader is not None
        else forget_test_loader
    )

    mia_nonmember_loader = (
        nonmember_loader
        if nonmember_loader is not None
        else remember_test_loader
    )

    mia_acc = compute_mia_accuracy(
        model=fallback_model,
        member_loader=mia_member_loader,
        nonmember_loader=mia_nonmember_loader,
        device=device,
    )

    final_metrics = {
        "eval/unlearning/RA": ra,
        "eval/unlearning/FA": fa,
        "eval/unlearning/ReA": rea,
        "eval/unlearning/MIA_acc": mia_acc,
        "eval/unlearning/remember_macro_f1": (
            remember_classification["macro_f1"]
        ),
        "eval/unlearning/forget_macro_f1": (
            forget_classification["macro_f1"]
        ),
        "eval/unlearning/remember_weighted_f1": (
            remember_classification["weighted_f1"]
        ),
        "eval/unlearning/forget_weighted_f1": (
            forget_classification["weighted_f1"]
        ),
    }

    for metric_name, value in final_metrics.items():
        logger.log_scalar(metric_name, value, step)
        logger.set_final_metric(metric_name, value)

    logger.info(
        f"[eval/unlearning/fedbn_retrain] "
        f"RA={ra:.4f} "
        f"FA={fa:.4f} "
        f"ReA={rea:.4f} "
        f"MIA_acc={mia_acc:.4f}"
    )

    return {
        "global": global_results,
        "global_classification": global_classification,
        "remember": remember_results,
        "remember_classification": remember_classification,
        "forget": forget_results,
        "forget_classification": forget_classification,
        "RA": ra,
        "FA": fa,
        "ReA": rea,
        "MIA_acc": mia_acc,
    }