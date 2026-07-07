"""
High-level evaluation orchestration. Any phase (FL/FU/retrain) calls
`run_full_evaluation()` with whatever loaders are relevant to it, and gets
back a flat dict of namespaced metrics ready to log + persist. This keeps
metric NAMING identical across phases (see README "Metric naming
convention") regardless of which script produced them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES
from src.eval.metrics import (
    compute_mia_accuracy,
    compute_ra_fa,
    compute_relearn_accuracy,
    evaluate_overall,
    evaluate_per_class,
    evaluate_per_hospital,
)
from src.utils.logger import ExperimentLogger


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[List[int], List[int], List[str]]:
    """Collect labels, predictions, and image paths from a dataloader.

    This is evaluation-only and does not affect training. It is used to log
    precision, recall, F1-score, confusion matrix, and per-image predictions.
    """
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    image_paths: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            if "image_path" in batch:
                paths = batch["image_path"]
                if isinstance(paths, (list, tuple)):
                    image_paths.extend([str(p) for p in paths])
                else:
                    image_paths.extend([str(paths)] * labels.size(0))
            else:
                image_paths.extend([""] * labels.size(0))

    return y_true, y_pred, image_paths


def run_standard_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    num_classes: int,
    logger: ExperimentLogger,
    step: int,
    tag_prefix: str = "eval",
) -> Dict:
    """Overall + per-hospital + per-class metrics. Used by every phase after
    training completes (and optionally at intermediate checkpoints)."""
    overall = evaluate_overall(model, test_loader, device)
    per_hospital = evaluate_per_hospital(model, test_loader, device)
    per_class = evaluate_per_class(model, test_loader, device, num_classes)

    logger.log_scalar(f"{tag_prefix}/overall/acc", overall["acc"], step)
    logger.log_scalar(f"{tag_prefix}/overall/loss", overall["loss"], step)

    logger.set_final_metric(f"{tag_prefix}/overall/acc", overall["acc"])
    logger.set_final_metric(f"{tag_prefix}/overall/loss", overall["loss"])

    for hospital, m in per_hospital.items():
        logger.log_scalar(f"{tag_prefix}/per_hospital/{hospital}/acc", m["acc"], step)
        logger.log_scalar(f"{tag_prefix}/per_hospital/{hospital}/loss", m["loss"], step)
        logger.set_final_metric(f"{tag_prefix}/per_hospital/{hospital}/acc", m["acc"])
        logger.set_final_metric(f"{tag_prefix}/per_hospital/{hospital}/loss", m["loss"])

    for class_name, m in per_class.items():
        logger.log_scalar(f"{tag_prefix}/per_class/{class_name}/acc", m["acc"], step)
        logger.log_scalar(f"{tag_prefix}/per_class/{class_name}/precision", m["precision"], step)
        logger.log_scalar(f"{tag_prefix}/per_class/{class_name}/recall", m["recall"], step)
        logger.log_scalar(f"{tag_prefix}/per_class/{class_name}/f1", m["f1"], step)

        logger.set_final_metric(f"{tag_prefix}/per_class/{class_name}/acc", m["acc"])
        logger.set_final_metric(f"{tag_prefix}/per_class/{class_name}/precision", m["precision"])
        logger.set_final_metric(f"{tag_prefix}/per_class/{class_name}/recall", m["recall"])
        logger.set_final_metric(f"{tag_prefix}/per_class/{class_name}/f1", m["f1"])

    y_true, y_pred, image_paths = _collect_predictions(model, test_loader, device)

    classification = logger.log_classification_results(
        tag=f"{tag_prefix}/classification",
        y_true=y_true,
        y_pred=y_pred,
        image_paths=image_paths,
        class_names=CLASS_NAMES,
        step=step,
    )

    logger.set_final_metric(
        f"{tag_prefix}/classification/macro_precision",
        classification["macro_precision"],
    )
    logger.set_final_metric(
        f"{tag_prefix}/classification/macro_recall",
        classification["macro_recall"],
    )
    logger.set_final_metric(
        f"{tag_prefix}/classification/macro_f1",
        classification["macro_f1"],
    )
    logger.set_final_metric(
        f"{tag_prefix}/classification/weighted_precision",
        classification["weighted_precision"],
    )
    logger.set_final_metric(
        f"{tag_prefix}/classification/weighted_recall",
        classification["weighted_recall"],
    )
    logger.set_final_metric(
        f"{tag_prefix}/classification/weighted_f1",
        classification["weighted_f1"],
    )

    hospital_summary = ", ".join(
        f"{h}: {m['acc']:.4f}" for h, m in per_hospital.items()
    )

    logger.info(
        f"[{tag_prefix}] overall_acc={overall['acc']:.4f} "
        f"overall_loss={overall['loss']:.4f} | "
        f"macro_precision={classification['macro_precision']:.4f} "
        f"macro_recall={classification['macro_recall']:.4f} "
        f"macro_f1={classification['macro_f1']:.4f} "
        f"weighted_f1={classification['weighted_f1']:.4f} | "
        f"per_hospital={{ {hospital_summary} }}"
    )

    return {
        "overall": overall,
        "per_hospital": per_hospital,
        "per_class": per_class,
        "classification": classification,
    }


def run_unlearning_evaluation(
    model: nn.Module,
    remember_loader: DataLoader,
    forget_loader: DataLoader,
    device: str,
    logger: ExperimentLogger,
    step: int,
    relearn_steps: int = 50,
    relearn_lr: float = 1e-3,
    nonmember_loader: Optional[DataLoader] = None,
    before_unlearning_acc: Optional[float] = None,
) -> Dict:
    """RA/FA/ReA/MIA + before/after accuracy, namespaced under
    eval/unlearning/*. Used by run_fu.py and run_retrain.py — NOT by run_fl.py
    (plain FL training has no forget/remember notion)."""
    ra_fa = compute_ra_fa(model, remember_loader, forget_loader, device)

    logger.log_scalar("eval/unlearning/RA", ra_fa["RA"], step)
    logger.log_scalar("eval/unlearning/FA", ra_fa["FA"], step)
    logger.set_final_metric("eval/unlearning/RA", ra_fa["RA"])
    logger.set_final_metric("eval/unlearning/FA", ra_fa["FA"])

    remember_true, remember_pred, remember_paths = _collect_predictions(
        model, remember_loader, device
    )
    remember_classification = logger.log_classification_results(
        tag="eval/unlearning/remember_classification",
        y_true=remember_true,
        y_pred=remember_pred,
        image_paths=remember_paths,
        class_names=CLASS_NAMES,
        step=step,
    )

    forget_true, forget_pred, forget_paths = _collect_predictions(
        model, forget_loader, device
    )
    forget_classification = logger.log_classification_results(
        tag="eval/unlearning/forget_classification",
        y_true=forget_true,
        y_pred=forget_pred,
        image_paths=forget_paths,
        class_names=CLASS_NAMES,
        step=step,
    )

    logger.set_final_metric(
        "eval/unlearning/remember_macro_f1",
        remember_classification["macro_f1"],
    )
    logger.set_final_metric(
        "eval/unlearning/remember_weighted_f1",
        remember_classification["weighted_f1"],
    )
    logger.set_final_metric(
        "eval/unlearning/forget_macro_f1",
        forget_classification["macro_f1"],
    )
    logger.set_final_metric(
        "eval/unlearning/forget_weighted_f1",
        forget_classification["weighted_f1"],
    )

    re_a = compute_relearn_accuracy(
        model,
        forget_loader,
        device,
        relearn_steps,
        relearn_lr,
    )

    logger.log_scalar("eval/unlearning/ReA", re_a, step)
    logger.set_final_metric("eval/unlearning/ReA", re_a)

    result = {
        "RA": ra_fa["RA"],
        "FA": ra_fa["FA"],
        "ReA": re_a,
        "remember_classification": remember_classification,
        "forget_classification": forget_classification,
    }

    if nonmember_loader is not None:
        mia_acc = compute_mia_accuracy(model, forget_loader, nonmember_loader, device)
        logger.log_scalar("eval/unlearning/MIA_acc", mia_acc, step)
        logger.set_final_metric("eval/unlearning/MIA_acc", mia_acc)
        result["MIA_acc"] = mia_acc

    if before_unlearning_acc is not None:
        logger.log_scalar("eval/unlearning/before_acc", before_unlearning_acc, step)
        logger.log_scalar("eval/unlearning/after_acc", ra_fa["FA"], step)
        logger.set_final_metric("eval/unlearning/before_acc", before_unlearning_acc)
        logger.set_final_metric("eval/unlearning/after_acc", ra_fa["FA"])

    logger.info(
        f"[eval/unlearning] RA={ra_fa['RA']:.4f} "
        f"FA={ra_fa['FA']:.4f} "
        f"ReA={re_a:.4f} "
        f"remember_macro_f1={remember_classification['macro_f1']:.4f} "
        f"forget_macro_f1={forget_classification['macro_f1']:.4f}"
        + (
            f" MIA_acc={result.get('MIA_acc', float('nan')):.4f}"
            if "MIA_acc" in result
            else ""
        )
    )

    return result