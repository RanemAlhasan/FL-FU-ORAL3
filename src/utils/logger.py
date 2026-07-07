"""Experiment logger with TensorBoard, JSON metrics, text logs, and prediction reports."""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, run_id: str, log_dir: str, tb_dir: str):
        self.run_id = run_id
        self.log_dir = log_dir
        self.tb_dir = tb_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        self.metrics_path = os.path.join(log_dir, "metrics.json")
        self.predictions_path = os.path.join(log_dir, "predictions.csv")
        self.confusion_matrix_path = os.path.join(log_dir, "confusion_matrix.csv")
        self.classification_report_path = os.path.join(log_dir, "classification_report.json")

        self._metrics: Dict[str, Any] = {
            "run_id": run_id,
            "scalars": {},
            "final": {},
            "classification": {},
        }

        self.writer = SummaryWriter(log_dir=tb_dir)
        self._text_logger = self._build_text_logger()
        self._start_time = time.time()
        self._text_logger.info(f"=== Run started: {run_id} ===")

    def _build_text_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.run_id)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        file_handler = logging.FileHandler(os.path.join(self.log_dir, "train.log"))
        stream_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def info(self, msg: str) -> None:
        self._text_logger.info(msg)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, global_step=step)
        self._metrics["scalars"].setdefault(tag, []).append(
            {"step": int(step), "value": float(value)}
        )

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        flat = "\n".join(f"  {k}: {v}" for k, v in hparams.items())
        self.writer.add_text("hparams", flat)
        self.info(f"Run hyperparameters:\n{flat}")

    def set_final_metric(self, key: str, value: Any) -> None:
        self._metrics["final"][key] = self._to_json_safe(value)

    def log_classification_results(
        self,
        tag: str,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        image_paths: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Log classification metrics and per-image predictions.

        Produces:
            - metrics.json classification section
            - TensorBoard scalars
            - predictions.csv
            - confusion_matrix.csv
            - classification_report.json
        """
        y_true = [int(v) for v in y_true]
        y_pred = [int(v) for v in y_pred]

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have same length. "
                f"Got {len(y_true)} and {len(y_pred)}."
            )

        if image_paths is not None and len(image_paths) != len(y_true):
            raise ValueError(
                f"image_paths must have same length as predictions. "
                f"Got {len(image_paths)} and {len(y_true)}."
            )

        if image_paths is None:
            image_paths = [""] * len(y_true)
        else:
            image_paths = [str(p) for p in image_paths]

        labels = list(range(len(class_names))) if class_names is not None else sorted(set(y_true) | set(y_pred))

        num_samples = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / max(1, num_samples)

        confusion_matrix = self._compute_confusion_matrix(y_true, y_pred, labels)
        per_class = self._compute_per_class_metrics(y_true, y_pred, labels, class_names)

        macro_precision = self._mean([m["precision"] for m in per_class.values()])
        macro_recall = self._mean([m["recall"] for m in per_class.values()])
        macro_f1 = self._mean([m["f1"] for m in per_class.values()])

        total_support = sum(m["support"] for m in per_class.values())
        weighted_precision = sum(m["precision"] * m["support"] for m in per_class.values()) / max(1, total_support)
        weighted_recall = sum(m["recall"] * m["support"] for m in per_class.values()) / max(1, total_support)
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / max(1, total_support)

        results: Dict[str, Any] = {
            "tag": tag,
            "num_samples": num_samples,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "labels": labels,
            "class_names": list(class_names) if class_names is not None else None,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix,
        }

        self._metrics["classification"][tag] = self._to_json_safe(results)

        scalar_step = 0 if step is None else int(step)
        self.log_scalar(f"{tag}/accuracy", accuracy, scalar_step)
        self.log_scalar(f"{tag}/macro_precision", macro_precision, scalar_step)
        self.log_scalar(f"{tag}/macro_recall", macro_recall, scalar_step)
        self.log_scalar(f"{tag}/macro_f1", macro_f1, scalar_step)
        self.log_scalar(f"{tag}/weighted_precision", weighted_precision, scalar_step)
        self.log_scalar(f"{tag}/weighted_recall", weighted_recall, scalar_step)
        self.log_scalar(f"{tag}/weighted_f1", weighted_f1, scalar_step)

        self.set_final_metric(f"{tag}/accuracy", accuracy)
        self.set_final_metric(f"{tag}/macro_precision", macro_precision)
        self.set_final_metric(f"{tag}/macro_recall", macro_recall)
        self.set_final_metric(f"{tag}/macro_f1", macro_f1)
        self.set_final_metric(f"{tag}/weighted_precision", weighted_precision)
        self.set_final_metric(f"{tag}/weighted_recall", weighted_recall)
        self.set_final_metric(f"{tag}/weighted_f1", weighted_f1)

        self.info(
            f"[{tag}] accuracy={accuracy:.4f} "
            f"macro_precision={macro_precision:.4f} "
            f"macro_recall={macro_recall:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"weighted_f1={weighted_f1:.4f}"
        )

        self._append_predictions_csv(
            tag=tag,
            y_true=y_true,
            y_pred=y_pred,
            image_paths=image_paths,
            class_names=class_names,
        )
        self._write_confusion_matrix_csv(
            tag=tag,
            labels=labels,
            class_names=class_names,
            confusion_matrix=confusion_matrix,
        )
        self._write_classification_report_json(tag=tag, results=results)

        return results

    def _compute_confusion_matrix(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        labels: Sequence[int],
    ) -> List[List[int]]:
        matrix: List[List[int]] = []

        for true_label in labels:
            row = []
            for pred_label in labels:
                row.append(
                    sum(
                        1
                        for t, p in zip(y_true, y_pred)
                        if int(t) == int(true_label) and int(p) == int(pred_label)
                    )
                )
            matrix.append(row)

        return matrix

    def _compute_per_class_metrics(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        labels: Sequence[int],
        class_names: Optional[Sequence[str]],
    ) -> Dict[str, Dict[str, Any]]:
        per_class: Dict[str, Dict[str, Any]] = {}

        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            support = sum(1 for t in y_true if t == label)

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

            class_key = self._class_name(label, class_names)

            per_class[class_key] = {
                "class_id": int(label),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

        return per_class

    def _append_predictions_csv(
        self,
        tag: str,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        image_paths: Sequence[str],
        class_names: Optional[Sequence[str]],
    ) -> None:
        file_exists = os.path.exists(self.predictions_path)

        with open(self.predictions_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tag",
                    "image_path",
                    "true_class_id",
                    "true_class_name",
                    "predicted_class_id",
                    "predicted_class_name",
                    "correct",
                ],
            )

            if not file_exists:
                writer.writeheader()

            for path, true_id, pred_id in zip(image_paths, y_true, y_pred):
                writer.writerow(
                    {
                        "tag": tag,
                        "image_path": path,
                        "true_class_id": int(true_id),
                        "true_class_name": self._class_name(int(true_id), class_names),
                        "predicted_class_id": int(pred_id),
                        "predicted_class_name": self._class_name(int(pred_id), class_names),
                        "correct": int(int(true_id) == int(pred_id)),
                    }
                )

    def _write_confusion_matrix_csv(
        self,
        tag: str,
        labels: Sequence[int],
        class_names: Optional[Sequence[str]],
        confusion_matrix: Sequence[Sequence[int]],
    ) -> None:
        safe_tag = self._safe_filename(tag)
        path = self._tagged_path(self.confusion_matrix_path, safe_tag)

        header = ["true\\pred"] + [self._class_name(label, class_names) for label in labels]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for label, row in zip(labels, confusion_matrix):
                writer.writerow([self._class_name(label, class_names)] + list(row))

    def _write_classification_report_json(self, tag: str, results: Dict[str, Any]) -> None:
        safe_tag = self._safe_filename(tag)
        path = self._tagged_path(self.classification_report_path, safe_tag)

        with open(path, "w") as f:
            json.dump(self._to_json_safe(results), f, indent=2)

    def flush(self) -> None:
        self.writer.flush()

        with open(self.metrics_path, "w") as f:
            json.dump(self._to_json_safe(self._metrics), f, indent=2)

    def close(self) -> None:
        elapsed = time.time() - self._start_time
        self.info(f"=== Run finished: {self.run_id} (elapsed {elapsed:.1f}s) ===")
        self.flush()
        self.writer.close()

        for handler in list(self._text_logger.handlers):
            handler.close()
            self._text_logger.removeHandler(handler)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    @staticmethod
    def _class_name(class_id: int, class_names: Optional[Sequence[str]]) -> str:
        if class_names is not None and 0 <= int(class_id) < len(class_names):
            return str(class_names[int(class_id)])
        return str(class_id)

    @staticmethod
    def _safe_filename(tag: str) -> str:
        return (
            tag.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

    @staticmethod
    def _tagged_path(base_path: str, safe_tag: str) -> str:
        root, ext = os.path.splitext(base_path)
        return f"{root}_{safe_tag}{ext}"

    def _to_json_safe(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_json_safe(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        try:
            return obj.item()
        except AttributeError:
            return str(obj)


def build_logger(run_id: str, log_dir: str, tb_dir: str) -> ExperimentLogger:
    return ExperimentLogger(run_id, log_dir, tb_dir)