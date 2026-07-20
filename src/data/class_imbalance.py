"""Backward-compatible class-imbalance loss utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


SUPPORTED_IMBALANCE_METHODS = {
    "standard_ce",
    "weighted_sampler",
    "global_weighted_ce",
    "local_class_balanced_ce",
}


def resolve_imbalance_method(config: Dict) -> str:
    """
    New configs use `imbalance_method`.

    Backward compatibility:
    - Old handle_class_imbalance=true  -> weighted_sampler
    - Old handle_class_imbalance=false -> standard_ce
    """
    explicit_method = config.get("imbalance_method")

    if explicit_method is None:
        return (
            "weighted_sampler"
            if bool(config.get("handle_class_imbalance", True))
            else "standard_ce"
        )

    method = str(explicit_method).strip().lower()

    if method not in SUPPORTED_IMBALANCE_METHODS:
        raise ValueError(
            f"Unsupported imbalance_method='{method}'. "
            f"Expected one of {sorted(SUPPORTED_IMBALANCE_METHODS)}."
        )

    return method


def count_dataset_classes(dataset, num_classes: int) -> List[int]:
    """Count training samples per class."""
    samples = getattr(dataset, "samples", None)

    if samples is None:
        raise TypeError(
            f"Dataset {type(dataset).__name__} has no `.samples` attribute."
        )

    counts = [0] * num_classes

    for sample in samples:
        label = int(sample.label)

        if label < 0 or label >= num_classes:
            raise ValueError(
                f"Found class label {label}, but num_classes={num_classes}."
            )

        counts[label] += 1

    return counts


def combine_class_counts(
    datasets: Sequence,
    num_classes: int,
) -> List[int]:
    """Combine class counts from all FL clients."""
    combined = [0] * num_classes

    for dataset in datasets:
        counts = count_dataset_classes(dataset, num_classes)

        for class_id, count in enumerate(counts):
            combined[class_id] += count

    return combined


def inverse_frequency_weights(
    counts: Sequence[int],
) -> List[float]:
    """Standard inverse-frequency weights: N / (C * n_c)."""
    total = sum(counts)
    present_classes = sum(count > 0 for count in counts)

    if total <= 0 or present_classes <= 0:
        raise ValueError(f"Invalid class counts: {list(counts)}")

    return [
        total / (present_classes * count) if count > 0 else 0.0
        for count in counts
    ]


def class_balanced_weights(
    counts: Sequence[int],
    beta: float = 0.999,
    max_weight: Optional[float] = None,
) -> List[float]:
    """Class-balanced weights based on effective sample counts."""
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}.")

    raw_weights: List[float] = []

    for count in counts:
        if count <= 0:
            raw_weights.append(0.0)
            continue

        effective_number = 1.0 - (beta ** count)
        weight = (1.0 - beta) / max(effective_number, 1e-12)
        raw_weights.append(weight)

    positive_weights = [
        weight for weight in raw_weights if weight > 0.0
    ]

    if not positive_weights:
        raise ValueError(f"Invalid class counts: {list(counts)}")

    mean_weight = sum(positive_weights) / len(positive_weights)

    normalized = [
        weight / mean_weight if weight > 0.0 else 0.0
        for weight in raw_weights
    ]

    if max_weight is not None:
        normalized = [
            min(weight, float(max_weight))
            for weight in normalized
        ]

    return normalized


def build_class_weights(
    method: str,
    counts: Optional[Sequence[int]],
    beta: float = 0.999,
    max_weight: Optional[float] = None,
) -> Optional[List[float]]:
    """
    Return loss weights.

    None means ordinary CrossEntropyLoss.
    """
    if method in {"standard_ce", "weighted_sampler"}:
        return None

    if counts is None:
        raise ValueError(
            f"Class counts are required for method='{method}'."
        )

    if method == "global_weighted_ce":
        return inverse_frequency_weights(counts)

    if method == "local_class_balanced_ce":
        return class_balanced_weights(
            counts,
            beta=beta,
            max_weight=max_weight,
        )

    raise ValueError(f"Unsupported imbalance method: {method}")