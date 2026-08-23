"""
Critical Layer Identification (CLI) — paper Section 4.1, Eq. 11-13.

The original paper-faithful mode uses a tensor-wise Manhattan SUM.

This module additionally supports:

1. Mean absolute parameter change, removing tensor-size bias.
2. Domain-aware scoring using normalized FedBN domain sensitivity.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.fl.core import SGD_MOMENTUM, SGD_WEIGHT_DECAY


VALID_DIFF_REDUCTIONS = (
    "sum",
    "mean",
)

VALID_CLI_SCORE_MODES = (
    "raw_sum",
    "mean_abs",
    "domain_aware",
)


def _named_float_params(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Return learnable parameters only.

    BatchNorm running_mean, running_var and num_batches_tracked are
    buffers and are deliberately excluded from adapter candidates.
    """
    return {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
    }


def _check_diff_reduction(
    reduction: str,
) -> str:
    reduction = reduction.lower()

    if reduction not in VALID_DIFF_REDUCTIONS:
        raise ValueError(
            f"Unknown CLI diff reduction {reduction!r}; "
            f"expected one of {VALID_DIFF_REDUCTIONS}."
        )

    return reduction


def _check_cli_score_mode(
    score_mode: str,
) -> str:
    score_mode = score_mode.lower()

    if score_mode not in VALID_CLI_SCORE_MODES:
        raise ValueError(
            f"Unknown CLI score mode {score_mode!r}; "
            f"expected one of {VALID_CLI_SCORE_MODES}."
        )

    return score_mode


def compute_layer_diffs(
    source_model: nn.Module,
    client_models: List[nn.Module],
    client_data_sizes: List[int],
    reduction: str = "sum",
) -> Dict[str, float]:
    """
    Compute the data-size-weighted local-update difference for each
    parameter.

    reduction="sum":
        Original paper-style Manhattan sum.

    reduction="mean":
        Mean absolute parameter change. This removes the advantage
        received by tensors merely because they contain more elements.
    """
    reduction = _check_diff_reduction(
        reduction
    )

    if len(client_models) != len(client_data_sizes):
        raise ValueError(
            "client_models and client_data_sizes must have "
            "equal length."
        )

    if not client_models:
        raise ValueError(
            "At least one client model is required for CLI."
        )

    total_size = sum(
        client_data_sizes
    )

    if total_size <= 0:
        raise ValueError(
            "The total CLI client-data size must be positive."
        )

    source_params = _named_float_params(
        source_model
    )

    diffs: Dict[str, float] = {
        name: 0.0
        for name in source_params
    }

    for client_model, data_size in zip(
        client_models,
        client_data_sizes,
    ):
        if data_size < 0:
            raise ValueError(
                "Client data sizes must be non-negative, "
                f"got {data_size}."
            )

        client_params = _named_float_params(
            client_model
        )

        client_weight = (
            data_size / total_size
        )

        for name, source_parameter in source_params.items():
            if name not in client_params:
                continue

            client_parameter = client_params[name]

            if (
                client_parameter.device
                != source_parameter.device
            ):
                client_parameter = client_parameter.to(
                    source_parameter.device
                )

            absolute_change = torch.abs(
                client_parameter
                - source_parameter
            )

            if reduction == "sum":
                client_score = (
                    absolute_change.sum().item()
                )
            else:
                client_score = (
                    absolute_change.mean().item()
                )

            diffs[name] += (
                client_weight
                * client_score
            )

    return diffs


def min_max_normalize_scores(
    scores: Mapping[str, float],
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """
    Min-max normalize a score mapping to [0, 1].
    """
    if not scores:
        return {}

    values = [
        float(value)
        for value in scores.values()
    ]

    if any(
        not math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "CLI scores must all be finite."
        )

    minimum = min(values)
    maximum = max(values)
    denominator = maximum - minimum

    if denominator <= epsilon:
        return {
            name: 0.0
            for name in scores
        }

    return {
        name: (
            float(value) - minimum
        ) / denominator
        for name, value in scores.items()
    }


def build_critical_layer_scores(
    cli_diffs: Mapping[str, float],
    cli_score_mode: str,
    domain_scores: Optional[
        Mapping[str, float]
    ] = None,
    domain_lambda: float = 0.5,
) -> Tuple[
    Dict[str, float],
    List[Dict[str, object]],
]:
    """
    Build final scores used to select critical layers.

    raw_sum:
        Original raw Manhattan-sum ranking.

    mean_abs:
        Mean absolute parameter-change ranking.

    domain_aware:
        normalized_mean_absolute_CLI
        + lambda * normalized_domain_score
    """
    cli_score_mode = _check_cli_score_mode(
        cli_score_mode
    )

    if domain_lambda < 0.0:
        raise ValueError(
            "domain_lambda must be non-negative."
        )

    if (
        cli_score_mode == "domain_aware"
        and not domain_scores
    ):
        raise ValueError(
            "domain_aware CLI selection requires a "
            "non-empty domain_scores mapping."
        )

    normalized_cli_scores = (
        min_max_normalize_scores(
            cli_diffs
        )
    )

    final_scores: Dict[str, float] = {}
    rows: List[Dict[str, object]] = []

    for parameter_name, cli_score_value in (
        cli_diffs.items()
    ):
        cli_score = float(
            cli_score_value
        )

        normalized_cli = (
            normalized_cli_scores[
                parameter_name
            ]
        )

        domain_score = float(
            (domain_scores or {}).get(
                parameter_name,
                0.0,
            )
        )

        if not math.isfinite(
            domain_score
        ):
            raise ValueError(
                "Domain score for "
                f"{parameter_name!r} must be finite, "
                f"got {domain_score}."
            )

        domain_score = min(
            1.0,
            max(0.0, domain_score),
        )

        if cli_score_mode == "raw_sum":
            final_score = cli_score
            effective_domain_score = 0.0

        elif cli_score_mode == "mean_abs":
            # Min-max normalization does not change ranking,
            # but gives a consistent and interpretable scale.
            final_score = normalized_cli
            effective_domain_score = 0.0

        else:
            effective_domain_score = (
                domain_score
            )

            final_score = (
                normalized_cli
                + domain_lambda
                * domain_score
            )

        final_scores[
            parameter_name
        ] = final_score

        rows.append(
            {
                "parameter": parameter_name,
                "cli_score_mode": cli_score_mode,
                "cli_score": cli_score,
                "normalized_cli_score": (
                    normalized_cli
                ),
                "domain_score": (
                    effective_domain_score
                ),
                "domain_lambda": (
                    domain_lambda
                    if cli_score_mode
                    == "domain_aware"
                    else 0.0
                ),
                "selection_score": (
                    final_score
                ),
            }
        )

    rows.sort(
        key=lambda row: float(
            row["selection_score"]
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row["rank"] = rank

    return final_scores, rows


def select_top_k_critical_layers(
    diffs: Mapping[str, float],
    num_unlearning_layers: int,
) -> List[str]:
    """
    Sort descending and return the first K parameter names.
    """
    if num_unlearning_layers <= 0:
        raise ValueError(
            "num_unlearning_layers must be positive."
        )

    sorted_layers = sorted(
        diffs.items(),
        key=lambda item: -float(
            item[1]
        ),
    )

    k = min(
        num_unlearning_layers,
        len(sorted_layers),
    )

    return [
        name
        for name, _ in sorted_layers[:k]
    ]


def run_critical_layer_identification(
    source_model: nn.Module,
    remember_client_loaders: List[
        DataLoader
    ],
    remember_client_data_sizes: List[int],
    device: str,
    local_epochs: int = 1,
    learning_rate: float = 0.005,
    diff_reduction: str = "sum",
) -> Dict[str, float]:
    """
    Train one source-model clone per supplied client, then compute
    parameter differences against the untouched source model.
    """
    diff_reduction = _check_diff_reduction(
        diff_reduction
    )

    criteria = nn.CrossEntropyLoss()

    client_models: List[nn.Module] = []

    for loader in remember_client_loaders:
        model = copy.deepcopy(
            source_model
        ).to(device)

        model.train()

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=SGD_MOMENTUM,
            weight_decay=SGD_WEIGHT_DECAY,
        )

        for _ in range(
            local_epochs
        ):
            for data, target in loader:
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                loss = criteria(
                    model(data),
                    target,
                )

                loss.backward()
                optimizer.step()

        model.to("cpu")

        client_models.append(
            model
        )

    return compute_layer_diffs(
        source_model,
        client_models,
        remember_client_data_sizes,
        reduction=diff_reduction,
    )