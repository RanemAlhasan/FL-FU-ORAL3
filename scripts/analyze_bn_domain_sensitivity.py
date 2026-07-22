#!/usr/bin/env python3
"""
Standalone FedBN domain-sensitivity analysis.

This script does NOT modify or train any model.

For every BatchNorm layer, it compares the forgotten hospital's:

    running_mean
    running_var

against pooled statistics from the retained hospitals.

Outputs:
    outputs/domain_sensitivity/<source_run>/<forgotten_hospital>/
        domain_sensitivity.csv
        domain_sensitivity.json

Example:
    python3 scripts/analyze_bn_domain_sensitivity.py \
        --source_run fl_fedbn_oral_XXXXX \
        --forget_client Canada_Dataset \
        --metric sym_kl \
        --pooling equal_hospital \
        --top_k 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

# Allow imports from the repository root.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.data.dataset import index_dataset
from src.data.partition import build_client_partitions
from src.models.backbone import build_model
from src.utils.checkpoint import load_checkpoint_into_new_model
from src.utils.config import load_source_run_config


EPS = 1e-8

BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


@dataclass
class GaussianStats:
    """Diagonal-Gaussian approximation of one BN layer's activations."""

    mean: torch.Tensor
    var: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure forgotten-hospital domain sensitivity using the "
            "per-hospital BatchNorm states saved by a FedBN source run."
        )
    )

    parser.add_argument(
        "--source_run",
        required=True,
        help="Phase-1 FedBN source run ID.",
    )

    parser.add_argument(
        "--forget_client",
        required=True,
        help="Hospital to treat as forgotten, for example Canada_Dataset.",
    )

    parser.add_argument(
        "--metric",
        choices=["sym_kl", "normalized_l2"],
        default="sym_kl",
        help="Primary score used for ranking BN layers.",
    )

    parser.add_argument(
        "--pooling",
        choices=["equal_hospital", "sample_weighted"],
        default="equal_hospital",
        help=(
            "How retained hospitals are pooled. "
            "equal_hospital prevents the largest hospital from dominating."
        ),
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of highest-scoring BN layers to print.",
    )

    parser.add_argument(
        "--output_root",
        default="outputs/domain_sensitivity",
        help="Root directory for CSV and JSON results.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used only while loading checkpoints.",
    )

    return parser.parse_args()


def extract_bn_stats(model: nn.Module) -> Dict[str, GaussianStats]:
    """
    Extract running_mean and running_var from every BatchNorm module.

    Affine BN parameters gamma/beta are not needed for the initial
    domain-distribution diagnostic.
    """
    results: Dict[str, GaussianStats] = {}

    for module_name, module in model.named_modules():
        if not isinstance(module, BATCHNORM_TYPES):
            continue

        if module.running_mean is None or module.running_var is None:
            continue

        results[module_name] = GaussianStats(
            mean=module.running_mean.detach().cpu().double().clone(),
            var=(
                module.running_var.detach()
                .cpu()
                .double()
                .clone()
                .clamp_min(EPS)
            ),
        )

    if not results:
        raise ValueError(
            f"No BatchNorm running statistics found in "
            f"{type(model).__name__}."
        )

    return results


def normalize_weights(raw_weights: Sequence[float]) -> List[float]:
    clean_weights = [max(0.0, float(weight)) for weight in raw_weights]
    total = sum(clean_weights)

    if total <= 0.0:
        return [1.0 / len(clean_weights)] * len(clean_weights)

    return [weight / total for weight in clean_weights]


def pool_gaussians(
    stats: Sequence[GaussianStats],
    raw_weights: Sequence[float],
) -> GaussianStats:
    """
    Pool diagonal Gaussian statistics.

    The pooled variance uses the law of total variance:

        Var(X) = E[Var(X | client)] + Var(E[X | client])

    This is preferable to directly averaging running_var tensors.
    """
    if not stats:
        raise ValueError("Cannot pool an empty list of Gaussian statistics.")

    if len(stats) != len(raw_weights):
        raise ValueError(
            "stats and raw_weights must have the same length."
        )

    weights = normalize_weights(raw_weights)

    reference_shape = stats[0].mean.shape

    for item in stats:
        if item.mean.shape != reference_shape:
            raise ValueError(
                "Cannot pool BN statistics with different channel shapes."
            )

    pooled_mean = torch.zeros_like(stats[0].mean, dtype=torch.float64)

    for weight, item in zip(weights, stats):
        pooled_mean += weight * item.mean

    pooled_var = torch.zeros_like(stats[0].var, dtype=torch.float64)

    for weight, item in zip(weights, stats):
        mean_offset = item.mean - pooled_mean
        pooled_var += weight * (
            item.var + mean_offset.pow(2)
        )

    return GaussianStats(
        mean=pooled_mean,
        var=pooled_var.clamp_min(EPS),
    )


def aggregate_hospital_stats(
    hospital_client_stats: Dict[
        str,
        List[Tuple[str, Dict[str, GaussianStats], float]],
    ],
) -> Tuple[
    Dict[str, Dict[str, GaussianStats]],
    Dict[str, float],
]:
    """
    Aggregate one or more client/shard checkpoints into one hospital state.

    Returns:
        hospital_stats:
            hospital -> BN layer -> pooled statistics

        hospital_sample_counts:
            hospital -> total training samples
    """
    hospital_stats: Dict[str, Dict[str, GaussianStats]] = {}
    hospital_sample_counts: Dict[str, float] = {}

    for hospital, client_entries in hospital_client_stats.items():
        if not client_entries:
            continue

        client_layer_sets = [
            set(client_stats.keys())
            for _, client_stats, _ in client_entries
        ]

        common_layers = set.intersection(*client_layer_sets)

        if not common_layers:
            raise ValueError(
                f"No common BatchNorm layers found for hospital {hospital}."
            )

        hospital_sample_counts[hospital] = sum(
            sample_count
            for _, _, sample_count in client_entries
        )

        aggregated: Dict[str, GaussianStats] = {}

        for layer_name in sorted(common_layers):
            layer_stats = [
                client_stats[layer_name]
                for _, client_stats, _ in client_entries
            ]

            layer_weights = [
                sample_count
                for _, _, sample_count in client_entries
            ]

            aggregated[layer_name] = pool_gaussians(
                layer_stats,
                layer_weights,
            )

        hospital_stats[hospital] = aggregated

    return hospital_stats, hospital_sample_counts


def symmetric_kl(
    forget_stats: GaussianStats,
    remember_stats: GaussianStats,
) -> float:
    """
    Symmetric KL divergence between diagonal Gaussian distributions.

    Returns the mean divergence across BN channels.
    """
    mean_f = forget_stats.mean
    var_f = forget_stats.var.clamp_min(EPS)

    mean_r = remember_stats.mean
    var_r = remember_stats.var.clamp_min(EPS)

    mean_difference_sq = (mean_f - mean_r).pow(2)

    kl_f_to_r = 0.5 * (
        torch.log(var_r / var_f)
        + (var_f + mean_difference_sq) / var_r
        - 1.0
    )

    kl_r_to_f = 0.5 * (
        torch.log(var_f / var_r)
        + (var_r + mean_difference_sq) / var_f
        - 1.0
    )

    score = 0.5 * (kl_f_to_r + kl_r_to_f)

    return float(score.mean().item())


def normalized_l2(
    forget_stats: GaussianStats,
    remember_stats: GaussianStats,
) -> float:
    """
    Scale-aware distance between two BN distributions.

    Mean differences are normalized by retained variance.
    Variance differences are measured in log-variance space.
    """
    mean_term = (
        (forget_stats.mean - remember_stats.mean).pow(2)
        / remember_stats.var.clamp_min(EPS)
    ).mean()

    log_var_term = (
        torch.log(forget_stats.var.clamp_min(EPS))
        - torch.log(remember_stats.var.clamp_min(EPS))
    ).pow(2).mean()

    return float((mean_term + log_var_term).item())


def mean_shift(
    forget_stats: GaussianStats,
    remember_stats: GaussianStats,
) -> float:
    return float(
        torch.sqrt(
            torch.mean(
                (forget_stats.mean - remember_stats.mean).pow(2)
            )
        ).item()
    )


def log_variance_shift(
    forget_stats: GaussianStats,
    remember_stats: GaussianStats,
) -> float:
    return float(
        torch.sqrt(
            torch.mean(
                (
                    torch.log(forget_stats.var.clamp_min(EPS))
                    - torch.log(
                        remember_stats.var.clamp_min(EPS)
                    )
                ).pow(2)
            )
        ).item()
    )


def infer_associated_parameter(
    model: nn.Module,
    bn_layer_name: str,
) -> str:
    """
    Map a ResNet BN module to the convolution parameter immediately
    preceding it.

    Examples:
        bn1                    -> conv1.weight
        layer2.0.bn1           -> layer2.0.conv1.weight
        layer4.1.bn2           -> layer4.1.conv2.weight
        layer2.0.downsample.1  -> layer2.0.downsample.0.weight

    Returns an empty string when no valid parameter is found.
    """
    parameter_names = set(dict(model.named_parameters()).keys())
    candidates: List[str] = []

    if bn_layer_name == "bn1":
        candidates.append("conv1.weight")

    match = re.match(
        r"^(.*)\.bn([123])$",
        bn_layer_name,
    )

    if match:
        prefix = match.group(1)
        conv_number = match.group(2)
        candidates.append(
            f"{prefix}.conv{conv_number}.weight"
        )

    if bn_layer_name.endswith(".downsample.1"):
        candidates.append(
            bn_layer_name[: -len(".1")] + ".0.weight"
        )

    for candidate in candidates:
        if candidate in parameter_names:
            return candidate

    return ""


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if abs(maximum - minimum) <= EPS:
        return [0.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def build_domain_rows(
    *,
    model: nn.Module,
    hospital_stats: Dict[str, Dict[str, GaussianStats]],
    hospital_sample_counts: Dict[str, float],
    forget_hospital: str,
    metric: str,
    pooling: str,
) -> List[Dict[str, object]]:
    if forget_hospital not in hospital_stats:
        raise ValueError(
            f"Forgotten hospital '{forget_hospital}' has no BN state."
        )

    remember_hospitals = [
        hospital
        for hospital in hospital_stats
        if hospital != forget_hospital
    ]

    if not remember_hospitals:
        raise ValueError(
            "At least one retained hospital is required."
        )

    forget_layers = hospital_stats[forget_hospital]

    common_layers = set(forget_layers.keys())

    for hospital in remember_hospitals:
        common_layers &= set(hospital_stats[hospital].keys())

    if not common_layers:
        raise ValueError(
            "No common BN layers exist across all hospitals."
        )

    if pooling == "equal_hospital":
        remember_weights = {
            hospital: 1.0
            for hospital in remember_hospitals
        }
    elif pooling == "sample_weighted":
        remember_weights = {
            hospital: hospital_sample_counts[hospital]
            for hospital in remember_hospitals
        }
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    rows: List[Dict[str, object]] = []

    for layer_name in sorted(common_layers):
        remember_layer_stats = [
            hospital_stats[hospital][layer_name]
            for hospital in remember_hospitals
        ]

        remember_layer_weights = [
            remember_weights[hospital]
            for hospital in remember_hospitals
        ]

        pooled_remember = pool_gaussians(
            remember_layer_stats,
            remember_layer_weights,
        )

        forget_layer = forget_layers[layer_name]

        sym_kl_score = symmetric_kl(
            forget_layer,
            pooled_remember,
        )

        normalized_l2_score = normalized_l2(
            forget_layer,
            pooled_remember,
        )

        selected_score = (
            sym_kl_score
            if metric == "sym_kl"
            else normalized_l2_score
        )

        rows.append(
            {
                "bn_layer": layer_name,
                "associated_parameter": infer_associated_parameter(
                    model,
                    layer_name,
                ),
                "channels": int(forget_layer.mean.numel()),
                "sym_kl": sym_kl_score,
                "normalized_l2": normalized_l2_score,
                "mean_shift": mean_shift(
                    forget_layer,
                    pooled_remember,
                ),
                "log_variance_shift": log_variance_shift(
                    forget_layer,
                    pooled_remember,
                ),
                "selected_score": selected_score,
            }
        )

    normalized_scores = min_max_normalize(
        [
            float(row["selected_score"])
            for row in rows
        ]
    )

    for row, normalized_score in zip(rows, normalized_scores):
        row["normalized_domain_score"] = normalized_score

    rows.sort(
        key=lambda row: float(row["selected_score"]),
        reverse=True,
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def save_results(
    *,
    rows: List[Dict[str, object]],
    metadata: Dict[str, object],
    output_dir: str,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        "domain_sensitivity.csv",
    )

    json_path = os.path.join(
        output_dir,
        "domain_sensitivity.json",
    )

    if not rows:
        raise ValueError("No domain-sensitivity rows were produced.")

    csv_columns = [
        "rank",
        "bn_layer",
        "associated_parameter",
        "channels",
        "selected_score",
        "normalized_domain_score",
        "sym_kl",
        "normalized_l2",
        "mean_shift",
        "log_variance_shift",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_columns,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in csv_columns
                }
            )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "metadata": metadata,
                "layers": rows,
            },
            file,
            indent=2,
        )

    return csv_path, json_path


def main() -> None:
    args = parse_args()

    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[warning] CUDA requested but unavailable; using CPU."
        )
        device = "cpu"

    config = load_source_run_config(
        args.source_run,
        logs_root="logs/fl",
    )

    source_algorithm = str(
        config.get("algorithm", "")
    ).strip().lower()

    if source_algorithm != "fedbn":
        raise ValueError(
            "Domain-sensitivity analysis requires a FedBN source run. "
            f"Received algorithm='{source_algorithm}'."
        )

    hospitals = config["hospitals"]

    if args.forget_client not in hospitals:
        raise ValueError(
            f"Unknown forgotten hospital '{args.forget_client}'. "
            f"Expected one of {hospitals}."
        )

    checkpoint_root = os.path.join(
        config["checkpoints_root"],
        args.source_run,
    )

    per_hospital_dir = os.path.join(
        checkpoint_root,
        "per_hospital",
    )

    if not os.path.isdir(per_hospital_dir):
        raise FileNotFoundError(
            "Per-hospital FedBN checkpoint directory not found: "
            f"{per_hospital_dir}"
        )

    train_samples = index_dataset(
        config["dataset_path"],
        "Train",
        hospitals,
    )

    partitions = build_client_partitions(
        train_samples,
        hospitals,
        config["client_split"],
        config.get("clients_per_hospital", 1),
        config["seed"],
    )

    def model_builder() -> nn.Module:
        # Full checkpoint weights are loaded immediately, so pretrained
        # initialization is unnecessary here.
        return build_model(
            config.get("model", "resnet18"),
            num_classes=config["num_classes"],
            pretrained=False,
        )

    hospital_client_stats: Dict[
        str,
        List[Tuple[str, Dict[str, GaussianStats], float]],
    ] = {
        hospital: []
        for hospital in hospitals
    }

    reference_model: nn.Module | None = None

    print("=" * 90)
    print("FEDBN DOMAIN-SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(f"Source run:       {args.source_run}")
    print(f"Forgot hospital:  {args.forget_client}")
    print(f"Metric:           {args.metric}")
    print(f"Pooling:          {args.pooling}")
    print(f"Checkpoints:      {per_hospital_dir}")
    print()

    for partition in partitions:
        checkpoint_path = os.path.join(
            per_hospital_dir,
            f"{partition.client_id}.pt",
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Missing FedBN client checkpoint: "
                f"{checkpoint_path}"
            )

        model = load_checkpoint_into_new_model(
            model_builder,
            checkpoint_path,
            device=device,
        )

        if reference_model is None:
            reference_model = model

        client_stats = extract_bn_stats(model)

        hospital_client_stats[partition.hospital].append(
            (
                partition.client_id,
                client_stats,
                float(len(partition.samples)),
            )
        )

        print(
            f"Loaded {partition.client_id:<35} "
            f"hospital={partition.hospital:<20} "
            f"samples={len(partition.samples):>5} "
            f"bn_layers={len(client_stats):>3}"
        )

        model.to("cpu")
        del model

    if reference_model is None:
        raise RuntimeError("No FedBN checkpoint was loaded.")

    hospital_stats, hospital_sample_counts = (
        aggregate_hospital_stats(
            hospital_client_stats
        )
    )

    rows = build_domain_rows(
        model=reference_model,
        hospital_stats=hospital_stats,
        hospital_sample_counts=hospital_sample_counts,
        forget_hospital=args.forget_client,
        metric=args.metric,
        pooling=args.pooling,
    )

    output_dir = os.path.join(
        args.output_root,
        args.source_run,
        args.forget_client.replace("_Dataset", ""),
    )

    metadata = {
        "source_run": args.source_run,
        "source_algorithm": source_algorithm,
        "forget_client": args.forget_client,
        "remember_hospitals": [
            hospital
            for hospital in hospitals
            if hospital != args.forget_client
        ],
        "metric": args.metric,
        "pooling": args.pooling,
        "hospital_sample_counts": hospital_sample_counts,
        "model": config.get("model", "resnet18"),
        "client_split": config["client_split"],
        "clients_per_hospital": config.get(
            "clients_per_hospital",
            1,
        ),
    }

    csv_path, json_path = save_results(
        rows=rows,
        metadata=metadata,
        output_dir=output_dir,
    )

    print()
    print("=" * 120)
    print(
        f"{'Rank':>4}  "
        f"{'BN layer':<32} "
        f"{'Associated parameter':<38} "
        f"{'Score':>12} "
        f"{'Normalized':>12}"
    )
    print("-" * 120)

    top_k = min(args.top_k, len(rows))

    for row in rows[:top_k]:
        print(
            f"{int(row['rank']):>4}  "
            f"{str(row['bn_layer']):<32} "
            f"{str(row['associated_parameter']):<38} "
            f"{float(row['selected_score']):>12.6f} "
            f"{float(row['normalized_domain_score']):>12.6f}"
        )

    print("=" * 120)
    print()
    print(f"CSV saved to:  {csv_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()