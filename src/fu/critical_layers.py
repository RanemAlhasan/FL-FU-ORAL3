"""
Critical Layer Identification (CLI), reproducing FUSED Section 4.1 (Eq. 11-13).

For each layer l, after one federated round of local training starting from
the (frozen) source model M^r, we compute:

    Diff_l = sum_n  (|D_1| / sum_k |D_k|) * Diff_l^n(p_l^n, p_l)

where Diff_l^n is the Manhattan distance between client n's locally-trained
layer-l parameters and the original (pre-training) layer-l parameters, and
the weights are each client's relative data volume (Eq. 13 in the paper:
note the paper's weighting uses |D_1| in the numerator for both terms, which
we interpret as a typesetting artifact of an FedAvg-style weighted average —
we implement it as the standard data-volume-weighted average, i.e. weight_n =
|D_n| / sum_k |D_k|, which is what Eq. 13's described intent ("the
aggregation method... where |D_i| represents the data volume of client i")
specifies).

The paper notes (Sec 5.2) that the layer-sensitivity gap is most pronounced
after a SINGLE federated round/epoch, so CLI should be run with a short,
dedicated mini-federation (default: 1 round) rather than reusing the full FL
training's final-round Diff values.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.backbone import get_named_layers


def manhattan_layer_diff(param_a: torch.Tensor, param_b: torch.Tensor) -> float:
    """Eq. 12: Diff(p^n_l, p_l) = sum_i sum_j |p^n_{l,ij} - p_{l,ij}|"""
    return torch.sum(torch.abs(param_a - param_b)).item()


def compute_layer_diffs(
    source_model: nn.Module,
    client_models: Dict[str, nn.Module],
    client_data_sizes: Dict[str, int],
) -> Dict[str, float]:
    """Compute Diff_l for every layer l (Eq. 11 + 13), aggregated across all
    provided client_models (each assumed to be one round of local training
    starting from `source_model`'s parameters).

    Returns: {layer_name: diff_value}, where layer_name matches the naming
    used by get_named_layers() (so it can be mapped back to actual nn.Module
    objects later when building sparse adapters).
    """
    source_layers = dict(get_named_layers(source_model))
    total_data = sum(client_data_sizes.values())
    if total_data == 0:
        raise ValueError("client_data_sizes sum to zero; cannot weight CLI by data volume.")

    diffs: Dict[str, float] = {name: 0.0 for name in source_layers}

    for client_id, client_model in client_models.items():
        client_layers = dict(get_named_layers(client_model))
        weight = client_data_sizes[client_id] / total_data

        for layer_name, source_module in source_layers.items():
            if layer_name not in client_layers:
                continue
            client_module = client_layers[layer_name]
            # Sum the diff over every parameter tensor owned directly by this
            # layer (e.g. both .weight and .bias for Conv2d/Linear/BatchNorm).
            layer_diff = 0.0
            source_params = dict(source_module.named_parameters(recurse=False))
            client_params = dict(client_module.named_parameters(recurse=False))
            for p_name, source_p in source_params.items():
                if p_name not in client_params:
                    continue
                layer_diff += manhattan_layer_diff(client_params[p_name].detach(),
                                                    source_p.detach())
            diffs[layer_name] += weight * layer_diff

    return diffs


def rank_layers_by_sensitivity(diffs: Dict[str, float]) -> List[str]:
    """Return layer names sorted from MOST sensitive (largest Diff, i.e. LS[0]
    in the paper's notation) to LEAST sensitive. Matches:
        LS = [argmax_l {Diff_l}, ..., argmin_l {Diff_l}]
    """
    return sorted(diffs.keys(), key=lambda name: diffs[name], reverse=True)


def select_top_k_critical_layers(diffs: Dict[str, float], k: int) -> List[str]:
    """The first K indices of LS — the layers designated for unlearning
    (i.e. where sparse adapters will be attached)."""
    ranked = rank_layers_by_sensitivity(diffs)
    if k > len(ranked):
        raise ValueError(
            f"Requested K={k} critical layers but the model only has {len(ranked)} "
            f"candidate layers. Reduce 'num_unlearning_layers' in your FU config."
        )
    return ranked[:k]
