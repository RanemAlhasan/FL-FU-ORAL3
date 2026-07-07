"""
Critical Layer Identification (CLI) — paper Section 4.1, Eq. 11-13.

Generic (not Flower/hospital-specific): works with a plain list of
per-client nn.Module clones and their data sizes, so the SAME code is
usable for CIFAR-10 (client_loaders: List[DataLoader]) and oral3
(hospital loaders bridged the same way).

Algorithm (matches the paper exactly):
  1. Given a distributed global model M^r = {p_1, ..., p_L} (L layers)
     and N clients, each client trains its OWN clone for one local
     round.
  2. For each layer l, compute the Manhattan distance between each
     client's trained layer and the ORIGINAL (untouched) global layer
     (Eq. 12): Diff(p_l^n, p_l) = sum_ij |p_l,ij^n - p_l,ij|.
  3. Aggregate across clients via a DATA-SIZE-WEIGHTED average (Eq. 13):
     Diff_l = sum_n (|D_n| / sum_k |D_k|) * Diff_l^n.
  4. LS = argsort(Diff_l) descending — the first K layers are the most
     sensitive to client-specific knowledge and are designated as
     unlearning layers.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.fl.core import SGD_MOMENTUM, SGD_WEIGHT_DECAY


def _named_float_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Only real nn.Parameters (weight/bias tensors) are valid CLI
    candidates / adapter targets. Buffers (BatchNorm running_mean,
    running_var, num_batches_tracked) are DELIBERATELY EXCLUDED: they are
    running statistics, not learnable weights, and PyTorch's batch_norm
    explicitly refuses to backpropagate through them (this caused a real
    "cudnn_batch_norm is not differentiable w.r.t. running_var" crash
    when a BN buffer was previously allowed to be selected as a critical
    layer and wrapped in a trainable sparse adapter delta). BatchNorm's
    AFFINE parameters (weight/bias, i.e. gamma/beta) are still eligible —
    only running statistics are excluded."""
    return {name: p.detach() for name, p in model.named_parameters()}


def compute_layer_diffs(
    source_model: nn.Module,
    client_models: List[nn.Module],
    client_data_sizes: List[int],
) -> Dict[str, float]:
    """Eq. 11-13. `client_models` must each be a clone of `source_model`
    that has undergone one local round of training — this function does
    NOT do any training itself (see run_critical_layer_identification for
    that), it only computes the diffs.

    Returns {layer_name: weighted_diff_value}, one entry per named
    parameter/buffer key (a "layer" in the paper's sense corresponds to
    one nn.Parameter/buffer tensor here — e.g. 'layer4.1.conv2.weight').
    """
    total_size = sum(client_data_sizes)
    source_params = _named_float_params(source_model)

    diffs: Dict[str, float] = {name: 0.0 for name in source_params}

    for client_model, data_size in zip(client_models, client_data_sizes):
        client_params = _named_float_params(client_model)
        weight = data_size / total_size
        for name, source_p in source_params.items():
            if name not in client_params:
                continue
            client_p = client_params[name]
            if client_p.device != source_p.device:
                client_p = client_p.to(source_p.device)
            manhattan = torch.sum(torch.abs(client_p - source_p)).item()
            diffs[name] += weight * manhattan

    return diffs


def select_top_k_critical_layers(diffs: Dict[str, float], num_unlearning_layers: int) -> List[str]:
    """LS = argsort(Diff_l) descending; return the first K names."""
    sorted_layers = sorted(diffs.items(), key=lambda kv: -kv[1])
    k = min(num_unlearning_layers, len(sorted_layers))
    return [name for name, _ in sorted_layers[:k]]


def run_critical_layer_identification(
    source_model: nn.Module,
    remember_client_loaders: List[DataLoader],
    remember_client_data_sizes: List[int],
    device: str,
    local_epochs: int = 1,
    learning_rate: float = 0.005,
) -> Dict[str, float]:
    """One short federated round (paper Sec 5.2: "the gap is most
    pronounced when Epoch=1") where each REMEMBER client trains its own
    clone of `source_model` for `local_epochs` (default 1, matching the
    paper), then Eq. 11-13 diffs are computed against the untouched
    `source_model`. `source_model` itself is NEVER mutated — every client
    trains on an independent deepcopy.
    """
    criteria = nn.CrossEntropyLoss()
    client_models = []

    for loader in remember_client_loaders:
        model = copy.deepcopy(source_model).to(device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                               momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
        for _ in range(local_epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criteria(model(data), target)
                loss.backward()
                optimizer.step()
        model.to("cpu")
        client_models.append(model)

    return compute_layer_diffs(source_model, client_models, remember_client_data_sizes)
