"""
Domain-adaptation-aware version of fused_training.py.

Identical in structure and evaluation to fused_training.py's train_normal
and forget_client_train — same test_client_forget calls, same history
format, same logging — the ONLY difference is that global_train_once /
fedavg are replaced with their core_domain.py equivalents, which accept
an `algorithm` ("fedavg" | "fedbn" | "fedprox") and thread a persistent
`client_bn_states` dict across all rounds of ONE call (i.e. across the
whole train_normal or forget_client_train run — matching how Engine A's
Flower FedBN strategy persists BN state across all rounds of one FL
simulation).

algorithm="fedavg" (default) makes both functions byte-for-byte
equivalent to fused_training.py's originals — this file is a strict
superset, not a fork with diverging baseline behavior.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.client_forget_eval import test_client_forget
from src.fl.core_domain import (assemble_representative_bn_model, evaluate,
                                 fedavg_domain, global_train_once_domain)
from src.models.resnet_lora import build_lora_adapter


def train_normal_domain(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    global_epochs: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    algorithm: str = "fedavg",
    fedprox_mu: float = 0.01,
    fedmoon_mu: float = 1.0,
    fedmoon_temperature: float = 0.5,
    client_data_sizes: List[int] = None,
    logger=None,
) -> Tuple[nn.Module, dict]:
    """Domain-adaptation-aware train_normal. See fused_training.py's
    train_normal for the base contract (identical here when
    algorithm='fedavg').

    `client_data_sizes`: optional, one entry per entry of `client_loaders`
    (same order) — each client's local dataset size, used to weight
    aggregation the same way Phase 1's Flower FedAvg strategy does
    (weighted by num_examples). Omit to keep the original unweighted
    mean."""
    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}
    client_bn_states: dict = {}
    client_prev_models: dict = {}

    for epoch in range(global_epochs):
        client_models = global_train_once_domain(
            global_model, client_loaders, local_epochs, learning_rate, device,
            algorithm=algorithm, mu=fedprox_mu, moon_mu=fedmoon_mu, moon_temperature=fedmoon_temperature,
            client_bn_states=client_bn_states, client_prev_models=client_prev_models,
        )
        global_model = fedavg_domain(client_models, algorithm=algorithm, weights=client_data_sizes)

        eval_model = global_model
        if algorithm == "fedbn" and client_bn_states:
            eval_model = assemble_representative_bn_model(global_model, client_bn_states)

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            eval_model, test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = f"[train_normal:{algorithm}] Epoch={epoch}, avg_f_acc={avg_f_acc:.4f}, avg_r_acc={avg_r_acc:.4f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("fl/forget_client_acc", avg_f_acc, epoch)
            logger.log_scalar("fl/remember_client_acc", avg_r_acc, epoch)

    final_model = global_model
    if algorithm == "fedbn" and client_bn_states:
        final_model = assemble_representative_bn_model(global_model, client_bn_states)

    return final_model, history


def forget_client_train_domain(
    trained_global_model: nn.Module,
    all_clean_client_loaders: List[DataLoader],
    attacked_test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    fused_iterations: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    algorithm: str = "fedavg",
    fedprox_mu: float = 0.01,
    fedmoon_mu: float = 1.0,
    fedmoon_temperature: float = 0.5,
    client_data_sizes: List[int] = None,
    logger=None,
) -> Tuple[nn.Module, dict]:
    """Domain-adaptation-aware forget_client_train. Same contract as
    fused_training.py's forget_client_train (identical when
    algorithm='fedavg') — see that function's docstring for the
    clean-train / attacked-test asymmetry, which is UNCHANGED here.

    algorithm='fedbn': the LoRA adapter model's BatchNorm stays local per
    REMEMBER client across all `fused_iterations` rounds, never
    aggregated. algorithm='fedprox': adds the proximal term to each
    remember client's adapter loss. algorithm='fedmoon': adds the
    model-contrastive term, persisting each remember client's own
    previous-round adapter model across rounds (client_prev_models) —
    same persistence pattern as FedBN's BN state.

    `client_data_sizes`: optional, one entry per entry of
    `all_clean_client_loaders` (ALL clients, same order/indexing as
    `forget_client_idx` — NOT pre-filtered to remember clients; this
    function filters it down internally, same as the loaders). Used to
    weight remember-client aggregation the same way Phase 1's Flower
    FedAvg strategy does (weighted by num_examples). Omit to keep the
    original unweighted mean.
    """
    num_clients = len(all_clean_client_loaders)
    remember_idx = [i for i in range(num_clients) if i not in forget_client_idx]
    remember_client_loaders = [all_clean_client_loaders[i] for i in remember_idx]
    remember_data_sizes = (
        [client_data_sizes[i] for i in remember_idx] if client_data_sizes is not None else None
    )

    fused_model = build_lora_adapter(copy.deepcopy(trained_global_model))

    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}
    client_bn_states: dict = {}
    client_prev_models: dict = {}

    for epoch in range(fused_iterations):
        fused_model.train()
        client_models = global_train_once_domain(
            fused_model, remember_client_loaders, local_epochs, learning_rate, device,
            algorithm=algorithm, mu=fedprox_mu, moon_mu=fedmoon_mu, moon_temperature=fedmoon_temperature,
            client_bn_states=client_bn_states, client_prev_models=client_prev_models,
        )
        fused_model = fedavg_domain(client_models, algorithm=algorithm, weights=remember_data_sizes)
        fused_model.eval()

        eval_model = fused_model
        if algorithm == "fedbn" and client_bn_states:
            eval_model = assemble_representative_bn_model(fused_model, client_bn_states)

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            eval_model, attacked_test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = (f"[FUSED forget_client_train:{algorithm}] Epoch={epoch}, "
               f"avg_r_acc={avg_r_acc:.4f}, avg_f_acc={avg_f_acc:.4f}")
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("fu/forget_client_acc", avg_f_acc, epoch)
            logger.log_scalar("fu/remember_client_acc", avg_r_acc, epoch)

    final_model = fused_model
    if algorithm == "fedbn" and client_bn_states:
        final_model = assemble_representative_bn_model(fused_model, client_bn_states)

    return final_model, history
