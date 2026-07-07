"""
Domain-adaptation-aware version of retrain.py's fl_retrain.

Same contract, evaluation, and history format as retrain.py — only
global_train_once/fedavg are swapped for their core_domain.py
equivalents. algorithm="fedavg" (default) is byte-for-byte identical to
retrain.py's original.

Existing FedAvg-sourced retrain checkpoints (retrain_spain_oral_330047,
etc.) are NOT reusable as the baseline for this comparison, since those
were trained with plain FedAvg throughout. To fairly compare "FUSED with
FedBN/FedProx" against "Retrain with the SAME FedBN/FedProx setting", you
need a matching retrain_domain run at the same algorithm setting — invoke
this module's fl_retrain_domain() directly, or via run_fu_lora_domain.py's
/ run_fu_cli_domain.py's --run_retrain_baseline flag (there is no separate
standalone run_retrain_domain.py script).
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.client_forget_eval import test_client_forget
from src.fl.core_domain import assemble_representative_bn_model, fedavg_domain, global_train_once_domain


def fl_retrain_domain(
    init_global_model: nn.Module,
    all_clean_client_loaders: List[DataLoader],
    attacked_test_loaders: List[DataLoader],
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
    """Domain-adaptation-aware fl_retrain. Identical contract to
    retrain.py's fl_retrain (identical behavior when algorithm='fedavg').

    NOTE: for the standalone Retrain baseline, prefer scripts/run_retrain.py
    (--set algorithm=FedBN/FedProx/FedMoon) instead of this function —
    that path already uses the verified Flower engine (src/fl/simulation.py)
    with FedBN/FedProx/FedMoon fully implemented and tested. This function
    exists only so run_fu_lora_domain.py's optional --run_retrain_baseline
    flag has a matching-engine baseline available in the SAME script/run.

    `client_data_sizes`: optional, one entry per entry of
    `all_clean_client_loaders` (ALL clients, same indexing as
    `forget_client_idx` — this function filters it down to remember
    clients internally, same as the loaders). Used to weight remember-
    client aggregation the same way Phase 1's Flower FedAvg strategy does
    (weighted by num_examples). Omit to keep the original unweighted mean.
    """
    global_model = copy.deepcopy(init_global_model)
    num_clients = len(all_clean_client_loaders)
    remember_idx = [i for i in range(num_clients) if i not in forget_client_idx]
    remember_client_loaders = [all_clean_client_loaders[i] for i in remember_idx]
    remember_data_sizes = (
        [client_data_sizes[i] for i in remember_idx] if client_data_sizes is not None else None
    )

    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}
    client_bn_states: dict = {}
    client_prev_models: dict = {}

    for epoch in range(global_epochs):
        client_models = global_train_once_domain(
            global_model, remember_client_loaders, local_epochs, learning_rate, device,
            algorithm=algorithm, mu=fedprox_mu, moon_mu=fedmoon_mu, moon_temperature=fedmoon_temperature,
            client_bn_states=client_bn_states, client_prev_models=client_prev_models,
        )
        global_model = fedavg_domain(client_models, algorithm=algorithm, weights=remember_data_sizes)

        eval_model = global_model
        if algorithm == "fedbn" and client_bn_states:
            eval_model = assemble_representative_bn_model(global_model, client_bn_states)

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            eval_model, attacked_test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = f"[Retrain:{algorithm}] Epoch={epoch}, avg_r_acc={avg_r_acc:.4f}, avg_f_acc={avg_f_acc:.4f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("retrain/forget_client_acc", avg_f_acc, epoch)
            logger.log_scalar("retrain/remember_client_acc", avg_r_acc, epoch)

    final_model = global_model
    if algorithm == "fedbn" and client_bn_states:
        final_model = assemble_representative_bn_model(global_model, client_bn_states)

    return final_model, history
