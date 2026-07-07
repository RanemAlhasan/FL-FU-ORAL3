"""
Faithful port of FUSED-Code's algs/fused_unlearning.py — FUSED class,
specialized to forget_paradigm='client' (the paper's Table 1 client-
unlearning scenario on CIFAR-10).

Two phases, matching FUSED.train_normal() and FUSED.forget_client_train():

  Phase A (train_normal): plain FL training for `global_epoch` rounds,
  using the BYZANTINE-ATTACKED client data (the forget client's labels are
  already shifted before this phase begins — see main.py's flow: the
  attack is applied via baizhanting_attack() BEFORE train_normal is called).

  Phase B (forget_client_train): reload the Phase-A global model fresh
  from disk, wrap it in a LoRA adapter (build_lora_adapter), then run
  `global_epoch` MORE federated rounds — but training the adapter ONLY on
  the (clean, non-attacked) remember clients' data, with the forget
  client(s) entirely excluded from `client_data_loaders`. Each round:
  local-train the LoRA model per remember client, FedAvg the resulting
  models, evaluate test_client_forget.

NOTE on epoch counts: the original reuses `args.global_epoch` for BOTH
phases (same CLI flag controls both the initial FL training round count
AND the FUSED adapter-training round count) — there is no separate
"--fused_iterations" knob in the source. We replicate this: one
`global_epochs` config value drives both phases by default, though our
config schema allows overriding the adapter phase's round count
separately if you want to deviate from the original for your own analysis.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.client_forget_eval import test_client_forget
from src.fl.core import evaluate, fedavg, global_train_once
from src.models.resnet_lora import build_lora_adapter


def train_normal(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    global_epochs: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    logger=None,
) -> Tuple[nn.Module, dict]:
    """Faithful port of FUSED.train_normal() for forget_paradigm='client'.
    `client_loaders`/`test_loaders` should already have the Byzantine
    attack applied (caller's responsibility — see
    scripts/run_fused_cifar10.py for the exact ordering, matching main.py).

    Returns (final_global_model, history) where history has per-round
    avg_f_acc/avg_r_acc lists for logging/plotting.
    """
    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}

    for epoch in range(global_epochs):
        client_models = global_train_once(global_model, client_loaders, local_epochs, learning_rate, device)
        global_model = fedavg(client_models)

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            global_model, test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = f"[train_normal] Epoch={epoch}, avg_f_acc={avg_f_acc:.4f}, avg_r_acc={avg_r_acc:.4f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("fl/forget_client_acc", avg_f_acc, epoch)
            logger.log_scalar("fl/remember_client_acc", avg_r_acc, epoch)

    return global_model, history


def forget_client_train(
    trained_global_model: nn.Module,
    all_clean_client_loaders: List[DataLoader],
    attacked_test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    fused_iterations: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    logger=None,
) -> Tuple[nn.Module, dict]:
    """Faithful port of FUSED.forget_client_train(). Matches the original's
    exact call contract from main.py:

        unlearning_model = case.forget_client_train(
            copy.deepcopy(model),               # trained_global_model
            copy.deepcopy(client_all_loaders),    # all_clean_client_loaders (CLEAN, unattacked, ALL clients)
            test_loaders_process,                 # attacked_test_loaders (ATTACKED test data)
        )

    Internally, exactly like the source:
        selected_clients = [i for i in range(num_user) if i not in forget_client_idx]
        select_client_loaders = select_part_sample(args, client_all_loaders, selected_clients)
    i.e. THIS function does the remember-client filtering itself, given ALL
    clients' clean loaders plus the forget indices — callers should NOT
    pre-filter before calling this. (`select_part_sample` with
    `cut_sample=1.0`, the default, is a no-op resampling of the same data —
    we skip reproducing that no-op for cut_sample=1.0 and only filter by
    client index, since cut_sample<1.0 partial-data experiments are not
    part of this reproduction's scope.)

    Evaluation uses `attacked_test_loaders` for ALL clients (including
    remember clients) — this asymmetry (clean train, attacked test) is
    intentional in the original and faithfully replicated here: FA measures
    whether the model still predicts the ATTACKED (shifted) labels for the
    forget client, while RA measures accuracy on remember clients' own
    (unattacked) test data, since baizhanting_attack only ever touches the
    forget client's loaders — remember clients' "attacked_test_loaders"
    entries are actually unchanged from their original clean state.
    """
    num_clients = len(all_clean_client_loaders)
    remember_client_loaders = [
        all_clean_client_loaders[i] for i in range(num_clients) if i not in forget_client_idx
    ]

    fused_model = build_lora_adapter(copy.deepcopy(trained_global_model))

    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}

    for epoch in range(fused_iterations):
        fused_model.train()
        client_models = global_train_once(fused_model, remember_client_loaders, local_epochs, learning_rate, device)
        fused_model = fedavg(client_models)
        fused_model.eval()

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            fused_model, attacked_test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = f"[FUSED forget_client_train] Epoch={epoch}, avg_r_acc={avg_r_acc:.4f}, avg_f_acc={avg_f_acc:.4f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("fu/forget_client_acc", avg_f_acc, epoch)
            logger.log_scalar("fu/remember_client_acc", avg_r_acc, epoch)

    return fused_model, history