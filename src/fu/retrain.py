"""
Faithful port of FUSED-Code's algs/fl_base.py::Base.FL_Retrain(), specialized
to forget_paradigm='client'.

Unlike FUSED's forget_client_train (which starts from the ALREADY-TRAINED
Phase-A model and only trains a small LoRA adapter), Retrain starts from
the SAME initial model checkpoint, then runs `global_epoch` full federated
rounds training the WHOLE model (every parameter, not just LoRA), using
ONLY the remember clients (forget client excluded from client SELECTION,
not from a held-out split — i.e. retraining never sees the forget client's
data again at all, simulating "as if the forget client never participated
in FL training to begin with").

NOTE: faithfully reproducing one further original detail —
`FL_Retrain` calls `select_part_sample(args, client_all_loaders,
selected_clients)` on the SELECTED (remember-only) clients with
`cut_sample=1.0` by default, which (per fused_training.py's docstring) is a
no-op resampling at cut_sample=1.0 — we skip reproducing that no-op
identically and just use the remember clients' loaders directly.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.client_forget_eval import test_client_forget
from src.fl.core import fedavg, global_train_once


def fl_retrain(
    init_global_model: nn.Module,
    all_clean_client_loaders: List[DataLoader],
    attacked_test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    global_epochs: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    logger=None,
) -> Tuple[nn.Module, dict]:
    """Faithful port of Base.FL_Retrain() for forget_paradigm='client'.

    `init_global_model` should be a FRESH (untrained, or identically
    initialized) model — matching the original's
    `global_model = copy.deepcopy(init_global_model)` starting point. This
    is the FU "upper bound" baseline: full retraining from scratch with the
    forget client's data never seen, full model trainable (no LoRA/adapter
    restriction), evaluated identically to FUSED via test_client_forget so
    RA/FA/comp/comm are directly comparable.
    """
    global_model = copy.deepcopy(init_global_model)
    num_clients = len(all_clean_client_loaders)
    remember_client_loaders = [
        all_clean_client_loaders[i] for i in range(num_clients) if i not in forget_client_idx
    ]

    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}

    for epoch in range(global_epochs):
        client_models = global_train_once(global_model, remember_client_loaders, local_epochs, learning_rate, device)
        global_model = fedavg(client_models)

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            global_model, attacked_test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = f"[Retrain] Epoch={epoch}, avg_r_acc={avg_r_acc:.4f}, avg_f_acc={avg_f_acc:.4f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("retrain/forget_client_acc", avg_f_acc, epoch)
            logger.log_scalar("retrain/remember_client_acc", avg_r_acc, epoch)

    return global_model, history
