"""
Faithful port of FUSED-Code's algs/fused_unlearning.py::
relearn_unlearning_knowledge() (also duplicated identically in
fl_base.py::Base, used by both FUSED and Retrain), specialized to
forget_paradigm='client'.

ReA measures whether the model can quickly RE-learn the forgotten
knowledge if given access to the forget client's data again — a LOW ReA
after a fixed, small number of relearning rounds means the knowledge was
genuinely erased (hard to recover), not just hidden.

Faithful to the original: relearning trains ONLY on the forget client's
data (not the original model's full client set), for `global_epoch`
additional rounds (the original reuses the same epoch-count config for
this phase too), then re-evaluates test_client_forget.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.client_forget_eval import test_client_forget
from src.fl.core import fedavg, global_train_once


def relearn_unlearning_knowledge(
    unlearned_model: nn.Module,
    all_clean_client_loaders: List[DataLoader],
    attacked_test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    relearn_rounds: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    logger=None,
) -> Tuple[nn.Module, dict]:
    """Faithful port of relearn_unlearning_knowledge() for
    forget_paradigm='client'. Trains ONLY on the forget client(s)' clean
    data, starting from the already-unlearned model, for `relearn_rounds`
    federated rounds (FedAvg across forget clients if there are multiple),
    then evaluates RA/FA again — the post-relearn FA is the headline ReA
    number reported in Table 1."""
    global_model = unlearned_model
    forget_loaders = [all_clean_client_loaders[i] for i in forget_client_idx]

    history = {"round": [], "avg_f_acc": [], "avg_r_acc": []}

    for epoch in range(relearn_rounds):
        client_models = global_train_once(global_model, forget_loaders, local_epochs, learning_rate, device)
        global_model = fedavg(client_models)

        avg_f_acc, avg_r_acc, _ = test_client_forget(
            global_model, attacked_test_loaders, forget_client_idx, device, test_batch_size,
        )
        history["round"].append(epoch)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = f"[Relearn] Round={epoch}, avg_f_acc={avg_f_acc:.4f} (ReA), avg_r_acc={avg_r_acc:.4f}"
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("relearn/forget_client_acc_ReA", avg_f_acc, epoch)
            logger.log_scalar("relearn/remember_client_acc", avg_r_acc, epoch)

    final_rea = history["avg_f_acc"][-1] if history["avg_f_acc"] else float("nan")
    return global_model, {"history": history, "ReA": final_rea}
