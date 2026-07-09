"""
Faithful port of FUSED-Code's algs/fl_base.py — the core FL primitives used
by both plain FL training and FUSED's adapter-only training.

Key fidelity points, copied verbatim from the source:
  - Optimizer: SGD(lr=args.lr, momentum=0.9, weight_decay=5e-4) — NOT Adam,
    despite our oral-cancer framework defaulting to Adam everywhere. For
    CIFAR-10, args.lr = 0.005 (see utils.py::model_init's cifar10 branch).
  - fedavg(): UNWEIGHTED mean across client models — the original's
    data-size-weighted averaging code is present but COMMENTED OUT:
        # avg_state_dict[layer] += local_state_dicts[client_idx][layer]*self.args.datasize_ls[client_idx]
        # if 'num_batches_tracked' in layer:
        #     avg_state_dict[layer] = (avg_state_dict[layer]/sum(self.args.datasize_ls)).long()
        # else:
        #     avg_state_dict[layer] /= sum(self.args.datasize_ls)
    The ACTIVE code just below it does a plain unweighted mean:
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)
    This is a real, faithfully-reproduced quirk: standard FedAvg weights by
    client data volume; this implementation does not, despite the paper's
    Eq. 13 describing data-volume weighting (that weighting is used for the
    CLI Diff aggregation in the paper's math, NOT for the actual FedAvg
    model averaging in this code — these are two different uses of
    "weighted averaging" that the paper conflates and the code keeps
    separate, with the FedAvg one ultimately unweighted).
  - local_train(): standard SGD loop, args.local_epoch passes over the
    client's DataLoader, CrossEntropyLoss.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 5e-4


def local_train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    local_epochs: int,
    device: str,
) -> nn.Module:
    """Faithful port of Base.local_train (non-text branch only — this
    reproduction never uses the text/AdamW path)."""
    criteria = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criteria(pred, target)
            loss.backward()
            optimizer.step()
    return model


def global_train_once(
    global_model: nn.Module,
    client_data_loaders: List[DataLoader],
    local_epochs: int,
    learning_rate: float,
    device: str,
) -> List[nn.Module]:
    """Faithful port of Base.global_train_once (non-text, non-infocom22
    branch only): each client gets an independent deep copy of the current
    global model, trains it locally via SGD, and the list of resulting
    client models is returned (for fedavg() to aggregate).

    NOTE: trainable-parameter filtering happens automatically here, since
    `optim.SGD(model.parameters(), ...)` would error on frozen (requires_
    grad=False) params being passed with no gradient — for FUSED's LoRA
    adapter training, the caller is expected to pass a model where only
    LoRA params have requires_grad=True (peft.get_peft_model already
    enforces this), and we explicitly filter to trainable params here for
    robustness, matching the *effective* behavior of the original (which
    relies on `model.parameters()` only including effectively-trainable
    params being updated, since frozen params simply receive zero gradient
    and SGD updates --- relevant only when frozen params are NOT explicitly
    excluded from the optimizer, where SGD would still be technically valid
    but wasteful; we exclude them outright for clarity and efficiency).
    """
    device_cpu = torch.device("cpu")
    client_models = []

    for client_loader in client_data_loaders:
        model = copy.deepcopy(global_model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(trainable_params, lr=learning_rate,
                               momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
        model.train()
        model.to(device)
        model = local_train(model, optimizer, client_loader, local_epochs, device)
        model.to(device_cpu)
        client_models.append(model)

    return client_models


def fedavg(local_models: List[nn.Module]) -> nn.Module:
    """Faithful port of Base.fedavg(): UNWEIGHTED mean of state_dicts
    across all provided local models. See module docstring for the
    weighted-vs-unweighted discrepancy this faithfully preserves.

    Matches the original's pattern of zeroing each layer's tensor in place
    (`avg_state_dict[layer] = 0`, which in the original relies on Python/
    PyTorch broadcasting a Python int 0 against the existing tensor's
    dtype/shape) then summing every client's value into it, then dividing
    by client count — special-casing `num_batches_tracked` buffers (BN
    running-stat counters) to integer division, exactly as the source does.
    """
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()
    local_state_dicts = [model.state_dict() for model in local_models]

    for layer in avg_state_dict.keys():
        summed = torch.zeros_like(local_state_dicts[0][layer], dtype=torch.float32)
        for client_idx in range(len(local_models)):
            summed += local_state_dicts[client_idx][layer].float()

        if "num_batches_tracked" in layer:
            avg_state_dict[layer] = (summed / len(local_models)).long()
        else:
            avg_state_dict[layer] = (summed / len(local_models)).to(local_state_dicts[0][layer].dtype)

    global_model.load_state_dict(avg_state_dict)
    return global_model


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[float, float]:
    """Faithful port of Base.test() (non-text, non-sample-paradigm branch):
    returns (test_loss, test_acc)."""
    model.eval()
    model.to(device)
    criteria = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # BUG FIX: criteria(...) defaults to reduction='mean', i.e. each
        # term is already a PER-BATCH mean loss. The original code summed
        # these per-batch means and then divided by the total SAMPLE count
        # (`len(test_loader.dataset)`), which is dividing a sum-of-means by
        # the wrong denominator — it systematically deflates test_loss by
        # roughly a factor of batch_size and has no effect on test_acc
        # (accuracy bookkeeping below was always correct). Weighting each
        # batch's mean loss by its own batch size and dividing by the total
        # sample count gives the correct overall mean loss, matching how
        # src/fl/client.py::_local_evaluate already computes it.
        test_loss += criteria(output, target).item() * len(target)
        pred = torch.argmax(output, dim=1)
        correct += torch.sum(torch.eq(pred.cpu(), target.cpu())).item()
        total += len(target)

    test_loss /= max(1, total)
    test_acc = correct / max(1, total)
    return test_loss, test_acc
