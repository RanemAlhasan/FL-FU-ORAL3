"""
Faithful port of FUSED-Code's utils.py::test_client_forget().

For client unlearning, "remember accuracy" (RA) and "forget accuracy" (FA)
are NOT computed as one overall accuracy number per client. Instead, the
original breaks EVERY client's test set down BY CLASS LABEL, evaluates
accuracy on each (client, label) subset separately, then averages:
    - FA = mean over all (client, label) pairs where client IS in
      forget_client_idx
    - RA = mean over all (client, label) pairs where client is NOT in
      forget_client_idx

This is a meaningfully different definition from "accuracy on the forget
client's whole test set" — it's a per-class-conditional average, which
changes how much weight any single class/client contributes (a (client,
label) pair with few samples counts equally to one with many, since we're
averaging accuracy VALUES, not pooling raw correct/total counts).
We replicate this exact averaging behavior.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fl.core import evaluate


def _split_loader_by_label(loader: DataLoader, test_batch_size: int) -> dict:
    """Faithful port of the original's per-label DataLoader splitting
    inside test_client_forget:
        label_data_dict = {}
        for data, target in test_loader:
            ...
            label_data_dict[label].append((data, label))
        for label, data_list in label_data_dict.items():
            class_loader = DataLoader(data_list, batch_size=len(data_list), shuffle=True)

    NOTE: the original uses batch_size=len(data_list) (i.e. the WHOLE
    per-label subset in a single batch) for client-forget evaluation,
    unlike test_class_forget which uses args.test_batch_size. We replicate
    this exactly — it only affects evaluation batching, not correctness of
    the accuracy/loss computation itself, but we keep it faithful in case
    batch-size-dependent behavior (e.g. BatchNorm in eval mode uses running
    stats, so this is actually inconsequential either way)."""
    label_data_dict: dict = {}
    for data, target in loader:
        for i in range(data.size(0)):
            label = int(target[i].item())
            label_data_dict.setdefault(label, []).append((data[i], target[i]))

    label_loaders = {}
    for label, data_list in label_data_dict.items():
        images = torch.stack([d[0] for d in data_list])
        labels = torch.stack([d[1] for d in data_list])
        label_loaders[label] = DataLoader(
            TensorDataset(images, labels), batch_size=len(data_list), shuffle=True,
        )
    return label_loaders


def test_client_forget(
    model: torch.nn.Module,
    test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    device: str,
    test_batch_size: int = 64,
) -> Tuple[float, float, List[list]]:
    """Faithful port of test_client_forget. Returns (avg_f_acc, avg_r_acc,
    test_result_ls), where test_result_ls rows are
    [client_id, class_id, label_num, test_acc, test_loss] (the original
    also includes an `epoch` column as the first element; we omit it here
    since this function is epoch-agnostic — callers can prepend epoch when
    logging, see scripts/run_fused_cifar10.py)."""
    forget_acc_ls, remember_acc_ls = [], []
    test_result_ls = []

    for client_id, test_loader in enumerate(test_loaders):
        label_loaders = _split_loader_by_label(test_loader, test_batch_size)

        for label, loader in label_loaders.items():
            test_loss, test_acc = evaluate(model, loader, device)
            label_num = len(loader.dataset)
            test_result_ls.append([client_id, label, label_num, test_acc, test_loss])

            if client_id in forget_client_idx:
                forget_acc_ls.append(test_acc)
            else:
                remember_acc_ls.append(test_acc)

    avg_f_acc = sum(forget_acc_ls) / len(forget_acc_ls) if forget_acc_ls else float("nan")
    avg_r_acc = sum(remember_acc_ls) / len(remember_acc_ls) if remember_acc_ls else float("nan")
    return avg_f_acc, avg_r_acc, test_result_ls
