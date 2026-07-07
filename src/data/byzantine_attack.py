"""
Faithful port of FUSED-Code's utils.py::baizhanting_attack().

This is the paper's "client affected by Byzantine attack... mode of attack
is label flipping" (Sec 5.3) made concrete: for each client designated as
"to be forgotten," EVERY sample's label is shifted by +1 (mod num_classes):
    if label < num_classes - 1: label += 1
    elif label == num_classes - 1: label = 0

This is applied to BOTH that client's train data AND its (per-client) test
data, replacing the original DataLoaders in place. "Forgetting" this client
later means removing the influence of this corrupted, mislabeled data from
the global model.

NOTE: despite "label flipping" suggesting randomness, this is a
DETERMINISTIC cyclic shift, not a random reassignment — every sample of
class c becomes class (c+1) % num_classes, with no randomness involved.
We replicate this exactly.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _shift_loader_labels(loader: DataLoader, num_classes: int, batch_size: int) -> DataLoader:
    """Rebuild a DataLoader with every label cyclically shifted by +1 mod
    num_classes, preserving images unchanged. Faithful to the original's
    manual extend-then-rebuild-DataLoader pattern."""
    images, labels = [], []
    for x, y in loader:
        images.extend(x)
        labels.extend(y)

    shifted_labels = []
    for label in labels:
        label_int = int(label)
        if label_int < num_classes - 1:
            shifted_labels.append(label_int + 1)
        else:  # label_int == num_classes - 1
            shifted_labels.append(0)

    images_arr = torch.stack([img if isinstance(img, torch.Tensor) else torch.tensor(img) for img in images]).float()
    labels_arr = torch.tensor(shifted_labels, dtype=torch.long)

    dataset = TensorDataset(images_arr, labels_arr)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def baizhanting_attack(
    client_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    num_classes: int,
    local_batch_size: int,
) -> tuple:
    """Apply the label-shift Byzantine attack to every client in
    `forget_client_idx`, for both their train (client_loaders) and test
    (test_loaders) data. Returns NEW lists with those clients' loaders
    replaced — does not mutate the input lists in place, to keep the
    "clean" (un-attacked) data available for use elsewhere (e.g. building
    the retrain baseline, which the original also threads `client_all_loaders`
    — the clean copy — through separately via `copy.deepcopy`)."""
    new_client_loaders = list(client_loaders)
    new_test_loaders = list(test_loaders)

    for user in forget_client_idx:
        new_client_loaders[user] = _shift_loader_labels(client_loaders[user], num_classes, local_batch_size)
        new_test_loaders[user] = _shift_loader_labels(test_loaders[user], num_classes, local_batch_size)

    return new_client_loaders, new_test_loaders
