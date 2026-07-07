"""
Faithful port of FUSED-Code's dataset/data_utils.py::split_proxy() and
split_data().

split_proxy carves off `proxy_frac` (default 0.2 in main.py) of EACH
client's data, per class, as a held-out "proxy" set used later for
membership-inference-attack shadow-model training (utils.py::
train_shadow_model). The remaining (1 - proxy_frac) becomes that client's
actual train+test data.

split_data then does a per-client train_test_split:
  - train_size = 0.7 when forget_paradigm == 'client'   (NOTE: 70/30, not
    a conventional 80/20 or a fixed external test set — this is specific
    to client-unlearning experiments in the original code)
  - train_size = 0.99 otherwise (class/sample unlearning paradigms)

Both functions operate on already-Dirichlet-partitioned per-client
(image, label) arrays — they don't know or care how clients were formed.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def split_proxy(
    client_images: List[np.ndarray],
    client_labels: List[np.ndarray],
    num_clients: int,
    num_classes: int,
    proxy_frac: float = 0.2,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """For each client, carve off `proxy_frac` of each class's samples into
    a separate "proxy" pool; the rest stays as that client's main data.
    Returns (client_x, client_y, proxy_x, proxy_y), each a list-per-client
    of numpy arrays (images, labels respectively).

    Faithful to the original's per-class proxy carving (not a flat
    per-client split), preserving FUSED-Code's class-balance-aware
    intent for the shadow-model / MIA pipeline.
    """
    client_x, client_y = [], []
    proxy_x, proxy_y = [], []

    for client in range(num_clients):
        dataset_image = client_images[client]
        dataset_label = client_labels[client]
        idxs = np.arange(len(dataset_label))

        all_class_x, all_class_y = [], []
        all_class_x_proxy, all_class_y_proxy = [], []

        for c in range(num_classes):
            idx_for_class = idxs[dataset_label == c]
            num_class_proxy = int(len(idx_for_class) * proxy_frac)
            if len(idx_for_class) == 0:
                continue
            idx_class_proxy = np.random.choice(idx_for_class, num_class_proxy, replace=False) \
                if num_class_proxy > 0 else np.array([], dtype=int)
            idx_class_client = np.array(sorted(set(idx_for_class.tolist()) - set(idx_class_proxy.tolist())))

            if len(idx_class_proxy) > 0:
                all_class_x_proxy.extend(dataset_image[idx_class_proxy])
                all_class_y_proxy.extend(dataset_label[idx_class_proxy])
            if len(idx_class_client) > 0:
                all_class_x.extend(dataset_image[idx_class_client])
                all_class_y.extend(dataset_label[idx_class_client])

        client_x.append(np.array(all_class_x))
        client_y.append(np.array(all_class_y))
        proxy_x.append(np.array(all_class_x_proxy))
        proxy_y.append(np.array(all_class_y_proxy))

    return client_x, client_y, proxy_x, proxy_y


def split_data(
    X: List[np.ndarray],
    y: List[np.ndarray],
    batch_size: int,
    test_batch_size: int,
    forget_paradigm: str = "client",
    num_workers: int = 2,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Per-client train/test DataLoader construction, matching the
    original's train_size selection (0.7 for client unlearning, 0.99
    otherwise) and DataLoader batch sizes (local_batch_size for train,
    test_batch_size for test). `num_workers` defaults to 2 to match the
    original; set to 0 in constrained/sandboxed environments where
    multiprocessing DataLoader workers can hang — purely a throughput
    knob, no effect on correctness.

    DEVIATION FROM THE ORIGINAL: training loaders use `drop_last=True`,
    which the original code does not set. Without it, a client whose
    training-shard size isn't evenly divisible by `batch_size` yields a
    final mini-batch smaller than batch_size on the last iteration of each
    epoch — and if that remainder is exactly 1, ResNet's BatchNorm2d layers
    raise `ValueError: Expected more than 1 value per channel when
    training` (variance is undefined for a single sample). With 50
    Dirichlet-partitioned clients of uneven, often-small size, this is a
    real failure mode hit in practice. `drop_last=True` discards that
    final partial batch instead of crashing. Test loaders are NOT changed
    (no drop_last) since evaluation should see every sample, and a size-1
    eval batch is harmless — model.eval() uses BatchNorm's running
    statistics, not the batch's own statistics, so no crash occurs there
    regardless of batch size.
    """
    train_size = 0.7 if forget_paradigm == "client" else 0.99

    client_loaders, test_loaders = [], []
    for i in range(len(y)):
        if len(y[i]) == 0:
            # Degenerate client with no data after proxy carving — skip
            # gracefully rather than crashing train_test_split on an
            # empty array (the original code does not explicitly guard
            # this; we add the guard defensively without changing behavior
            # for any non-degenerate client).
            client_loaders.append(DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)),
                                              batch_size=batch_size))
            test_loaders.append(DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)),
                                            batch_size=test_batch_size))
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True,
        )

        train_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(yy, dtype=torch.long))
                      for x, yy in zip(X_train, y_train)]
        test_data = [(torch.tensor(x, dtype=torch.float32), torch.tensor(yy, dtype=torch.long))
                     for x, yy in zip(X_test, y_test)]

        client_loaders.append(DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, drop_last=True))
        test_loaders.append(DataLoader(test_data, batch_size=test_batch_size, shuffle=True))

    return client_loaders, test_loaders
