"""
Top-level data pipeline, faithful port of FUSED-Code's dataset/
generate_data.py::data_init(), specialized to data_name='cifar10' and
forget_paradigm='client' (the paper's Table 1 client-unlearning scenario).

Original flow (data_init):
  1. Load CIFAR-10 train/test sets via torchvision (already-transformed
     tensors).
  2. Flatten BOTH train and test sets into one pooled (dataset_x, dataset_y)
     array (note: for forget_paradigm == 'client', test images get pooled
     in too — see the `if FL_params.forget_paradigm == 'client':` branch in
     the original; this means the "test set" CIFAR-10 normally provides is
     entirely absorbed into the federated client pool, and each client's
     own held-out test data comes only from `split_data`'s per-client 70/30
     split, NOT from torchvision's separate 10k-image test set).
  3. Dirichlet-partition the pooled data across num_clients.
  4. split_proxy: carve a proxy_frac slice off each client for MIA shadow
     training.
  5. split_data: per-client 70/30 train/test split (client paradigm).

This module reproduces that exact flow. It deliberately does NOT support
other forget_paradigms or datasets — see the standalone oral-cancer
framework (src/data/dataset.py etc.) for the general-purpose version; this
module exists solely to faithfully reproduce the CIFAR-10 Table 1 numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cifar10 import load_cifar10
from src.data.dirichlet_partition import dirichlet_partition, print_partition_summary
from src.data.proxy_split import split_data, split_proxy


@dataclass
class CIFAR10FederatedData:
    client_loaders: List[DataLoader]       # per-client train loaders (70% of that client's data)
    test_loaders: List[DataLoader]         # per-client test loaders (30% of that client's data)
    proxy_client_loaders: List[DataLoader]  # per-client proxy train loaders (for MIA shadow models)
    proxy_test_loaders: List[DataLoader]    # per-client proxy test loaders
    client_data_sizes: List[int]            # len(X[client]) per client, BEFORE proxy carving
                                            # (matches FL_params.datasize_ls in the original)


def _flatten_dataset(loader: DataLoader) -> tuple:
    """Pull every (image, label) pair out of a DataLoader into flat numpy
    arrays, exactly mirroring the original's manual accumulation loop:
        for train_data in train_loader:
            x_train, y_train = train_data
            dataset_x.extend(x_train.cpu().detach().numpy())
            dataset_y.extend(y_train.cpu().detach().numpy())
    """
    xs, ys = [], []
    for batch_x, batch_y in loader:
        xs.extend(batch_x.cpu().detach().numpy())
        ys.extend(batch_y.cpu().detach().numpy())
    return np.array(xs), np.array(ys)


def build_cifar10_federated_data(
    dataset_root: str,
    num_clients: int,
    num_classes: int,
    alpha: float,
    local_batch_size: int,
    test_batch_size: int,
    proxy_frac: float = 0.2,
    forget_paradigm: str = "client",
    seed: int = None,
    verbose: bool = True,
    num_workers: int = 2,
) -> CIFAR10FederatedData:
    """End-to-end faithful reproduction of FUSED-Code's data_init() for
    data_name='cifar10', forget_paradigm='client'.

    `num_workers` defaults to 2 to match the original
    (kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {} —
    actually the ORIGINAL uses num_workers=0 via that kwargs dict, but
    explicitly passes num_workers=2 as an override in the DataLoader calls
    themselves; we preserve that effective value of 2). Set to 0 if running
    in an environment where multiprocessing workers misbehave (e.g. some
    sandboxed/restricted containers) — this has no effect on correctness,
    only on data-loading throughput.
    """
    trainset, testset = load_cifar10(dataset_root)

    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=num_workers)
    train_loader = DataLoader(trainset, batch_size=local_batch_size, shuffle=True, num_workers=num_workers)

    train_x, train_y = _flatten_dataset(train_loader)
    dataset_x_list = [train_x]
    dataset_y_list = [train_y]

    if forget_paradigm == "client":
        # Original: test images get pooled into the federated client data
        # too, for the client-unlearning scenario specifically.
        test_x, test_y = _flatten_dataset(test_loader)
        dataset_x_list.append(test_x)
        dataset_y_list.append(test_y)

    dataset_x = np.concatenate(dataset_x_list, axis=0)
    dataset_y = np.concatenate(dataset_y_list, axis=0)

    if verbose:
        print(f"Pooled dataset: {dataset_x.shape[0]} samples "
              f"(train{' + test' if forget_paradigm == 'client' else ''}), "
              f"partitioning across {num_clients} clients via Dirichlet(alpha={alpha})...")

    dataidx_map = dirichlet_partition(dataset_y, num_clients, num_classes, alpha, seed=seed)
    if verbose:
        print_partition_summary(dataidx_map, dataset_y)

    client_images = [dataset_x[dataidx_map[c]] for c in range(num_clients)]
    client_labels = [dataset_y[dataidx_map[c]] for c in range(num_clients)]
    client_data_sizes = [len(c) for c in client_images]

    client_x, client_y, proxy_x, proxy_y = split_proxy(
        client_images, client_labels, num_clients, num_classes, proxy_frac,
    )

    client_loaders, test_loaders = split_data(
        client_x, client_y, local_batch_size, test_batch_size, forget_paradigm, num_workers,
    )
    proxy_client_loaders, proxy_test_loaders = split_data(
        proxy_x, proxy_y, local_batch_size, test_batch_size, forget_paradigm, num_workers,
    )

    return CIFAR10FederatedData(
        client_loaders=client_loaders,
        test_loaders=test_loaders,
        proxy_client_loaders=proxy_client_loaders,
        proxy_test_loaders=proxy_test_loaders,
        client_data_sizes=client_data_sizes,
    )
