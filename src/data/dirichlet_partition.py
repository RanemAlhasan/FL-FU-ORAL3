"""
Faithful port of FUSED-Code's dataset/data_utils.py::separate_data(),
'dir' (Dirichlet) branch ONLY (the 'pat' branch is not used by the paper's
main CIFAR-10 client-unlearning experiments, which use alpha=1.0 Dirichlet
partitioning per main.py's --partition='dir' default).

Original source (verbatim logic, lightly restructured for clarity/typing):

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = len(classes_ls)
        N = len(dataset_label)
        try_cnt = 1
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, num_clients))
                proportions = np.array([p*(len(idx_j) < N/num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

`least_samples = 100` is a module-level constant in the original
(dataset/data_utils.py), retained here with the same name/value — it is the
MINIMUM size of the SMALLEST individual client's shard that must be cleared
before the partition is accepted; the while-loop retries the entire
Dirichlet draw (for all classes) until every client's shard has at least
100 samples. With CIFAR-10's actual scale (60,000 images across 50 clients,
~1,200 samples/client on average), this threshold is comfortably satisfiable
and the retry loop terminates quickly in practice; it is NOT satisfiable
with very small synthetic test datasets (e.g. a few hundred samples across
many clients) — see this reproduction's test suite, which uses a
realistically-scaled synthetic dataset for exactly this reason.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

LEAST_SAMPLES = 100  # matches FUSED-Code's dataset/data_utils.py module constant


def dirichlet_partition(
    dataset_labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int = None,
) -> Dict[int, List[int]]:
    """Partition sample indices across `num_clients` clients via a
    Dirichlet(alpha) distribution per class, retrying until every client's
    shard clears LEAST_SAMPLES. Returns {client_idx: [sample indices]}.

    This is an index-only partitioner — it does not touch the actual
    image/label tensors, matching the separation of concerns in the
    original (`separate_data` assigns X[client]/y[client] using the
    returned `dataidx_map` immediately after this loop in the source).
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    try:
        classes_ls = list(range(num_classes))
        K = len(classes_ls)
        N = len(dataset_labels)

        min_size = 0
        try_cnt = 1
        idx_batch: List[List[int]] = [[] for _ in range(num_clients)]

        while min_size < LEAST_SAMPLES:
            if try_cnt > 1:
                print(
                    f"Client data size does not meet the minimum requirement "
                    f"{LEAST_SAMPLES}. Try allocating again for the {try_cnt}-th time."
                )
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([
                    p * (len(idx_j) < N / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min(len(idx_j) for idx_j in idx_batch)
            try_cnt += 1

        dataidx_map = {j: idx_batch[j] for j in range(num_clients)}
        return dataidx_map
    finally:
        if seed is not None:
            np.random.set_state(rng_state)


def print_partition_summary(dataidx_map: Dict[int, List[int]], dataset_labels: np.ndarray) -> None:
    """Mirrors the original's per-client print statements in separate_data():
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", ...)
        print(f"\t\t Samples of labels: ", [...])
    """
    for client, idxs in dataidx_map.items():
        client_labels = dataset_labels[idxs]
        unique_labels = np.unique(client_labels)
        print(f"Client {client}\t Size of data: {len(idxs)}\t Labels: ", unique_labels)
        stats = [(int(c), int(np.sum(client_labels == c))) for c in unique_labels]
        print(f"\t\t Samples of labels: ", stats)
        print("-" * 50)
