"""
Faithful port of FUSED-Code's utils.py::membership_inference_attack() +
train_shadow_model(), specialized to forget_paradigm='client'.

The MIA protocol in the original (verified against the actual repo,
Zhong-Zhengyi/FUSED-Code):

  1. Train `n_shadow` (default 5) "shadow" unlearning models by re-running
     the SAME unlearning procedure (FUSED's forget_client_train, or
     Retrain's FL_Retrain) on PROXY data, starting from the SAME
     already-trained Phase-A checkpoint used for the real run (the shadow
     step does NOT redo Phase-A training from scratch — see
     train_shadow_model()'s 'fused'/'retrain' branches, which call
     `case.forget_client_train(copy.deepcopy(model), ...)` /
     `case.FL_Retrain(copy.deepcopy(model), ...)` directly on the model
     handed to membership_inference_attack()).
  2. For each shadow model, record its OUTPUT LOGITS over its proxy data —
     BOTH the proxy client (train) loaders AND the proxy test loaders are
     used (utils.py's train_shadow_model has two accumulation loops: one
     over `proxy_client_loaders_bk[user]`, one over
     `proxy_test_loaders[user]`) — labeled 1 if that sample's client is a
     "remember" client (member-like) or 0 if it's a "forget" client
     (non-member-like). Both proxy loader groups passed in are the
     BYZANTINE-ATTACKED ones (main.py passes `proxy_client_loaders_process`
     as `proxy_client_loaders_bk` and `proxy_test_loaders_process` as
     `proxy_test_loaders` into membership_inference_attack for the client
     scenario).
  3. Balance the pooled shadow-collected (logit, membership) pairs via
     `reduce_ones()` and train a small FCNet classifier (FCNet:
     Linear(num_classes->20)->Linear(20->20)->Linear(20->2)) via
     SGD(lr=0.01, momentum=0.9, weight_decay=5e-4), for `args.global_epoch`
     epochs (the original reuses the FL global-epoch count for attack-model
     training — there is no separate "attack_epochs" hyperparameter).
  4. Evaluate that attack classifier, EVERY epoch, on the REAL unlearned
     model's logits over its OWN client (train) loaders AND test loaders —
     collected PER CLIENT into one DataLoader per client (utils.py builds
     `test_loaders[client]` from `test_x_user[client]`/`test_y_user[client]`,
     which are populated from BOTH `client_all_loaders_bk[client]` (attacked
     train loaders) and `test_loaders[client]` (the ORIGINAL, un-attacked
     test loaders — main.py passes `test_loaders`, not `test_loaders_process`,
     as this argument for the client scenario) — this train/test-loader
     asymmetry (attacked train source, clean test source) is faithfully
     reproduced here). Crucially, `reduce_ones()` is only ever applied to
     the FLAT shadow-collected training pool; the per-client evaluation
     pools are used UNBALANCED, exactly as in the source.
  5. The reported "MIA" value is the attack classifier's accuracy at
     distinguishing forget-client samples from remember-client samples.
     utils.py's `membership_inference_attack()` itself only logs a
     per-client, per-epoch CSV and returns nothing — there is no single
     scalar computed inside that function. To get a single number
     comparable to the paper's Table 1 (and to make this function usable
     as a normal Python return value), we report the mean of the
     per-client accuracies at the FINAL epoch as the headline "MIA_acc",
     while still returning the full per-client breakdown for inspection.

`reduce_ones()` in the original re-balances the 0/1 label classes before
training the attack model ("assumes more training than testing examples...
1 as over-represented class is hardcoded in here" — for client-unlearning,
label 0 = forget client, typically the minority class out of 50 clients,
so in practice this balancing step is usually close to a no-op, but we
replicate it for fidelity).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FCNet(nn.Module):
    """Faithful port of utils.py::FCNet — the MIA attack classifier."""

    def __init__(self, num_classes: int, dim_hidden: int = 20, dim_out: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(num_classes, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def reduce_ones(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port of utils.py::reduce_ones (label 1 = over-represented
    class hardcoded, per the original's comment). We drop the original's
    parallel `classes` array since it is only consumed by the class-forget
    paradigm's per-class test-loader split, which is out of scope here."""
    idx_to_keep = np.where(y == 0)[0]
    idx_to_reduce = np.where(y == 1)[0]
    if len(idx_to_reduce) == 0 or len(idx_to_keep) == 0:
        return x, y
    num_to_reduce = (y.shape[0] - idx_to_reduce.shape[0]) * 2
    num_to_reduce = min(num_to_reduce, idx_to_reduce.shape[0])
    idx_sample = np.random.choice(idx_to_reduce, num_to_reduce, replace=False)
    keep_idx = np.concatenate([idx_to_keep, idx_sample, idx_to_keep])
    return x[keep_idx], y[keep_idx]


@torch.no_grad()
def _collect_flat(
    model: nn.Module,
    loader_groups: List[List[DataLoader]],
    forget_client_idx: List[int],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool logits/membership across ALL clients and ALL loader groups
    (e.g. [train_loaders, test_loaders]) into one flat array. Faithful to
    the original's two back-to-back accumulation loops (one per loader
    group) both appending into the same flat `attack_x_train`/
    `attack_y_train` (or `test_x`/`test_y`) lists."""
    model.eval()
    model.to(device)
    all_logits, all_membership = [], []

    for loaders in loader_groups:
        for client_id, loader in enumerate(loaders):
            membership_label = 0.0 if client_id in forget_client_idx else 1.0
            for images, _labels in loader:
                images = images.to(device)
                outputs = model(images)
                all_logits.extend(outputs.cpu().numpy())
                all_membership.extend([membership_label] * images.size(0))

    return np.array(all_logits, dtype="float32"), np.array(all_membership, dtype="int32")


@torch.no_grad()
def _collect_per_client(
    model: nn.Module,
    loader_groups: List[List[DataLoader]],
    forget_client_idx: List[int],
    device: str,
    num_clients: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Faithful port of the original's `test_x_user`/`test_y_user`
    per-client accumulation: for each client, pool its logits/membership
    across ALL given loader groups (e.g. [attacked_train, clean_test]),
    UNBALANCED (no reduce_ones applied — the original never rebalances
    these per-client eval pools, only the flat shadow-training pool)."""
    model.eval()
    model.to(device)
    per_client_logits: Dict[int, list] = {i: [] for i in range(num_clients)}
    per_client_membership: Dict[int, list] = {i: [] for i in range(num_clients)}

    for loaders in loader_groups:
        for client_id, loader in enumerate(loaders):
            membership_label = 0.0 if client_id in forget_client_idx else 1.0
            for images, _labels in loader:
                images = images.to(device)
                outputs = model(images)
                per_client_logits[client_id].extend(outputs.cpu().numpy())
                per_client_membership[client_id].extend([membership_label] * images.size(0))

    return {
        client_id: (
            np.array(per_client_logits[client_id], dtype="float32"),
            np.array(per_client_membership[client_id], dtype="int32"),
        )
        for client_id in range(num_clients)
    }


def train_shadow_models(
    unlearning_fn: Callable[[], nn.Module],
    proxy_loader_groups: List[List[DataLoader]],
    forget_client_idx: List[int],
    n_shadow: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port of train_shadow_model(): repeat `unlearning_fn()`
    (a zero-arg closure that re-runs ONLY the unlearning phase — e.g.
    forget_client_train starting from the already Phase-A-trained model —
    on proxy data and returns the resulting shadow model) `n_shadow`
    times, pooling (logits, membership) pairs collected across BOTH proxy
    loader groups (proxy client/train loaders + proxy test loaders) from
    each run."""
    all_logits, all_membership = [], []
    for shadow_idx in range(n_shadow):
        print(f"[MIA] Training shadow model {shadow_idx + 1}/{n_shadow}...")
        shadow_model = unlearning_fn()
        logits, membership = _collect_flat(shadow_model, proxy_loader_groups, forget_client_idx, device)
        all_logits.append(logits)
        all_membership.append(membership)

    return np.concatenate(all_logits, axis=0), np.concatenate(all_membership, axis=0)


def membership_inference_attack(
    unlearned_model: nn.Module,
    real_loader_groups: List[List[DataLoader]],
    proxy_loader_groups: List[List[DataLoader]],
    forget_client_idx: List[int],
    num_classes: int,
    n_shadow: int,
    unlearning_fn: Callable[[], nn.Module],
    device: str,
    attack_epochs: int,
    test_batch_size: int = 64,
) -> Tuple[float, Dict[int, float]]:
    """Faithful port of membership_inference_attack().

    `real_loader_groups` / `proxy_loader_groups` are each a list of
    per-client loader lists to pool together — pass
    `[attacked_client_loaders, clean_test_loaders]` for the real side and
    `[attacked_proxy_client_loaders, attacked_proxy_test_loaders]` for the
    proxy side to match main.py's exact call contract for
    forget_paradigm='client':

        membership_inference_attack(args, unlearning_model, case, model,
            client_all_loaders_process,   # -> real_loader_groups[0]  (ATTACKED)
            test_loaders,                 # -> real_loader_groups[1]  (CLEAN)
            proxy_client_loaders_process, # -> proxy_loader_groups[0] (ATTACKED)
            proxy_client_loaders,         # (unused for forget_paradigm='client';
                                           #  train_shadow_model's 'client' branch
                                           #  only reads proxy_client_loaders_bk)
            proxy_test_loaders_process)   # -> proxy_loader_groups[1] (ATTACKED)

    `attack_epochs` should be `args.global_epoch` to match the original
    (there is no separate attack-epoch hyperparameter upstream).

    Returns (mean_final_epoch_accuracy, per_client_final_epoch_accuracy) —
    the first value is the single "MIA" scalar to log against Table 1; the
    original itself returns nothing and only logs a per-client/per-epoch
    CSV, so this aggregate is our (documented) addition on top of a
    faithful per-client evaluation loop.
    """
    num_clients = len(real_loader_groups[0])

    # Step 1: shadow models trained on proxy data -> attack training set
    attack_x_train, attack_y_train = train_shadow_models(
        unlearning_fn, proxy_loader_groups, forget_client_idx, n_shadow, device,
    )
    attack_x_train, attack_y_train = reduce_ones(attack_x_train, attack_y_train)

    # Step 2: real model's logits, per client, over (train+test) real data
    per_client_real = _collect_per_client(
        unlearned_model, real_loader_groups, forget_client_idx, device, num_clients,
    )
    per_client_test_loaders = {
        client_id: DataLoader(
            TensorDataset(torch.tensor(x), torch.tensor(y)),
            batch_size=test_batch_size, shuffle=True,
        )
        for client_id, (x, y) in per_client_real.items()
    }

    # Step 3: train the attack classifier for `attack_epochs` epochs,
    # evaluating per-client accuracy every epoch (faithful to utils.py's
    # train() helper, which evaluates a dict of per-client test loaders).
    attack_model = FCNet(num_classes=num_classes)
    optimizer = optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    attack_model.to(device)
    attack_model.train()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(attack_x_train), torch.tensor(attack_y_train)),
        batch_size=test_batch_size, shuffle=True,
    )

    criteria = nn.CrossEntropyLoss()
    per_client_acc: Dict[int, float] = {}
    for epoch in range(attack_epochs):
        attack_model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            pred = attack_model(data)
            loss = criteria(pred, target)
            loss.backward()
            optimizer.step()

        attack_model.eval()
        per_client_acc = {}
        with torch.no_grad():
            for client_id, loader in per_client_test_loaders.items():
                correct, total = 0, 0
                for data, target in loader:
                    data, target = data.to(device), target.to(device).long()
                    pred = attack_model(data)
                    predicted = torch.argmax(pred, dim=1)
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
                per_client_acc[client_id] = correct / max(1, total)

        avg_acc = sum(per_client_acc.values()) / max(1, len(per_client_acc))
        print(f"[MIA] Attack model epoch={epoch}, mean_client_accuracy={avg_acc:.4f}")

    mean_final_acc = sum(per_client_acc.values()) / max(1, len(per_client_acc))
    return mean_final_acc, per_client_acc