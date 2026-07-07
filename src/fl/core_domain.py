"""
Domain-adaptation-aware extension of core.py.

WHY THIS FILE EXISTS
---------------------
core.py (the plain-loop engine underneath FUSED's forget_client_train and
the Retrain baseline) only ever does ONE thing: unweighted FedAvg over
every parameter, including BatchNorm. That is exactly correct for
CIFAR-10 (i.i.d.-ish clients, paper-faithful reproduction) but is the
wrong thing for cross-hospital medical images, where each hospital is a
different imaging domain (scanner, protocol, demographics). Averaging BN
statistics across domains blurs them into a "no domain fits well" state
— this is the exact problem FedBN (Li et al., ICLR 2021) was built to
fix, and it's already correctly implemented for the FL *training* phase
(src/fl/client.py + src/fl/strategies.py, Flower-based). This file brings
the SAME ideas into the (non-Flower) engine that drives FUSED's
unlearning phase and the Retrain baseline, which previously had none of
them:

  1. FedBN: exclude BatchNorm params/buffers from cross-client averaging;
     persist each client's own BN state across rounds instead.
  2. FedProx: add a (mu/2)||w - w_global||^2 proximal term to each
     client's local loss, pulling local updates back toward the
     round-start global model.
  3. FedMoon: add a model-contrastive term that pulls each client's
     penultimate features toward the round-start GLOBAL model's features
     and pushes them away from that SAME client's PREVIOUS-round local
     features — requires persisting one extra thing per client (its own
     previous-round model) across rounds, same pattern as FedBN's BN
     state persistence.

This file does NOT modify core.py. Everything here is additive — pass
`algorithm="fedavg"` (the default) and you get IDENTICAL behavior to
plain core.py, so this is a strict superset, not a replacement.

FIDELITY NOTE: BN-key-detection (FedBN) is imported directly from
src/models/fedbn.py::split_federated_and_local_params, and the
feature-extraction hook + contrastive-loss formula (FedMoon) are ported
directly from src/fl/client.py::_forward_with_features /
_find_classifier_module / _fedmoon_term — the exact same logic already
verified correct for the Flower FL phase, reused here rather than
reimplemented, to avoid two subtly-different copies of the same math.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.fl.client import _find_classifier_module
from src.fl.core import SGD_MOMENTUM, SGD_WEIGHT_DECAY, evaluate  # noqa: F401 (evaluate re-exported)
from src.models.fedbn import split_federated_and_local_params

VALID_ALGORITHMS = ("fedavg", "fedbn", "fedprox", "fedmoon")


def _check_algorithm(algorithm: str) -> str:
    algorithm = algorithm.lower()
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use one of {VALID_ALGORITHMS}.")
    return algorithm


def _forward_with_features(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Faithful port of src/fl/client.py::_forward_with_features — run the
    model and also capture the penultimate feature vector (input to the
    final classifier Linear layer) via a forward pre-hook. Reused
    verbatim (same helper, same _find_classifier_module) so FedMoon's
    feature definition is identical between Engine A (Flower FL phase)
    and this engine."""
    features_holder = {}

    def pre_hook(module, inputs):
        features_holder["features"] = torch.flatten(inputs[0], 1)

    classifier_module, _ = _find_classifier_module(model)
    handle = classifier_module.register_forward_pre_hook(pre_hook)

    try:
        logits = model(images)
    finally:
        handle.remove()

    features = features_holder.get("features", torch.zeros(images.size(0), 1, device=images.device))
    return logits, features


def local_train_domain(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    local_epochs: int,
    device: str,
    mu: float = 0.0,
    global_model_snapshot: Optional[nn.Module] = None,
    moon_mu: float = 1.0,
    moon_temperature: float = 0.5,
    prev_local_model: Optional[nn.Module] = None,
    use_moon: bool = False,
) -> nn.Module:
    """Same as core.py::local_train, plus optional FedProx proximal term
    and/or FedMoon contrastive term. Both default off, making this
    byte-for-byte equivalent to core.py::local_train's loss when
    mu=0.0 and use_moon=False.

    FedProx term matches src/fl/client.py::_fedprox_term:
        (mu / 2) * sum((local_p - global_p) ** 2)

    FedMoon term matches src/fl/client.py::_fedmoon_term exactly: pull
    current local features toward the (frozen) global snapshot's
    features, push away from this SAME client's previous-round local
    model's features (or the global snapshot again, if no previous-round
    model exists yet — matching the original's `else: prev_features =
    global_features` fallback for a client's very first participating
    round).
    """
    criteria = nn.CrossEntropyLoss()
    use_prox = mu > 0.0 and global_model_snapshot is not None

    for _ in range(local_epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            if use_moon:
                pred, local_features = _forward_with_features(model, data)
            else:
                pred = model(data)

            loss = criteria(pred, target)

            if use_prox:
                prox_term = torch.tensor(0.0, device=device)
                for local_p, global_p in zip(model.parameters(), global_model_snapshot.parameters()):
                    prox_term = prox_term + torch.sum((local_p - global_p) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            if use_moon:
                with torch.no_grad():
                    _, global_features = _forward_with_features(global_model_snapshot, data)
                if prev_local_model is not None:
                    with torch.no_grad():
                        _, prev_features = _forward_with_features(prev_local_model, data)
                else:
                    prev_features = global_features

                pos_sim = F.cosine_similarity(local_features, global_features) / moon_temperature
                neg_sim = F.cosine_similarity(local_features, prev_features) / moon_temperature
                moon_logits = torch.stack([pos_sim, neg_sim], dim=1)
                moon_targets = torch.zeros(moon_logits.size(0), dtype=torch.long, device=device)
                loss = loss + moon_mu * F.cross_entropy(moon_logits, moon_targets)

            loss.backward()
            optimizer.step()

    return model


def global_train_once_domain(
    global_model: nn.Module,
    client_data_loaders: List[DataLoader],
    local_epochs: int,
    learning_rate: float,
    device: str,
    algorithm: str = "fedavg",
    mu: float = 0.0,
    moon_mu: float = 1.0,
    moon_temperature: float = 0.5,
    client_bn_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    client_prev_models: Optional[Dict[int, nn.Module]] = None,
) -> List[nn.Module]:
    """Same contract as core.py::global_train_once, plus:

    - algorithm="fedbn": each client restores/persists its own BatchNorm
      state across rounds via `client_bn_states` (caller-owned dict,
      create `{}` once before your round loop, pass the SAME object every
      call — mirrors src/fl/strategies.py's FedBNTrackingFedAvg).

    - algorithm="fedprox": FedProx proximal term against a frozen
      round-start global snapshot.

    - algorithm="fedmoon": FedMoon contrastive term against the frozen
      round-start global snapshot AND each client's own previous-round
      local model, persisted via `client_prev_models` (same
      caller-owned-dict pattern as `client_bn_states`).

    - algorithm="fedavg" (default): identical to core.py::global_train_once.
    """
    algorithm = _check_algorithm(algorithm)
    domain_adaptation = algorithm == "fedbn"
    use_prox = algorithm == "fedprox"
    use_moon = algorithm == "fedmoon"
    device_cpu = torch.device("cpu")

    global_model_snapshot = None
    if use_prox or use_moon:
        global_model_snapshot = copy.deepcopy(global_model).to(device)
        global_model_snapshot.eval()
        for p in global_model_snapshot.parameters():
            p.requires_grad = False

    client_models = []

    for client_idx, client_loader in enumerate(client_data_loaders):
        model = copy.deepcopy(global_model)

        if domain_adaptation and client_bn_states is not None and client_idx in client_bn_states:
            current_state = model.state_dict()
            for key, value in client_bn_states[client_idx].items():
                if key in current_state:
                    current_state[key] = value.to(dtype=current_state[key].dtype)
            model.load_state_dict(current_state, strict=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(trainable_params, lr=learning_rate,
                               momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
        model.train()
        model.to(device)

        prev_local_model = None
        if use_moon and client_prev_models is not None:
            prev_local_model = client_prev_models.get(client_idx)
            if prev_local_model is not None:
                prev_local_model = prev_local_model.to(device)

        model = local_train_domain(
            model, optimizer, client_loader, local_epochs, device,
            mu=mu if use_prox else 0.0,
            global_model_snapshot=global_model_snapshot,
            moon_mu=moon_mu, moon_temperature=moon_temperature,
            prev_local_model=prev_local_model, use_moon=use_moon,
        )

        if domain_adaptation and client_bn_states is not None:
            local_keys = split_federated_and_local_params(model, True)["local"]
            full_state = model.state_dict()
            client_bn_states[client_idx] = {k: full_state[k].detach().cpu() for k in local_keys}

        if use_moon and client_prev_models is not None:
            prev_snapshot = copy.deepcopy(model).to(device_cpu)
            prev_snapshot.eval()
            for p in prev_snapshot.parameters():
                p.requires_grad = False
            client_prev_models[client_idx] = prev_snapshot

        model.to(device_cpu)
        client_models.append(model)

    return client_models


def fedavg_domain(
    local_models: List[nn.Module],
    algorithm: str = "fedavg",
    weights: Optional[List[float]] = None,
) -> nn.Module:
    """Same contract as core.py::fedavg, plus:

    - algorithm="fedbn": BatchNorm keys (per
      src/models/fedbn.py::split_federated_and_local_params) are EXCLUDED
      from averaging — see module docstring / assemble_representative_bn_model.

    - algorithm="fedavg" / "fedprox" / "fedmoon": identical to
      core.py::fedavg (FedProx and FedMoon only change the client-side
      loss, never server-side aggregation — matches
      src/fl/strategies.py, where both reuse plain FedAvg aggregation).

    - `weights`: optional per-client aggregation weights (pass each
      client's local dataset size), matching Flower's default FedAvg
      strategy used in Phase 1 (src/fl/strategies.py), which weights by
      num_examples. Normalized internally, so absolute scale doesn't
      matter. When omitted (default), falls back to the original plain
      unweighted mean over `local_models` — every existing caller that
      doesn't pass `weights` keeps its exact prior behavior.
    """
    algorithm = _check_algorithm(algorithm)
    domain_adaptation = algorithm == "fedbn"

    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()
    local_state_dicts = [model.state_dict() for model in local_models]

    if weights is not None:
        if len(weights) != len(local_models):
            raise ValueError(
                f"weights must have one entry per model in local_models "
                f"({len(weights)} given, {len(local_models)} expected)."
            )
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
    else:
        norm_weights = [1.0 / len(local_models)] * len(local_models)

    if domain_adaptation:
        bn_keys = set(split_federated_and_local_params(local_models[0], True)["local"])
    else:
        bn_keys = set()

    for layer in avg_state_dict.keys():
        if layer in bn_keys:
            continue

        weighted_sum = torch.zeros_like(local_state_dicts[0][layer], dtype=torch.float32)
        for client_idx in range(len(local_models)):
            weighted_sum += local_state_dicts[client_idx][layer].float() * norm_weights[client_idx]

        if "num_batches_tracked" in layer:
            avg_state_dict[layer] = weighted_sum.round().long()
        else:
            avg_state_dict[layer] = weighted_sum.to(local_state_dicts[0][layer].dtype)

    global_model.load_state_dict(avg_state_dict)
    return global_model


def assemble_representative_bn_model(
    aggregated_model: nn.Module,
    client_bn_states: Dict[int, Dict[str, torch.Tensor]],
    representative_client_idx: int = 0,
) -> nn.Module:
    """For algorithm="fedbn" runs: build ONE exportable/evaluable model by
    taking the aggregated (BN-excluded) federated params and merging in
    ONE client's local BN state, chosen deterministically. Mirrors
    src/fl/simulation.py's exact same "representative client" pattern
    used to assemble a single checkpoint from a FedBN Flower run.

    For per-hospital evaluation instead (recommended when reporting
    per-hospital RA/FA — the "correct" way to evaluate a FedBN model, per
    the paper), build one model per client_idx via this same function
    with each client's own idx, rather than relying on one representative
    checkpoint for everyone.
    """
    if representative_client_idx not in client_bn_states:
        representative_client_idx = sorted(client_bn_states.keys())[0]

    model = copy.deepcopy(aggregated_model)
    current_state = model.state_dict()
    for key, value in client_bn_states[representative_client_idx].items():
        if key in current_state:
            current_state[key] = value.to(dtype=current_state[key].dtype)
    model.load_state_dict(current_state, strict=True)
    return model
