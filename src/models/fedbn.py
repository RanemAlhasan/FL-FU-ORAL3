"""
FedBN-style domain adaptation.

Core idea (Li et al., FedBN): every client keeps its own BatchNorm
statistics/affine parameters *local* — they are never sent to the server
and never overwritten by aggregation. Only non-BN parameters are federated.
Because each hospital here is a distinct imaging domain (different scanners,
lighting, demographics), this directly combats domain shift without any
extra modeling machinery.

This module provides the parameter split (federated vs. local-only) used by:
  - src/fl/client.py   (excludes BN params from what's sent to the server)
  - src/fl/strategies.py (FedBN aggregation: skip BN keys during averaging)

It is a no-op / unused when `domain_adaptation: false` in config — in that
case all parameters (including BN) are federated normally (this is what
plain FedAvg/FedProx/FedMOON do).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.models.backbone import get_batchnorm_layer_names


def split_federated_and_local_params(
    model: nn.Module, domain_adaptation: bool
) -> Dict[str, List[str]]:
    """Return {"federated": [...state_dict keys...], "local": [...]}.

    When domain_adaptation=False, all keys are federated and "local" is empty
    — i.e. standard FL behavior, BN included.
    """
    if not domain_adaptation:
        return {"federated": list(model.state_dict().keys()), "local": []}

    bn_layer_names = set(get_batchnorm_layer_names(model))
    if not bn_layer_names:
        # BUG FIX: previously silently returned local_keys=[] here, so a
        # "fedbn" run against a backbone with zero BatchNorm modules (e.g.
        # model=vit_b16, which normalizes exclusively via nn.LayerNorm)
        # federated 100% of parameters like plain FedAvg, produced
        # byte-identical "per-hospital" checkpoints, and reported spurious
        # domain-adaptation results — all with no warning anywhere. Fail
        # loudly instead: FedBN has nothing to keep local for this backbone.
        raise ValueError(
            f"domain_adaptation=True (FedBN) but {type(model).__name__} has no "
            f"BatchNorm1d/2d/3d layers to keep local (see "
            f"src/models/backbone.py::get_batchnorm_layer_names) — FedBN would "
            f"silently degrade to plain FedAvg. This backbone likely normalizes "
            f"via LayerNorm instead (e.g. vit_b16); FedBN as implemented here "
            f"is not applicable to it. Use a BatchNorm-based backbone (resnet18/"
            f"resnet50/densenet121/efficientnet_b0) for FedBN runs."
        )
    federated_keys, local_keys = [], []
    for key in model.state_dict().keys():
        # state_dict keys look like "bn1.weight", "layer1.0.bn1.running_mean", etc.
        # A key belongs to a BN layer if its module-name prefix matches one of
        # the registered BN layer names.
        module_prefix = key.rsplit(".", 1)[0]
        if module_prefix in bn_layer_names:
            local_keys.append(key)
        else:
            federated_keys.append(key)
    return {"federated": federated_keys, "local": local_keys}


def extract_federated_state_dict(
    model: nn.Module, domain_adaptation: bool
) -> Dict[str, torch.Tensor]:
    """The subset of a model's state_dict that should be sent to the server
    this round. Under FedBN, BN params are excluded (stay on the client)."""
    split = split_federated_and_local_params(model, domain_adaptation)
    full_state = model.state_dict()
    return {k: full_state[k] for k in split["federated"]}


def merge_local_bn_into_global(
    model: nn.Module,
    global_federated_state: Dict[str, torch.Tensor],
    domain_adaptation: bool,
) -> None:
    """Load the server's aggregated federated parameters into `model` IN
    PLACE, while leaving that model's own local BN parameters untouched
    (i.e. this client keeps its own BN stats across rounds, per FedBN)."""
    current_state = model.state_dict()
    current_state.update(global_federated_state)
    model.load_state_dict(current_state, strict=True)
    
    # New Addition 
def build_retained_bn_reference_model(
    global_model: nn.Module,
    client_models: Dict[str, nn.Module],
    client_weights: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """
    Build one FedBN fallback model using only retained-client BN states.

    The model keeps the final global non-BN parameters and uses a weighted
    average of retained clients' BatchNorm parameters and running statistics.

    This is used for evaluating a forgotten hospital because exact FedBN
    retraining has no personalized BN state for that hospital.

    Backward compatible:
    - Existing FedBN functions are unchanged.
    - This helper is only used when explicitly called.
    """
    if not client_models:
        raise ValueError(
            "Cannot build retained BN reference model: "
            "client_models is empty."
        )

    reference_model = copy.deepcopy(global_model)
    reference_state = reference_model.state_dict()

    local_bn_keys = split_federated_and_local_params(
        reference_model,
        domain_adaptation=True,
    )["local"]

    available_client_ids = sorted(client_models.keys())

    if client_weights is None:
        normalized_weights = {
            client_id: 1.0 / len(available_client_ids)
            for client_id in available_client_ids
        }
    else:
        raw_weights = {
            client_id: float(client_weights.get(client_id, 0.0))
            for client_id in available_client_ids
        }

        total_weight = sum(raw_weights.values())

        if total_weight <= 0:
            normalized_weights = {
                client_id: 1.0 / len(available_client_ids)
                for client_id in available_client_ids
            }
        else:
            normalized_weights = {
                client_id: weight / total_weight
                for client_id, weight in raw_weights.items()
            }

    client_states = {
        client_id: model.state_dict()
        for client_id, model in client_models.items()
    }

    for key in local_bn_keys:
        tensors = [
            client_states[client_id][key].detach()
            for client_id in available_client_ids
        ]

        reference_tensor = reference_state[key]

        if torch.is_floating_point(reference_tensor):
            averaged_tensor = torch.zeros_like(
                reference_tensor,
                dtype=torch.float64,
            )

            for client_id, tensor in zip(
                available_client_ids,
                tensors,
            ):
                averaged_tensor += (
                    tensor.to(
                        device=averaged_tensor.device,
                        dtype=torch.float64,
                    )
                    * normalized_weights[client_id]
                )

            reference_state[key] = averaged_tensor.to(
                dtype=reference_tensor.dtype,
            )
        else:
            # BatchNorm num_batches_tracked is an integer tensor.
            # Use the largest retained value rather than averaging integers.
            stacked = torch.stack(
                [
                    tensor.to(reference_tensor.device)
                    for tensor in tensors
                ],
                dim=0,
            )

            reference_state[key] = stacked.max(dim=0).values.to(
                dtype=reference_tensor.dtype,
            )

    reference_model.load_state_dict(reference_state, strict=True)
    return reference_model