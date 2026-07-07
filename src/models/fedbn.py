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

from typing import Dict, List

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
