"""
Utilities for loading matched FL source models for federated unlearning.

Backward compatibility:
- Non-FedBN runs keep loading best.pt exactly as before.
- FedBN also keeps loading best.pt unless fedbn_source_mode is explicitly
  set to "retained_bn_average".
"""

from __future__ import annotations

import copy
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.models.fedbn import (
    build_retained_bn_reference_model,
    split_federated_and_local_params,
)
from src.utils.checkpoint import (
    get_checkpoint_path,
    load_checkpoint_into_new_model,
)


SUPPORTED_FEDBN_SOURCE_MODES = {
    "global",
    "retained_bn_average",
}


def _log(logger, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def load_fu_source_model(
    *,
    source_checkpoint_dir: str,
    source_algorithm: str,
    forget_client: str,
    client_partitions,
    model_builder,
    device: str,
    fedbn_source_mode: str = "global",
    logger=None,
) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """
    Load the source model used by federated unlearning.

    Returns:
        source_model:
            Model used as the main FUSED source model.

        retained_client_models:
            For matched FedBN mode, maps retained client IDs to their
            hospital-specific Phase-1 models. Empty for all other cases.
    """
    mode = str(fedbn_source_mode).strip().lower()

    if mode not in SUPPORTED_FEDBN_SOURCE_MODES:
        raise ValueError(
            f"Unsupported fedbn_source_mode='{mode}'. "
            f"Expected one of {sorted(SUPPORTED_FEDBN_SOURCE_MODES)}."
        )

    global_checkpoint_path = get_checkpoint_path(
        source_checkpoint_dir,
        "best",
    )

    global_model = load_checkpoint_into_new_model(
        model_builder,
        global_checkpoint_path,
        device=device,
    )

    normalized_algorithm = str(source_algorithm).strip().lower()

    if (
        normalized_algorithm != "fedbn"
        or mode == "global"
    ):
        _log(
            logger,
            "Loaded source model from the normal global checkpoint: "
            f"{global_checkpoint_path}",
        )
        return global_model, {}

    per_hospital_dir = os.path.join(
        source_checkpoint_dir,
        "per_hospital",
    )

    if not os.path.isdir(per_hospital_dir):
        raise FileNotFoundError(
            "Matched FedBN source mode requires the per-hospital "
            f"checkpoint folder, but it was not found: {per_hospital_dir}"
        )

    retained_client_models: Dict[str, nn.Module] = {}
    retained_client_weights: Dict[str, float] = {}

    for partition in client_partitions:
        if partition.hospital == forget_client:
            continue

        checkpoint_path = os.path.join(
            per_hospital_dir,
            f"{partition.client_id}.pt",
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Missing retained FedBN checkpoint for "
                f"client_id='{partition.client_id}': {checkpoint_path}"
            )

        retained_client_models[partition.client_id] = (
            load_checkpoint_into_new_model(
                model_builder,
                checkpoint_path,
                device=device,
            )
        )

        retained_client_weights[partition.client_id] = float(
            len(partition.samples)
        )

    if not retained_client_models:
        raise ValueError(
            "No retained FedBN client checkpoints were loaded. "
            f"forget_client={forget_client}"
        )

    source_model = build_retained_bn_reference_model(
        global_model=global_model,
        client_models=retained_client_models,
        client_weights=retained_client_weights,
    ).to(device)

    _log(
        logger,
        "Loaded matched FedBN source model using retained-only BN states. "
        f"forget_client={forget_client}, "
        f"retained_clients={sorted(retained_client_models)}",
    )

    return source_model, retained_client_models


def _matches_critical_layer(
    state_key: str,
    critical_layers: Iterable[str],
) -> bool:
    for layer_name in critical_layers:
        if (
            state_key == layer_name
            or state_key.startswith(f"{layer_name}.")
        ):
            return True

    return False


def build_personalized_fu_models(
    *,
    unlearned_model: nn.Module,
    retained_source_models: Dict[str, nn.Module],
    critical_layers: List[str],
) -> Dict[str, nn.Module]:
    """
    Build retained-hospital FU models.

    Shared non-BN parameters come from the final unlearned model.
    Hospital-specific BN parameters remain personalized.

    When a selected critical layer is itself a BN parameter, the value
    from the final unlearned model is also copied. This preserves the
    behavior of the existing representative FedBN adapter output.
    """
    if not retained_source_models:
        return {}

    unlearned_state = unlearned_model.state_dict()

    federated_keys = set(
        split_federated_and_local_params(
            unlearned_model,
            domain_adaptation=True,
        )["federated"]
    )

    personalized_models: Dict[str, nn.Module] = {}

    for client_id, source_client_model in retained_source_models.items():
        personalized_model = copy.deepcopy(source_client_model)
        personalized_state = personalized_model.state_dict()

        for key, destination_tensor in personalized_state.items():
            should_copy = (
                key in federated_keys
                or _matches_critical_layer(key, critical_layers)
            )

            if not should_copy:
                continue

            if key not in unlearned_state:
                raise KeyError(
                    f"State key '{key}' was not found in unlearned model."
                )

            personalized_state[key] = (
                unlearned_state[key]
                .detach()
                .to(
                    device=destination_tensor.device,
                    dtype=destination_tensor.dtype,
                )
                .clone()
            )

        personalized_model.load_state_dict(
            personalized_state,
            strict=True,
        )

        personalized_models[client_id] = personalized_model

    return personalized_models