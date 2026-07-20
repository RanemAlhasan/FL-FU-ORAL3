"""
A single Flower client implementation that is algorithm-agnostic: the FL
*strategy* (FedAvg/FedProx/FedBN/FedMOON) determines server-side aggregation
behavior (see strategies.py) and a couple of client-side hooks (proximal term
for FedProx, contrastive term for FedMOON, BN exclusion for FedBN).
"""
from __future__ import annotations

import base64
import copy
import io
import time
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.fedbn import (
    extract_federated_state_dict,
    merge_local_bn_into_global,
    split_federated_and_local_params,
)


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(keys: List[str], arrays: List[np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(a) for k, a in zip(keys, arrays)}


def encode_state_dict_b64(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Serialize a state_dict to a base64 string so it can travel through
    Flower's metrics/config channel.

    This is especially important for FedBN because Ray/Flower simulation may
    recreate client objects between rounds, so we cannot rely on local Python
    object persistence to preserve BatchNorm state.
    """
    buffer = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state_dict_b64(encoded: str) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(base64.b64decode(encoded))
    return torch.load(buffer, map_location="cpu")


class OralCancerFlowerClient(fl.client.NumPyClient):
    """One Flower client = one hospital or one simulated hospital shard."""

    def __init__(
        self,
        client_id: str,
        hospital: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        algorithm_config: Dict,
        local_epochs: int,
        learning_rate: float,
        # New Addition
        classification_class_weights: Optional[List[float]] = None,
        imbalance_method: str = "standard_ce",
    ):
        self.client_id = client_id
        self.hospital = hospital
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.algo = algorithm_config
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        # New Addition
        self.imbalance_method = imbalance_method

        if classification_class_weights is None:
            self.classification_criterion = nn.CrossEntropyLoss()
        else:
            class_weight_tensor = torch.tensor(
                classification_class_weights,
                dtype=torch.float32,
                device=device,
            )

            self.classification_criterion = nn.CrossEntropyLoss(
                weight=class_weight_tensor,
            )

        self._prev_local_model: Optional[nn.Module] = None
        self._global_model_snapshot: Optional[nn.Module] = None

        self._domain_adaptation = bool(self.algo.get("domain_adaptation", False))

        self._federated_keys = list(
            extract_federated_state_dict(
                self.model,
                self._domain_adaptation,
            ).keys()
        )

    # ------------------------------------------------------------------
    # FedBN helpers
    # ------------------------------------------------------------------

    def _restore_local_bn_from_config(self, config) -> None:
        """
        Restore this client's local BN state if the server sent it.

        This fixes FedBN under Ray/Flower simulation because client objects
        are not guaranteed to persist across rounds. Without this, each new
        client object may start from fresh/random/pretrained BN statistics,
        which breaks FedBN evaluation and can cause unstable loss.
        """
        if not self._domain_adaptation:
            return

        if config is None:
            return

        if "bn_state_b64" not in config:
            return

        bn_state = decode_state_dict_b64(config["bn_state_b64"])

        current_state = self.model.state_dict()

        for key, value in bn_state.items():
            if key in current_state:
                current_state[key] = value.to(
                    dtype=current_state[key].dtype,
                    device=current_state[key].device,
                )

        self.model.load_state_dict(current_state, strict=True)

    def _get_local_bn_state(self) -> Dict[str, torch.Tensor]:
        """
        Return this client's local BN parameters and buffers only.
        """
        local_keys = split_federated_and_local_params(self.model, True)["local"]
        full_state = self.model.state_dict()
        return {k: full_state[k].detach().cpu() for k in local_keys}

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config) -> List[np.ndarray]:
        fed_state = extract_federated_state_dict(
            self.model,
            self._domain_adaptation,
        )
        return state_dict_to_ndarrays(fed_state)

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, Dict]:
        # 1. Restore this client's own BN state from the previous round.
        #    This is only active for FedBN/domain_adaptation=True.
        self._restore_local_bn_from_config(config)

        # 2. Load server aggregated federated parameters.
        #    For FedBN, this intentionally excludes BN parameters/buffers,
        #    so the local BN state remains hospital-specific.
        global_federated_state = ndarrays_to_state_dict(self._federated_keys, parameters)
        merge_local_bn_into_global(
            self.model,
            global_federated_state,
            self._domain_adaptation,
        )

        # 3. Snapshot received global model for FedProx/FedMOON.
        if self.algo["name"] in ("fedprox", "fedmoon"):
            self._global_model_snapshot = copy.deepcopy(self.model)
            self._global_model_snapshot.eval()
            for p in self._global_model_snapshot.parameters():
                p.requires_grad = False

        comp_start = time.time()
        train_loss = self._local_train()
        comp_time = time.time() - comp_start

        if self.algo["name"] == "fedmoon":
            self._prev_local_model = copy.deepcopy(self.model)
            self._prev_local_model.eval()
            for p in self._prev_local_model.parameters():
                p.requires_grad = False

        fed_state = extract_federated_state_dict(
            self.model,
            self._domain_adaptation,
        )

        num_examples = len(self.train_loader.dataset)
        fed_arrays = state_dict_to_ndarrays(fed_state)
        comm_bytes = sum(arr.nbytes for arr in fed_arrays)

        metrics = {
            "client_id": self.client_id,
            "hospital": self.hospital,
            "train_loss": train_loss,
            "comp_time_sec": comp_time,
            "comm_bytes": comm_bytes,
        }

        if self._domain_adaptation:
            metrics["bn_state_b64"] = encode_state_dict_b64(self._get_local_bn_state())

        return fed_arrays, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, Dict]:
        # Restore hospital-specific BN before evaluation.
        self._restore_local_bn_from_config(config)

        global_federated_state = ndarrays_to_state_dict(self._federated_keys, parameters)
        merge_local_bn_into_global(
            self.model,
            global_federated_state,
            self._domain_adaptation,
        )

        loss, acc = self._local_evaluate()
        num_examples = len(self.val_loader.dataset)

        return float(loss), num_examples, {
            "client_id": self.client_id,
            "hospital": self.hospital,
            "accuracy": acc,
        }

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _local_train(self) -> float:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        total_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                logits, features = self._forward_with_features(self.model, images)
                
                # New Modification
                loss = self.classification_criterion(logits, labels)

                if self.algo["name"] == "fedprox" and self._global_model_snapshot is not None:
                    loss = loss + self._fedprox_term()

                if self.algo["name"] == "fedmoon" and self._global_model_snapshot is not None:
                    loss = loss + self._fedmoon_term(images, features)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(1, total_batches)

    def _fedprox_term(self) -> torch.Tensor:
        mu = self.algo.get("mu", 0.01)

        prox_term = torch.tensor(0.0, device=self.device)

        for local_p, global_p in zip(
            self.model.parameters(),
            self._global_model_snapshot.parameters(),
        ):
            prox_term = prox_term + torch.sum((local_p - global_p) ** 2)

        return (mu / 2.0) * prox_term

    def _fedmoon_term(self, images: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        """
        FedMOON model-contrastive loss:
        pull current local features toward global model features,
        push away from previous local model features.
        """
        moon_mu = self.algo.get("moon_mu", 1.0)
        temperature = self.algo.get("moon_temperature", 0.5)

        with torch.no_grad():
            _, global_features = self._forward_with_features(
                self._global_model_snapshot,
                images,
            )

        if self._prev_local_model is not None:
            with torch.no_grad():
                _, prev_features = self._forward_with_features(
                    self._prev_local_model,
                    images,
                )
        else:
            prev_features = global_features

        pos_sim = F.cosine_similarity(local_features, global_features) / temperature
        neg_sim = F.cosine_similarity(local_features, prev_features) / temperature

        logits = torch.stack([pos_sim, neg_sim], dim=1)
        contrastive_targets = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=self.device,
        )

        contrastive_loss = F.cross_entropy(logits, contrastive_targets)
        return moon_mu * contrastive_loss

    @staticmethod
    def _forward_with_features(
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run model and also return the penultimate feature vector.
        This is used by FedMOON.
        """
        features_holder = {}

        def pre_hook(module, inputs):
            features_holder["features"] = torch.flatten(inputs[0], 1)

        classifier_module, _ = _find_classifier_module(model)
        handle = classifier_module.register_forward_pre_hook(pre_hook)

        try:
            logits = model(images)
        finally:
            handle.remove()

        features = features_holder.get(
            "features",
            torch.zeros(images.size(0), 1, device=images.device),
        )

        return logits, features

    def _local_evaluate(self) -> Tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item() * images.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)

        return avg_loss, acc


_PEFT_LORA_ADAPTER_ATTRS = {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}


def _find_classifier_module(model: nn.Module) -> Tuple[nn.Module, str]:
    """Locate the final Linear classifier layer generically across backbones.

    BUG FIX: this used to keep whichever nn.Linear was found LAST while
    walking named_modules(), with no filtering. For a plain backbone that
    correctly resolves to the true `fc`/`classifier` head. But for a model
    wrapped by src/models/resnet_lora.py::build_lora_adapter (peft's
    get_peft_model), PEFT's LoraLayer registers `base_layer` (the real
    frozen classifier) BEFORE its `lora_A`/`lora_B` ModuleDicts, so the
    unfiltered "last Linear wins" scan ends up on `fc.lora_B.default` — a
    (r=16 -> num_classes) Linear that is the LoRA adapter's own up-
    projection, not the model's true penultimate-feature classifier. Every
    caller of this function (FedMoon's feature hook, here and in
    src/fl/core_domain.py / src/fu/fused_cli_training.py) would then treat
    that 16-dim LoRA projection's input as "features," corrupting the
    contrastive term whenever FedMoon runs against a LoRA-adapted model
    (src/fu/fused_training_domain.py's algorithm="fedmoon" path). Skipping
    any Linear nested under a `lora_A`/`lora_B`/`lora_embedding_A`/
    `lora_embedding_B` container fixes this for LoRA models while leaving
    plain (non-PEFT) backbones' behavior unchanged.
    """
    last_linear_name = None
    last_linear = None

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _PEFT_LORA_ADAPTER_ATTRS & set(name.split(".")):
            continue
        last_linear_name = name
        last_linear = module

    if last_linear is None:
        raise ValueError("Could not find a Linear classifier head in model.")

    return last_linear, last_linear_name
