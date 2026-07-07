"""
A single Flower client implementation that is algorithm-agnostic: the FL
*strategy* (FedAvg/FedProx/FedBN/FedMOON) determines server-side aggregation
behavior (see strategies.py) and a couple of client-side hooks (proximal term
for FedProx, contrastive term for FedMOON, BN exclusion for FedBN). The
client itself doesn't branch on "if algorithm == ...": it reads small,
declarative `algorithm_config` flags and applies the corresponding loss term
generically. This is what makes adding a 5th algorithm later a matter of
adding flags + one loss term, not forking the client class.
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

from src.models.fedbn import extract_federated_state_dict, merge_local_bn_into_global, \
    split_federated_and_local_params


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(keys: List[str], arrays: List[np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.tensor(a) for k, a in zip(keys, arrays)}


def encode_state_dict_b64(state_dict: Dict[str, torch.Tensor]) -> str:
    """Serialize a (small) state_dict to a base64 string so it can travel
    inside Flower's Dict[str, Scalar] fit() metrics — the only channel
    guaranteed to cross the simulation backend's process/actor boundary
    (Ray runs each client_fn invocation in a separate worker process, so
    plain Python object mutation in the driver process, e.g. a dict closed
    over by client_fn, is NOT visible after the simulation returns).
    Used specifically to ship each client's local BN buffers back to the
    driver under FedBN, where BN parameters are intentionally excluded from
    the normal federated parameter array."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_state_dict_b64(encoded: str) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(base64.b64decode(encoded))
    return torch.load(buffer, map_location="cpu")


class OralCancerFlowerClient(fl.client.NumPyClient):
    """One Flower client = one hospital (or one simulated shard of a hospital).

    `algorithm_config` (set per-experiment from the YAML config) controls:
        name: "fedavg" | "fedprox" | "fedbn" | "fedmoon"
        mu: float                  # FedProx proximal term weight
        moon_mu: float              # FedMOON contrastive term weight
        moon_temperature: float
        domain_adaptation: bool      # whether BN params stay local (FedBN-style)
    """

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
        # NOTE: deliberately no logger reference here. Flower's simulation
        # backend (Ray) pickles client_fn's closure to ship clients to
        # worker actors; an ExperimentLogger holds live logging.Logger file
        # handles (containing unpicklable thread locks), so capturing it
        # anywhere in the client construction path breaks serialization.
        # All per-round metric logging instead happens server-side via the
        # on_fit_metrics/on_evaluate_metrics strategy callbacks (see
        # src/fl/simulation.py), which run in the main process where the
        # logger lives.

        # FedMOON needs the previous round's local model and the (frozen)
        # global model to compute its model-contrastive loss term.
        self._prev_local_model: Optional[nn.Module] = None
        self._global_model_snapshot: Optional[nn.Module] = None

        self._federated_keys = list(
            extract_federated_state_dict(self.model, self.algo.get("domain_adaptation", False)).keys()
        )

    # -- Flower NumPyClient interface -------------------------------------

    def get_parameters(self, config) -> List[np.ndarray]:
        fed_state = extract_federated_state_dict(self.model, self.algo.get("domain_adaptation", False))
        return state_dict_to_ndarrays(fed_state)

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, Dict]:
        # 1. Load the server's aggregated federated parameters; keep this
        #    client's own BN stats untouched if domain_adaptation is on.
        global_federated_state = ndarrays_to_state_dict(self._federated_keys, parameters)
        merge_local_bn_into_global(self.model, global_federated_state,
                                    self.algo.get("domain_adaptation", False))

        # 2. Snapshot the just-received global model (frozen) for FedProx /
        #    FedMOON regularization terms.
        if self.algo["name"] in ("fedprox", "fedmoon"):
            self._global_model_snapshot = copy.deepcopy(self.model)
            for p in self._global_model_snapshot.parameters():
                p.requires_grad = False

        comp_start = time.time()
        train_loss = self._local_train()
        comp_time = time.time() - comp_start

        if self.algo["name"] == "fedmoon":
            # Save this round's locally-trained model as "previous" for the
            # next round's contrastive term.
            self._prev_local_model = copy.deepcopy(self.model)
            for p in self._prev_local_model.parameters():
                p.requires_grad = False

        fed_state = extract_federated_state_dict(self.model, self.algo.get("domain_adaptation", False))
        num_examples = len(self.train_loader.dataset)
        comm_bytes = sum(arr.nbytes for arr in state_dict_to_ndarrays(fed_state))

        metrics = {
            "client_id": self.client_id,
            "hospital": self.hospital,
            "train_loss": train_loss,
            "comp_time_sec": comp_time,
            "comm_bytes": comm_bytes,
        }

        if self.algo.get("domain_adaptation", False):
            # Ship this client's own LOCAL (non-federated) BN buffers back
            # to the driver process via the metrics channel — see
            # encode_state_dict_b64's docstring for why this, rather than
            # any direct object reference, is required when running under
            # Ray-backed simulation. The driver (src/fl/simulation.py)
            # decodes this on the LAST round to assemble per-hospital
            # deployable checkpoints.
            local_keys = split_federated_and_local_params(self.model, True)["local"]
            full_state = self.model.state_dict()
            local_bn_state = {k: full_state[k] for k in local_keys}
            metrics["bn_state_b64"] = encode_state_dict_b64(local_bn_state)

        return state_dict_to_ndarrays(fed_state), num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, Dict]:
        global_federated_state = ndarrays_to_state_dict(self._federated_keys, parameters)
        merge_local_bn_into_global(self.model, global_federated_state,
                                    self.algo.get("domain_adaptation", False))

        loss, acc = self._local_evaluate()
        num_examples = len(self.val_loader.dataset)
        return float(loss), num_examples, {
            "client_id": self.client_id,
            "hospital": self.hospital,
            "accuracy": acc,
        }

    # -- training internals ------------------------------------------------

    def _local_train(self) -> float:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        total_loss, total_batches = 0.0, 0

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits, features = self._forward_with_features(self.model, images)
                loss = F.cross_entropy(logits, labels)

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
        prox_term = 0.0
        for local_p, global_p in zip(self.model.parameters(), self._global_model_snapshot.parameters()):
            prox_term = prox_term + torch.sum((local_p - global_p) ** 2)
        return (mu / 2.0) * prox_term

    def _fedmoon_term(self, images: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        """Model-contrastive loss: pull current local features toward the
        global model's features on the same batch, push away from the
        previous round's local features (the negative)."""
        moon_mu = self.algo.get("moon_mu", 1.0)
        temperature = self.algo.get("moon_temperature", 0.5)

        with torch.no_grad():
            _, global_features = self._forward_with_features(self._global_model_snapshot, images)
        if self._prev_local_model is not None:
            with torch.no_grad():
                _, prev_features = self._forward_with_features(self._prev_local_model, images)
        else:
            prev_features = global_features  # first round: no real negative yet

        pos_sim = F.cosine_similarity(local_features, global_features) / temperature
        neg_sim = F.cosine_similarity(local_features, prev_features) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        contrastive_targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        contrastive_loss = F.cross_entropy(logits, contrastive_targets)
        return moon_mu * contrastive_loss

    @staticmethod
    def _forward_with_features(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model and also return its penultimate-layer feature vector
        (needed for FedMOON's contrastive loss). Works generically across the
        backbone zoo by registering a forward PRE-hook on the final Linear
        classifier layer, capturing its input tensor (the pooled feature
        vector) regardless of backbone architecture — no backbone-specific
        surgery required."""
        features_holder = {}

        def pre_hook(module, inputs):
            features_holder["features"] = torch.flatten(inputs[0], 1)

        classifier_module, _ = _find_classifier_module(model)
        handle = classifier_module.register_forward_pre_hook(pre_hook)
        try:
            logits = model(images)
        finally:
            handle.remove()
        return logits, features_holder.get("features", torch.zeros(images.size(0), 1, device=images.device))

    def _local_evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
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


def _find_classifier_module(model: nn.Module) -> Tuple[nn.Module, str]:
    """Locate the final Linear classifier layer generically across backbones."""
    last_linear_name, last_linear = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name, last_linear = name, module
    if last_linear is None:
        raise ValueError("Could not find a Linear classifier head in model.")
    return last_linear, last_linear_name
