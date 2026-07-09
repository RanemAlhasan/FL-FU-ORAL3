"""
ORPHANED / NOT A LIVE ENTRY POINT — DO NOT IMPORT OR RUN.

Flower client used during the FUSED unlearning phase. Unlike the FL-phase
client (src/fl/client.py), this client:
  - Never updates the frozen base model's parameters.
  - Only trains its share of the sparse adapter deltas (Eq. 14).
  - Is only ever instantiated for REMEMBER clients — forget clients are
    excluded entirely from this phase (Algorithm 1, lines 4-7).

Only used by src/fu/fused_runner.py, which is itself orphaned (see that
file's module docstring for the full explanation). Not imported by any
script today. For any real Phase-2 FUSED run, use
scripts/run_fu_cli_domain.py or scripts/run_fu_lora_domain.py instead.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# FIX (was a broken import — src/fu/sparse_adapter.py doesn't exist, only
# sparse_adapter_generic.py does): this now imports without raising
# ModuleNotFoundError, but the generic SparseAdapterSet's API
# (`.deltas`/`.masks` dicts) does NOT match how this file uses it below
# (`.adapters.values()`, per-adapter `.delta`) — see fused_runner.py's
# module docstring for the full explanation. This file is dead/orphaned
# code and still cannot actually run without a real port.
from src.fu.sparse_adapter_generic import SparseAdapterSet


def adapter_deltas_to_ndarrays(adapter_set: SparseAdapterSet) -> List[np.ndarray]:
    """Flatten ONLY the trainable delta tensors (not masks, not base params —
    masks are static/shared and base params never move) into the array list
    Flower sends over the wire. This is what gives FUSED its tiny
    communication footprint (Table 1 "Comm" column)."""
    return [adapter.delta.detach().cpu().numpy() for adapter in adapter_set.adapters.values()]


def ndarrays_to_adapter_deltas(adapter_set: SparseAdapterSet, arrays: List[np.ndarray]) -> None:
    with torch.no_grad():
        for adapter, array in zip(adapter_set.adapters.values(), arrays):
            adapter.delta.copy_(torch.tensor(array, dtype=adapter.delta.dtype))
    adapter_set.apply_masks()


class FusedAdapterClient(fl.client.NumPyClient):
    """One Flower client per REMEMBER partition during the FUSED unlearning
    phase. Holds its own SparseAdapterSet instance (independent trainable
    deltas, same mask/critical-layer structure as every other client) and
    its own frozen copy of the base model for the forward pass."""

    def __init__(
        self,
        client_id: str,
        hospital: str,
        frozen_model: torch.nn.Module,
        adapter_set: SparseAdapterSet,
        train_loader: DataLoader,
        device: str,
        local_epochs: int,
        learning_rate: float,
    ):
        self.client_id = client_id
        self.hospital = hospital
        self.frozen_model = frozen_model.to(device)
        for p in self.frozen_model.parameters():
            p.requires_grad = False
        self.adapter_set = adapter_set.to(device)
        self.train_loader = train_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config) -> List[np.ndarray]:
        return adapter_deltas_to_ndarrays(self.adapter_set)

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, Dict]:
        ndarrays_to_adapter_deltas(self.adapter_set, parameters)

        optimizer = torch.optim.Adam(self.adapter_set.trainable_parameters(), lr=self.learning_rate)

        comp_start = time.time()
        total_loss, total_batches = 0.0, 0
        self.frozen_model.train()  # BN running stats still update; weights are frozen via requires_grad=False

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Eq. 15: merge (base + delta) into the model before forward.
                self.adapter_set.merge_into(self.frozen_model)

                optimizer.zero_grad()
                logits = self.frozen_model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                self.adapter_set.apply_masks()  # keep deltas sparse (Eq. 14 constraint)

                # Put the model's real nn.Parameters back so it remains a
                # normal, serializable nn.Module between batches (e.g. for
                # client_registry inspection or checkpointing elsewhere).
                self.adapter_set.restore_modules()

                total_loss += loss.item()
                total_batches += 1

        comp_time = time.time() - comp_start
        deltas = adapter_deltas_to_ndarrays(self.adapter_set)
        comm_bytes = sum(arr.nbytes for arr in deltas)

        metrics = {
            "client_id": self.client_id,
            "hospital": self.hospital,
            "adapter_train_loss": total_loss / max(1, total_batches),
            "comp_time_sec": comp_time,
            "comm_bytes": comm_bytes,
        }
        return deltas, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, Dict]:
        # Adapter-phase clients are not evaluated directly by the strategy in
        # this pipeline (evaluation happens centrally in run_fu.py against
        # held-out remember/forget test sets after the FU phase completes),
        # but we implement this for Flower API completeness / debugging.
        ndarrays_to_adapter_deltas(self.adapter_set, parameters)
        self.adapter_set.merge_into(self.frozen_model)
        self.frozen_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.frozen_model(images)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += images.size(0)
        self.adapter_set.restore_modules()
        return float(total_loss / max(1, total)), total, {"accuracy": correct / max(1, total)}
