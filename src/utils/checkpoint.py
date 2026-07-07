"""
Checkpoint save/load utilities.

Critical invariant: loading a checkpoint from a *source* run (e.g. FU loading
an FL checkpoint, or a later FU run reloading an earlier FU's base model)
NEVER writes back into that source run's checkpoint directory, and the
returned state_dict is deep-copied into a freshly constructed model — never
the literal object graph that might be shared/mutated elsewhere in-process.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a model's state_dict (+ optional extra metadata) under
    checkpoint_dir/name.pt. Returns the saved path."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint_into_new_model(
    model_builder,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Build a brand-new model via `model_builder()` and load weights from
    `checkpoint_path` (read-only file access) into it. This is the ONLY
    sanctioned way for FU / retrain scripts to obtain a "copy" of a source
    FL model — it guarantees no shared parameter tensors with whatever
    object might exist in the source run's process (there shouldn't be one,
    since phases run in separate processes, but this also protects against
    accidental reuse within a single script, e.g. running CLI + adapter
    training back to back against "the same" model object).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = model_builder()
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    # Deep copy defensively: torch.load already allocates fresh tensors, but
    # we copy again to make the "never aliases another run's weights"
    # invariant explicit and robust to future refactors.
    state_dict = copy.deepcopy(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    return model


def clone_model(model: torch.nn.Module, model_builder, device: str = "cpu") -> torch.nn.Module:
    """Produce an independent deep copy of `model` via a fresh build + state_dict
    copy, rather than `copy.deepcopy(model)`, to keep behavior consistent with
    load_checkpoint_into_new_model and avoid copying optimizer/hook state."""
    clone = model_builder()
    state_dict = copy.deepcopy(model.state_dict())
    clone.load_state_dict(state_dict)
    clone.to(device)
    return clone


def get_checkpoint_path(checkpoint_dir: str, name: str = "best") -> str:
    return os.path.join(checkpoint_dir, f"{name}.pt")
