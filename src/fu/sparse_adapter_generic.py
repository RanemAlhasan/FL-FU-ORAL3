"""
Sparse unlearning adapters — paper Section 4.2, Eq. 14-15, Algorithm 1.

THIS IS NOT LoRA. The paper's adapter is a raw, randomly-sparsified
DELTA over the SAME shape as each critical layer's original parameter
tensor — "we discard most of the parameters in a random manner and leave
only a small portion, forming a sparse parameter matrix A^f" — not a
low-rank decomposition. This module implements exactly that.

For each critical layer l (selected by critical_layers_generic.py):
  - A boolean mask M_l is drawn once (fixed for the adapter's lifetime),
    with `sparsity` fraction of entries True (trainable) and the rest
    False (permanently zero delta).
  - A trainable delta tensor A^f_l is created, same shape as the
    original layer, initialized to zero, and masked: only M_l entries
    ever receive gradient / ever get written by the optimizer step
    (Eq. 14 constraint).
  - At inference/forward time, the layer's effective parameter is
    (original_frozen_value + A^f_l * M_l) (Eq. 15's "p_Af + p_Lf").
  - The ORIGINAL model's frozen parameters are NEVER modified in place
    for critical layers, and are never trained for non-critical layers
    either (only the sparse deltas are ever in the optimizer).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class SparseAdapterSet:
    """Holds one sparse delta + mask per critical layer, for ONE model
    instance. Not shared across clients — each client gets its own
    SparseAdapterSet built with `SparseAdapterSet.build(...)`, sharing
    only the SAME critical_layers list and sparsity (so masks can differ
    in their random draw per client, matching the paper's "random
    manner" — this mirrors the actual repo's per-client independent
    adapter, only the DELTAS get aggregated via FedAvg afterward, not
    the masks)."""

    def __init__(self, critical_layers: List[str], sparsity: float):
        self.critical_layers = list(critical_layers)
        self.sparsity = sparsity
        self.masks: Dict[str, torch.Tensor] = {}
        self.deltas: Dict[str, nn.Parameter] = {}
        self._original_values: Dict[str, torch.Tensor] = {}

    @classmethod
    def build(cls, model: nn.Module, critical_layers: List[str], sparsity: float,
              seed: Optional[int] = None) -> "SparseAdapterSet":
        adapter_set = cls(critical_layers, sparsity)
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        state_dict = model.state_dict()
        for name in critical_layers:
            if name not in state_dict:
                continue
            shape = state_dict[name].shape
            if generator is not None:
                rand = torch.rand(shape, generator=generator)
            else:
                rand = torch.rand(shape)
            mask = (rand < sparsity).float()
            adapter_set.masks[name] = mask
            adapter_set.deltas[name] = nn.Parameter(torch.zeros(shape), requires_grad=True)
        return adapter_set

    def to(self, device) -> "SparseAdapterSet":
        for name in list(self.deltas.keys()):
            self.deltas[name] = nn.Parameter(self.deltas[name].data.to(device), requires_grad=True)
            self.masks[name] = self.masks[name].to(device)
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return list(self.deltas.values())

    def num_trainable_elements(self) -> int:
        return int(sum(mask.sum().item() for mask in self.masks.values()))

    def num_total_elements(self) -> int:
        return int(sum(mask.numel() for mask in self.masks.values()))

    def apply_masks(self) -> None:
        """Re-zero any delta entries outside their mask, IN PLACE — call
        this after every optimizer.step() to enforce the Eq. 14 sparsity
        constraint (an SGD step on a masked parameter can, in principle,
        still nudge masked-out entries via weight_decay; this guarantees
        they stay exactly zero regardless)."""
        with torch.no_grad():
            for name, delta in self.deltas.items():
                delta.mul_(self.masks[name])

    def functional_forward(self, model: nn.Module, *args, **kwargs):
        """Run `model(*args, **kwargs)` with each critical layer's real
        parameter/buffer REPLACED, for this call only, by
        (original_frozen_value + delta * mask) — Eq. 15's
        M_n(i,e) = (p_Af + p_Lf) o p_Lr — computed as a genuine
        differentiable tensor op via torch.func.functional_call, so
        gradients correctly flow back into `self.deltas` on
        loss.backward(). This REPLACES the old merge_into()/
        restore_modules() pair, which mutated the model's real
        parameters via an in-place .copy_() under torch.no_grad() —
        that approach silently breaks autograd (the in-place copy is not
        a differentiable operation, so backward() has nothing to flow
        gradients through), causing "does not require grad and does not
        have a grad_fn" errors. functional_call avoids ever touching the
        model's real storage: `model` is left completely untouched by
        this call, nothing to restore afterward.
        """
        state_dict = model.state_dict()
        overrides = {}
        for name in self.critical_layers:
            if name not in self.deltas:
                continue
            overrides[name] = state_dict[name] + self.deltas[name] * self.masks[name]

        if hasattr(torch, "func") and hasattr(torch.func, "functional_call"):
            functional_call = torch.func.functional_call
        else:  # older torch versions
            from torch.nn.utils.stateless import functional_call  # type: ignore

        return functional_call(model, overrides, args=args, kwargs=kwargs)

    def finalize_into(self, model: nn.Module) -> nn.Module:
        """Permanently bake (original + delta * mask) into `model`'s real
        parameters — use this ONCE, at the very end of training, to
        produce the final exportable/checkpointable unlearned model
        M^f = merge(M^r, A^f). No further backward pass through these
        deltas is needed after this call."""
        state_dict = model.state_dict()
        for name in self.critical_layers:
            if name not in self.deltas:
                continue
            state_dict[name] = (state_dict[name] + self.deltas[name] * self.masks[name]).detach()
        model.load_state_dict(state_dict, strict=True)
        return model


def average_adapter_deltas(
    adapter_sets: List[SparseAdapterSet],
    weights: Optional[List[float]] = None,
) -> SparseAdapterSet:
    """FedAvg over ONLY the delta tensors (Algorithm 1, line 16: "Server
    aggregates adapter deltas via FedAvg") — masks are per-client and
    NOT averaged (each client's mask stays fixed for its own lifetime;
    only the resulting delta VALUES get averaged, exactly like averaging
    any other trainable parameter under FedAvg). All adapter_sets must
    share the same critical_layers/sparsity (i.e. all built from the
    same SparseAdapterSet.build(..., critical_layers=..., sparsity=...)
    call structure, just possibly different random seeds per client).

    `weights`: optional per-client aggregation weights (pass each
    remember-client's local dataset size), matching Phase 1's Flower
    FedAvg strategy, which weights by num_examples. Normalized
    internally. Omit to fall back to the original plain unweighted mean
    (every existing caller that doesn't pass `weights` keeps its exact
    prior behavior)."""
    reference = adapter_sets[0]
    averaged = SparseAdapterSet(reference.critical_layers, reference.sparsity)
    averaged.masks = {name: mask.clone() for name, mask in reference.masks.items()}

    if weights is not None:
        if len(weights) != len(adapter_sets):
            raise ValueError(
                f"weights must have one entry per adapter set in adapter_sets "
                f"({len(weights)} given, {len(adapter_sets)} expected)."
            )
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
    else:
        norm_weights = [1.0 / len(adapter_sets)] * len(adapter_sets)

    for name in reference.deltas:
        weighted_sum = torch.zeros_like(reference.deltas[name].data)
        for adapter_set, w in zip(adapter_sets, norm_weights):
            weighted_sum += adapter_set.deltas[name].data * w
        averaged.deltas[name] = nn.Parameter(weighted_sum, requires_grad=True)

    return averaged
