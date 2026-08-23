"""
FUSED Algorithm 1 — the paper's ACTUAL method (CLI + sparse adapters),
not the LoRA stand-in used in fused_training.py / fused_training_domain.py.
This is a NEW, separate pipeline; it does not modify or import anything
from those files, and existing LoRA-based runs are entirely unaffected.

Pipeline (matches Algorithm 1 in the paper exactly):
  1. Critical Layer Identification (critical_layers_generic.py): one
     short federated round on REMEMBER clients, diff each client's
     trained clone against the untouched source model, pick the top-K
     most sensitive layers.
  2. Build a SparseAdapterSet (sparse_adapter_generic.py) for those K
     layers: random dropout mask + zero-initialized trainable delta per
     layer, `sparsity` fraction trainable.
  3. For `fused_iterations` rounds: each REMEMBER client merges its own
     adapter into a frozen clone of the source model, trains ONLY the
     unmasked delta entries (source model's real parameters are NEVER
     touched), then the deltas are FedAvg'd across remember clients
     (forget clients never participate — Algorithm 1, lines 4-7).
  4. Final adapters are merged permanently into the source model to
     produce M^f.

DOMAIN-ADAPTATION AWARENESS (this project's contribution, added on top
of the paper's plain-FedAvg Algorithm 1, exactly as it was added for the
LoRA pipeline in core_domain.py):
  - algorithm="fedprox": adds (mu/2)||delta_local - delta_global||^2 to
    each client's adapter loss, where delta_global is the ROUND-START
    averaged adapter (the sparse-adapter analogue of FedProx's proximal
    term against the global model).
  - algorithm="fedmoon": adds the model-contrastive term between (a)
    features from the client's CURRENT adapter-merged model, (b)
    features from the ROUND-START globally-averaged adapter merged in
    (the "global" positive pair), and (c) features from this SAME
    client's own PREVIOUS-round adapter-merged model (the "negative"
    pair) — persisted per client across rounds, same pattern as
    core_domain.py.
  - algorithm="fedbn": if any selected critical layer is a BatchNorm
    parameter (uncommon but possible — the paper found some ResNet18 BN
    layers among the most sensitive), that layer's adapter delta is kept
    LOCAL per remember-client (never averaged) instead of FedAvg'd,
    matching FedBN's spirit applied at the adapter-delta level. Ordinary
    (non-BN) critical layers still aggregate normally. If NO critical
    layer happens to be a BN parameter, algorithm="fedbn" is functionally
    identical to "fedavg" for that run (documented, not silently wrong).
  - algorithm="fedavg" (default): exactly Algorithm 1 as written in the
    paper, no additions.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.eval.client_forget_eval import test_client_forget
from src.fl.client import _find_classifier_module
from src.fl.core import SGD_MOMENTUM, SGD_WEIGHT_DECAY
from src.fu.critical_layers_generic import (
    VALID_CLI_SCORE_MODES,
    build_critical_layer_scores,
    run_critical_layer_identification,
    select_top_k_critical_layers,
)
from src.fu.sparse_adapter_generic import SparseAdapterSet, average_adapter_deltas
from src.models.backbone import get_batchnorm_layer_names

VALID_ALGORITHMS = ("fedavg", "fedbn", "fedprox", "fedmoon")


def _check_algorithm(algorithm: str) -> str:
    algorithm = algorithm.lower()
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use one of {VALID_ALGORITHMS}.")
    return algorithm

def _normalized_cli_candidate_names(
    model: nn.Module,
) -> set[str]:
    """
    Return convolution-style weight tensors eligible for normalized
    and domain-aware critical-layer selection.

    Excludes:
      - bias tensors;
      - one-dimensional BatchNorm affine parameters;
      - the final classification head.

    Includes:
      - ordinary convolution weights;
      - residual downsample convolution weights.
    """
    candidates: set[str] = set()

    for name, parameter in model.named_parameters():
        if not name.endswith(".weight"):
            continue

        # BN affine weights are one-dimensional. Convolution weights
        # have at least two dimensions.
        if parameter.ndim < 2:
            continue

        # The classifier contains class-discriminative information and
        # is not a hospital-domain feature extractor.
        if name.startswith("fc.") or ".fc." in name:
            continue

        candidates.add(name)

    if not candidates:
        raise RuntimeError(
            "No eligible convolutional parameters were found for "
            "normalized/domain-aware CLI selection."
        )

    return candidates

def _save_selection_scores(
    path: str,
    payload: dict,
) -> None:
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

def _forward_with_features(
    model: nn.Module, images: torch.Tensor, overrides: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same pattern as core_domain.py::_forward_with_features — capture
    the penultimate feature vector via a forward pre-hook on the final
    classifier layer, reusing the exact same helper as the rest of the
    project (src/fl/client.py::_find_classifier_module).

    If `overrides` is given (a {param_name: tensor} dict, e.g. from
    SparseAdapterSet.functional_forward's construction), the forward
    pass is run via torch.func.functional_call with those parameter
    values substituted — keeping the computation differentiable w.r.t.
    whatever `overrides`' tensors depend on (e.g. adapter deltas) — the
    hook still fires normally since functional_call still executes the
    module's real forward method, only the parameter VALUES differ for
    this call.
    """
    features_holder = {}

    def pre_hook(module, inputs):
        features_holder["features"] = torch.flatten(inputs[0], 1)

    classifier_module, _ = _find_classifier_module(model)
    handle = classifier_module.register_forward_pre_hook(pre_hook)

    try:
        if overrides is not None:
            if hasattr(torch, "func") and hasattr(torch.func, "functional_call"):
                functional_call = torch.func.functional_call
            else:  # older torch versions
                from torch.nn.utils.stateless import functional_call  # type: ignore
            logits = functional_call(model, overrides, args=(images,))
        else:
            logits = model(images)
    finally:
        handle.remove()

    features = features_holder.get("features", torch.zeros(images.size(0), 1, device=images.device))
    return logits, features


def _identify_bn_critical_layers(model: nn.Module, critical_layers: List[str]) -> set:
    bn_names = set(get_batchnorm_layer_names(model))
    return {name for name in critical_layers if any(name.startswith(bn) for bn in bn_names)}


def _representative_bn_adapter(
    global_adapter: SparseAdapterSet,
    client_local_bn_deltas: Dict[int, Dict[str, torch.Tensor]],
    bn_critical_layers: set,
    device: str,
) -> SparseAdapterSet:
    """BUG FIX: for algorithm="fedbn", per-client BN-critical adapter
    deltas were tracked in `client_local_bn_deltas` across rounds (used
    only to seed each client's OWN next round) but never actually reached
    the evaluated/saved model — both the per-round eval model and the
    final merged model baked in `global_adapter`, the cross-client FedAvg
    of BN-critical deltas, which this file's own comments already flag as
    "NOT meaningfully averaged." That defeated the point of the FedBN
    ablation for this pipeline. Build one representative adapter instead:
    identical to `global_adapter` everywhere, except BN-critical layers are
    overridden with ONE client's own tracked local delta (deterministic:
    lowest client index that has reported), mirroring the same
    "representative client" pattern src/fl/core_domain.py::
    assemble_representative_bn_model already uses for the LoRA/FedBN path.
    """
    if not bn_critical_layers or not client_local_bn_deltas:
        return global_adapter

    representative_idx = sorted(client_local_bn_deltas.keys())[0]
    representative_deltas = client_local_bn_deltas[representative_idx]

    merged = SparseAdapterSet(global_adapter.critical_layers, global_adapter.sparsity)
    merged.masks = {k: v.clone() for k, v in global_adapter.masks.items()}
    merged.deltas = {k: nn.Parameter(v.data.clone(), requires_grad=False)
                      for k, v in global_adapter.deltas.items()}
    for name in bn_critical_layers:
        if name in representative_deltas:
            merged.deltas[name] = nn.Parameter(
                representative_deltas[name].clone().to(device), requires_grad=False)
    return merged


def run_fused_cli_unlearning(
    source_model: nn.Module,
    all_clean_client_loaders: List[DataLoader],
    attacked_test_loaders: List[DataLoader],
    forget_client_idx: List[int],
    client_data_sizes: List[int],
    num_unlearning_layers: int,
    adapter_sparsity: float,
    fused_iterations: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    test_batch_size: int,
    algorithm: str = "fedavg",
    fedprox_mu: float = 0.01,
    fedmoon_mu: float = 1.0,
    fedmoon_temperature: float = 0.5,
    cli_local_epochs: int = 1,
    cli_use_all_clients: bool = True,
    cli_score_mode: str = "raw_sum",
    domain_scores: Optional[
        Dict[str, float]
    ] = None,
    domain_lambda: float = 0.5,
    selection_scores_output_path: Optional[
        str
    ] = None,
    selection_only: bool = False,
    selection_log_top_k: int = 10,
    seed: int = 42,
    logger=None,
) -> Tuple[nn.Module, dict, List[str]]:
    """Full Algorithm 1 (CLI + sparse adapter FUSED unlearning), plus
    optional domain-adaptation awareness — see module docstring.

    Returns (unlearned_model, history, critical_layer_names).
    `source_model` is treated as READ-ONLY throughout (never mutated in
    place) until the very last step, which merges into a fresh deepcopy.

    `cli_use_all_clients` (default True, paper-faithful): Eq 11-13 defines
    the CLI diff as a data-volume-weighted sum over ALL N clients (n =
    1, ..., N), which — in the client-unlearning setting — includes the
    client that's about to be forgotten. That's intentional: CLI's stated
    purpose (Sec 4.1) is to find "the layers that are sensitive to
    [the] knowledge" being removed, so the forget client's own local
    update is exactly the signal that tells you which layers actually
    hold ITS knowledge, not just which layers are sensitive to remember-
    client heterogeneity. The forget client's data is used ONLY for this
    one diagnostic federated round (one local-epoch pass to measure a
    Manhattan distance) — it is never used again afterward; all actual
    adapter training below still uses remember clients exclusively.
    Set this to False if your deployment's right-to-be-forgotten policy
    requires the forget client's data to never be touched at all, even
    for this one measurement pass — you'll deviate from Eq 11-13, but
    stay strictly compliant.
    """
    algorithm = _check_algorithm(
        algorithm
    )

    cli_score_mode = (
        cli_score_mode.lower()
    )

    if (
        cli_score_mode
        not in VALID_CLI_SCORE_MODES
    ):
        raise ValueError(
            f"Unknown cli_score_mode "
            f"{cli_score_mode!r}; expected one of "
            f"{VALID_CLI_SCORE_MODES}."
        )

    if domain_lambda < 0.0:
        raise ValueError(
            "domain_lambda must be non-negative."
        )

    if selection_log_top_k < 0:
        raise ValueError(
            "selection_log_top_k must be "
            "non-negative."
        )

    num_clients = len(
        all_clean_client_loaders
    )
    remember_idx = [i for i in range(num_clients) if i not in forget_client_idx]
    remember_loaders = [all_clean_client_loaders[i] for i in remember_idx]
    remember_data_sizes = [client_data_sizes[i] for i in remember_idx]

    # --- Step 1: Critical Layer Identification (paper Sec 4.1, Eq 11-13) ---
    # Eq 13 sums over ALL N clients — including the forget client — not just
    # remember clients (see cli_use_all_clients docstring above for why this
    # matters mechanically, not just for literal paper-fidelity).
    cli_loaders = all_clean_client_loaders if cli_use_all_clients else remember_loaders
    cli_data_sizes = client_data_sizes if cli_use_all_clients else remember_data_sizes
    if logger is not None:
        scope = "ALL clients (paper Eq 11-13)" if cli_use_all_clients else "remember clients only (compliance mode)"
        logger.info(f"Running Critical Layer Identification (one federated round, {scope})...")
        diff_reduction = (
        "sum"
        if cli_score_mode == "raw_sum"
        else "mean"
    )

    diffs = run_critical_layer_identification(
        source_model,
        cli_loaders,
        cli_data_sizes,
        device,
        local_epochs=cli_local_epochs,
        learning_rate=learning_rate,
        diff_reduction=diff_reduction,
    )

    # Preserve the original paper-style candidate space for raw_sum.
    # For mean_abs and domain_aware, compare only convolutional feature
    # weights so small bias and BN affine tensors cannot dominate.
    if cli_score_mode == "raw_sum":
        candidate_diffs = diffs
    else:
        eligible_names = _normalized_cli_candidate_names(
            source_model
        )

        candidate_diffs = {
            name: score
            for name, score in diffs.items()
            if name in eligible_names
        }

        if logger is not None:
            logger.info(
                "Restricted normalized/domain-aware CLI selection to "
                f"{len(candidate_diffs)} convolutional weight tensors "
                f"from {len(diffs)} total learnable parameters."
            )

    (
        selection_scores,
        selection_rows,
    ) = build_critical_layer_scores(
        candidate_diffs,
        cli_score_mode=cli_score_mode,
        domain_scores=domain_scores,
        domain_lambda=domain_lambda,
    )

    critical_layers = (
        select_top_k_critical_layers(
            selection_scores,
            num_unlearning_layers,
        )
    )

    selected_set = set(
        critical_layers
    )

    for row in selection_rows:
        row["selected"] = (
            row["parameter"]
            in selected_set
        )

    selection_payload = {
        "cli_score_mode": cli_score_mode,
        "diff_reduction": diff_reduction,
        "domain_lambda": (
            domain_lambda
            if cli_score_mode
            == "domain_aware"
            else 0.0
        ),
        "num_unlearning_layers": (
            num_unlearning_layers
        ),
        "critical_layers": (
            critical_layers
        ),
        "candidates": selection_rows,
    }

    if (
        selection_scores_output_path
        is not None
    ):
        _save_selection_scores(
            selection_scores_output_path,
            selection_payload,
        )

    selection_message = (
        f"Selected {len(critical_layers)} "
        "critical layers using "
        f"cli_score_mode={cli_score_mode}: "
        f"{critical_layers}"
    )

    if logger is not None:
        logger.info(
            selection_message
        )

        for row in selection_rows[
            :selection_log_top_k
        ]:
            logger.info(
                "[CLI score] "
                f"rank={row['rank']} "
                f"selected={row['selected']} "
                f"parameter={row['parameter']} "
                f"cli={row['cli_score']:.8g} "
                "cli_norm="
                f"{row['normalized_cli_score']:.6f} "
                f"domain={row['domain_score']:.6f} "
                f"final={row['selection_score']:.6f}"
            )
    else:
        print(
            f"[CLI] {selection_message}"
        )

    bn_critical_layers = (
        _identify_bn_critical_layers(
            source_model,
            critical_layers,
        )
        if algorithm == "fedbn"
        else set()
    )

    if (
        algorithm == "fedbn"
        and logger is not None
    ):
        logger.info(
            "FedBN-local (never-aggregated) "
            "critical layers among selection: "
            f"{bn_critical_layers or 'none'}"
        )

    if selection_only:
        history = {
            "round": [],
            "avg_f_acc": [],
            "avg_r_acc": [],
            "critical_layer_selection": (
                selection_payload
            ),
        }

        return (
            copy.deepcopy(source_model),
            history,
            critical_layers,
        )
    # --- Step 2-4: sparse adapter construction + FUSED federation ------
    # BUG FIX: this used to reassign `source_model = source_model.to(device)`
    # and then freeze `.requires_grad` on those SAME parameter objects —
    # `.to(device)` returns the same module (in-place), so this mutated the
    # CALLER's source_model, contradicting this function's own "read-only"
    # docstring above. Harmless as long as nothing reused source_model after
    # this call — until the shadow-model MIA (src/eval/mia.py, wired in via
    # run_fu_cli_domain.py's shadow_fn) started doing exactly that: reusing
    # source_model post-call to build each shadow model. Once frozen here,
    # every shadow's fresh Critical Layer Identification pass got a model
    # with requires_grad=False everywhere, so loss.backward() had no
    # gradient path: "element 0 of tensors does not require grad and does
    # not have a grad_fn". Deep-copy here so freezing only ever touches a
    # local copy, making this function actually read-only as documented.
    source_model = copy.deepcopy(source_model).to(device)
    source_model.eval()
    for p in source_model.parameters():
        p.requires_grad = False

    global_adapter = SparseAdapterSet.build(source_model, critical_layers, adapter_sparsity, seed=seed).to(device)

    history = {
        "round": [],
        "avg_f_acc": [],
        "avg_r_acc": [],
        "critical_layer_selection": (
            selection_payload
        ),
    }
    client_prev_merged_models: Dict[int, nn.Module] = {}
    client_local_bn_deltas: Dict[int, Dict[str, torch.Tensor]] = {}

    for round_idx in range(fused_iterations):
        round_start_adapter = global_adapter  # frozen reference for FedProx/FedMoon this round

        # Precompute the round-start globally-averaged adapter's features
        # ONCE per round (not per-batch) as a real, finalized, no-grad
        # snapshot — this is FedMoon's "positive"/global reference model.
        global_snapshot_model = None
        if algorithm == "fedmoon":
            global_snapshot_model = copy.deepcopy(source_model)
            round_start_adapter.finalize_into(global_snapshot_model)
            global_snapshot_model.eval()
            for p in global_snapshot_model.parameters():
                p.requires_grad = False

        client_adapters: List[SparseAdapterSet] = []

        for local_client_idx, loader in enumerate(remember_loaders):
            client_idx = remember_idx[local_client_idx]  # original client index, for prev-model persistence

            client_adapter = SparseAdapterSet(critical_layers, adapter_sparsity)
            client_adapter.masks = {k: v.clone() for k, v in round_start_adapter.masks.items()}
            client_adapter.deltas = {
                k: nn.Parameter(v.data.clone(), requires_grad=True)
                for k, v in round_start_adapter.deltas.items()
            }

            if algorithm == "fedbn" and client_idx in client_local_bn_deltas:
                for name in bn_critical_layers:
                    if name in client_local_bn_deltas[client_idx]:
                        client_adapter.deltas[name] = nn.Parameter(
                            client_local_bn_deltas[client_idx][name].clone().to(device), requires_grad=True)

            optimizer = optim.SGD(client_adapter.trainable_parameters(), lr=learning_rate,
                                   momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
            criteria = nn.CrossEntropyLoss()

            use_moon = algorithm == "fedmoon"
            prev_merged_model = client_prev_merged_models.get(client_idx) if use_moon else None

            for _ in range(local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()

                    # Differentiable forward: delta stays in the autograd
                    # graph (functional_call), unlike the old in-place
                    # merge_into()/restore_modules() approach.
                    if use_moon:
                        state_dict = source_model.state_dict()
                        overrides = {
                            name: state_dict[name] + client_adapter.deltas[name] * client_adapter.masks[name]
                            for name in client_adapter.critical_layers if name in client_adapter.deltas
                        }
                        pred, local_features = _forward_with_features(source_model, data, overrides=overrides)
                    else:
                        pred = client_adapter.functional_forward(source_model, data)
                    loss = criteria(pred, target)

                    if algorithm == "fedprox":
                        prox_term = torch.tensor(0.0, device=device)
                        for name, delta in client_adapter.deltas.items():
                            prox_term = prox_term + torch.sum((delta - round_start_adapter.deltas[name]) ** 2)
                        loss = loss + (fedprox_mu / 2.0) * prox_term

                    if use_moon:
                        with torch.no_grad():
                            _, global_features = _forward_with_features(global_snapshot_model, data)
                            if prev_merged_model is not None:
                                _, prev_features = _forward_with_features(prev_merged_model, data)
                            else:
                                prev_features = global_features

                        pos_sim = F.cosine_similarity(local_features, global_features) / fedmoon_temperature
                        neg_sim = F.cosine_similarity(local_features, prev_features) / fedmoon_temperature
                        moon_logits = torch.stack([pos_sim, neg_sim], dim=1)
                        moon_targets = torch.zeros(moon_logits.size(0), dtype=torch.long, device=device)
                        loss = loss + fedmoon_mu * F.cross_entropy(moon_logits, moon_targets)

                    loss.backward()
                    optimizer.step()
                    client_adapter.apply_masks()

            if algorithm == "fedbn":
                client_local_bn_deltas.setdefault(client_idx, {})
                for name in bn_critical_layers:
                    client_local_bn_deltas[client_idx][name] = client_adapter.deltas[name].detach().cpu().clone()

            if use_moon:
                snapshot = copy.deepcopy(source_model)
                client_adapter.finalize_into(snapshot)
                snapshot.eval()
                for p in snapshot.parameters():
                    p.requires_grad = False
                client_prev_merged_models[client_idx] = snapshot

            client_adapters.append(client_adapter)

        # --- Server aggregation: FedAvg over adapter deltas only -------
        global_adapter = average_adapter_deltas(client_adapters, weights=remember_data_sizes)
        if algorithm == "fedbn":
            # BN-critical deltas are NOT meaningfully averaged across
            # clients (each client keeps its own via client_local_bn_deltas
            # above); leave the averaged value in global_adapter as a
            # reasonable shared initialization for any client that hasn't
            # participated yet, but it is overwritten per-client next round.
            pass

        # --- Evaluate (attacked test set, matching the LoRA pipeline) --
        # No gradients needed here, so a real (non-functional) finalized
        # copy is simplest and correct.
        eval_adapter = global_adapter
        if algorithm == "fedbn":
            eval_adapter = _representative_bn_adapter(
                global_adapter, client_local_bn_deltas, bn_critical_layers, device)
        eval_model = copy.deepcopy(source_model)
        eval_adapter.finalize_into(eval_model)
        eval_model.eval()
        avg_f_acc, avg_r_acc, _ = test_client_forget(
            eval_model, attacked_test_loaders, forget_client_idx, device, test_batch_size,
        )
        del eval_model

        history["round"].append(round_idx)
        history["avg_f_acc"].append(avg_f_acc)
        history["avg_r_acc"].append(avg_r_acc)

        msg = (f"[FUSED-CLI:{algorithm}] Round={round_idx}, "
               f"avg_r_acc={avg_r_acc:.4f}, avg_f_acc={avg_f_acc:.4f}, "
               f"adapter_density={global_adapter.num_trainable_elements()}/{global_adapter.num_total_elements()}")
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.log_scalar("fu_cli/forget_client_acc", avg_f_acc, round_idx)
            logger.log_scalar("fu_cli/remember_client_acc", avg_r_acc, round_idx)

    # --- Final merge: M^f = merge(M^r, A^f) ----------------------------
    final_adapter = global_adapter
    if algorithm == "fedbn":
        final_adapter = _representative_bn_adapter(
            global_adapter, client_local_bn_deltas, bn_critical_layers, device)
    unlearned_model = copy.deepcopy(source_model)
    final_adapter.finalize_into(unlearned_model)

    for p in unlearned_model.parameters():
        p.requires_grad = True

    return unlearned_model, history, critical_layers
