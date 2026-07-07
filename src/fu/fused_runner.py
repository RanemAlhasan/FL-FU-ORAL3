"""
Orchestrates the full FUSED unlearning phase (Algorithm 1 in the paper),
end to end:

  1. Load the source FL model — READ-ONLY, into a freshly built model
     instance (never the literal object from any other phase/process).
  2. Run a short CLI mini-federation (1 round by default) to compute
     per-layer Diff values and select the top-K critical layers.
  3. Build sparse adapters for those critical layers.
  4. Run I FUSED federated iterations: remember clients train their adapter
     shares locally (Eq. 14), server aggregates adapter deltas via FedAvg
     (Algorithm 1, line 16) — forget clients never participate.
  5. Merge final adapters into the frozen base model to get M^f.
  6. Return M^f (a brand-new model instance) plus the trained adapter set
     (saved separately so removal/reversibility is possible later without
     retraining).

This module is invoked exclusively by scripts/run_fu.py. It NEVER mutates
the source FL run's checkpoint or logs — see src/utils/checkpoint.py for the
enforced "build fresh, load weights" pattern used to obtain every model
instance here.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Context
from flwr.common.constant import PARTITION_ID_KEY
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.partition import ClientPartition
from src.fu.critical_layers import compute_layer_diffs, select_top_k_critical_layers
from src.fu.fused_client import (FusedAdapterClient, adapter_deltas_to_ndarrays,
                                  ndarrays_to_adapter_deltas)
from src.fu.sparse_adapter import SparseAdapterSet
from src.utils.checkpoint import clone_model
from src.utils.logger import ExperimentLogger


def _build_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def run_critical_layer_identification(
    source_model: nn.Module,
    model_builder,
    remember_partitions: List[ClientPartition],
    remember_datasets: List,
    device: str,
    batch_size: int,
    learning_rate: float,
    logger: ExperimentLogger,
) -> Dict[str, float]:
    """One short federated round (FUSED Sec 5.2: "the gap is most pronounced
    when Epoch=1") where each REMEMBER client trains its OWN clone of the
    source model for one local epoch, after which we compute layer-wise
    Diff values (Eq. 11-13) between each client's trained clone and the
    untouched source model.

    Critically: each client clone is produced via `clone_model`, which does
    a fresh build + state_dict copy — so this step cannot, even in principle,
    accidentally mutate `source_model`'s parameters."""
    logger.info("Running Critical Layer Identification (CLI) mini-federation (1 round)...")
    client_models: Dict[str, nn.Module] = {}
    client_data_sizes: Dict[str, int] = {}

    for partition, dataset in zip(remember_partitions, remember_datasets):
        client_model = clone_model(source_model, model_builder, device)
        loader = _build_loader(dataset, batch_size)
        optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)

        client_model.train()
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(client_model(images), labels)
            loss.backward()
            optimizer.step()

        client_models[partition.client_id] = client_model
        client_data_sizes[partition.client_id] = len(dataset)

    diffs = compute_layer_diffs(source_model, client_models, client_data_sizes)
    logger.info(f"CLI complete. Layer Diff values computed for {len(diffs)} candidate layers.")
    return diffs


def run_fused_unlearning(
    config: Dict,
    source_model: nn.Module,
    model_builder,
    remember_partitions: List[ClientPartition],
    remember_train_datasets: List,
    device: str,
    logger: ExperimentLogger,
) -> Tuple[nn.Module, SparseAdapterSet, List[str]]:
    """Run the full FUSED Algorithm 1 unlearning process. Returns
    (unlearned_model, trained_adapter_set, critical_layer_names).

    `source_model` is treated as READ-ONLY throughout: we never call
    `.backward()` or any in-place op on ITS parameters. All training
    happens on per-client clones and on the adapter deltas, which are only
    merged into a freshly-cloned model at the very end (step 6).
    """
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_unlearning_layers = config["num_unlearning_layers"]
    sparsity = config.get("adapter_sparsity", 0.05)
    fused_iterations = config["fused_iterations"]
    local_epochs = config.get("local_epochs", 1)
    seed = config.get("seed", 42)

    logger.log_hparams({
        "fu_method": "FUSED", "num_unlearning_layers": num_unlearning_layers,
        "adapter_sparsity": sparsity, "fused_iterations": fused_iterations,
        "local_epochs": local_epochs, "num_remember_clients": len(remember_partitions),
    })

    # --- Step 1-3: CLI + sparse adapter construction -----------------------
    diffs = run_critical_layer_identification(
        source_model, model_builder, remember_partitions, remember_train_datasets,
        device, batch_size, learning_rate, logger,
    )
    for layer_name, diff_value in sorted(diffs.items(), key=lambda x: -x[1])[:10]:
        logger.log_scalar(f"fu/cli_diff/{layer_name}", diff_value, 0)

    critical_layers = select_top_k_critical_layers(diffs, num_unlearning_layers)
    logger.info(f"Selected critical layers for unlearning adapters: {critical_layers}")

    adapter_set = SparseAdapterSet.build(source_model, critical_layers, sparsity=sparsity, seed=seed)
    total_params = adapter_set.num_total_elements()
    trainable_params = adapter_set.num_trainable_elements()
    logger.info(
        f"Sparse adapters built: {trainable_params}/{total_params} parameters "
        f"trainable ({100 * trainable_params / max(1, total_params):.3f}% density)."
    )
    logger.set_final_metric("fu/adapter_density", trainable_params / max(1, total_params))
    logger.set_final_metric("fu/adapter_trainable_params", trainable_params)

    # --- Step 4: I federated iterations, adapters only, remember clients only
    def client_fn(context: Context) -> fl.client.Client:
        idx = int(context.node_config[PARTITION_ID_KEY])
        partition = remember_partitions[idx]
        # Independent clone of the frozen base model + a fresh
        # SparseAdapterSet sharing the SAME masks/critical-layer structure
        # as `adapter_set` but with its own (initially zero) delta storage,
        # so every client trains independently before aggregation — exactly
        # mirroring Algorithm 1's per-client A^f_n(i, e).
        client_frozen_model = clone_model(source_model, model_builder, device)
        client_adapter_set = SparseAdapterSet.build(
            client_frozen_model, critical_layers, sparsity=sparsity, seed=seed,
        )
        loader = _build_loader(remember_train_datasets[idx], batch_size)
        numpy_client = FusedAdapterClient(
            client_id=partition.client_id, hospital=partition.hospital,
            frozen_model=client_frozen_model, adapter_set=client_adapter_set,
            train_loader=loader, device=device, local_epochs=local_epochs,
            learning_rate=learning_rate,
        )
        return numpy_client.to_client()

    def fit_metrics_aggregation_fn(metrics_list):
        # Flower expects fit_metrics_aggregation_fn(List[(num_examples, metrics)])
        # -> Dict. We log here and return a simple weighted-average loss.
        total_examples = sum(n for n, _ in metrics_list)
        total_comm, total_comp, weighted_loss = 0, 0.0, 0.0
        for n, m in metrics_list:
            hospital = m.get("hospital", m.get("client_id", "unknown"))
            weighted_loss += m["adapter_train_loss"] * n
            total_comm += m["comm_bytes"]
            total_comp += m["comp_time_sec"]
        return {
            "adapter_train_loss": weighted_loss / max(1, total_examples),
            "comm_bytes": total_comm,
            "comp_time_sec": total_comp,
        }

    initial_arrays = adapter_deltas_to_ndarrays(adapter_set)
    initial_parameters = fl.common.ndarrays_to_parameters(initial_arrays)

    iteration_logs: List[Dict] = []

    class _FusedStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            per_client_metrics = [(res.num_examples, res.metrics) for _, res in results]
            agg_metrics = fit_metrics_aggregation_fn(per_client_metrics)
            iteration_logs.append({"round": server_round, **agg_metrics})
            for _, res in results:
                hospital = res.metrics.get("hospital", res.metrics.get("client_id", "unknown"))
                logger.log_scalar(f"fu/adapter_train_loss/client_{hospital}",
                                   res.metrics["adapter_train_loss"], server_round)
            logger.log_scalar("fu/comm_cost_bytes/iteration", agg_metrics["comm_bytes"], server_round)
            logger.log_scalar("fu/comp_time_sec/iteration", agg_metrics["comp_time_sec"], server_round)
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None and aggregated[0] is not None:
                self.last_aggregated_parameters = aggregated[0]
            return aggregated

    strategy = _FusedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=len(remember_partitions),
        min_available_clients=len(remember_partitions),
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=lambda metrics_list: {},  # real handling is in aggregate_fit above
    )
    strategy.last_aggregated_parameters = None

    logger.info(f"Starting FUSED adapter-training federation: "
                f"{len(remember_partitions)} remember clients, {fused_iterations} iterations.")

    num_clients = len(remember_partitions)
    # See matching comment in src/fl/simulation.py: base.yaml sets
    # `gpus_per_client: null` explicitly, so the key is always PRESENT with
    # value None rather than absent — .get()'s default only triggers on
    # absence, so we check for None explicitly here.
    gpus_per_client = config.get("gpus_per_client")
    if gpus_per_client is None:
        gpus_per_client = (1.0 / num_clients) if device == "cuda" else 0.0

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=fused_iterations),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": gpus_per_client},
    )

    if strategy.last_aggregated_parameters is None:
        raise RuntimeError(
            "FUSED adapter federation completed but no aggregated parameters were "
            "captured. Check that remember clients have non-empty datasets."
        )

    final_arrays = fl.common.parameters_to_ndarrays(strategy.last_aggregated_parameters)
    ndarrays_to_adapter_deltas(adapter_set, final_arrays)

    # --- Step 5-6: merge adapters into a FRESH clone of the source model ---
    # (never the literal `source_model` object — see module docstring).
    # finalize_into() permanently bakes (base + delta*mask) into real,
    # checkpointable nn.Parameters — appropriate here since adapter training
    # is complete and no further backward pass through these deltas is needed.
    unlearned_model = clone_model(source_model, model_builder, device)
    adapter_set.finalize_into(unlearned_model)
    logger.info("FUSED unlearning complete: adapters merged into final unlearned model M^f.")

    return unlearned_model, adapter_set, critical_layers
