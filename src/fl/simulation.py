"""
Builds and runs a Flower simulation for the FL training phase ONLY.

This module is invoked exclusively by scripts/run_fl.py. It has no knowledge
of FUSED, retraining, or any other phase — its only job is to produce a
trained global model checkpoint plus FL-phase metrics (fl/... and
eval/... tags) for one algorithm/backbone/config combination.
"""
from __future__ import annotations

from typing import Dict, List

import flwr as fl
from flwr.common import Context
from flwr.common.constant import PARTITION_ID_KEY

from src.data.partition import ClientPartition
from src.data.sampler import build_loader as _build_loader
from src.fl.client import (OralCancerFlowerClient, decode_state_dict_b64,
                            ndarrays_to_state_dict, state_dict_to_ndarrays)
from src.fl.strategies import build_strategy
from src.models.backbone import build_model
from src.models.fedbn import extract_federated_state_dict, merge_local_bn_into_global
from src.utils.logger import ExperimentLogger


def run_federated_learning(
    config: Dict,
    client_partitions: List[ClientPartition],
    train_datasets,
    val_datasets,
    device: str,
    logger: ExperimentLogger,
):
    """Run the FL phase to completion. Returns
    (trained_model, per_hospital_models):
      - trained_model: a fresh nn.Module with the final round's aggregated
        parameters loaded in (the single "global" checkpoint to save).
      - per_hospital_models: dict client_id -> nn.Module, each with that
        client's own final local state (including its own BN stats under
        FedBN) — useful for per-hospital evaluation/deployment.

    This function does NOT perform any unlearning or retraining — it is the
    entire content of Phase 1.
    """

    algorithm_name = config["algorithm"].lower()
    model_name = config["model"]
    num_classes = config["num_classes"]
    global_rounds = config["global_epochs"]
    local_epochs = config["local_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    # Derived from algorithm_name, NOT read from a separate config key: FedBN
    # is the only algorithm that keeps BatchNorm local, so this must always
    # agree with which strategy build_strategy() below actually picks.
    # Previously this read config.get("domain_adaptation", False) as an
    # independent yaml key — if a run overrode --set algorithm=FedBN without
    # also flipping domain_adaptation (e.g. starting from fl_fedavg.yaml,
    # which ships domain_adaptation: false), the server-side strategy would
    # correctly become FedBNTrackingFedAvg, but every client's
    # self._domain_adaptation would stay False, so BatchNorm params/buffers
    # would be federated normally and no bn_state_b64 would ever be sent —
    # a silent downgrade to plain FedAvg for a run labeled "fedbn"
    # everywhere in its logs/checkpoints. core_domain.py already derives
    # this the same way; this brings the Flower path in line with it.
    domain_adaptation = algorithm_name == "fedbn"
    handle_imbalance = config.get("handle_class_imbalance", True)
    pretrained = config.get("pretrained", True)

    algorithm_config = {
        "name": algorithm_name,
        "mu": config.get("fedprox_mu", 0.01),
        "moon_mu": config.get("fedmoon_mu", 1.0),
        "moon_temperature": config.get("fedmoon_temperature", 0.5),
        "domain_adaptation": domain_adaptation,
    }

    logger.log_hparams({
        "algorithm": algorithm_name, "model": model_name, "num_clients": len(client_partitions),
        "global_epochs": global_rounds, "local_epochs": local_epochs,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "domain_adaptation": domain_adaptation,
    })

    # Reference model: built once, used only to (a) seed initial federated
    # parameters and (b) act as the structural template the final
    # checkpoint is loaded into. Each Flower client builds and owns its OWN
    # separate model instance inside client_fn — nothing here is shared
    # with / mutated by client-side training.
    reference_model = build_model(model_name, num_classes, pretrained=pretrained).to(device)

    # Captures each client's final-round LOCAL BN state, decoded from the
    # "bn_state_b64" fit metric (see src/fl/client.py::encode_state_dict_b64
    # for why this travels through metrics rather than a shared Python
    # object: Ray runs client_fn in separate worker processes, so ordinary
    # closure mutation from inside client_fn is never visible back here).
    final_round_bn_states: Dict[str, Dict[str, "object"]] = {}

    def client_fn(context: Context) -> fl.client.Client:
        # Modern Flower (>=1.13) passes a Context instead of a raw cid string
        # to client_fn; the partition index is exposed via the stable
        # PARTITION_ID_KEY ("partition-id") node_config entry, which is how
        # Flower's own legacy-compatibility shim derives "cid" internally.
        idx = int(context.node_config[PARTITION_ID_KEY])
        partition = client_partitions[idx]
        client_model = build_model(model_name, num_classes, pretrained=pretrained).to(device)
        train_loader = _build_loader(train_datasets[idx], batch_size, train=True,
                                      handle_imbalance=handle_imbalance)
        val_loader = _build_loader(val_datasets[idx], batch_size, train=False,
                                    handle_imbalance=False)
        numpy_client = OralCancerFlowerClient(
            client_id=partition.client_id,
            hospital=partition.hospital,
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            algorithm_config=algorithm_config,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
        )
        return numpy_client.to_client()

    def on_fit_metrics(server_round: int, per_client_metrics: List[Dict]):
        round_total_comm, round_total_comp = 0, 0.0
        for m in per_client_metrics:
            hospital = m.get("hospital", m.get("client_id", "unknown"))
            logger.log_scalar(f"fl/train_loss/client_{hospital}", m["train_loss"], server_round)
            logger.log_scalar(f"fl/comp_time_sec/client_{hospital}", m["comp_time_sec"], server_round)
            round_total_comm += m["comm_bytes"]
            round_total_comp += m["comp_time_sec"]
            if "bn_state_b64" in m:
                # Overwrite on every round so we end up with whichever round
                # ran last (== final_round_bn_states reflects the FINAL
                # trained local BN buffers once the simulation completes).
                client_id = m.get("client_id", hospital)
                final_round_bn_states[client_id] = decode_state_dict_b64(m["bn_state_b64"])
        logger.log_scalar("fl/comm_cost_bytes/round", round_total_comm, server_round)
        logger.log_scalar("fl/comp_time_sec/round", round_total_comp, server_round)
        logger.info(f"[FL round {server_round}] comm={round_total_comm}B comp={round_total_comp:.2f}s")

    def on_evaluate_metrics(server_round: int, per_client_metrics: List[Dict]):
        for m in per_client_metrics:
            hospital = m.get("hospital", m.get("client_id", "unknown"))
            logger.log_scalar(f"eval/per_hospital/{hospital}/acc", m["accuracy"], server_round)

    initial_fed_state = extract_federated_state_dict(reference_model, domain_adaptation)
    initial_parameters = fl.common.ndarrays_to_parameters(state_dict_to_ndarrays(initial_fed_state))
    federated_keys = list(initial_fed_state.keys())

    strategy = build_strategy(
        algorithm_name,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(client_partitions),
        min_evaluate_clients=len(client_partitions),
        min_available_clients=len(client_partitions),
        initial_parameters=initial_parameters,
        on_fit_metrics=on_fit_metrics,
        on_evaluate_metrics=on_evaluate_metrics,
    )

    logger.info(f"Starting Flower simulation: algorithm={algorithm_name}, "
                f"clients={len(client_partitions)}, rounds={global_rounds}")

    num_clients = len(client_partitions)
    # config.get(key, default) only falls back when the key is ABSENT, but
    # base.yaml explicitly sets `gpus_per_client: null` (so users can see
    # the knob exists), meaning the key is always PRESENT with value None.
    # We therefore check for None explicitly to get "auto" behavior.
    gpus_per_client = config.get("gpus_per_client")
    if gpus_per_client is None:
        gpus_per_client = (1.0 / num_clients) if device == "cuda" else 0.0

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=global_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": gpus_per_client},
    )

    # Retrieve the final round's aggregated (BN-excluded, if domain_adaptation)
    # parameters from the strategy and load them into a FRESH model instance.
    # We deliberately do not reuse `reference_model` as "the" trained model
    # object beyond this loading step, to keep the "build fresh, load
    # weights" pattern consistent with how FU/retrain scripts later load
    # this checkpoint (see src/utils/checkpoint.py).
    if strategy.last_aggregated_parameters is None:
        raise RuntimeError(
            "FL simulation completed but no aggregated parameters were captured. "
            "This usually means all clients failed during fit — check train.log."
        )
    final_arrays = fl.common.parameters_to_ndarrays(strategy.last_aggregated_parameters)
    final_fed_state = ndarrays_to_state_dict(federated_keys, final_arrays)

    trained_model = build_model(model_name, num_classes, pretrained=False).to(device)
    if domain_adaptation:
        # Federated (non-BN) params come from server aggregation; BN params
        # are, by FedBN design, never aggregated. We complete the exportable
        # checkpoint with one representative client's locally-trained BN
        # buffers (received via the fit-metrics channel — see
        # final_round_bn_states above). NOTE: this representative choice
        # only affects this single exported checkpoint's BN values;
        # per-hospital evaluation during FL already used each client's own
        # correct BN stats (see eval/per_hospital/* above), so domain-shift
        # measurement is unaffected. For per-hospital deployment, use
        # `per_hospital_models` (returned below) instead of this checkpoint.
        if not final_round_bn_states:
            raise RuntimeError(
                "domain_adaptation=True but no client reported BN state via "
                "fit metrics — check that OralCancerFlowerClient.fit() is "
                "attaching 'bn_state_b64' (src/fl/client.py)."
            )
        first_client_id = client_partitions[0].client_id
        if first_client_id not in final_round_bn_states:
            # Fall back to whichever client DID report, deterministically
            # (sorted by client_id) — guards against a client transiently
            # failing on the very last round.
            first_client_id = sorted(final_round_bn_states.keys())[0]
        full_client_bn_state = final_round_bn_states[first_client_id]
        merged_state = dict(final_fed_state)
        for key, value in full_client_bn_state.items():
            merged_state[key] = value
        trained_model.load_state_dict(merged_state, strict=True)
        logger.info(
            f"FedBN checkpoint assembled: non-BN params from server aggregation, "
            f"BN params taken from client '{first_client_id}' (representative)."
        )
    else:
        trained_model.load_state_dict(final_fed_state, strict=True)
        logger.info("Checkpoint assembled from fully-federated parameters (no domain adaptation).")

    # Expose each client's final full model (BN included) so callers (e.g.
    # run_fl.py) can optionally persist per-hospital checkpoints — useful
    # under FedBN where the "correct" evaluation/deployment model for a
    # given hospital uses THAT hospital's own BN statistics, not the
    # single representative checkpoint above.
    per_hospital_models: Dict[str, "object"] = {}
    for client_id, bn_state in final_round_bn_states.items():
        hospital_model = build_model(model_name, num_classes, pretrained=False).to(device)
        merged = dict(final_fed_state)
        merged.update(bn_state)
        hospital_model.load_state_dict(merged, strict=True)
        per_hospital_models[client_id] = hospital_model

    return trained_model, per_hospital_models
