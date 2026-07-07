"""
FL algorithm strategies.

FedAvg, FedProx, and FedMOON use weighted-average aggregation on the server.
FedProx and FedMOON differ mainly in the client-side loss.

FedBN also uses FedAvg-style aggregation, but only over BN-excluded parameters.
The important extra fix here is that FedBN must remember each client's local
BatchNorm state and send it back to the same client in later rounds. This is
required because Ray/Flower simulation may recreate client objects between
rounds.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import EvaluateIns, FitIns, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


def _default_fit_metrics_aggregation_fn(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """
    Weighted-average aggregation of numeric fit metrics only.

    Non-numeric metrics such as FedBN's bn_state_b64 are intentionally ignored
    by Flower's internal History aggregation.
    """
    total_examples = sum(n for n, _ in metrics_list)

    if total_examples == 0:
        return {}

    numeric_keys = [
        k
        for k, v in metrics_list[0][1].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]

    aggregated: Dict[str, Scalar] = {}

    for key in numeric_keys:
        aggregated[key] = (
            sum(n * m.get(key, 0.0) for n, m in metrics_list) / total_examples
        )

    return aggregated


def _default_evaluate_metrics_aggregation_fn(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    return _default_fit_metrics_aggregation_fn(metrics_list)


class MetricsTrackingFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg subclass that logs per-client fit/evaluate metrics through callbacks
    and stores the last aggregated model parameters.
    """

    def __init__(
        self,
        on_fit_metrics: Optional[Callable] = None,
        on_evaluate_metrics: Optional[Callable] = None,
        **kwargs,
    ):
        kwargs.setdefault(
            "fit_metrics_aggregation_fn",
            _default_fit_metrics_aggregation_fn,
        )
        kwargs.setdefault(
            "evaluate_metrics_aggregation_fn",
            _default_evaluate_metrics_aggregation_fn,
        )

        super().__init__(**kwargs)

        self.on_fit_metrics = on_fit_metrics
        self.on_evaluate_metrics = on_evaluate_metrics
        self.last_aggregated_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        if self.on_fit_metrics is not None:
            per_client_metrics = [res.metrics for _, res in results]
            self.on_fit_metrics(server_round, per_client_metrics)

        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None and aggregated[0] is not None:
            self.last_aggregated_parameters = aggregated[0]

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        if self.on_evaluate_metrics is not None:
            per_client_metrics = [res.metrics for _, res in results]
            self.on_evaluate_metrics(server_round, per_client_metrics)

        return super().aggregate_evaluate(server_round, results, failures)


class FedBNTrackingFedAvg(MetricsTrackingFedAvg):
    """
    FedBN strategy.

    Aggregation is still FedAvg over BN-excluded parameters.

    The extra FedBN-specific responsibility is to persist each client's local
    BN state on the server and send it back to the same Flower client ID in
    later fit/evaluate calls. This prevents local BN statistics from being
    reset when Ray/Flower recreates client objects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn_state_by_cid: Dict[str, str] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)

        patched_instructions = []

        for client_proxy, fit_ins in instructions:
            config = dict(fit_ins.config)

            if client_proxy.cid in self.bn_state_by_cid:
                config["bn_state_b64"] = self.bn_state_by_cid[client_proxy.cid]

            patched_instructions.append(
                (
                    client_proxy,
                    FitIns(
                        parameters=fit_ins.parameters,
                        config=config,
                    ),
                )
            )

        return patched_instructions

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(
            server_round,
            parameters,
            client_manager,
        )

        patched_instructions = []

        for client_proxy, evaluate_ins in instructions:
            config = dict(evaluate_ins.config)

            if client_proxy.cid in self.bn_state_by_cid:
                config["bn_state_b64"] = self.bn_state_by_cid[client_proxy.cid]

            patched_instructions.append(
                (
                    client_proxy,
                    EvaluateIns(
                        parameters=evaluate_ins.parameters,
                        config=config,
                    ),
                )
            )

        return patched_instructions

    def aggregate_fit(self, server_round, results, failures):
        for client_proxy, fit_res in results:
            encoded_bn = fit_res.metrics.get("bn_state_b64")

            if encoded_bn is not None:
                self.bn_state_by_cid[client_proxy.cid] = encoded_bn

        return super().aggregate_fit(server_round, results, failures)


def build_fedavg_strategy(**kwargs) -> fl.server.strategy.Strategy:
    return MetricsTrackingFedAvg(**kwargs)


def build_fedprox_strategy(**kwargs) -> fl.server.strategy.Strategy:
    return MetricsTrackingFedAvg(**kwargs)


def build_fedbn_strategy(**kwargs) -> fl.server.strategy.Strategy:
    return FedBNTrackingFedAvg(**kwargs)


def build_fedmoon_strategy(**kwargs) -> fl.server.strategy.Strategy:
    return MetricsTrackingFedAvg(**kwargs)


STRATEGY_REGISTRY: Dict[str, Callable[..., fl.server.strategy.Strategy]] = {
    "fedavg": build_fedavg_strategy,
    "fedprox": build_fedprox_strategy,
    "fedbn": build_fedbn_strategy,
    "fedmoon": build_fedmoon_strategy,
}


def build_strategy(name: str, **kwargs) -> fl.server.strategy.Strategy:
    name = name.lower()

    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown FL algorithm '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    return STRATEGY_REGISTRY[name](**kwargs)
