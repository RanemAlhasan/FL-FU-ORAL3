"""
FL algorithm strategies, registered in STRATEGY_REGISTRY so that adding a
5th algorithm is a matter of writing one factory function here and pointing
a config's `algorithm:` field at it.

Implementation note: FedProx and FedMOON's actual *novelty* (proximal term,
contrastive term) is implemented client-side in src/fl/client.py, because
those terms are local-loss modifications, not aggregation-rule changes. On
the server side, FedProx and FedMOON both use plain weighted-average
aggregation (Flower's built-in FedAvg strategy) — this matches how both
methods are specified in their original papers (the server aggregation rule
is unchanged; only the client objective changes).

FedBN's novelty IS an aggregation-side behavior in the sense that BatchNorm
parameters must never be aggregated — but we implement that by simply never
sending BN parameters to the server in the first place (see
src/models/fedbn.py + src/fl/client.py), so the server-side strategy for
FedBN is also plain FedAvg operating over a reduced (BN-free) parameter set.
This keeps the server-side strategy code uniform across all four algorithms
and pushes all algorithm-specific behavior to well-documented, swappable
client-side hooks — the cleanest extension point for new algorithms.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


def _default_fit_metrics_aggregation_fn(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Weighted-average aggregation of the (JSON/Scalar-safe) numeric fit
    metrics, satisfying Flower's expected `fit_metrics_aggregation_fn`
    contract so it stops warning that none was provided. This only affects
    what's stored in Flower's own internal `History` object — our pipeline's
    actual per-round logging happens via the `on_fit_metrics` callback above,
    which has already run by the time this is called."""
    total_examples = sum(n for n, _ in metrics_list)
    if total_examples == 0:
        return {}
    numeric_keys = [
        k for k, v in metrics_list[0][1].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    aggregated: Dict[str, Scalar] = {}
    for key in numeric_keys:
        aggregated[key] = sum(n * m.get(key, 0.0) for n, m in metrics_list) / total_examples
    return aggregated


def _default_evaluate_metrics_aggregation_fn(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Same as above, for evaluate() metrics (e.g. per-client accuracy)."""
    return _default_fit_metrics_aggregation_fn(metrics_list)


class MetricsTrackingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg subclass that also collects per-client fit/eval metrics into a
    callback, so run_fl.py can log per-hospital training loss, comm cost,
    and comp time every round under properly namespaced TensorBoard tags.

    It also remembers the most recently aggregated parameters
    (`self.last_aggregated_parameters`), since Flower does not expose a
    public, version-stable way to retrieve "the final global model" after
    `start_simulation` returns. run_fl.py reads this attribute once the
    simulation completes to build the saved checkpoint."""

    def __init__(self, on_fit_metrics: Optional[Callable] = None,
                 on_evaluate_metrics: Optional[Callable] = None, **kwargs):
        kwargs.setdefault("fit_metrics_aggregation_fn", _default_fit_metrics_aggregation_fn)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", _default_evaluate_metrics_aggregation_fn)
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


def build_fedavg_strategy(**kwargs) -> fl.server.strategy.Strategy:
    return MetricsTrackingFedAvg(**kwargs)


def build_fedprox_strategy(**kwargs) -> fl.server.strategy.Strategy:
    # Server-side aggregation is identical to FedAvg; the proximal term lives
    # in the client's local loss (src/fl/client.py::_fedprox_term).
    return MetricsTrackingFedAvg(**kwargs)


def build_fedbn_strategy(**kwargs) -> fl.server.strategy.Strategy:
    # Server-side aggregation is identical to FedAvg, but operating only over
    # the BN-excluded parameter set the clients send (src/models/fedbn.py).
    return MetricsTrackingFedAvg(**kwargs)


def build_fedmoon_strategy(**kwargs) -> fl.server.strategy.Strategy:
    # Server-side aggregation is identical to FedAvg; the contrastive term
    # lives in the client's local loss (src/fl/client.py::_fedmoon_term).
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
