"""
Unlearning scope definitions. The brief asks for client/hospital unlearning
first, with class and sample unlearning as future extensions sharing the
same CLI + sparse adapter machinery. This module is the seam: CLI
(critical_layers.py) and adapter training (fused_runner.py) only ever see
"forget clients" (which clients/partitions to EXCLUDE from adapter
training) and "remember clients" (everyone else) — they don't know or care
whether that exclusion happened because of client-, class-, or
sample-level forgetting. Adding a new scope means adding one function here
that returns the right ClientForgetRememberSplit; nothing in fu/ changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.data.dataset import OralCancerDataset
from src.data.partition import ClientPartition


@dataclass
class ForgetRememberSplit:
    """The output every scope-builder produces. `forget_partitions` are
    excluded entirely from adapter training (Algorithm 1: "the server
    distributes both the unlearning adapters and the original model to the
    clients that contain the remembered dataset... excluding clients that
    only contain the forgetting dataset"). `remember_partitions` participate
    normally. For class/sample scope, a single client may need its DATA
    filtered (not the whole client excluded) — that's handled by
    `class_filter`/passing already-filtered ClientPartitions in, see below.
    """
    forget_partitions: List[ClientPartition]
    remember_partitions: List[ClientPartition]
    scope: str                  # "client" | "class" | "sample"
    description: str


def build_client_unlearning_split(
    all_partitions: List[ClientPartition],
    forget_client_hospital: str,
) -> ForgetRememberSplit:
    """Client/hospital unlearning: forget_client = the named hospital (and
    ALL of its simulated sub-shards, if client_split=="simulated" — a
    hospital being forgotten means every shard of that hospital is forgotten,
    since they're the same domain/institution)."""
    forget = [p for p in all_partitions if p.hospital == forget_client_hospital]
    remember = [p for p in all_partitions if p.hospital != forget_client_hospital]

    if not forget:
        available = sorted({p.hospital for p in all_partitions})
        raise ValueError(
            f"forget_client '{forget_client_hospital}' does not match any "
            f"partition's hospital. Available hospitals: {available}"
        )
    if not remember:
        raise ValueError(
            "Client unlearning would leave zero remember clients — cannot "
            "train adapters with no remaining data. Check your client_split config."
        )

    return ForgetRememberSplit(
        forget_partitions=forget,
        remember_partitions=remember,
        scope="client",
        description=f"Forgetting hospital/client '{forget_client_hospital}' "
                     f"({len(forget)} partition(s)); remembering "
                     f"{len(remember)} partition(s).",
    )


def build_class_unlearning_split(
    all_partitions: List[ClientPartition],
    forget_class_indices: List[int],
) -> ForgetRememberSplit:
    """Class unlearning (stub for future extension, Eq. 8-10 in the paper):
    every client keeps participating, but each client's local dataset is
    filtered to REMOVE samples of the forgotten class(es) before adapter
    training. We represent this by returning new ClientPartition objects
    whose `.samples` already exclude the forgotten classes — so from
    fused_runner.py's perspective these are just "remember partitions" with
    smaller sample lists; no separate forget-client concept is needed since
    every client retains non-forgotten-class data.

    NOTE: this is provided as a working stub per the brief's extensibility
    requirement; it has not been validated against the paper's class-unlearning
    experimental protocol (Table 1 "Class Unlearning" columns) and should be
    checked against your own class-imbalance handling before using results.
    """
    remember_partitions = []
    for p in all_partitions:
        filtered_samples = [s for s in p.samples if s.label not in forget_class_indices]
        remember_partitions.append(
            ClientPartition(client_id=p.client_id, hospital=p.hospital, samples=filtered_samples)
        )
    # "Forget" here is represented as empty-client placeholders carrying just
    # the removed samples, useful for FA (forget accuracy) evaluation later.
    forget_partitions = []
    for p in all_partitions:
        forgotten_samples = [s for s in p.samples if s.label in forget_class_indices]
        if forgotten_samples:
            forget_partitions.append(
                ClientPartition(client_id=f"{p.client_id}__forgotten_classes",
                                 hospital=p.hospital, samples=forgotten_samples)
            )

    return ForgetRememberSplit(
        forget_partitions=forget_partitions,
        remember_partitions=remember_partitions,
        scope="class",
        description=f"Forgetting class indices {forget_class_indices} across "
                     f"all {len(all_partitions)} clients.",
    )


def build_sample_unlearning_split(
    all_partitions: List[ClientPartition],
    forget_client_hospital: str,
    forget_fraction: float = 0.1,
    seed: int = 42,
) -> ForgetRememberSplit:
    """Sample unlearning (stub for future extension): forget a random
    fraction of ONE client's samples, keep the rest of that client (plus all
    other clients) as remember data. Mirrors the paper's framing ("Sample
    unlearning means forgetting a portion of data within a client. It is
    similar to client unlearning.").

    NOTE: provided as a working stub per the brief's extensibility
    requirement; not yet validated against the paper's sample-unlearning
    protocol (Table 1 "Sample Unlearning" columns, which evaluate 0A/PS
    rather than RA/FA — see src/eval/metrics.py for those definitions).
    """
    import random
    rng = random.Random(seed)

    remember_partitions, forget_partitions = [], []
    for p in all_partitions:
        if p.hospital != forget_client_hospital:
            remember_partitions.append(p)
            continue
        shuffled = list(p.samples)
        rng.shuffle(shuffled)
        n_forget = int(len(shuffled) * forget_fraction)
        forget_samples = shuffled[:n_forget]
        remember_samples = shuffled[n_forget:]
        remember_partitions.append(
            ClientPartition(client_id=p.client_id, hospital=p.hospital, samples=remember_samples)
        )
        if forget_samples:
            forget_partitions.append(
                ClientPartition(client_id=f"{p.client_id}__forgotten_samples",
                                 hospital=p.hospital, samples=forget_samples)
            )

    return ForgetRememberSplit(
        forget_partitions=forget_partitions,
        remember_partitions=remember_partitions,
        scope="sample",
        description=f"Forgetting {forget_fraction:.0%} of samples from "
                     f"'{forget_client_hospital}'.",
    )


def build_forget_remember_split(scope: str, all_partitions: List[ClientPartition],
                                 config: dict) -> ForgetRememberSplit:
    """Single dispatch point used by scripts/run_fu.py. Add new scopes here."""
    if scope == "client":
        return build_client_unlearning_split(all_partitions, config["forget_client"])
    elif scope == "class":
        return build_class_unlearning_split(all_partitions, config["forget_class_indices"])
    elif scope == "sample":
        return build_sample_unlearning_split(
            all_partitions, config["forget_client"],
            config.get("forget_fraction", 0.1), config.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown unlearning scope '{scope}'. Expected client/class/sample.")
