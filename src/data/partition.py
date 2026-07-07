"""
Federated client partitioning.

Two modes, selected via config `client_split`:

  "hospital_based" (default): one Flower client per hospital. Client count
      == number of hospitals (3 here: Spain/Canada/India).

  "simulated": each hospital is further split into `clients_per_hospital`
      smaller clients (e.g. to stress-test FL with more clients than
      hospitals), while every sub-client still carries its parent hospital's
      domain label. This keeps FedBN's domain-specific BatchNorm and the
      per-hospital evaluation breakdown meaningful even when "client" and
      "hospital" are no longer 1:1.

The output of either mode is a list of `ClientPartition` objects, each with
a stable `client_id` and `hospital` field. Downstream FL/FU code should
always group/report by `hospital`, not by `client_id`, when it cares about
domain-level results — multiple client_ids can map to the same hospital
under "simulated" mode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from src.data.dataset import OralCancerDataset, Sample


@dataclass
class ClientPartition:
    client_id: str
    hospital: str
    samples: List[Sample] = field(default_factory=list)


def partition_hospital_based(samples: List[Sample], hospitals: List[str]) -> List[ClientPartition]:
    partitions = []
    for hospital in hospitals:
        hospital_samples = [s for s in samples if s.hospital == hospital]
        partitions.append(ClientPartition(client_id=hospital, hospital=hospital,
                                           samples=hospital_samples))
    return partitions


def partition_simulated(
    samples: List[Sample],
    hospitals: List[str],
    clients_per_hospital: int,
    seed: int = 42,
) -> List[ClientPartition]:
    """Split each hospital's samples into `clients_per_hospital` roughly equal,
    randomly-shuffled shards. Each shard becomes its own Flower client but
    keeps the parent hospital as its domain label, so FedBN / per-hospital
    metrics still aggregate correctly across the sub-clients of one hospital.
    """
    rng = random.Random(seed)
    partitions: List[ClientPartition] = []

    for hospital in hospitals:
        hospital_samples = [s for s in samples if s.hospital == hospital]
        rng.shuffle(hospital_samples)

        n = len(hospital_samples)
        k = max(1, clients_per_hospital)
        shard_size = n // k
        remainder = n % k

        start = 0
        for shard_idx in range(k):
            size = shard_size + (1 if shard_idx < remainder else 0)
            shard = hospital_samples[start:start + size]
            start += size
            client_id = f"{hospital}__shard{shard_idx}"
            partitions.append(ClientPartition(client_id=client_id, hospital=hospital,
                                               samples=shard))
    return partitions


def build_client_partitions(
    samples: List[Sample],
    hospitals: List[str],
    client_split: str = "hospital_based",
    clients_per_hospital: int = 1,
    seed: int = 42,
) -> List[ClientPartition]:
    if client_split == "hospital_based":
        return partition_hospital_based(samples, hospitals)
    elif client_split == "simulated":
        return partition_simulated(samples, hospitals, clients_per_hospital, seed)
    else:
        raise ValueError(
            f"Unknown client_split mode '{client_split}'. "
            f"Expected 'hospital_based' or 'simulated'."
        )


def carve_proxy_partitions(
    partitions: List[ClientPartition],
    proxy_frac: float,
    seed: int = 42,
) -> Tuple[List[ClientPartition], List[ClientPartition]]:
    """Split each partition's samples into a "main" partition
    ((1 - proxy_frac) fraction) and a "proxy" partition (proxy_frac
    fraction), via a seeded per-partition shuffle. Returns (main_partitions,
    proxy_partitions), same order/client_id/hospital as the input.

    The main partitions are what the real FL/FU run trains and evaluates
    on; the proxy partitions are held out, structurally identical (same
    hospital/client layout), and used ONLY for MIA shadow-model training —
    never touched by the real run, so the shadow models' "membership"
    ground truth stays uncontaminated by data the real model has actually
    seen. Same purpose as src/data/proxy_split.py's CIFAR-10 proxy carving,
    adapted here to work on ClientPartition/Sample objects directly instead
    of raw tensors, since the oral-cancer pipeline doesn't pool images into
    numpy arrays the way the CIFAR-10 reproduction does.
    """
    rng = random.Random(seed)
    main_partitions: List[ClientPartition] = []
    proxy_partitions: List[ClientPartition] = []

    for partition in partitions:
        samples = list(partition.samples)
        rng.shuffle(samples)
        n_proxy = int(len(samples) * proxy_frac)
        proxy_samples = samples[:n_proxy]
        main_samples = samples[n_proxy:]
        main_partitions.append(ClientPartition(
            client_id=partition.client_id, hospital=partition.hospital, samples=main_samples,
        ))
        proxy_partitions.append(ClientPartition(
            client_id=partition.client_id, hospital=partition.hospital, samples=proxy_samples,
        ))

    return main_partitions, proxy_partitions


def partitions_to_datasets(
    partitions: List[ClientPartition],
    transform,
    load_metadata: bool,
    hospital_to_idx: dict,
) -> List[OralCancerDataset]:
    """Wrap each ClientPartition's raw samples in an OralCancerDataset, sharing
    one global hospital_to_idx mapping across all clients so domain label
    indices are consistent FL-wide."""
    return [
        OralCancerDataset(
            samples=p.samples,
            transform=transform,
            load_metadata=load_metadata,
            hospital_to_idx=hospital_to_idx,
        )
        for p in partitions
    ]
