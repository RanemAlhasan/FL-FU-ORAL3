"""
Class-imbalance-aware sampling, kept in its own module (separate from
src/data/dataset.py and every loader-building call site) specifically so
class balancing can be toggled on/off for an experiment without touching
any training code — flip `handle_class_imbalance` in the run's YAML config
(or `--set handle_class_imbalance=false`) and re-run; nothing else changes.

Used by:
  - src/fl/simulation.py (Phase 1 FL + Phase 3 retrain, both go through
    run_federated_learning) — already wired in before this module existed;
    simulation.py now delegates here instead of duplicating the logic.
  - scripts/run_fu_lora_domain.py / scripts/run_fu_cli_domain.py (Phase 2
    unlearning) — previously built plain, unweighted DataLoaders for every
    client's training data regardless of `handle_class_imbalance`, which
    made Phase 2 inconsistent with Phase 1/3 whenever a hospital's class
    distribution is skewed. Both now call build_loader() from here too.

Two independent things can be imbalanced in this dataset and this module
only ever addresses the first:
  - Per-hospital CLASS imbalance (e.g. a hospital with far fewer Malignant
    than Benign samples) — this is what a WeightedRandomSampler directly
    fixes: within one client's own DataLoader, draw each class with equal
    expected frequency regardless of how rare it is in that client's data.
  - Cross-hospital DATASET SIZE imbalance (e.g. Spain has far more samples
    than Canada) — this is a cross-client concern, not a per-loader one,
    and is already handled at aggregation time (FedAvg weighted by each
    client's dataset size — see src/fl/strategies.py's Flower FedAvg for
    Phase 1/3, and the `weights=`/`client_data_sizes` parameters threaded
    through src/fl/core_domain.py::fedavg_domain and
    src/fu/sparse_adapter_generic.py::average_adapter_deltas for Phase 2).
    A WeightedRandomSampler cannot fix this (it resamples within one
    client's own data; it cannot manufacture samples from a different
    client's dataset), so it is intentionally out of scope here.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def class_sample_weights(dataset: Dataset) -> List[float]:
    """Inverse-class-frequency weight per sample: weight(sample) = total /
    (num_classes_present * count_of_that_sample's_class). A
    WeightedRandomSampler built from these weights draws every class with
    equal expected frequency, independent of how skewed the underlying
    counts are.

    Works with any Dataset exposing `.samples` (a list of objects with a
    `.label_name` attribute, e.g. src/data/dataset.py::Sample) — i.e. any
    OralCancerDataset, including hospital/proxy/forget-remember subsets
    produced by src/data/partition.py, since they all wrap Sample objects
    the same way. Reuses OralCancerDataset.class_sample_weights() directly
    when available (single source of truth for the exact formula), falling
    back to the same computation for any other `.samples`-bearing dataset.
    """
    if hasattr(dataset, "class_sample_weights"):
        return dataset.class_sample_weights()

    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError(
            f"class_sample_weights() needs a dataset with a `.samples` list "
            f"(each with `.label_name`) or its own `.class_sample_weights()` "
            f"method; got {type(dataset).__name__}."
        )

    counts = Counter(s.label_name for s in samples)
    total = sum(counts.values())
    class_weight = {name: total / (len(counts) * count) for name, count in counts.items()}
    return [class_weight[s.label_name] for s in samples]


def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """WeightedRandomSampler over inverse class frequency, drawing
    len(dataset) samples per epoch with replacement (so every class is
    still seen roughly `len(dataset) / num_classes` times per epoch,
    rather than however many times it naturally occurs)."""
    weights = class_sample_weights(dataset)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    train: bool,
    handle_imbalance: bool,
    num_workers: int = 2,
) -> DataLoader:
    """Canonical train/eval DataLoader builder used across every phase
    (FL, retrain, FUSED unlearning). Class-balances via
    WeightedRandomSampler only when BOTH `train` and `handle_imbalance`
    are true and the dataset is non-empty — eval/test/val loaders must
    never be balanced (that would distort reported accuracy), and an
    empty dataset has no classes to weight.

    `handle_imbalance` should come straight from the run config's
    `handle_class_imbalance` key (see configs/base.yaml) so a single YAML
    flag (or `--set handle_class_imbalance=false`) toggles this for an
    entire run, letting you produce a clean with/without-sampler ablation
    pair without editing any code.
    """
    if train and handle_imbalance and len(dataset) > 0:
        sampler = build_weighted_sampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def build_tensor_pair_loader(
    tensor_pair_dataset: Dataset,
    oral_dataset: Dataset,
    batch_size: int,
    train: bool,
    handle_imbalance: bool,
    num_workers: int = 2,
) -> DataLoader:
    """Same contract as build_loader(), for the FUSED-phase scripts'
    "(image, label) tuple" loaders: `tensor_pair_dataset` (a
    TensorPairDataset wrapping `oral_dataset`) is what's actually iterated,
    but sample weights must be computed from `oral_dataset` (the
    TensorPairDataset bridge has no `.samples`/`.class_sample_weights()` of
    its own). The sampler indexes into `tensor_pair_dataset` by position,
    which is valid since TensorPairDataset preserves `oral_dataset`'s
    sample order 1:1.
    """
    if train and handle_imbalance and len(oral_dataset) > 0:
        weights = class_sample_weights(oral_dataset)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(tensor_pair_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return DataLoader(tensor_pair_dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
