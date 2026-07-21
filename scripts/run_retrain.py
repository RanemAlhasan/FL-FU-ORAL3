#!/usr/bin/env python3
"""
Phase 3: Retrain-from-scratch baseline — the FU "upper bound" (per the
FUSED paper: "Retraining is the upper bound of unlearning; it can achieve
the effect of the model having never encountered the forgotten data.").

Usage:
    python scripts/run_retrain.py --config configs/retrain_baseline.yaml

    # Override any config value from the command line with --set key=value
    # (repeatable). --forget_client remains available as a shorthand for
    # the common case and is applied AFTER --set, so it always wins if both
    # are given:
    python scripts/run_retrain.py --config configs/retrain_baseline.yaml \
        --set algorithm=FedProx --set global_epochs=20 --forget_client Canada_Dataset

This is a COMPLETE, INDEPENDENT FL run from a freshly-initialized model —
it does not load any other run's checkpoint. The only difference from a
normal Phase-1 FL run is that the forget client/class/sample's data is
EXCLUDED from training from round 0, by construction, rather than learned
and then unlearned. Its checkpoint and metrics are saved exactly like a
Phase-1 run (under eval/*) but ALSO get eval/unlearning/* metrics (RA/FA/
ReA/MIA) for direct comparison against a FUSED run via
scripts/compare_runs.py.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import index_dataset, OralCancerDataset
from src.data.partition import build_client_partitions, partitions_to_datasets
from src.data.transforms import build_transforms

from src.eval.evaluator import (
    run_standard_evaluation,
    run_unlearning_evaluation,
)
from src.eval.fedbn_retrain import (
    build_hospital_model_map,
    run_fedbn_retrain_evaluation,
)
from src.models.fedbn import build_retained_bn_reference_model

from src.fl.simulation import run_federated_learning
from src.fu.scope import build_forget_remember_split
from src.utils.checkpoint import save_checkpoint

from src.utils.config import (
    apply_overrides,
    load_config,
    load_source_run_config,
    make_run_id,
    parse_set_overrides,
    resolve_run_dirs,
    save_config_snapshot,
)

from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: retrain-from-scratch baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_retrain.py --config configs/retrain_baseline.yaml
  python scripts/run_retrain.py --config configs/retrain_baseline.yaml \
      --set algorithm=FedProx --set global_epochs=20 --forget_client Canada_Dataset
""",
    )
    parser.add_argument("--config", required=True, help="Path to retrain config YAML.")
    parser.add_argument("--forget_client", default=None,
                         help="Shorthand override for 'forget_client'. Applied after --set, "
                              "so it wins if both are given.")
    parser.add_argument("--run_id", default=None, help="Override the auto-generated run_id.")
    parser.add_argument(
        "--set", dest="set_overrides", action="append", default=None, metavar="KEY=VALUE",
        help="Override a config value, e.g. --set global_epochs=20. Repeatable.",
    )
    
    # New Addition
    parser.add_argument(
    "--source_run",
    default=None,
    help=(
        "Optional Phase-1 run ID. Only its training settings are copied. "
        "Its model checkpoint is never loaded."
    ),
    )

    parser.add_argument(
        "--source_logs_root",
        default=None,
        help=(
            "Folder containing Phase-1 run logs. "
            "Defaults to logs/fl."
        ),
    )
    
    return parser.parse_args()

# New Addition

SOURCE_TRAINING_KEYS = (
    "algorithm",
    "model",
    "num_classes",
    "pretrained",
    "dataset_path",
    "hospitals",
    "client_split",
    "clients_per_hospital",
    "load_metadata",
    "image_size",
    "augmentation",
    "seed",
    "global_epochs",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "handle_class_imbalance",
    "imbalance_method",
    "class_balance_beta",
    "class_weight_clip",
    "fedprox_mu",
    "fedmoon_mu",
    "fedmoon_temperature",
)

def apply_source_training_settings(
    retrain_config,
    source_config,
):
    """
    Copy only training settings from the selected Phase-1 run.

    Retrain output folders, unlearning settings, and forget-client
    settings remain controlled by the retrain config.
    """
    merged = dict(retrain_config)

    for key in SOURCE_TRAINING_KEYS:
        if key in source_config:
            merged[key] = source_config[key]

    return merged

def main():
    args = parse_args()
    
    config = load_config(args.config)

    source_run = (
        args.source_run
        or config.get("source_run")
    )

    source_logs_root = (
        args.source_logs_root
        or config.get("source_logs_root", "logs/fl")
    )

    if source_run:
        source_config = load_source_run_config(
            source_run_id=source_run,
            logs_root=source_logs_root,
        )

        config = apply_source_training_settings(
            retrain_config=config,
            source_config=source_config,
        )

        config["source_run"] = source_run
        config["source_logs_root"] = source_logs_root

        print(
            f"Copied training settings from source run: "
            f"{source_run}"
        )

    overrides = parse_set_overrides(args.set_overrides)
    config = apply_overrides(config, overrides)

    if overrides:
        print(f"Applied --set overrides: {overrides}")

    if args.forget_client:
        config["forget_client"] = args.forget_client
    
    dataset_tag = "oralcancer"
    algorithm_tag = f"retrain-{config['unlearning_scope']}"
    run_id = args.run_id or make_run_id(
        f"{algorithm_tag}_{config['model'].lower()}_{dataset_tag}"
    )
    dirs = resolve_run_dirs(run_id, config["logs_root"], config["checkpoints_root"], config["outputs_root"])
    save_config_snapshot(config, os.path.join(dirs["log_dir"], "config.snapshot.yaml"))

    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])
    logger.info(f"Config loaded from {args.config}")
    
    if source_run:
        logger.info(
            f"Matched Phase-1 source run: {source_run}"
        )
        logger.info(
            "The source checkpoint is NOT loaded. "
            "Retraining starts from a fresh model."
        )

    logger.info(
        f"Retrain experiment: "
        f"algorithm={config['algorithm']}, "
        f"imbalance_method={config.get('imbalance_method')}, "
        f"handle_class_imbalance="
        f"{config.get('handle_class_imbalance')}, "
        f"forget_client={config.get('forget_client')}"
    )
    
    logger.info(f"Retrain baseline: this is an INDEPENDENT FL run with the forget "
                f"client/scope excluded from round 0. No source checkpoint is loaded.")

    device = config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    torch.manual_seed(config["seed"])

    # --- Data ----------------------------------------------------------
    train_samples = index_dataset(config["dataset_path"], "Train", config["hospitals"])
    test_samples = index_dataset(config["dataset_path"], "Test", config["hospitals"])
    global_hospital_to_idx = {h: i for i, h in enumerate(config["hospitals"])}

    train_transform = build_transforms(config["image_size"], train=True, augmentation=config["augmentation"])
    eval_transform = build_transforms(config["image_size"], train=False)

    all_train_partitions = build_client_partitions(
        train_samples, config["hospitals"], config["client_split"],
        config.get("clients_per_hospital", 1), config["seed"],
    )
    train_split = build_forget_remember_split(config["unlearning_scope"], all_train_partitions, config)
    logger.info(f"Unlearning scope: {train_split.scope}. {train_split.description}")
    logger.info("Retraining FL model using ONLY remember partitions (forget data excluded from round 0).")

    remember_train_datasets = partitions_to_datasets(
        train_split.remember_partitions, train_transform, config["load_metadata"], global_hospital_to_idx,
    )

    all_test_partitions = build_client_partitions(
        test_samples, config["hospitals"], config["client_split"],
        config.get("clients_per_hospital", 1), config["seed"],
    )
    test_split = build_forget_remember_split(config["unlearning_scope"], all_test_partitions, config)
    remember_test_datasets = partitions_to_datasets(
        test_split.remember_partitions, eval_transform, config["load_metadata"], global_hospital_to_idx,
    )
    forget_test_datasets = partitions_to_datasets(
        test_split.forget_partitions, eval_transform, config["load_metadata"], global_hospital_to_idx,
    )

    def concat_loader(datasets, batch_size, shuffle=False):
        if not datasets:
            return DataLoader(OralCancerDataset([], transform=eval_transform,
                                                 hospital_to_idx=global_hospital_to_idx),
                               batch_size=batch_size)
        return DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=shuffle, num_workers=2)

    remember_test_loader = concat_loader(remember_test_datasets, config["batch_size"])
    forget_test_loader = concat_loader(forget_test_datasets, config["batch_size"])

    global_test_dataset = OralCancerDataset(
        test_samples, transform=eval_transform, load_metadata=config["load_metadata"],
        hospital_to_idx=global_hospital_to_idx,
    )
    global_test_loader = DataLoader(global_test_dataset, batch_size=config["batch_size"],
                                     shuffle=False, num_workers=2)

    # Validation loaders for per-round eval/per_hospital metrics during the
    # retrain FL run — built only over remember partitions, matching the
    # remember-only training data.
    val_split_for_remember = build_client_partitions(
        test_samples, config["hospitals"], config["client_split"],
        config.get("clients_per_hospital", 1), config["seed"],
    )
    val_split_for_remember = build_forget_remember_split(
        config["unlearning_scope"], val_split_for_remember, config,
    ).remember_partitions
    val_datasets = partitions_to_datasets(
        val_split_for_remember, eval_transform, config["load_metadata"], global_hospital_to_idx,
    )

    # --- Retrain FL from scratch on remember-only data (Phase 3's actual work)
    retrained_model, per_hospital_models = run_federated_learning(
        config=config,
        client_partitions=train_split.remember_partitions,
        train_datasets=remember_train_datasets,
        val_datasets=val_datasets,
        device=device,
        logger=logger,
    )

    # --- Evaluation: standard + unlearning-style RA/FA/ReA/MIA --------------
    is_fedbn = config["algorithm"].lower() == "fedbn"

    fedbn_fallback_model = None

    if is_fedbn:
        retained_client_weights = {
            partition.client_id: len(partition.samples)
            for partition in train_split.remember_partitions
        }

        fedbn_fallback_model = build_retained_bn_reference_model(
            global_model=retrained_model,
            client_models=per_hospital_models,
            client_weights=retained_client_weights,
        ).to(device)

        retained_hospital_models = build_hospital_model_map(
            per_client_models=per_hospital_models,
            client_partitions=train_split.remember_partitions,
            logger=logger,
        )

        run_fedbn_retrain_evaluation(
            hospital_models=retained_hospital_models,
            fallback_model=fedbn_fallback_model,
            global_test_loader=global_test_loader,
            remember_test_loader=remember_test_loader,
            forget_test_loader=forget_test_loader,
            device=device,
            num_classes=config["num_classes"],
            logger=logger,
            step=config["global_epochs"],
            relearn_steps=config.get("relearn_steps", 50),
            relearn_lr=config.get("relearn_lr", 1e-3),
            nonmember_loader=remember_test_loader,
        )

    else:
        run_standard_evaluation(
            retrained_model,
            global_test_loader,
            device,
            config["num_classes"],
            logger,
            step=config["global_epochs"],
            tag_prefix="eval",
        )

        run_unlearning_evaluation(
            retrained_model,
            remember_test_loader,
            forget_test_loader,
            device,
            logger,
            step=config["global_epochs"],
            relearn_steps=config.get("relearn_steps", 50),
            relearn_lr=config.get("relearn_lr", 1e-3),
            nonmember_loader=remember_test_loader,
            before_unlearning_acc=None,
        )

    # --- Save ---------------------------------------------------------------
    checkpoint_extra = {
        "config": config,
        "run_id": run_id,
        "source_run": source_run,
        "forget_client": config.get("forget_client"),
        "algorithm": config.get("algorithm"),
        "imbalance_method": config.get("imbalance_method"),
    }

    if is_fedbn:
        # Main deployable reference:
        # global non-BN parameters + retained-only averaged BN state.
        save_checkpoint(
            fedbn_fallback_model,
            dirs["checkpoint_dir"],
            "retrained_model",
            extra=checkpoint_extra,
        )

        # Keep the raw server/global model separately.
        save_checkpoint(
            retrained_model,
            dirs["checkpoint_dir"],
            "retrained_global_model",
            extra=checkpoint_extra,
        )

        per_hospital_dir = os.path.join(
            dirs["checkpoint_dir"],
            "per_hospital",
        )
        os.makedirs(per_hospital_dir, exist_ok=True)

        for client_id, model in per_hospital_models.items():
            save_checkpoint(
                model,
                per_hospital_dir,
                client_id,
                extra=checkpoint_extra,
            )

        logger.info(
            f"Saved {len(per_hospital_models)} retained FedBN "
            f"personalized checkpoints to {per_hospital_dir}"
        )

    else:
        save_checkpoint(
            retrained_model,
            dirs["checkpoint_dir"],
            "retrained_model",
            extra=checkpoint_extra,
        )
    logger.info(f"Saved retrained baseline checkpoint to {dirs['checkpoint_dir']}")

    logger.close()
    print(f"\nDone. run_id = {run_id}")


if __name__ == "__main__":
    main()
