#!/usr/bin/env python3
"""
Resume script: loads existing fused_model.pt / retrain_model.pt checkpoints
from a completed fused_cifar10_full run and re-executes ONLY Step 6 (ReA),
with Retrain now actually relearning from retrain_model (not a duplicate
FUSED run). Skips Phase A, Phase B, and MIA entirely — MIA's original
numbers (eval/unlearning/{FUSED,Retrain}/MIA_acc) are untouched and still
valid.

ReA is re-run on the BYZANTINE-ATTACKED loaders, matching main.py's exact
call `case.relearn_unlearning_knowledge(unlearning_model,
client_all_loaders_process, test_loaders_process)` — i.e. it measures how
quickly the label-flipped/poisoned knowledge comes back if the forget
client's attacked data is handed back to the model, NOT how quickly clean
knowledge comes back. (An earlier version of this script used
data.client_loaders/data.test_loaders directly — clean, unattacked data —
which measures a different, easier quantity and is not faithful to the
paper's ReA; that has been fixed below.)

SAFE TO RESUME: ReA only depends on the seeded Dirichlet client partition
(data.client_loaders, data.test_loaders) plus the deterministic Byzantine
label-shift (baizhanting_attack has no randomness — it's a fixed +1 mod
num_classes label permutation), so it's fully reproducible given the same
seed/config as the original run. No proxy-split dependency here, so the
unseeded proxy carve issue (see resume_rea_mia.py if it exists) does not
apply.

Usage:
    python scripts/resume_rea_only.py \\
        --config configs/cifar10_client_unlearning.yaml \\
        --source_run fused_cifar10_full_327966 \\
        --run_id fused_cifar10_resume_rea_<jobid>
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.data.byzantine_attack import baizhanting_attack
from src.data.pipeline import build_cifar10_federated_data
from src.fu.relearn import relearn_unlearning_knowledge
from src.models.resnet_lora import build_resnet18_cifar10, build_lora_adapter, count_trainable_parameters
from src.utils.config import (apply_overrides, load_config, make_run_id,
                               parse_set_overrides, resolve_run_dirs, save_config_snapshot)
from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Resume ReA-only from existing FUSED CIFAR-10 checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_run", required=True,
                         help="run_id of the completed fused_cifar10_full run whose checkpoints to load.")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--set", dest="set_overrides", action="append", default=None, metavar="KEY=VALUE")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    overrides = parse_set_overrides(args.set_overrides)
    config = apply_overrides(config, overrides)
    if overrides:
        print(f"Applied --set overrides: {overrides}")

    run_id = args.run_id or make_run_id("fused_cifar10_resume_rea")
    dirs = resolve_run_dirs(run_id, config["logs_root"], config["checkpoints_root"], config["outputs_root"])
    save_config_snapshot(config, os.path.join(dirs["log_dir"], "config.snapshot.yaml"))

    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])
    logger.log_hparams(config)

    device = config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    if device != config["device"]:
        logger.info(f"Requested device '{config['device']}' unavailable; falling back to CPU.")
    torch.manual_seed(config["seed"])

    forget_client_idx = config["forget_client_idx"]

    # --- Rebuild data, SAME seed as original run -------------------------
    logger.info(f"Rebuilding CIFAR-10 federated data (seed={config['seed']}, "
                f"matches original run {args.source_run}'s Dirichlet client partition).")
    data = build_cifar10_federated_data(
        dataset_root=config["dataset_root"],
        num_clients=config["num_clients"],
        num_classes=config["num_classes"],
        alpha=config["alpha"],
        local_batch_size=config["local_batch_size"],
        test_batch_size=config["test_batch_size"],
        proxy_frac=config["proxy_frac"],
        forget_paradigm="client",
        seed=config["seed"],
        num_workers=config.get("num_workers", 2),
    )
    logger.info(f"Built {config['num_clients']} clients. "
                f"Client data sizes (pre-proxy-carve): {data.client_data_sizes}")

    # Rebuild the SAME Byzantine-attacked loaders the original run used for
    # Phase A/ReA — ReA must be re-run on the attacked (poisoned) data, per
    # main.py's exact call contract (see module docstring above).
    logger.info(f"Applying Byzantine label-shift attack to forget_client_idx={forget_client_idx}...")
    attacked_client_loaders, attacked_test_loaders = baizhanting_attack(
        data.client_loaders, data.test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )

    # --- Load existing checkpoints ----------------------------------------
    source_ckpt_dir = os.path.join(config["checkpoints_root"], args.source_run)
    logger.info(f"Loading checkpoints from {source_ckpt_dir}")

    fused_model = build_lora_adapter(
        build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"])
    ).to(device)
    fused_model.load_state_dict(torch.load(
        os.path.join(source_ckpt_dir, "fused_model.pt"), map_location=device))
    fused_model.eval()

    retrain_model = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
    retrain_model.load_state_dict(torch.load(
        os.path.join(source_ckpt_dir, "retrain_model.pt"), map_location=device))
    retrain_model.eval()

    trainable, total = count_trainable_parameters(fused_model)
    logger.info(f"Loaded FUSED adapter: {trainable}/{total} trainable parameters "
                f"({100 * trainable / total:.3f}% density).")

    # --- Step 6: ReA for both FUSED and Retrain -----------------------------
    logger.info("=== ReA: relearn_unlearning_knowledge (FUSED) ===")
    _, fused_relearn = relearn_unlearning_knowledge(
        copy.deepcopy(fused_model), attacked_client_loaders, attacked_test_loaders, forget_client_idx,
        config["relearn_rounds"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    logger.set_final_metric("eval/unlearning/FUSED/ReA", fused_relearn["ReA"])
    logger.info(f"FUSED ReA = {fused_relearn['ReA']:.4f}")

    logger.info("=== ReA: relearn_unlearning_knowledge (Retrain) ===")
    _, retrain_relearn = relearn_unlearning_knowledge(
        copy.deepcopy(retrain_model), attacked_client_loaders, attacked_test_loaders, forget_client_idx,
        config["relearn_rounds"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    logger.set_final_metric("eval/unlearning/Retrain/ReA", retrain_relearn["ReA"])
    logger.info(f"Retrain ReA = {retrain_relearn['ReA']:.4f}")

    # --- Summary ------------------------------------------------------------
    logger.info("")
    logger.info("=== RESUME RUN SUMMARY (ReA fix only; MIA untouched, use original run's MIA numbers) ===")
    logger.info(f"{'Metric':<10} {'Retrain':>10} {'FUSED':>10}")
    logger.info(f"{'ReA':<10} {retrain_relearn['ReA']:>10.4f} {fused_relearn['ReA']:>10.4f}")

    logger.close()
    print(f"\nDone. run_id = {run_id}")


if __name__ == "__main__":
    main()