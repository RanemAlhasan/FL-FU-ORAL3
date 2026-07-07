#!/usr/bin/env python3
"""
Resume script: loads existing phase_a_model.pt / fused_model.pt /
retrain_model.pt checkpoints from a completed fused_cifar10_full run and
re-executes ONLY Step 7 (MIA), using the fixed membership_inference_attack
(src/eval/mia.py): shadow models re-run only the unlearning phase from the
already-trained Phase-A checkpoint (no wasted full retrain-from-scratch per
shadow), on the BYZANTINE-ATTACKED proxy loaders, and both real/proxy
evaluation pools combine train+test loader logits per client — see
src/eval/mia.py's module docstring for the full rationale.

Skips Phase A, Phase B, and ReA entirely. Pair this with
scripts/resume_rea_only.py (fixed) to fully refresh Step 6+7 without
redoing the expensive Phase A/B training.

NOTE on the Retrain shadow's starting model: the original repo's shadow
retraining does not need to start from the exact same initial weights as
the real Retrain baseline (see src/eval/mia.py's docstring — shadow models
exist only to train the ATTACK classifier, not to reproduce the real run
bit-for-bit) — so this script builds a FRESH init model for the Retrain
shadow rather than trying to reconstruct the original run's
`init_model_snapshot` (which was never checkpointed to disk).

SAFE TO RESUME: MIA depends on the seeded Dirichlet client/proxy partition
(data.client_loaders, data.test_loaders, data.proxy_client_loaders,
data.proxy_test_loaders) plus the deterministic Byzantine label-shift,
both fully reproducible given the same seed/config as the original run.

Usage:
    python scripts/resume_mia_only.py \\
        --config configs/cifar10_client_unlearning.yaml \\
        --source_run fused_cifar10_full_327966 \\
        --run_id fused_cifar10_resume_mia_<jobid>
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
from src.eval.mia import membership_inference_attack
from src.fu.fused_training import forget_client_train
from src.fu.retrain import fl_retrain
from src.models.resnet_lora import build_resnet18_cifar10, build_lora_adapter, count_trainable_parameters
from src.utils.config import (apply_overrides, load_config, make_run_id,
                               parse_set_overrides, resolve_run_dirs, save_config_snapshot)
from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Resume MIA-only from existing FUSED CIFAR-10 checkpoints.")
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

    run_id = args.run_id or make_run_id("fused_cifar10_resume_mia")
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

    logger.info(f"Applying Byzantine label-shift attack to forget_client_idx={forget_client_idx}...")
    attacked_client_loaders, attacked_test_loaders = baizhanting_attack(
        data.client_loaders, data.test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )
    attacked_proxy_client_loaders, attacked_proxy_test_loaders = baizhanting_attack(
        data.proxy_client_loaders, data.proxy_test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )

    # --- Load existing checkpoints ----------------------------------------
    source_ckpt_dir = os.path.join(config["checkpoints_root"], args.source_run)
    logger.info(f"Loading checkpoints from {source_ckpt_dir}")

    trained_model = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
    trained_model.load_state_dict(torch.load(
        os.path.join(source_ckpt_dir, "phase_a_model.pt"), map_location=device))
    trained_model.eval()

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

    # --- Step 7: MIA for both FUSED and Retrain -----------------------------
    # See scripts/run_fused_cifar10.py's Step 7 comment block for the full
    # rationale (shadow models start from the already-trained checkpoint,
    # train only on attacked proxy data, real+proxy eval pools combine
    # train+test loader logits, attack classifier trains for global_epochs).
    logger.info("=== MIA: membership_inference_attack (FUSED) ===")

    def fused_shadow_fn():
        shadow_unlearned, _ = forget_client_train(
            copy.deepcopy(trained_model), attacked_proxy_client_loaders, attacked_proxy_test_loaders,
            forget_client_idx, config["fused_iterations"], config["local_epochs"], config["learning_rate"],
            device, config["test_batch_size"],
        )
        return shadow_unlearned

    fused_mia_acc, fused_mia_per_client = membership_inference_attack(
        fused_model,
        [attacked_client_loaders, data.test_loaders],
        [attacked_proxy_client_loaders, attacked_proxy_test_loaders],
        forget_client_idx, config["num_classes"], config["n_shadow"], fused_shadow_fn, device,
        config["global_epochs"], config["test_batch_size"],
    )
    logger.set_final_metric("eval/unlearning/FUSED/MIA_acc", fused_mia_acc)
    logger.info(f"FUSED MIA accuracy = {fused_mia_acc:.4f} (per-client: {fused_mia_per_client})")

    logger.info("=== MIA: membership_inference_attack (Retrain) ===")

    def retrain_shadow_fn():
        # Fresh init model for the shadow — does NOT need to match the
        # original run's init_model_snapshot bit-for-bit (that snapshot
        # was never checkpointed, and the shadow only needs to produce a
        # plausible "retrain from scratch" model to train the attack
        # classifier against — see this script's module docstring).
        shadow_init = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
        shadow_unlearned, _ = fl_retrain(
            shadow_init, attacked_proxy_client_loaders, attacked_proxy_test_loaders,
            forget_client_idx, config["global_epochs"], config["local_epochs"], config["learning_rate"],
            device, config["test_batch_size"],
        )
        return shadow_unlearned

    retrain_mia_acc, retrain_mia_per_client = membership_inference_attack(
        retrain_model,
        [attacked_client_loaders, data.test_loaders],
        [attacked_proxy_client_loaders, attacked_proxy_test_loaders],
        forget_client_idx, config["num_classes"], config["n_shadow"], retrain_shadow_fn, device,
        config["global_epochs"], config["test_batch_size"],
    )
    logger.set_final_metric("eval/unlearning/Retrain/MIA_acc", retrain_mia_acc)
    logger.info(f"Retrain MIA accuracy = {retrain_mia_acc:.4f} (per-client: {retrain_mia_per_client})")

    # --- Summary ------------------------------------------------------------
    logger.info("")
    logger.info("=== RESUME RUN SUMMARY (MIA fix only; ReA untouched here, "
                "use scripts/resume_rea_only.py for that) ===")
    logger.info(f"{'Metric':<10} {'Retrain':>10} {'FUSED':>10}")
    logger.info(f"{'MIA':<10} {retrain_mia_acc:>10.4f} {fused_mia_acc:>10.4f}")

    logger.close()
    print(f"\nDone. run_id = {run_id}")


if __name__ == "__main__":
    main()