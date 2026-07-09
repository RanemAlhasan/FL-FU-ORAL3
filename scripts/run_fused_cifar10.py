#!/usr/bin/env python3
"""
Faithful, end-to-end reproduction of FUSED-Code's CIFAR-10 client-unlearning
experiment (paper's Table 1, "Cifar10-ResNet18" column: Retrain vs FUSED).

This single script runs the ENTIRE pipeline in one process, matching
main.py's flow exactly (unlike the oral-cancer framework, where FL/FU/
retrain are deliberately separate scripts — that separation doesn't apply
here, since faithfully reproducing main.py means faithfully reproducing its
single-process, single-run structure too):

  1. Load CIFAR-10, Dirichlet-partition across num_clients, carve proxy data.
  2. Apply the Byzantine label-shift attack to the forget client(s)' data.
  3. Phase A: train_normal — plain FL training on ATTACKED data
     (global_epochs rounds).
  4. Phase B (FUSED): forget_client_train — LoRA adapter training on CLEAN
     remember-client data only (fused_iterations rounds), evaluated against
     ATTACKED test data.
  5. Phase B (Retrain): fl_retrain — full retraining from the SAME initial
     model, on CLEAN remember-client data only (global_epochs rounds),
     for direct comparison.
  6. ReA: relearn_unlearning_knowledge for BOTH FUSED's and Retrain's
     unlearned models — train on the forget client's BYZANTINE-ATTACKED
     data (same poisoned data as Phase A, not clean data) for
     relearn_rounds, report resulting forget-client accuracy. This matches
     main.py's `case.relearn_unlearning_knowledge(unlearning_model,
     client_all_loaders_process, test_loaders_process)` — ReA measures how
     quickly the POISONED knowledge comes back, not clean knowledge.
  7. MIA: membership_inference_attack for BOTH FUSED's and Retrain's
     unlearned models. Shadow models do NOT redo Phase-A training from
     scratch — they start from the already-trained checkpoint and only
     re-run the unlearning phase on BYZANTINE-ATTACKED proxy data
     (matching utils.py's train_shadow_model, which calls
     `case.forget_client_train(copy.deepcopy(model), proxy_client_loaders_bk,
     proxy_test_loaders)` directly on the Phase-A model). Both the real and
     proxy evaluation pools combine TRAIN-loader and TEST-loader logits per
     client (attacked train + clean test for the real side; attacked
     train + attacked test for the proxy side), and the attack classifier
     trains for `global_epochs` epochs (reusing the FL epoch count, as the
     original does — there is no separate attack-epoch hyperparameter).

All metrics are logged under the SAME naming convention as our oral-cancer
framework (eval/unlearning/RA, FA, ReA, MIA_acc) so scripts/compare_runs.py
-style tooling could in principle be pointed at either project's logs/
directory, even though the two codebases are otherwise independent.

Usage:
    python scripts/run_fused_cifar10.py --config configs/cifar10_client_unlearning.yaml
    python scripts/run_fused_cifar10.py --config configs/cifar10_client_unlearning.yaml \\
        --set global_epochs=20 --set fused_iterations=20 --set n_shadow=2
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
from src.fu.fused_training import forget_client_train, train_normal
from src.fu.relearn import relearn_unlearning_knowledge
from src.fu.retrain import fl_retrain
from src.models.resnet_lora import build_resnet18_cifar10, count_trainable_parameters
from src.utils.config import (apply_overrides, load_config, make_run_id,
                               parse_set_overrides, resolve_run_dirs, save_config_snapshot)
from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Faithful FUSED CIFAR-10 client-unlearning reproduction.")
    parser.add_argument("--config", required=True)
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

    run_id = args.run_id or make_run_id("fused_cifar10_client")
    dirs = resolve_run_dirs(run_id, config["logs_root"], config["checkpoints_root"], config["outputs_root"])
    save_config_snapshot(config, os.path.join(dirs["log_dir"], "config.snapshot.yaml"))

    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])
    logger.log_hparams(config)

    device = config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    if device != config["device"]:
        logger.info(f"Requested device '{config['device']}' unavailable; falling back to CPU.")
    torch.manual_seed(config["seed"])

    forget_client_idx = config["forget_client_idx"]

    # --- Step 1: data ----------------------------------------------------
    logger.info("Building CIFAR-10 federated data (Dirichlet partition + proxy split)...")
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

    # --- Step 2: Byzantine label-shift attack on the forget client(s) ----
    logger.info(f"Applying Byzantine label-shift attack to forget_client_idx={forget_client_idx}...")
    attacked_client_loaders, attacked_test_loaders = baizhanting_attack(
        data.client_loaders, data.test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )
    attacked_proxy_client_loaders, attacked_proxy_test_loaders = baizhanting_attack(
        data.proxy_client_loaders, data.proxy_test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )

    # --- Step 3: Phase A — plain FL training on attacked data ------------
    logger.info("=== Phase A: train_normal (plain FL on attacked data) ===")
    init_model = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
    init_model_snapshot = copy.deepcopy(init_model)  # kept for Retrain's "same starting point"

    trained_model, phase_a_history = train_normal(
        init_model, attacked_client_loaders, attacked_test_loaders, forget_client_idx,
        config["global_epochs"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    torch.save(trained_model.state_dict(), os.path.join(dirs["checkpoint_dir"], "phase_a_model.pt"))

    before_f_acc = phase_a_history["avg_f_acc"][-1]
    before_r_acc = phase_a_history["avg_r_acc"][-1]
    logger.set_final_metric("eval/before_unlearning/FA", before_f_acc)
    logger.set_final_metric("eval/before_unlearning/RA", before_r_acc)
    logger.info(f"Phase A complete. Before unlearning: FA={before_f_acc:.4f}, RA={before_r_acc:.4f}")

    # --- Step 4: Phase B (FUSED) — LoRA adapter on clean remember data ---
    logger.info("=== Phase B (FUSED): forget_client_train (LoRA adapter) ===")
    fused_model, fused_history = forget_client_train(
        trained_model, data.client_loaders, attacked_test_loaders, forget_client_idx,
        config["fused_iterations"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    trainable, total = count_trainable_parameters(fused_model)
    logger.info(f"FUSED adapter: {trainable}/{total} trainable parameters "
                f"({100 * trainable / total:.3f}% density).")
    logger.set_final_metric("fu/adapter_trainable_params", trainable)
    logger.set_final_metric("fu/adapter_density", trainable / total)

    fused_fa = fused_history["avg_f_acc"][-1]
    fused_ra = fused_history["avg_r_acc"][-1]
    logger.set_final_metric("eval/unlearning/FUSED/FA", fused_fa)
    logger.set_final_metric("eval/unlearning/FUSED/RA", fused_ra)
    logger.info(f"FUSED unlearning complete: FA={fused_fa:.4f}, RA={fused_ra:.4f}")
    torch.save(fused_model.state_dict(), os.path.join(dirs["checkpoint_dir"], "fused_model.pt"))

    # --- Step 5: Phase B (Retrain) — full retrain on clean remember data -
    logger.info("=== Phase B (Retrain): fl_retrain (full model, upper-bound baseline) ===")
    retrain_model, retrain_history = fl_retrain(
        init_model_snapshot, data.client_loaders, attacked_test_loaders, forget_client_idx,
        config["global_epochs"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    retrain_fa = retrain_history["avg_f_acc"][-1]
    retrain_ra = retrain_history["avg_r_acc"][-1]
    logger.set_final_metric("eval/unlearning/Retrain/FA", retrain_fa)
    logger.set_final_metric("eval/unlearning/Retrain/RA", retrain_ra)
    logger.info(f"Retrain baseline complete: FA={retrain_fa:.4f}, RA={retrain_ra:.4f}")
    torch.save(retrain_model.state_dict(), os.path.join(dirs["checkpoint_dir"], "retrain_model.pt"))

    # NOTE: relearn faithfully re-runs on the BYZANTINE-ATTACKED loaders,
    # matching main.py's exact call:
    #   case.relearn_unlearning_knowledge(unlearning_model,
    #       client_all_loaders_process, test_loaders_process)
    # i.e. ReA measures whether the model can quickly RELEARN the
    # label-flipped/poisoned knowledge if handed the forget client's
    # attacked data again — NOT whether it relearns clean labels. Passing
    # data.client_loaders/data.test_loaders here (as an earlier version of
    # this script did) measures a different, easier quantity and is not
    # faithful to the paper's ReA.
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

    # --- Step 7: MIA for both FUSED and Retrain ---------------------------
    # Faithful to utils.py's train_shadow_model()/membership_inference_attack():
    #   - Shadow models do NOT redo Phase-A training from scratch. They
    #     start from the SAME already-trained checkpoint used for the real
    #     run (`trained_model` for FUSED, `init_model_snapshot` for
    #     Retrain — matching each method's own real-run starting point)
    #     and only re-run the unlearning phase itself, on PROXY data.
    #   - The proxy data fed into that shadow unlearning step is the
    #     BYZANTINE-ATTACKED proxy client/test loaders (main.py passes
    #     `proxy_client_loaders_bk`/`proxy_test_loaders`, both attacked,
    #     into `case.forget_client_train(...)` / `case.FL_Retrain(...)`
    #     inside train_shadow_model — NOT the clean `proxy_client_loaders`).
    #   - Both the real and proxy evaluation pools combine TRAIN-loader and
    #     TEST-loader logits (see mia.py's module docstring for the exact
    #     asymmetry: real side = [attacked_client_loaders, data.test_loaders]
    #     (attacked train, CLEAN test); proxy side =
    #     [attacked_proxy_client_loaders, attacked_proxy_test_loaders]
    #     (both attacked)).
    #   - The attack classifier trains for `mia_attack_epochs` epochs if set,
    #     else falls back to `global_epochs` (the original reuses
    #     args.global_epoch verbatim, with no dedicated attack-epoch
    #     hyperparameter upstream — this fallback preserves that faithful
    #     default while letting `--set mia_attack_epochs=N` actually shorten
    #     this expensive step, matching run_fu_cli_domain.py's
    #     `args.mia_attack_epochs or args.global_epoch` pattern).
    if config.get("run_mia", True):
        mia_attack_epochs = config.get("mia_attack_epochs") or config["global_epochs"]
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
            mia_attack_epochs, config["test_batch_size"],
        )
        logger.set_final_metric("eval/unlearning/FUSED/MIA_acc", fused_mia_acc)
        logger.info(f"FUSED MIA accuracy = {fused_mia_acc:.4f} (per-client: {fused_mia_per_client})")

        logger.info("=== MIA: membership_inference_attack (Retrain) ===")

        def retrain_shadow_fn():
            shadow_unlearned, _ = fl_retrain(
                copy.deepcopy(init_model_snapshot), attacked_proxy_client_loaders, attacked_proxy_test_loaders,
                forget_client_idx, config["global_epochs"], config["local_epochs"], config["learning_rate"],
                device, config["test_batch_size"],
            )
            return shadow_unlearned

        retrain_mia_acc, retrain_mia_per_client = membership_inference_attack(
            retrain_model,
            [attacked_client_loaders, data.test_loaders],
            [attacked_proxy_client_loaders, attacked_proxy_test_loaders],
            forget_client_idx, config["num_classes"], config["n_shadow"], retrain_shadow_fn, device,
            mia_attack_epochs, config["test_batch_size"],
        )
        logger.set_final_metric("eval/unlearning/Retrain/MIA_acc", retrain_mia_acc)
        logger.info(f"Retrain MIA accuracy = {retrain_mia_acc:.4f} (per-client: {retrain_mia_per_client})")

    # --- Final summary, matching Table 1's layout -------------------------
    logger.info("")
    logger.info("=== TABLE 1 STYLE SUMMARY (Cifar10-ResNet18, Client Unlearning) ===")
    logger.info(f"{'Metric':<10} {'Retrain':>10} {'FUSED':>10}")
    logger.info(f"{'RA':<10} {retrain_ra:>10.4f} {fused_ra:>10.4f}")
    logger.info(f"{'FA':<10} {retrain_fa:>10.4f} {fused_fa:>10.4f}")
    logger.info(f"{'ReA':<10} {retrain_relearn['ReA']:>10.4f} {fused_relearn['ReA']:>10.4f}")
    if config.get("run_mia", True):
        logger.info(f"{'MIA':<10} {retrain_mia_acc:>10.4f} {fused_mia_acc:>10.4f}")

    logger.close()
    print(f"\nDone. run_id = {run_id}")


if __name__ == "__main__":
    main()