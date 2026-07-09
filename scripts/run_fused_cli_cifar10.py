#!/usr/bin/env python3
"""
FUSED CIFAR-10 reproduction using the ACTUAL paper method: Critical Layer
Identification + random-sparsified adapters (Algorithm 1), NOT LoRA.

This is a NEW, separate pipeline from run_fused_cifar10.py / fused_training.py
(the LoRA-based reproduction) — nothing here imports or modifies those files,
so existing LoRA-based results and checkpoints are completely unaffected.

Steps:
  1. Load the existing Phase-A checkpoint (fused_cifar10_full_327966's
     phase_a_model.pt) — Phase A is NOT redone here, it's algorithm-agnostic
     w.r.t. what we're testing (same reasoning as run_fu_lora_domain.py for
     oral3: hold Phase 1 fixed at plain FedAvg, vary the algorithm only in
     Phase 2/3).
  2. Rebuild data + apply the Byzantine label-shift attack (same seed as
     the original run, for a fair comparison).
  3. Run FUSED unlearning via src/fu/fused_cli_training.py::run_fused_cli_unlearning
     (Algorithm 1 + optional FedBN/FedProx/FedMoon per --algorithm).
  4. ReA: relearn probe (src/fu/relearn.py, unchanged, algorithm-agnostic).
  5. Retrain baseline, matched algorithm (src/fu/retrain_domain.py::fl_retrain_domain).
  6. MIA: shadow models built by re-running Step 3/5's unlearning function
     on proxy data (src/eval/mia.py, unchanged).

Usage:
    python3 scripts/run_fused_cli_cifar10.py \\
        --source_run fused_cifar10_full_327966 \\
        --algorithm fedbn \\
        --config configs/cifar10_client_unlearning.yaml
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
from src.fu.fused_cli_training import run_fused_cli_unlearning
from src.fu.relearn import relearn_unlearning_knowledge
from src.fu.retrain_domain import fl_retrain_domain
from src.models.resnet_lora import build_resnet18_cifar10
from src.utils.config import (apply_overrides, load_config, make_run_id,
                               parse_set_overrides, resolve_run_dirs, save_config_snapshot)
from src.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(description="FUSED CIFAR-10 via the real CLI+sparse-adapter method.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_run", required=True,
                         help="run_id of a completed fused_cifar10_full run to load phase_a_model.pt from.")
    parser.add_argument("--algorithm", choices=["fedavg", "fedbn", "fedprox", "fedmoon"], default="fedavg")
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--fedmoon_mu", type=float, default=1.0)
    parser.add_argument("--fedmoon_temperature", type=float, default=0.5)
    parser.add_argument("--num_unlearning_layers", type=int, default=4,
                         help="K in the paper's CLI step (paper Fig. 2 discussion identifies ~4 sensitive "
                              "ResNet18 layers for the client-unlearning scenario).")
    parser.add_argument("--adapter_sparsity", type=float, default=0.05,
                         help="Fraction of each critical layer's parameters that are trainable.")
    parser.add_argument("--run_mia", action="store_true", default=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--source_checkpoints_root", default="checkpoints/cifar10",
                         help="Where to LOAD the existing Phase-A checkpoint from (the original "
                              "fused_cifar10_full run's checkpoints/, NOT this run's own --set "
                              "checkpoints_root, which is for SAVING this run's new outputs).")
    parser.add_argument("--set", dest="set_overrides", action="append", default=None, metavar="KEY=VALUE")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    overrides = parse_set_overrides(args.set_overrides)
    config = apply_overrides(config, overrides)

    run_id = args.run_id or make_run_id(f"fused_cli_{args.algorithm}_cifar10")
    dirs = resolve_run_dirs(run_id, "logs/cifar10_cli", "checkpoints/cifar10_cli", "outputs/cifar10_cli")
    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])

    merged_config = dict(config)
    merged_config.update({
        "algorithm": args.algorithm, "fedprox_mu": args.fedprox_mu,
        "fedmoon_mu": args.fedmoon_mu, "fedmoon_temperature": args.fedmoon_temperature,
        "num_unlearning_layers": args.num_unlearning_layers, "adapter_sparsity": args.adapter_sparsity,
        "source_run": args.source_run, "method": "FUSED-CLI (Algorithm 1, real paper method)",
    })
    save_config_snapshot(merged_config, os.path.join(dirs["log_dir"], "config.snapshot.yaml"))
    logger.log_hparams(merged_config)

    device = config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    torch.manual_seed(config["seed"])
    forget_client_idx = config["forget_client_idx"]

    logger.info(f"Rebuilding CIFAR-10 federated data (seed={config['seed']}, matches source run's partition).")
    data = build_cifar10_federated_data(
        dataset_root=config["dataset_root"], num_clients=config["num_clients"],
        num_classes=config["num_classes"], alpha=config["alpha"],
        local_batch_size=config["local_batch_size"], test_batch_size=config["test_batch_size"],
        proxy_frac=config["proxy_frac"], forget_paradigm="client", seed=config["seed"],
        num_workers=config.get("num_workers", 2),
    )

    logger.info(f"Applying Byzantine label-shift attack to forget_client_idx={forget_client_idx}...")
    attacked_client_loaders, attacked_test_loaders = baizhanting_attack(
        data.client_loaders, data.test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )
    attacked_proxy_client_loaders, attacked_proxy_test_loaders = baizhanting_attack(
        data.proxy_client_loaders, data.proxy_test_loaders, forget_client_idx,
        config["num_classes"], config["local_batch_size"],
    )

    source_ckpt_dir = os.path.join(args.source_checkpoints_root, args.source_run)
    trained_model = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
    trained_model.load_state_dict(torch.load(
        os.path.join(source_ckpt_dir, "phase_a_model.pt"), map_location=device))
    logger.info(f"Loaded Phase-A checkpoint (read-only) from {source_ckpt_dir}/phase_a_model.pt")

    # --- Step 3: FUSED unlearning (Algorithm 1, real CLI+sparse-adapter) --
    logger.info(f"=== FUSED-CLI unlearning (algorithm={args.algorithm}) ===")
    fused_model, fu_history, critical_layers = run_fused_cli_unlearning(
        source_model=copy.deepcopy(trained_model),
        # BUG FIX: was `attacked_client_loaders` — the label-shifted train
        # loaders — passed as the "clean" client loaders. Remember clients'
        # loaders are byte-identical either way (baizhanting_attack only
        # replaces forget_client_idx's own loader), so this never affected
        # actual adapter training, but it DID feed the forget client's
        # mislabeled data into the Critical Layer Identification diagnostic
        # pass (cli_use_all_clients=True, the default) instead of its true
        # clean labels, corrupting which layers get selected as "critical."
        # The sibling LoRA script (run_fused_cifar10.py) correctly passes
        # the genuinely clean `data.client_loaders` to the analogous
        # parameter — match that convention here.
        all_clean_client_loaders=data.client_loaders,
        attacked_test_loaders=attacked_test_loaders,
        forget_client_idx=forget_client_idx,
        client_data_sizes=data.client_data_sizes,
        num_unlearning_layers=args.num_unlearning_layers,
        adapter_sparsity=args.adapter_sparsity,
        fused_iterations=config["fused_iterations"],
        local_epochs=config["local_epochs"],
        learning_rate=config["learning_rate"],
        device=device, test_batch_size=config["test_batch_size"],
        algorithm=args.algorithm, fedprox_mu=args.fedprox_mu,
        fedmoon_mu=args.fedmoon_mu, fedmoon_temperature=args.fedmoon_temperature,
        seed=config["seed"], logger=logger,
    )
    logger.set_final_metric(f"eval/{args.algorithm}/FUSED_CLI/RA", fu_history["avg_r_acc"][-1])
    logger.set_final_metric(f"eval/{args.algorithm}/FUSED_CLI/FA", fu_history["avg_f_acc"][-1])

    # --- Step 4: ReA -----------------------------------------------------
    logger.info("=== ReA: relearn_unlearning_knowledge (algorithm-agnostic) ===")
    _, fused_relearn = relearn_unlearning_knowledge(
        copy.deepcopy(fused_model), attacked_client_loaders, attacked_test_loaders, forget_client_idx,
        config["relearn_rounds"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    logger.set_final_metric(f"eval/{args.algorithm}/FUSED_CLI/ReA", fused_relearn["ReA"])

    # --- Step 5: Retrain baseline, matched algorithm ---------------------
    logger.info(f"=== Retrain baseline (algorithm={args.algorithm}) ===")
    init_model_snapshot = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
    retrain_model, retrain_history = fl_retrain_domain(
        init_global_model=init_model_snapshot,
        all_clean_client_loaders=attacked_client_loaders,
        attacked_test_loaders=attacked_test_loaders,
        forget_client_idx=forget_client_idx,
        global_epochs=config["global_epochs"], local_epochs=config["local_epochs"],
        learning_rate=config["learning_rate"], device=device, test_batch_size=config["test_batch_size"],
        algorithm=args.algorithm, fedprox_mu=args.fedprox_mu,
        fedmoon_mu=args.fedmoon_mu, fedmoon_temperature=args.fedmoon_temperature, logger=logger,
    )
    logger.set_final_metric(f"eval/{args.algorithm}/Retrain/RA", retrain_history["avg_r_acc"][-1])
    logger.set_final_metric(f"eval/{args.algorithm}/Retrain/FA", retrain_history["avg_f_acc"][-1])

    _, retrain_relearn = relearn_unlearning_knowledge(
        copy.deepcopy(retrain_model), attacked_client_loaders, attacked_test_loaders, forget_client_idx,
        config["relearn_rounds"], config["local_epochs"], config["learning_rate"],
        device, config["test_batch_size"], logger,
    )
    logger.set_final_metric(f"eval/{args.algorithm}/Retrain/ReA", retrain_relearn["ReA"])

    # --- Step 6: MIA (optional) ------------------------------------------
    if args.run_mia:
        # Attack classifier epochs: `mia_attack_epochs` if set via --set,
        # else falls back to `global_epochs` (the original reuses
        # args.global_epoch verbatim, with no dedicated attack-epoch
        # hyperparameter upstream — this fallback preserves that faithful
        # default while letting `--set mia_attack_epochs=N` actually shorten
        # this expensive step, matching run_fu_cli_domain.py's
        # `args.mia_attack_epochs or args.global_epoch` pattern).
        mia_attack_epochs = config.get("mia_attack_epochs") or config["global_epochs"]
        logger.info(f"=== MIA (algorithm={args.algorithm}) ===")

        def fused_shadow_fn():
            shadow_model, _, _ = run_fused_cli_unlearning(
                source_model=copy.deepcopy(trained_model),
                all_clean_client_loaders=attacked_proxy_client_loaders,
                attacked_test_loaders=attacked_proxy_test_loaders,
                forget_client_idx=forget_client_idx,
                client_data_sizes=data.client_data_sizes,
                num_unlearning_layers=args.num_unlearning_layers, adapter_sparsity=args.adapter_sparsity,
                fused_iterations=config["fused_iterations"], local_epochs=config["local_epochs"],
                learning_rate=config["learning_rate"], device=device, test_batch_size=config["test_batch_size"],
                algorithm=args.algorithm, fedprox_mu=args.fedprox_mu,
                fedmoon_mu=args.fedmoon_mu, fedmoon_temperature=args.fedmoon_temperature,
                seed=config["seed"] + 1,  # different mask draw per shadow call
            )
            return shadow_model

        fused_mia_acc, fused_mia_per_client = membership_inference_attack(
            fused_model, [attacked_client_loaders, data.test_loaders],
            [attacked_proxy_client_loaders, attacked_proxy_test_loaders],
            forget_client_idx, config["num_classes"], config["n_shadow"], fused_shadow_fn, device,
            mia_attack_epochs, config["test_batch_size"],
        )
        logger.set_final_metric(f"eval/{args.algorithm}/FUSED_CLI/MIA_acc", fused_mia_acc)

        def retrain_shadow_fn():
            shadow_init = build_resnet18_cifar10(config["num_classes"], pretrained=config["pretrained"]).to(device)
            shadow_model, _ = fl_retrain_domain(
                shadow_init, attacked_proxy_client_loaders, attacked_proxy_test_loaders, forget_client_idx,
                config["global_epochs"], config["local_epochs"], config["learning_rate"],
                device, config["test_batch_size"], algorithm=args.algorithm, fedprox_mu=args.fedprox_mu,
                fedmoon_mu=args.fedmoon_mu, fedmoon_temperature=args.fedmoon_temperature,
            )
            return shadow_model

        retrain_mia_acc, retrain_mia_per_client = membership_inference_attack(
            retrain_model, [attacked_client_loaders, data.test_loaders],
            [attacked_proxy_client_loaders, attacked_proxy_test_loaders],
            forget_client_idx, config["num_classes"], config["n_shadow"], retrain_shadow_fn, device,
            mia_attack_epochs, config["test_batch_size"],
        )
        logger.set_final_metric(f"eval/{args.algorithm}/Retrain/MIA_acc", retrain_mia_acc)

    logger.info("")
    logger.info(f"=== SUMMARY (algorithm={args.algorithm}) ===")
    logger.info(f"{'Metric':<10} {'Retrain':>10} {'FUSED-CLI':>10}")
    logger.info(f"{'RA':<10} {retrain_history['avg_r_acc'][-1]:>10.4f} {fu_history['avg_r_acc'][-1]:>10.4f}")
    logger.info(f"{'FA':<10} {retrain_history['avg_f_acc'][-1]:>10.4f} {fu_history['avg_f_acc'][-1]:>10.4f}")
    logger.info(f"{'ReA':<10} {retrain_relearn['ReA']:>10.4f} {fused_relearn['ReA']:>10.4f}")
    if args.run_mia:
        logger.info(f"{'MIA':<10} {retrain_mia_acc:>10.4f} {fused_mia_acc:>10.4f}")
    logger.info(f"Critical layers selected: {critical_layers}")

    logger.close()
    print(f"\nDone. run_id = {run_id} (algorithm = {args.algorithm})")


if __name__ == "__main__":
    main()
