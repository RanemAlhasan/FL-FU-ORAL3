#!/usr/bin/env python3
"""
Phase 2 (DOMAIN-ADAPTATION-AWARE): FUSED unlearning on the 3-hospital
oral-cancer dataset, with FedBN/FedProx support added to the unlearning
phase itself (not just Phase 1).

This is the "contribution" script — it is NOT part of the paper-faithful
reproduction. It plays the same role the original run_fu_lora.py once did
(that script no longer exists in this repo — --algorithm fedavg here
reproduces its behavior exactly), with these differences from that
predecessor:

  - Imports src/fu/fused_training_domain.py and src/fu/retrain_domain.py
    instead of the plain fused_training.py / retrain.py.
  - New --algorithm {fedavg,fedbn,fedprox} flag, threaded down into
    forget_client_train_domain / fl_retrain_domain.
  - New --fedprox_mu flag (only used when --algorithm fedprox).
  - relearn.py is UNCHANGED and reused as-is: relearn only retrains the
    single forget client, so cross-client BN averaging / domain
    adaptation is not applicable there.

RECOMMENDED USAGE for a clean ablation: keep --source_run fixed to the
SAME plain-FedAvg Phase-1 checkpoint (e.g. fl_fedavg_oral_329811) across
all three --algorithm settings. That isolates the effect of Phase-2's
aggregation algorithm alone. Mixing a different Phase-1 source per
--algorithm confounds two variables at once — see the "_backbone"-suffixed
sibling scripts (scripts/research_run/domain_contribution_backbone/) for
that separate, deliberately-confounded experiment (forking from each
algorithm's own matching Phase-1 checkpoint instead).

Usage:
    python3 scripts/run_fu_lora_domain.py \\
        --source_run fl_fedavg_oral_329811 \\
        --forget_client Spain_Dataset \\
        --algorithm fedbn \\
        --global_epoch 50 --local_epoch 3 --batch_size 16

    python3 scripts/run_fu_lora_domain.py \\
        --source_run fl_fedavg_oral_329811 \\
        --forget_client Spain_Dataset \\
        --algorithm fedprox --fedprox_mu 0.01 \\
        --global_epoch 50 --local_epoch 3 --batch_size 16

Run once per hospital x algorithm for the full comparison grid.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.dataset import index_dataset, OralCancerDataset
from src.data.partition import build_client_partitions, carve_proxy_partitions, partitions_to_datasets
from src.data.transforms import build_transforms
from src.eval.evaluator import run_standard_evaluation, run_unlearning_evaluation
from src.eval.mia import membership_inference_attack
from src.fl.core import evaluate
from src.fu import relearn as relearn_mod
from src.fu.fused_training_domain import forget_client_train_domain
from src.fu.retrain_domain import fl_retrain_domain
from src.models.resnet_lora import build_resnet18_cifar10, count_trainable_parameters
from src.utils.checkpoint import get_checkpoint_path, load_checkpoint_into_new_model, save_checkpoint
from src.utils.config import (load_source_run_config, make_run_id, resolve_run_dirs,
                               save_config_snapshot)
from src.utils.logger import build_logger


class TensorPairDataset(Dataset):
    """Same bridge wrapper as run_fu_lora.py — see that file for rationale."""

    def __init__(self, oral_dataset: OralCancerDataset):
        self.oral_dataset = oral_dataset

    def __len__(self) -> int:
        return len(self.oral_dataset)

    def __getitem__(self, idx: int):
        item = self.oral_dataset[idx]
        return item["image"], item["label"]


def as_tensor_pair_loader(oral_dataset: OralCancerDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorPairDataset(oral_dataset), batch_size=batch_size,
                       shuffle=shuffle, num_workers=2)


def concat_as_dict_loader(oral_datasets, batch_size: int, shuffle: bool) -> DataLoader:
    """Same as concat_as_loader, but WITHOUT the TensorPairDataset bridge:
    yields OralCancerDataset's native dict batches ({"image", "label",
    "hospital", ...}), which is what src/eval/evaluator.py and
    src/eval/metrics.py expect. Only use this for loaders passed to
    run_standard_evaluation / run_unlearning_evaluation — everything else
    in this script (training, test_client_forget, relearn, the plain
    evaluate() call) expects (image, label) tuples via
    as_tensor_pair_loader/concat_as_loader and must keep using those."""
    combined = oral_datasets[0] if len(oral_datasets) == 1 else ConcatDataset(oral_datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 (domain-adaptation-aware): FUSED unlearning.")
    parser.add_argument("--source_run", required=True,
                         help="run_id of the completed Phase-1 FL run to fork from.")
    parser.add_argument("--forget_client", required=True,
                         help="Hospital name to forget, e.g. Spain_Dataset / Canada_Dataset / India_Dataset.")
    parser.add_argument("--algorithm", choices=["fedavg", "fedbn", "fedprox", "fedmoon"], default="fedavg",
                         help="Aggregation algorithm used DURING unlearning (Phase 2), independent of "
                              "whichever algorithm trained --source_run. Default 'fedavg' reproduces "
                              "run_fu_lora.py's original behavior exactly.")
    parser.add_argument("--fedprox_mu", type=float, default=0.01,
                         help="FedProx proximal term weight (only used when --algorithm fedprox). "
                              "Matches src/fl config's fedprox_mu default.")
    parser.add_argument("--fedmoon_mu", type=float, default=1.0,
                         help="FedMoon contrastive term weight (only used when --algorithm fedmoon). "
                              "Matches src/fl config's fedmoon_mu default.")
    parser.add_argument("--fedmoon_temperature", type=float, default=0.5,
                         help="FedMoon contrastive temperature (only used when --algorithm fedmoon). "
                              "Matches src/fl config's fedmoon_temperature default.")
    parser.add_argument("--global_epoch", type=int, default=50)
    parser.add_argument("--local_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_retrain_baseline", action="store_true",
                         help="Also run retrain_domain.fl_retrain_domain with the SAME --algorithm, "
                              "for a fair matched-algorithm baseline (existing plain-FedAvg retrain "
                              "checkpoints are NOT a fair baseline for fedbn/fedprox comparisons).")
    parser.add_argument("--run_relearn_probe", action="store_true", default=True,
                         help="Run the relearn (ReA) probe (src/fu/relearn.py, unchanged/algorithm-agnostic).")
    parser.add_argument("--relearn_rounds", type=int, default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--proxy_frac", type=float, default=0.2,
                         help="Fraction of each hospital's train/test samples held out as a "
                              "proxy pool for shadow-model MIA training (never used by the real "
                              "run). Matches src/data/proxy_split.py's CIFAR-10 default.")
    parser.add_argument("--n_shadow", type=int, default=3,
                         help="Number of shadow models to train for MIA (each re-runs the full "
                              "unlearning phase on proxy data — expensive; CIFAR-10 uses 5, we "
                              "default lower here given the extra per-round cost of real images).")
    parser.add_argument("--mia_attack_epochs", type=int, default=None,
                         help="Epochs to train the MIA attack classifier for. Defaults to "
                              "--global_epoch, matching the original FUSED-Code convention of "
                              "reusing the FL epoch count (no separate attack-epoch hyperparameter).")
    parser.add_argument("--run_shadow_mia", dest="run_shadow_mia", action="store_true", default=True,
                         help="Run the shadow-model MIA (src/eval/mia.py), in addition to the "
                              "existing lightweight loss-threshold MIA_acc. On by default.")
    parser.add_argument("--no_shadow_mia", dest="run_shadow_mia", action="store_false",
                         help="Skip the (expensive) shadow-model MIA step.")
    return parser.parse_args()


def main():
    args = parse_args()

    fl_config = load_source_run_config(args.source_run, logs_root="logs/fl")
    hospitals = fl_config["hospitals"]
    if args.forget_client not in hospitals:
        raise ValueError(
            f"--forget_client '{args.forget_client}' is not in source run's hospitals "
            f"{hospitals}. Pick one of these exactly."
        )
    forget_client_idx = [hospitals.index(args.forget_client)]

    run_id = args.run_id or make_run_id(
        f"fu_domain_{args.algorithm}_{args.forget_client.replace('_Dataset', '').lower()}_oral"
    )
    dirs = resolve_run_dirs(run_id, "logs/fu_domain", "checkpoints/fu_domain", "outputs/fu_domain")

    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])
    logger.info(f"Forking from source FL run: {args.source_run}")
    logger.info(f"Forgetting hospital: {args.forget_client} (index {forget_client_idx[0]} of {hospitals})")
    logger.info(f"Phase-2 algorithm: {args.algorithm}"
                + (f" (mu={args.fedprox_mu})" if args.algorithm == "fedprox" else ""))
    logger.info(f"global_epoch={args.global_epoch}, local_epoch={args.local_epoch}, "
                f"batch_size={args.batch_size}")

    device = fl_config["device"] if torch.cuda.is_available() and fl_config["device"] == "cuda" else "cpu"
    torch.manual_seed(fl_config["seed"])

    merged_config = dict(fl_config)
    merged_config.update({
        "forget_client": args.forget_client,
        "phase2_algorithm": args.algorithm,
        "phase2_fedprox_mu": args.fedprox_mu,
        "global_epoch": args.global_epoch,
        "local_epoch": args.local_epoch,
        "batch_size": args.batch_size,
        "source_run": args.source_run,
    })
    save_config_snapshot(merged_config, os.path.join(dirs["log_dir"], "config.snapshot.yaml"))

    source_checkpoint_dir = os.path.join(fl_config["checkpoints_root"], args.source_run)
    source_checkpoint_path = get_checkpoint_path(source_checkpoint_dir, "best")

    def model_builder():
        # This LoRA pipeline hardcodes target_modules for resnet18's layer
        # naming (see CIFAR10_TARGET_MODULES in src/models/resnet_lora.py:
        # "layer4.0.conv2", "layer4.1.conv1", ...), so it can only ever be
        # correct for a resnet18 source checkpoint. Fail loudly here rather
        # than silently building a resnet18 for a run that actually trained
        # a different architecture (fl_config["model"]) — use
        # run_fu_cli_domain.py (architecture-generic) for non-resnet18 runs.
        source_model_name = fl_config.get("model", "resnet18")
        if source_model_name.lower() != "resnet18":
            raise ValueError(
                f"run_fu_lora_domain.py only supports resnet18 source checkpoints "
                f"(LoRA target_modules are hardcoded to resnet18's layer names), "
                f"but --source_run '{args.source_run}' was trained with "
                f"model='{source_model_name}'. Use run_fu_cli_domain.py instead — "
                f"it's architecture-generic (Critical Layer Identification + "
                f"sparse adapters work over any backbone)."
            )
        return build_resnet18_cifar10(num_classes=fl_config["num_classes"],
                                       pretrained=fl_config.get("pretrained", True))

    source_model = load_checkpoint_into_new_model(model_builder, source_checkpoint_path, device=device)
    logger.info(f"Loaded source checkpoint (read-only) from {source_checkpoint_path}")

    train_samples = index_dataset(merged_config["dataset_path"], "Train", hospitals)
    test_samples = index_dataset(merged_config["dataset_path"], "Test", hospitals)
    hospital_to_idx = {h: i for i, h in enumerate(hospitals)}

    train_transform = build_transforms(merged_config["image_size"], train=True,
                                        augmentation=merged_config["augmentation"])
    eval_transform = build_transforms(merged_config["image_size"], train=False)

    train_partitions = build_client_partitions(
        train_samples, hospitals, merged_config["client_split"],
        merged_config.get("clients_per_hospital", 1), merged_config["seed"],
    )
    test_partitions = build_client_partitions(
        test_samples, hospitals, merged_config["client_split"],
        merged_config.get("clients_per_hospital", 1), merged_config["seed"],
    )

    # Carve out a held-out proxy pool per hospital, structurally identical
    # to the real train/test split, used ONLY for shadow-model MIA training
    # below. `train_partitions`/`test_partitions` are reassigned to the
    # "main" (post-carve) partitions, so every downstream use of them in
    # this script — real training, real evaluation — never touches proxy
    # data, keeping the shadow models' membership ground truth clean.
    train_partitions, proxy_train_partitions = carve_proxy_partitions(
        train_partitions, args.proxy_frac, merged_config["seed"],
    )
    test_partitions, proxy_test_partitions = carve_proxy_partitions(
        test_partitions, args.proxy_frac, merged_config["seed"],
    )

    train_oral_datasets = partitions_to_datasets(
        train_partitions, train_transform, merged_config["load_metadata"], hospital_to_idx,
    )
    test_oral_datasets = partitions_to_datasets(
        test_partitions, eval_transform, merged_config["load_metadata"], hospital_to_idx,
    )

    all_clean_client_loaders = [
        as_tensor_pair_loader(ds, merged_config["batch_size"], shuffle=True)
        for ds in train_oral_datasets
    ]
    all_test_loaders = [
        as_tensor_pair_loader(ds, merged_config["batch_size"], shuffle=False)
        for ds in test_oral_datasets
    ]
    client_data_sizes = [len(ds) for ds in train_oral_datasets]

    logger.info(f"Built {len(all_clean_client_loaders)} hospital train loaders, "
                f"{len(all_test_loaders)} hospital test loaders, in order {hospitals}, "
                f"sizes={client_data_sizes}.")

    # Proxy loaders — held-out data for shadow-model MIA training only (see
    # carve_proxy_partitions above). Same loader format (tuple-yielding) as
    # the real loaders above, since they feed the same forget_client_train_domain
    # call inside the shadow_fn closure below.
    proxy_train_oral_datasets = partitions_to_datasets(
        proxy_train_partitions, train_transform, merged_config["load_metadata"], hospital_to_idx,
    )
    proxy_test_oral_datasets = partitions_to_datasets(
        proxy_test_partitions, eval_transform, merged_config["load_metadata"], hospital_to_idx,
    )
    proxy_train_loaders = [
        as_tensor_pair_loader(ds, merged_config["batch_size"], shuffle=True)
        for ds in proxy_train_oral_datasets
    ]
    proxy_test_loaders = [
        as_tensor_pair_loader(ds, merged_config["batch_size"], shuffle=False)
        for ds in proxy_test_oral_datasets
    ]
    proxy_client_data_sizes = [len(ds) for ds in proxy_train_oral_datasets]
    logger.info(f"Built {len(proxy_train_loaders)} proxy hospital train loaders "
                f"(proxy_frac={args.proxy_frac}), sizes={proxy_client_data_sizes}.")

    logger.info(f"Running FUSED forget_client_train_domain (algorithm={args.algorithm})...")
    unlearned_model, fu_history = forget_client_train_domain(
        trained_global_model=source_model,
        all_clean_client_loaders=all_clean_client_loaders,
        attacked_test_loaders=all_test_loaders,
        forget_client_idx=forget_client_idx,
        fused_iterations=args.global_epoch,
        local_epochs=args.local_epoch,
        learning_rate=args.learning_rate,
        device=device,
        test_batch_size=args.batch_size,
        algorithm=args.algorithm,
        fedprox_mu=args.fedprox_mu,
        fedmoon_mu=args.fedmoon_mu,
        fedmoon_temperature=args.fedmoon_temperature,
        client_data_sizes=client_data_sizes,
        logger=logger,
    )

    trainable, total = count_trainable_parameters(unlearned_model)
    logger.info(f"FUSED unlearning complete. LoRA trainable params: {trainable}/{total} "
                f"({100 * trainable / max(1, total):.4f}%).")

    relearn_result = None
    if args.run_relearn_probe:
        relearn_rounds = args.relearn_rounds or args.global_epoch
        logger.info(f"Running relearn (ReA) probe for {relearn_rounds} rounds (algorithm-agnostic)...")
        _, relearn_result = relearn_mod.relearn_unlearning_knowledge(
            unlearned_model=unlearned_model,
            all_clean_client_loaders=all_clean_client_loaders,
            attacked_test_loaders=all_test_loaders,
            forget_client_idx=forget_client_idx,
            relearn_rounds=relearn_rounds,
            local_epochs=args.local_epoch,
            learning_rate=args.learning_rate,
            device=device,
            test_batch_size=args.batch_size,
            logger=logger,
        )
        logger.info(f"ReA (post-relearn forget-client accuracy) = {relearn_result['ReA']:.4f}")

    retrain_result = None
    if args.run_retrain_baseline:
        logger.info(f"Running MATCHED-algorithm retrain baseline ({args.algorithm}) "
                     f"for {args.global_epoch} rounds...")
        fresh_model = model_builder().to(device)
        retrained_model, retrain_result = fl_retrain_domain(
            init_global_model=fresh_model,
            all_clean_client_loaders=all_clean_client_loaders,
            attacked_test_loaders=all_test_loaders,
            forget_client_idx=forget_client_idx,
            global_epochs=args.global_epoch,
            local_epochs=args.local_epoch,
            learning_rate=args.learning_rate,
            device=device,
            test_batch_size=args.batch_size,
            algorithm=args.algorithm,
            fedprox_mu=args.fedprox_mu,
            fedmoon_mu=args.fedmoon_mu,
            fedmoon_temperature=args.fedmoon_temperature,
            client_data_sizes=client_data_sizes,
            logger=logger,
        )
        save_checkpoint(retrained_model, dirs["checkpoint_dir"], "retrain_baseline_model",
                         extra={"forget_client": args.forget_client, "algorithm": args.algorithm,
                                "history": retrain_result})

    # FIX: previously built from `test_samples` — the RAW, pre-carve sample
    # list indexed at the top of this function — which silently included
    # the proxy_frac slice that carve_proxy_partitions() carved out
    # specifically to stay isolated from the real run (shadow-MIA-only).
    # Build from the post-carve `test_oral_datasets` instead, exactly like
    # every other "real" evaluation set in this script, so the "final
    # global test" metric never touches proxy data.
    global_test_dataset = ConcatDataset(test_oral_datasets)
    global_test_loader = as_tensor_pair_loader(global_test_dataset, args.batch_size, shuffle=False)
    final_loss, final_acc = evaluate(unlearned_model, global_test_loader, device)
    logger.info(f"Final unlearned model: global test loss={final_loss:.4f}, acc={final_acc:.4f}")

    # FIX: symmetric final evaluation (see identical fix in run_fu_cli_domain.py).
    # NOTE: run_standard_evaluation/run_unlearning_evaluation expect dict-format
    # batches ({"image": ..., "label": ...}), unlike every other loader in this
    # script (which yields (image, label) tuples for the tuple-based training/
    # test_client_forget/evaluate() code paths). Use concat_as_dict_loader /
    # a plain DataLoader over the OralCancerDataset directly for these two calls
    # only — do not swap in the tuple-based loaders here (see the TypeError this
    # previously caused: batch["image"] on a tuple/list, not a dict).
    global_test_loader_dict = DataLoader(global_test_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=2)
    remember_idx = [i for i in range(len(hospitals)) if i != forget_client_idx[0]]
    remember_test_loader = concat_as_dict_loader(
        [test_oral_datasets[i] for i in remember_idx], args.batch_size, shuffle=False,
    )
    forget_test_loader = concat_as_dict_loader(
        [test_oral_datasets[forget_client_idx[0]]], args.batch_size, shuffle=False,
    )
    # FIX: MIA's member set must be data the model was actually TRAINED on
    # (see compute_mia_accuracy's docstring). forget_test_loader is the
    # forget hospital's TEST split, which — like every test split — was
    # never in any training set. The forget client's TRAIN split
    # (train_oral_datasets, post-carve), on the other hand, WAS trained on
    # during Phase 1 (source_model was forked from a Phase-1 FL run trained
    # over all hospitals, forget client included), so it's the correct
    # member set for asking "does the unlearned model still leak whether it
    # saw this data?".
    forget_train_loader_dict = concat_as_dict_loader(
        [train_oral_datasets[forget_client_idx[0]]], args.batch_size, shuffle=False,
    )
    run_standard_evaluation(
        unlearned_model, global_test_loader_dict, device, fl_config["num_classes"],
        logger, step=args.global_epoch, tag_prefix="eval",
    )
    unlearning_eval = run_unlearning_evaluation(
        unlearned_model, remember_test_loader, forget_test_loader, device, logger,
        step=args.global_epoch,
        relearn_steps=merged_config.get("relearn_steps", 50),
        relearn_lr=merged_config.get("relearn_lr", 1e-3),
        nonmember_loader=remember_test_loader,
        member_loader=forget_train_loader_dict,
        before_unlearning_acc=None,
    )
    logger.info(
        f"[symmetric eval] RA={unlearning_eval['RA']:.4f} FA={unlearning_eval['FA']:.4f} "
        f"ReA={unlearning_eval['ReA']:.4f}"
        + (f" MIA_acc={unlearning_eval['MIA_acc']:.4f}" if "MIA_acc" in unlearning_eval else "")
    )

    # Shadow-model MIA (src/eval/mia.py) — the paper-faithful, shadow-model
    # attack, stronger evidence for a privacy claim than the lightweight
    # loss-threshold MIA_acc above (kept as-is for continuity/comparison).
    # Shadow models start from the SAME already-trained source_model and
    # re-run ONLY forget_client_train_domain (same algorithm/hyperparameters
    # as the real run) on the held-out proxy data carved out earlier —
    # never touching the real train/test data used for the run above.
    mia_shadow_acc = None
    mia_shadow_per_client = None
    if args.run_shadow_mia:
        mia_attack_epochs = args.mia_attack_epochs or args.global_epoch
        logger.info(f"Running shadow-model MIA (n_shadow={args.n_shadow}, "
                    f"attack_epochs={mia_attack_epochs}, proxy_frac={args.proxy_frac})...")

        def shadow_fn():
            shadow_model, _ = forget_client_train_domain(
                trained_global_model=copy.deepcopy(source_model),
                all_clean_client_loaders=proxy_train_loaders,
                attacked_test_loaders=proxy_test_loaders,
                forget_client_idx=forget_client_idx,
                fused_iterations=args.global_epoch,
                local_epochs=args.local_epoch,
                learning_rate=args.learning_rate,
                device=device,
                test_batch_size=args.batch_size,
                algorithm=args.algorithm,
                fedprox_mu=args.fedprox_mu,
                fedmoon_mu=args.fedmoon_mu,
                fedmoon_temperature=args.fedmoon_temperature,
                client_data_sizes=proxy_client_data_sizes,
            )
            return shadow_model

        mia_shadow_acc, mia_shadow_per_client = membership_inference_attack(
            unlearned_model,
            [all_clean_client_loaders, all_test_loaders],
            [proxy_train_loaders, proxy_test_loaders],
            forget_client_idx, fl_config["num_classes"], args.n_shadow, shadow_fn, device,
            mia_attack_epochs, args.batch_size,
        )
        logger.log_scalar("eval/unlearning/MIA_acc_shadow", mia_shadow_acc, args.global_epoch)
        logger.set_final_metric("eval/unlearning/MIA_acc_shadow", mia_shadow_acc)
        logger.info(f"Shadow-model MIA accuracy = {mia_shadow_acc:.4f} (per-client: {mia_shadow_per_client})")

    save_checkpoint(
        unlearned_model, dirs["checkpoint_dir"], "unlearned_model",
        extra={
            "config": merged_config, "run_id": run_id, "source_run": args.source_run,
            "algorithm": args.algorithm, "fedprox_mu": args.fedprox_mu,
            "forget_client": args.forget_client, "forget_client_idx": forget_client_idx,
            "fu_history": fu_history, "relearn_result": relearn_result,
            "final_global_test_loss": final_loss, "final_global_test_acc": final_acc,
            "mia_shadow_acc": mia_shadow_acc, "mia_shadow_per_client": mia_shadow_per_client,
        },
    )
    logger.info(f"Saved unlearned model to {dirs['checkpoint_dir']}/unlearned_model.pt")

    logger.close()
    print(f"\nDone. run_id = {run_id} (source_run = {args.source_run}, algorithm = {args.algorithm}, "
          f"forgot {args.forget_client})")


if __name__ == "__main__":
    main()
