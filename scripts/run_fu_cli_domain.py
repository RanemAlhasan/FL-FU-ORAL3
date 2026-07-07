#!/usr/bin/env python3
"""
Phase 2 (REAL PAPER METHOD): FUSED unlearning on the 3-hospital oral-cancer
dataset via Critical Layer Identification + sparse adapters (Algorithm 1),
NOT LoRA. This is a NEW, separate pipeline from run_fu_lora_domain.py /
fused_training_domain.py — nothing here modifies those files, so existing
LoRA-based oral3 results are completely unaffected.

Same scope and CLI contract as run_fu_lora_domain.py (see that script's
docstring for the full rationale on --algorithm / --source_run choice) —
only the unlearning MECHANISM differs (CLI+sparse-adapter here vs. LoRA
there). relearn.py is reused unchanged (algorithm/mechanism-agnostic).

Usage:
    python3 scripts/run_fu_cli_domain.py \\
        --source_run fl_fedavg_oral_329811 \\
        --forget_client Spain_Dataset \\
        --algorithm fedbn \\
        --global_epoch 50 --local_epoch 3 --batch_size 16
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.dataset import index_dataset, OralCancerDataset
from src.data.partition import build_client_partitions, partitions_to_datasets
from src.data.transforms import build_transforms
from src.eval.evaluator import run_standard_evaluation, run_unlearning_evaluation
from src.fl.core import evaluate
from src.fu import relearn as relearn_mod
from src.fu.fused_cli_training import run_fused_cli_unlearning
from src.fu.retrain_domain import fl_retrain_domain
from src.models.resnet_lora import build_resnet18_cifar10
from src.utils.checkpoint import get_checkpoint_path, load_checkpoint_into_new_model, save_checkpoint
from src.utils.config import load_source_run_config, make_run_id, resolve_run_dirs, save_config_snapshot
from src.utils.logger import build_logger


class TensorPairDataset(Dataset):
    def __init__(self, oral_dataset: OralCancerDataset):
        self.oral_dataset = oral_dataset

    def __len__(self) -> int:
        return len(self.oral_dataset)

    def __getitem__(self, idx: int):
        item = self.oral_dataset[idx]
        return item["image"], item["label"]


def as_tensor_pair_loader(oral_dataset: OralCancerDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorPairDataset(oral_dataset), batch_size=batch_size, shuffle=shuffle, num_workers=2)


def concat_as_dict_loader(oral_datasets, batch_size: int, shuffle: bool) -> DataLoader:
    """Concatenate several per-hospital OralCancerDataset objects into a single
    DataLoader, WITHOUT the TensorPairDataset bridge: yields OralCancerDataset's
    native dict batches ({"image", "label", "hospital", ...}), which is what
    src/eval/evaluator.py and src/eval/metrics.py expect. Only use this for
    loaders passed to run_standard_evaluation / run_unlearning_evaluation —
    everything else in this script (training, test_client_forget, relearn, the
    plain evaluate() call) expects (image, label) tuples via
    as_tensor_pair_loader and must keep using that."""
    combined = oral_datasets[0] if len(oral_datasets) == 1 else ConcatDataset(oral_datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 (real CLI+sparse-adapter method): FUSED unlearning.")
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--forget_client", required=True)
    parser.add_argument("--algorithm", choices=["fedavg", "fedbn", "fedprox", "fedmoon"], default="fedavg")
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--fedmoon_mu", type=float, default=1.0)
    parser.add_argument("--fedmoon_temperature", type=float, default=0.5)
    parser.add_argument("--num_unlearning_layers", type=int, default=4)
    parser.add_argument("--adapter_sparsity", type=float, default=0.05)
    parser.add_argument("--global_epoch", type=int, default=50)
    parser.add_argument("--local_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_retrain_baseline", action="store_true")
    parser.add_argument("--relearn_rounds", type=int, default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()

    fl_config = load_source_run_config(args.source_run, logs_root="logs/fl")
    hospitals = fl_config["hospitals"]
    if args.forget_client not in hospitals:
        raise ValueError(f"--forget_client '{args.forget_client}' not in {hospitals}.")
    forget_client_idx = [hospitals.index(args.forget_client)]

    run_id = args.run_id or make_run_id(
        f"fu_cli_{args.algorithm}_{args.forget_client.replace('_Dataset', '').lower()}_oral"
    )
    dirs = resolve_run_dirs(run_id, "logs/fu_cli_domain", "checkpoints/fu_cli_domain", "outputs/fu_cli_domain")
    logger = build_logger(run_id, dirs["log_dir"], dirs["tb_dir"])
    logger.info(f"Forking from source FL run: {args.source_run}")
    logger.info(f"Forgetting: {args.forget_client} (index {forget_client_idx[0]} of {hospitals})")
    logger.info(f"Phase-2 mechanism: FUSED-CLI (Algorithm 1), algorithm={args.algorithm}")

    device = fl_config["device"] if torch.cuda.is_available() and fl_config["device"] == "cuda" else "cpu"
    torch.manual_seed(fl_config["seed"])

    merged_config = dict(fl_config)
    merged_config.update({
        "forget_client": args.forget_client, "phase2_algorithm": args.algorithm,
        "phase2_fedprox_mu": args.fedprox_mu, "phase2_fedmoon_mu": args.fedmoon_mu,
        "num_unlearning_layers": args.num_unlearning_layers, "adapter_sparsity": args.adapter_sparsity,
        "global_epoch": args.global_epoch, "local_epoch": args.local_epoch, "batch_size": args.batch_size,
        "source_run": args.source_run, "method": "FUSED-CLI (Algorithm 1, real paper method)",
    })
    save_config_snapshot(merged_config, os.path.join(dirs["log_dir"], "config.snapshot.yaml"))

    source_checkpoint_dir = os.path.join(fl_config["checkpoints_root"], args.source_run)
    source_checkpoint_path = get_checkpoint_path(source_checkpoint_dir, "best")

    def model_builder():
        return build_resnet18_cifar10(num_classes=fl_config["num_classes"], pretrained=fl_config.get("pretrained", True))

    source_model = load_checkpoint_into_new_model(model_builder, source_checkpoint_path, device=device)
    logger.info(f"Loaded source checkpoint (read-only) from {source_checkpoint_path}")

    train_samples = index_dataset(merged_config["dataset_path"], "Train", hospitals)
    test_samples = index_dataset(merged_config["dataset_path"], "Test", hospitals)
    hospital_to_idx = {h: i for i, h in enumerate(hospitals)}

    train_transform = build_transforms(merged_config["image_size"], train=True, augmentation=merged_config["augmentation"])
    eval_transform = build_transforms(merged_config["image_size"], train=False)

    train_partitions = build_client_partitions(
        train_samples, hospitals, merged_config["client_split"],
        merged_config.get("clients_per_hospital", 1), merged_config["seed"],
    )
    test_partitions = build_client_partitions(
        test_samples, hospitals, merged_config["client_split"],
        merged_config.get("clients_per_hospital", 1), merged_config["seed"],
    )

    train_oral_datasets = partitions_to_datasets(train_partitions, train_transform, merged_config["load_metadata"], hospital_to_idx)
    test_oral_datasets = partitions_to_datasets(test_partitions, eval_transform, merged_config["load_metadata"], hospital_to_idx)

    all_clean_client_loaders = [as_tensor_pair_loader(ds, merged_config["batch_size"], shuffle=True) for ds in train_oral_datasets]
    all_test_loaders = [as_tensor_pair_loader(ds, merged_config["batch_size"], shuffle=False) for ds in test_oral_datasets]
    client_data_sizes = [len(ds) for ds in train_oral_datasets]

    logger.info(f"Built {len(all_clean_client_loaders)} hospital train loaders in order {hospitals}, "
                f"sizes={client_data_sizes}")

    logger.info(f"Running FUSED-CLI unlearning (algorithm={args.algorithm})...")
    unlearned_model, fu_history, critical_layers = run_fused_cli_unlearning(
        source_model=source_model,
        all_clean_client_loaders=all_clean_client_loaders,
        attacked_test_loaders=all_test_loaders,
        forget_client_idx=forget_client_idx,
        client_data_sizes=client_data_sizes,
        num_unlearning_layers=args.num_unlearning_layers,
        adapter_sparsity=args.adapter_sparsity,
        fused_iterations=args.global_epoch,
        local_epochs=args.local_epoch,
        learning_rate=args.learning_rate,
        device=device, test_batch_size=args.batch_size,
        algorithm=args.algorithm, fedprox_mu=args.fedprox_mu,
        fedmoon_mu=args.fedmoon_mu, fedmoon_temperature=args.fedmoon_temperature,
        seed=fl_config["seed"], logger=logger,
    )
    logger.info(f"FUSED-CLI unlearning complete. Critical layers: {critical_layers}")

    relearn_rounds = args.relearn_rounds or args.global_epoch
    logger.info(f"Running relearn (ReA) probe for {relearn_rounds} rounds...")
    _, relearn_result = relearn_mod.relearn_unlearning_knowledge(
        unlearned_model=unlearned_model,
        all_clean_client_loaders=all_clean_client_loaders,
        attacked_test_loaders=all_test_loaders,
        forget_client_idx=forget_client_idx,
        relearn_rounds=relearn_rounds, local_epochs=args.local_epoch,
        learning_rate=args.learning_rate, device=device, test_batch_size=args.batch_size, logger=logger,
    )
    logger.info(f"ReA = {relearn_result['ReA']:.4f}")

    retrain_result = None
    if args.run_retrain_baseline:
        logger.info(f"Running MATCHED-algorithm retrain baseline ({args.algorithm})...")
        fresh_model = model_builder().to(device)
        retrained_model, retrain_result = fl_retrain_domain(
            init_global_model=fresh_model,
            all_clean_client_loaders=all_clean_client_loaders,
            attacked_test_loaders=all_test_loaders,
            forget_client_idx=forget_client_idx,
            global_epochs=args.global_epoch, local_epochs=args.local_epoch,
            learning_rate=args.learning_rate, device=device, test_batch_size=args.batch_size,
            algorithm=args.algorithm, fedprox_mu=args.fedprox_mu,
            fedmoon_mu=args.fedmoon_mu, fedmoon_temperature=args.fedmoon_temperature,
            client_data_sizes=client_data_sizes, logger=logger,
        )
        save_checkpoint(retrained_model, dirs["checkpoint_dir"], "retrain_baseline_model",
                         extra={"forget_client": args.forget_client, "algorithm": args.algorithm,
                                "history": retrain_result})

    global_test_dataset = OralCancerDataset(
        test_samples, transform=eval_transform, load_metadata=merged_config["load_metadata"],
        hospital_to_idx=hospital_to_idx,
    )
    global_test_loader = as_tensor_pair_loader(global_test_dataset, args.batch_size, shuffle=False)
    final_loss, final_acc = evaluate(unlearned_model, global_test_loader, device)
    logger.info(f"Final unlearned model: global test loss={final_loss:.4f}, acc={final_acc:.4f}")

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
        before_unlearning_acc=None,
    )
    logger.info(
        f"[symmetric eval] RA={unlearning_eval['RA']:.4f} FA={unlearning_eval['FA']:.4f} "
        f"ReA={unlearning_eval['ReA']:.4f}"
        + (f" MIA_acc={unlearning_eval['MIA_acc']:.4f}" if "MIA_acc" in unlearning_eval else "")
    )

    save_checkpoint(
        unlearned_model, dirs["checkpoint_dir"], "unlearned_model",
        extra={
            "config": merged_config, "run_id": run_id, "source_run": args.source_run,
            "algorithm": args.algorithm, "forget_client": args.forget_client,
            "forget_client_idx": forget_client_idx, "critical_layers": critical_layers,
            "fu_history": fu_history, "relearn_result": relearn_result,
            "final_global_test_loss": final_loss, "final_global_test_acc": final_acc,
        },
    )
    logger.info(f"Saved unlearned model to {dirs['checkpoint_dir']}/unlearned_model.pt")

    logger.close()
    print(f"\nDone. run_id = {run_id} (source_run = {args.source_run}, algorithm = {args.algorithm}, "
          f"forgot {args.forget_client}, critical_layers = {critical_layers})")


if __name__ == "__main__":
    main()
