"""
Evaluation metrics, mirroring the FUSED paper's evaluation protocol
(Sec 5.1 "Evaluations") adapted to our 3-class oral cancer setting.

  - Standard: overall accuracy/loss, per-hospital accuracy/loss
    (domain-shift diagnostic), per-class precision/recall/F1.
  - RA (Remember Accuracy): accuracy on the remember dataset (D^r).
    Should be high.
  - FA (Forget Accuracy): accuracy on the forget dataset (D^u).
    Should be low (ideally near chance level, NOT necessarily 0 — for a
    3-class problem chance level is ~0.33, unlike the paper's many-class
    Cifar100 setting where near-0 is achievable).
  - ReA (Relearn Accuracy): accuracy achieved after a SMALL number of
    fine-tuning steps on the forgotten data, starting from the unlearned
    model. Lower ReA after limited relearning = more thorough forgetting
    (the paper's framing: low ReA means the knowledge was truly erased,
    not just hidden).
  - MIA (Membership Inference Attack, simplified loss-threshold attacker):
    trains a simple threshold classifier on a held-out reference set's
    per-sample loss to decide "was this sample part of the training set",
    then reports its accuracy on the forget set vs a non-member control
    set. Lower MIA accuracy on the forgotten data = less privacy leakage.
    NOTE: this is a lightweight proxy for the paper's MIA evaluation, not a
    full shadow-model attack — adequate for relative comparison across your
    own FL/FU runs, but should not be over-interpreted as an absolute
    privacy guarantee.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES


@torch.no_grad()
def evaluate_overall(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}


@torch.no_grad()
def evaluate_per_hospital(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, Dict[str, float]]:
    model.eval()
    per_hospital_correct: Dict[str, int] = {}
    per_hospital_total: Dict[str, int] = {}
    per_hospital_loss: Dict[str, float] = {}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        hospitals = batch["hospital"]  # list[str], collated by default DataLoader
        logits = model(images)
        losses = F.cross_entropy(logits, labels, reduction="none")
        preds = logits.argmax(dim=1)

        for i, hospital in enumerate(hospitals):
            per_hospital_total[hospital] = per_hospital_total.get(hospital, 0) + 1
            per_hospital_correct[hospital] = per_hospital_correct.get(hospital, 0) + int(preds[i] == labels[i])
            per_hospital_loss[hospital] = per_hospital_loss.get(hospital, 0.0) + losses[i].item()

    return {
        hospital: {
            "acc": per_hospital_correct[hospital] / max(1, per_hospital_total[hospital]),
            "loss": per_hospital_loss[hospital] / max(1, per_hospital_total[hospital]),
            "n": per_hospital_total[hospital],
        }
        for hospital in per_hospital_total
    }


@torch.no_grad()
def evaluate_per_class(model: nn.Module, loader: DataLoader, device: str,
                        num_classes: int = 3) -> Dict[str, Dict[str, float]]:
    model.eval()
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        preds = model(images).argmax(dim=1)

        for c in range(num_classes):
            tp[c] += ((preds == c) & (labels == c)).sum().item()
            fp[c] += ((preds == c) & (labels != c)).sum().item()
            fn[c] += ((preds != c) & (labels == c)).sum().item()
            total_per_class[c] += (labels == c).sum().item()
            correct_per_class[c] += ((preds == c) & (labels == c)).sum().item()

    results = {}
    for c in range(num_classes):
        precision = tp[c] / max(1, tp[c] + fp[c])
        recall = tp[c] / max(1, tp[c] + fn[c])
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        acc = correct_per_class[c] / max(1, total_per_class[c])
        results[CLASS_NAMES[c]] = {
            "acc": float(acc), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "n": int(total_per_class[c]),
        }
    return results


def compute_ra_fa(
    model: nn.Module, remember_loader: DataLoader, forget_loader: DataLoader, device: str,
) -> Dict[str, float]:
    """RA = accuracy on remember data (higher better).
    FA = accuracy on forget data (lower better, relative to chance level)."""
    ra = evaluate_overall(model, remember_loader, device)["acc"] if len(remember_loader.dataset) else float("nan")
    fa = evaluate_overall(model, forget_loader, device)["acc"] if len(forget_loader.dataset) else float("nan")
    return {"RA": ra, "FA": fa}


def compute_relearn_accuracy(
    model: nn.Module,
    forget_loader: DataLoader,
    device: str,
    relearn_steps: int = 50,
    learning_rate: float = 1e-3,
) -> float:
    """ReA: fine-tune a TEMPORARY CLONE of the unlearned model on the forget
    set for a small, fixed number of steps, then measure accuracy on that
    same forget set. A low ReA means the model struggles to quickly relearn
    the forgotten knowledge (good — knowledge was truly overwritten, not
    just masked). This never modifies the actual unlearned model passed in;
    it operates on a throwaway deep copy."""
    import copy as _copy
    relearn_model = _copy.deepcopy(model)
    relearn_model.to(device)
    optimizer = torch.optim.Adam(relearn_model.parameters(), lr=learning_rate)

    relearn_model.train()
    step = 0
    while step < relearn_steps:
        for batch in forget_loader:
            if step >= relearn_steps:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(relearn_model(images), labels)
            loss.backward()
            optimizer.step()
            step += 1
        if len(forget_loader) == 0:
            break

    return evaluate_overall(relearn_model, forget_loader, device)["acc"]


@torch.no_grad()
def compute_mia_accuracy(
    model: nn.Module,
    member_loader: DataLoader,   # samples the model WAS trained on (e.g. forget set, pre-unlearning member)
    nonmember_loader: DataLoader,  # held-out samples the model was NEVER trained on
    device: str,
) -> float:
    """Simplified loss-threshold membership inference attack.

    For each sample, compute the model's per-sample cross-entropy loss.
    Members of the training set typically have LOWER loss than non-members
    (the classic MIA signal). We fit the threshold as the midpoint between
    the two populations' mean losses on a HELD-OUT HALF of each population,
    then report the attacker's classification accuracy on the OTHER half.

    Lower returned accuracy (closer to 0.5) = less privacy leakage.
    """
    model.eval()

    def per_sample_losses(loader: DataLoader) -> np.ndarray:
        losses = []
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            batch_losses = F.cross_entropy(logits, labels, reduction="none")
            losses.extend(batch_losses.cpu().numpy().tolist())
        return np.array(losses)

    member_losses = per_sample_losses(member_loader)
    nonmember_losses = per_sample_losses(nonmember_loader)

    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        return float("nan")

    # BUG FIX: this used to fit the threshold from `member_losses.mean()`/
    # `nonmember_losses.mean()` and then evaluate accuracy on those SAME
    # full arrays — i.e. the attacker was tuned and scored on identical
    # data, self-tuning away from a valid attack-train/attack-test split
    # and inflating (or otherwise biasing, in a sample-size-dependent way)
    # the reported MIA_acc. Split each population in half via a threshold-
    # FIT half and a separately-scored EVAL half instead. Falls back to the
    # old (fit==eval) behavior only when a population has exactly 1 sample
    # (too small to split) so this never crashes on tiny loaders.
    rng = np.random.RandomState(0)

    def fit_eval_split(losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(losses) < 2:
            return losses, losses
        idx = rng.permutation(len(losses))
        half = len(losses) // 2
        return losses[idx[:half]], losses[idx[half:]]

    member_fit, member_eval = fit_eval_split(member_losses)
    nonmember_fit, nonmember_eval = fit_eval_split(nonmember_losses)

    threshold = (member_fit.mean() + nonmember_fit.mean()) / 2.0
    # Attacker predicts "member" when loss < threshold.
    member_preds = member_eval < threshold          # should be True (correct) ideally
    nonmember_preds = nonmember_eval >= threshold    # should be True (correct) ideally

    correct = member_preds.sum() + nonmember_preds.sum()
    total = len(member_eval) + len(nonmember_eval)
    return float(correct / total)
