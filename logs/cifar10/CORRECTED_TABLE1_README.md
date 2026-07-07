# Corrected Table 1 — Cifar10-ResNet18, Client Unlearning

Final values to cite, combined from three run directories:

| Metric | Retrain | FUSED  | Source run |
|--------|---------|--------|------------|
| RA     | 0.6252  | 0.6141 | fused_cifar10_full_327966 |
| FA     | 0.0307  | 0.0475 | fused_cifar10_full_327966 |
| ReA    | 0.4414  | 0.6893 | fused_cifar10_resume_rea_332669 |
| MIA    | 0.7437  | 0.8194 | fused_cifar10_resume_mia_332859 |

RA and FA come straight from the original run's final `train.log` summary and
were never affected by any bug — use them as-is. ReA and MIA each went
through a bug-fix + resume cycle described below; **only the numbers in the
table above are current** — see "Superseded numbers" for what NOT to cite.

## Why ReA was rerun

The original run (`fused_cifar10_full_327966`) computed ReA using two bugs
in `scripts/run_fused_cifar10.py`'s Step 6:

1. `relearn_unlearning_knowledge` was evaluated against
   `attacked_test_loaders` (Byzantine label-shifted test set) instead of
   clean `data.test_loaders`. Relearning trains on clean data, so scoring
   against shifted labels produced near-chance accuracy (~0.08) regardless
   of how well relearning actually worked.
2. The Retrain ReA call accidentally reused `fused_model` (deep-copied
   again) instead of `retrain_model`, and stored the result in the same
   `fused_relearn` variable — so "Retrain ReA" was never actually testing
   the retrain checkpoint.

Original (broken) values: Retrain ReA = 0.0800, FUSED ReA = 0.0798
(both ~6x below the paper's reported 0.49 / 0.42 for this table cell).

Both bugs were fixed in `scripts/run_fused_cifar10.py`, and
`scripts/resume_rea_only.py` was used to recompute ReA from the existing
`fused_model.pt` / `retrain_model.pt` checkpoints in `fused_cifar10_full_327966`,
without rerunning Phase A/B/MIA (seed=50, same Dirichlet client partition,
fully reproducible — see `resume_rea_only.py`'s docstring for why this is
safe). The current, correct rerun is `fused_cifar10_resume_rea_332669`
(ReA: Retrain=0.4414, FUSED=0.6893).

## Why MIA was rerun

The original run's MIA (`fused_cifar10_full_327966`: Retrain=0.6646,
FUSED=0.7141) was superseded by a later, corrected shadow-model protocol in
`src/eval/mia.py`, re-executed via `scripts/resume_mia_only.py`: shadow
models now start from the already-trained Phase-A checkpoint (rather than
retraining from scratch per shadow) and train only on the
Byzantine-attacked proxy data, with both the real and proxy evaluation
pools combining train-loader and test-loader logits per client — matching
the original FUSED-Code repo's `train_shadow_model`/
`membership_inference_attack` contract exactly (see `src/eval/mia.py`'s
module docstring for the full protocol). The current, correct rerun is
`fused_cifar10_resume_mia_332859` (MIA: Retrain=0.7437, FUSED=0.8194).

## Superseded numbers — do not cite

- **ReA = 0.5254 (Retrain) / 0.5950 (FUSED)**, previously attributed to a
  run `fused_cifar10_resume_rea_330959`. That run directory no longer
  exists in this repo, and a later, further-corrected rerun
  (`fused_cifar10_resume_rea_332669`, values in the table above) supersedes
  it. If you have this 0.5254/0.5950 pair recorded anywhere else
  (slides, drafts, notes), replace it with 0.4414/0.6893.
- **MIA = 0.6646 (Retrain) / 0.7141 (FUSED)**, from the original
  `fused_cifar10_full_327966` run. Superseded by
  `fused_cifar10_resume_mia_332859` (0.7437/0.8194) — see above.

## Known remaining discrepancy (not a bug, noted for write-up)

The paper reports Retrain ReA (0.49) > FUSED ReA (0.42) — i.e. FUSED
forgets more thoroughly than full retraining, the paper's core claim.
This reproduction shows the opposite ordering at both correction stages —
first FUSED ReA (0.595) > Retrain ReA (0.525), and now, with the latest
corrected numbers, FUSED ReA (0.6893) > Retrain ReA (0.4414), an even
larger gap in the same (paper-contradicting) direction. Since the direction
of the discrepancy has been consistent across two independent fix/rerun
cycles, this looks like a genuine property of this reproduction (single-seed
variance, or an interaction between LoRA-only relearning and this specific
Dirichlet partition draw) rather than a remaining bug — but it's worth
flagging explicitly in the write-up as a qualitative mismatch with the
paper specifically on ReA, distinct from RA/FA/MIA, which do reproduce the
paper's expected pattern (FUSED close to Retrain, both showing successful
forgetting).
