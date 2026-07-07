# Corrected Table 1 — Cifar10-ResNet18, Client Unlearning

Final values to cite, combined from two run directories:

| Metric | Retrain | FUSED | Source run |
|--------|---------|-------|------------|
| RA     | 0.6252  | 0.6141 | fused_cifar10_full_327966 |
| FA     | 0.0307  | 0.0475 | fused_cifar10_full_327966 |
| ReA    | 0.5254  | 0.5950 | fused_cifar10_resume_rea_330959 (CORRECTED) |
| MIA    | 0.6646  | 0.7141 | fused_cifar10_full_327966 |

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

Both bugs were fixed in `scripts/run_fused_cifar10.py` (commit/edit dated
2026-06-29) and the resume script `scripts/resume_rea_only.py` was used to
recompute ReA from the existing `fused_model.pt` / `retrain_model.pt`
checkpoints in `fused_cifar10_full_327966`, without rerunning Phase A/B/MIA
(seed=50, same Dirichlet client partition, fully reproducible — see
resume_rea_only.py docstring for why this is safe). That rerun is
`fused_cifar10_resume_rea_330959`.

RA, FA, and MIA from the original run are unaffected by this bug and
remain valid as-is.

## Known remaining discrepancy (not a bug, noted for write-up)

The paper reports Retrain ReA (0.49) > FUSED ReA (0.42) — i.e. FUSED
forgets more thoroughly than full retraining, the paper's core claim.
This reproduction shows the opposite ordering: FUSED ReA (0.595) >
Retrain ReA (0.525). Magnitude gaps from the paper (FUSED: +17.5pt,
Retrain: +3.6pt) are in a similar range to this reproduction's other
RA/FA run-to-run variance. Not yet investigated further; candidate
explanations include single-seed variance or an interaction between
LoRA-only relearning and this specific Dirichlet partition draw.
