# Oral FL Learning-Curve Experiment

This package is designed for the current FL-FU-ORAL3 repository.

## What it tests

Train the current oral FL model with nested 20%, 40%, 60%, 80%, and 100%
subsets of the existing Train split while evaluating every run on the exact
same untouched Test split.

The subsets are stratified by hospital AND class and are nested:

`p20 ⊂ p40 ⊂ p60 ⊂ p80 ⊂ p100`

Two already-selected final training settings are supported:

- FedBN + standard cross entropy (`fedbn_stdce`)
- FedAvg + weighted random sampler (`fedavg_wrs`)

## Add the files

From the repository root:

- `configs/learning_curve_fedbn_stdce.yaml`
- `configs/learning_curve_fedavg_wrs.yaml`
- `scripts/learning_curve/create_learning_curve_datasets.py`
- `scripts/learning_curve/run_learning_curve.sh`
- `scripts/learning_curve/collect_learning_curve.py`

## 1. Build the dataset views once

```bash
cd ~/fl-fu-oral3

python3 scripts/learning_curve/create_learning_curve_datasets.py \
  --source dataset/oral3 \
  --output-root dataset/oral3_learning_curve_seed42 \
  --seed 42
```

This uses symbolic links. It does not duplicate or modify the original images.

Inspect the counts:

```bash
column -s, -t < dataset/oral3_learning_curve_seed42/subset_summary.csv | less -S
```

## 2. Prepare Slurm log folder

Do this BEFORE `sbatch` because Slurm opens the output/error paths before the
job script starts:

```bash
mkdir -p slurm_logs/learning_curve
```

## 3. Submit the five FedBN + StdCE runs

```bash
sbatch \
  --export=ALL,SETTING=fedbn_stdce,STUDY_TAG=lc_s42,TRAIN_SEED=42 \
  scripts/learning_curve/run_learning_curve.sh
```

The job is a Slurm array with tasks 20,40,60,80,100.

## 4. Submit the five FedAvg + WRS runs

```bash
sbatch \
  --export=ALL,SETTING=fedavg_wrs,STUDY_TAG=lc_s42,TRAIN_SEED=42 \
  scripts/learning_curve/run_learning_curve.sh
```

## 5. Monitor

```bash
squeue -u $USER
```

To inspect logs:

```bash
tail -f slurm_logs/learning_curve/oral_lc_<ARRAY_JOB_ID>_20.out
```

## 6. Collect results after the jobs finish

```bash
python3 scripts/learning_curve/collect_learning_curve.py \
  --study-tag lc_s42 \
  --settings fedbn_stdce fedavg_wrs
```

Outputs:

- `outputs/learning_curve/lc_s42_fl_learning_curve.csv`
- `outputs/learning_curve/lc_s42_overall_accuracy.png`
- `outputs/learning_curve/lc_s42_macro_f1.png`

## Interpretation

If accuracy and Macro-F1 are still improving substantially between 80% and
100%, the current experiment is likely data-limited and a genuinely larger
dataset is justified.

If the curve has already flattened by 60%-80%, simply adding more data from
the same distribution may provide only limited gains.

For a paper-quality conclusion, repeat training with additional training seeds
after the first screening curve. Keep the subset views fixed so that the only
change is training stochasticity.
