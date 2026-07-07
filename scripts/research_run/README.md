# Full-Scale Research Run Plan

10 real runs (50 global rounds / 3 local epochs each), comparing 4 FL
algorithms, then FUSED unlearning vs retraining across all 3 hospitals.

## Sequencing

### Step 1 — Submit all 4 Phase-1 jobs now, in parallel (independent of each other)

```bash
cd ~/fl-fu-oral3
sbatch scripts/research_run/01_fl_fedavg.sh
sbatch scripts/research_run/02_fl_fedprox.sh
sbatch scripts/research_run/03_fl_fedbn.sh
sbatch scripts/research_run/04_fl_fedmoon.sh
```

Track them:
```bash
squeue -u $USER
```

Each prints its own `run_id` at the end (e.g. `fl_fedavg_oral_<jobid>`) —
note all 4 run_ids from each job's `output_*.txt` once they finish.

### Step 2 — Compare the 4 Phase-1 results, pick a winner

Once all 4 have finished:
```bash
python3 scripts/compare_runs.py --run_ids \
    fl_fedavg_oral_<jobid1> \
    fl_fedprox_oral_<jobid2> \
    fl_fedbn_oral_<jobid3> \
    fl_fedmoon_oral_<jobid4> \
    --out outputs/phase1_comparison.csv
```

Also worth checking in TensorBoard directly — overlay all 4 runs and look at:
- `eval/overall/acc` — which algorithm wins overall?
- `eval/per_hospital/India_Dataset/acc` — which algorithm best closes the
  domain-shift gap seen in earlier smoke tests (India lagged well behind
  Spain/Canada)? This is the main question FedBN/FedProx/FedMOON are meant
  to help with.

Pick the run_id with the best overall accuracy AND the most balanced
per-hospital accuracy (a model that's great on Canada but terrible on
India isn't actually the "best" one for this multi-hospital problem, even
if its overall number looks good — overall accuracy is dominated by
whichever hospital has the most test samples).

### Step 3 — Edit and submit the 3 Phase-2 (FUSED) scripts

Open each of `05_fu_spain.sh`, `06_fu_canada.sh`, `07_fu_india.sh` and
replace:
```bash
SOURCE_RUN="REPLACE_WITH_CHOSEN_PHASE1_RUN_ID"
```
with your chosen Phase-1 run_id from Step 2 (the SAME run_id in all three,
so the three forgetting experiments are directly comparable to each other).

```bash
sbatch scripts/research_run/05_fu_spain.sh
sbatch scripts/research_run/06_fu_canada.sh
sbatch scripts/research_run/07_fu_india.sh
```

### Step 4 — Submit the 3 Phase-3 (retrain) scripts

These don't depend on Step 2/3 at all — submit them any time, even
alongside Step 1:
```bash
sbatch scripts/research_run/08_retrain_spain.sh
sbatch scripts/research_run/09_retrain_canada.sh
sbatch scripts/research_run/10_retrain_india.sh
```

### Step 5 — Final comparison across all 10 runs

```bash
python3 scripts/compare_runs.py --run_ids \
    fl_fedavg_oral_<jobid> fl_fedprox_oral_<jobid> fl_fedbn_oral_<jobid> fl_fedmoon_oral_<jobid> \
    fu_spain_oral_<jobid> fu_canada_oral_<jobid> fu_india_oral_<jobid> \
    retrain_spain_oral_<jobid> retrain_canada_oral_<jobid> retrain_india_oral_<jobid> \
    --out outputs/full_study_comparison.csv
```

## What this answers

| Question | Compare |
|---|---|
| Which FL algorithm is best for these 3 non-IID hospitals? | The 4 Phase-1 runs' `eval/overall/acc` and `eval/per_hospital/*/acc` |
| Does FedBN's domain adaptation actually help the India gap? | FedBN's `eval/per_hospital/India_Dataset/acc` vs the other 3 algorithms' |
| Does FUSED forget as well as full retraining, per hospital? | For each hospital: FU run's `eval/unlearning/FA` vs the matching retrain run's `eval/unlearning/FA` (lower = more forgotten; should be close between the two) |
| Does FUSED preserve remaining knowledge better than retraining? | FU's `eval/unlearning/RA` vs retrain's `eval/unlearning/RA` (higher = better; FUSED is *expected* to win here, per the paper's core claim) |
| Is FUSED actually cheaper, as claimed? | FU's `fu/comm_cost_bytes/iteration` + `fu/comp_time_sec/iteration` vs retrain's `fl/comm_cost_bytes/round` + `fl/comp_time_sec/round` (FUSED should be dramatically lower on both) |
| Does forgetting difficulty vary by hospital size/quality? | Compare FA/RA/ReA across the 3 forgotten hospitals (Spain=smallest, Canada=largest, India=worst baseline accuracy) |

## Timing expectations (GPU, post-CUDA-fix)

The smoke test (5 rounds, CPU) took ~1 hour per FL run. With CUDA now
working and 50 rounds (10x the smoke test's round count), expect roughly
proportional scaling on GPU rather than CPU — actual wall-clock will depend
on GPU utilization, but plan for these to take meaningfully longer than the
smoke test even with the speedup, simply because there's 10x the work.
Monitor the first job's pace via its `output_*.txt` round-by-round
timestamps before assuming all 4 will finish in a similar window.
