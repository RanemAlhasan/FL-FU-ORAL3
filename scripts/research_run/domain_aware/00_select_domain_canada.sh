#!/bin/bash -l

#SBATCH -J select_da_canada
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_cli_domain_aware/output_select_da_canada_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_cli_domain_aware/error_select_da_canada_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00

set -euo pipefail

PROJECT_DIR="/export/home/ralhasan/fl-fu-oral3"
PYTHON="/export/home/ralhasan/anaconda3/envs/fl_fu/bin/python3"

cd "${PROJECT_DIR}"

RUN_ID="select_domain_canada_conv_only_${SLURM_JOB_ID}"

echo "================================================================================"
echo "DOMAIN-AWARE CLI SELECTION-ONLY TEST"
echo "================================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Run ID:    ${RUN_ID}"
echo "Python:    ${PYTHON}"
echo "Started:   $(date --iso-8601=seconds)"
echo "================================================================================"

"${PYTHON}" -m py_compile \
  src/fu/fused_cli_training.py \
  src/fu/critical_layers_generic.py \
  src/fu/domain_sensitivity.py \
  scripts/run_fu_cli_domain.py

"${PYTHON}" -u scripts/run_fu_cli_domain.py \
  --source_run fl_fedbn_oral_344913 \
  --forget_client Canada_Dataset \
  --algorithm fedbn \
  --fedbn_source_mode retained_bn_average \
  --phase2_imbalance_method standard_ce \
  --cli_score_mode domain_aware \
  --domain_source_run fl_fedbn_oral_344913 \
  --domain_pooling equal_hospital \
  --domain_lambda 0.5 \
  --num_unlearning_layers 4 \
  --adapter_sparsity 0.05 \
  --global_epoch 50 \
  --local_epoch 3 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --expected_train_samples 2538 \
  --expected_test_samples 637 \
  --selection_log_top_k 15 \
  --cli_selection_only \
  --no_shadow_mia \
  --run_id "${RUN_ID}"

echo "Selection-only test completed: ${RUN_ID}"
