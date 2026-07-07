#!/bin/bash -l

#SBATCH -J fu_domain_india_fedprox
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_domain/output_fu_domain_india_fedprox_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_domain/error_fu_domain_india_fedprox_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# CONTRIBUTION: FUSED unlearning with domain-adaptation-aware aggregation
# (fedprox) applied DURING the unlearning phase itself, forgetting India_Dataset.
# Source checkpoint is the plain-FedAvg Phase-1 run (clean ablation: only
# Phase 2's algorithm varies, Phase 1 is held fixed).
# --fedprox_mu is always passed; it is silently ignored unless --algorithm
# is fedprox (see scripts/run_fu_lora_domain.py).
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
SOURCE_RUN="fl_fedavg_oral_329811"

cd "$PROJECT_DIR"

mkdir -p logs/fu_domain checkpoints/fu_domain outputs/fu_domain slurm_logs/fu_domain

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

if [ ! -f "checkpoints/fl/${SOURCE_RUN}/best.pt" ]; then
    echo "ERROR: checkpoints/fl/${SOURCE_RUN}/best.pt not found. Has that FL run completed?"
    exit 1
fi

echo ""
echo "=== FUSED unlearning, algorithm=fedprox, forget=India_Dataset ==="

python3 scripts/run_fu_lora_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client India_Dataset \
    --algorithm fedprox \
    --fedprox_mu 0.01 \
    --global_epoch 50 \
    --local_epoch 3 \
    --batch_size 16 \
    --run_id "fu_domain_fedprox_india_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fu_domain_fedprox_india_oral_${SLURM_JOB_ID}"
