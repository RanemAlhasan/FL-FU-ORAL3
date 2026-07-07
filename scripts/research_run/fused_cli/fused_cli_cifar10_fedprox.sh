#!/bin/bash -l

#SBATCH -J fused_cli_c10_fedprox
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fused_cli/output_fused_cli_cifar10_fedprox_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fused_cli/error_fused_cli_cifar10_fedprox_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --time 96:00:00

# =============================================================================
# FUSED CIFAR-10 via the REAL paper method (CLI + sparse adapters), algorithm=fedprox.
# Loads the existing Phase-A checkpoint (fused_cifar10_full_327966), so only
# Phase 2 (FUSED unlearning) + Retrain + ReA + MIA are run here.
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
DATASET_ROOT="${PROJECT_DIR}/dataset/cifar10"
SOURCE_RUN="fused_cifar10_full_327966"

cd "$PROJECT_DIR"

mkdir -p logs/cifar10_cli checkpoints/cifar10_cli outputs/cifar10_cli slurm_logs/fused_cli

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

if [ ! -f "checkpoints/cifar10/${SOURCE_RUN}/phase_a_model.pt" ]; then
    echo "ERROR: checkpoints/cifar10/${SOURCE_RUN}/phase_a_model.pt not found."
    exit 1
fi

echo ""
echo "=== FUSED-CLI CIFAR-10, algorithm=fedprox ==="

python3 scripts/run_fused_cli_cifar10.py \
    --config configs/cifar10_client_unlearning.yaml \
    --source_run "${SOURCE_RUN}" \
    --source_checkpoints_root checkpoints/cifar10 \
    --algorithm fedprox \
    --fedprox_mu 0.01 \
    --fedmoon_mu 1.0 \
    --fedmoon_temperature 0.5 \
    --num_unlearning_layers 4 \
    --adapter_sparsity 0.05 \
    --set dataset_root="${DATASET_ROOT}" \
    --set logs_root=logs/cifar10_cli \
    --set checkpoints_root=checkpoints/cifar10_cli \
    --set outputs_root=outputs/cifar10_cli \
    --run_id "fused_cli_fedprox_cifar10_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fused_cli_fedprox_cifar10_${SLURM_JOB_ID}"
