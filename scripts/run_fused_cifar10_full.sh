#!/bin/bash -l

#SBATCH -J fused_cifar10_full
#SBATCH -o output_fused_cifar10_full_%j.txt
#SBATCH -e error_fused_cifar10_full_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# Full paper-scale run of the FUSED CIFAR-10 client-unlearning reproduction.
#
# Uses configs/cifar10_client_unlearning.yaml's defaults as-is: 50 clients,
# Dirichlet alpha=1.0, 100 global rounds (Phase A), 100 FUSED adapter
# iterations (Phase B), 100 relearn rounds (ReA), MIA enabled with 5 shadow
# models. This is SUBSTANTIAL compute — the MIA step alone re-runs the
# ENTIRE Phase A + Phase B pipeline 5 times for FUSED and 5 times for
# Retrain (10 extra full pipeline runs on top of the main ones). Run the
# smoke test script first (run_fused_cifar10_smoketest.sh) to confirm
# everything works before committing to this.
#
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

# EDIT THIS to wherever you unzip fused_cifar10_repro on the cluster.
PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
cd "$PROJECT_DIR"

# EDIT THIS to your real CIFAR-10 dataset root (the folder that directly
# contains cifar-10-batches-py/).
DATASET_ROOT=/export/home/ralhasan/fl-fu-oral3/dataset/cifar10

mkdir -p logs checkpoints outputs

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Dataset root: ${DATASET_ROOT}"
nvidia-smi

echo ""
echo "=== FUSED CIFAR-10 full reproduction (50 clients, 100 rounds, MIA enabled) ==="
python3 scripts/run_fused_cifar10.py \
    --config configs/cifar10_client_unlearning.yaml \
    --set dataset_root="${DATASET_ROOT}" \
    --run_id "fused_cifar10_full_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fused_cifar10_full_${SLURM_JOB_ID}"
echo "Check logs/fused_cifar10_full_${SLURM_JOB_ID}/metrics.json for the final"
echo "Table-1-style RA/FA/ReA/MIA numbers (eval/unlearning/FUSED/* and eval/unlearning/Retrain/*)."
