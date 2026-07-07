#!/bin/bash -l

#SBATCH -J retrain_spain_oral
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/retrain/output_retrain_spain_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/retrain/error_retrain_spain_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# Phase 3, Run 1/3: Retrain-from-scratch baseline, excluding Spain_Dataset.
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
DATASET_PATH="${PROJECT_DIR}/dataset/oral3"

cd "$PROJECT_DIR"

mkdir -p logs/retrain checkpoints/retrain outputs/retrain slurm_logs/retrain

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Dataset path: ${DATASET_PATH}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

echo "Cleaning old Ray state..."
ray stop --force || true
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_REDIS_ADDRESS
export RAY_ADDRESS=""
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

if [ ! -d "${DATASET_PATH}/Train" ] || [ ! -d "${DATASET_PATH}/Test" ]; then
    echo "ERROR: ${DATASET_PATH} does not contain Train/ and Test/ subfolders."
    exit 1
fi

echo ""
echo "=== Phase 3: Retrain-from-scratch baseline (exclude=Spain_Dataset, full scale) ==="

python3 scripts/run_retrain.py \
    --config configs/retrain_baseline.yaml \
    --set dataset_path="${DATASET_PATH}" \
    --forget_client Spain_Dataset \
    --run_id "retrain_spain_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = retrain_spain_oral_${SLURM_JOB_ID}"