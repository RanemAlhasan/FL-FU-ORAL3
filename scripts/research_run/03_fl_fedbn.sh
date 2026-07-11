#!/bin/bash -l
#SBATCH -J fl_fedbn_oral
#SBATCH -o output_fl_fedbn_%j.txt
#SBATCH -e error_fl_fedbn_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
DATASET_PATH=/export/home/ralhasan/fl-fu-oral3/dataset/oral3
cd "$PROJECT_DIR"

mkdir -p logs checkpoints outputs

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Dataset path: ${DATASET_PATH}"
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
echo "=== Phase 1: FL training (FedBN, full scale) ==="
python3 scripts/run_fl.py \
    --config configs/fl_fedbn.yaml \
    --set dataset_path="${DATASET_PATH}" \
    --set global_epochs=50 \
    --set local_epochs=3 \
    --set batch_size=16 \
    --run_id "fl_fedbn_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fl_fedbn_oral_${SLURM_JOB_ID}"