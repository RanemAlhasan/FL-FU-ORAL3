#!/bin/bash -l
#SBATCH -J ret_canada_favg_wrs
#SBATCH -o output_retrain_canada_fedavg_wrs_%j.txt
#SBATCH -e error_retrain_canada_fedavg_wrs_%j.txt
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
echo "Source FL run: fl_fedavg_oral_343638"
echo "Forget client: Canada_Dataset"
nvidia-smi

echo "Cleaning old Ray state..."
ray stop --force || true
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_REDIS_ADDRESS
export RAY_ADDRESS=""
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

if [ ! -d "${DATASET_PATH}/Train" ] || [ ! -d "${DATASET_PATH}/Test" ]; then
    echo "ERROR: ${DATASET_PATH} does not contain Train/ and Test/ subfolders."
    exit 1
fi

echo ""
echo "=== Exact retraining: FedAvg + WRS, forget Canada ==="

python3 scripts/run_retrain.py \
  --config configs/retrain_fedavg_wrs.yaml \
  --source_run fl_fedavg_oral_343638 \
  --source_logs_root logs/fl \
  --forget_client Canada_Dataset \
  --set dataset_path="${DATASET_PATH}" \
  --set handle_class_imbalance=true \
  --set imbalance_method=weighted_sampler \
  --run_id "retrain_canada_fedavg_wrs_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = retrain_canada_fedavg_wrs_oral_${SLURM_JOB_ID}"