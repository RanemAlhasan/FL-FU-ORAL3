#!/bin/bash -l
#SBATCH --job-name=oral_lc
#SBATCH --partition=gpu-all
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=20,40,60,80,100
#SBATCH --output=slurm_logs/learning_curve/%x_%A_%a.out
#SBATCH --error=slurm_logs/learning_curve/%x_%A_%a.err

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1

# Avoid stale Ray state from a previous allocation.
ray stop --force >/dev/null 2>&1 || true
unset RAY_ADDRESS || true
unset RAY_JOB_ID || true
export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"

PCT="${SLURM_ARRAY_TASK_ID}"
SETTING="${SETTING:-fedbn_stdce}"
STUDY_TAG="${STUDY_TAG:-lc_s42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_DIR}/dataset/oral3_learning_curve_seed42}"
DATASET_PATH="${DATASET_ROOT}/p${PCT}"

case "${SETTING}" in
    fedbn_stdce)
        CONFIG="configs/learning_curve_fedbn_stdce.yaml"
        ;;
    fedavg_wrs)
        CONFIG="configs/learning_curve_fedavg_wrs.yaml"
        ;;
    *)
        echo "ERROR: unsupported SETTING='${SETTING}'."
        echo "Use fedbn_stdce or fedavg_wrs."
        exit 2
        ;;
esac

if [[ ! -d "${DATASET_PATH}/Train" ]]; then
    echo "ERROR: missing ${DATASET_PATH}/Train"
    exit 1
fi

if [[ ! -d "${DATASET_PATH}/Test" ]]; then
    echo "ERROR: missing ${DATASET_PATH}/Test"
    exit 1
fi

RUN_ID="${STUDY_TAG}_${SETTING}_p${PCT}_seed${TRAIN_SEED}_job${SLURM_ARRAY_JOB_ID}"

echo "============================================================"
echo "Learning-curve FL run"
echo "Run ID:       ${RUN_ID}"
echo "Setting:      ${SETTING}"
echo "Train data:   p${PCT}"
echo "Dataset path: ${DATASET_PATH}"
echo "Train seed:   ${TRAIN_SEED}"
echo "Config:       ${CONFIG}"
echo "SLURM job:    ${SLURM_JOB_ID}"
echo "============================================================"
nvidia-smi || true

python3 scripts/run_fl.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --set "dataset_path=${DATASET_PATH}" \
    --set "seed=${TRAIN_SEED}"

echo ""
echo "Done: ${RUN_ID}"
echo "Metrics: logs/fl/${RUN_ID}/metrics.json"
