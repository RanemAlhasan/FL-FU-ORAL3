#!/bin/bash

set -euo pipefail

: "${SOURCE_RUN:?SOURCE_RUN is required}"
: "${FORGET_CLIENT:?FORGET_CLIENT is required}"
: "${FU_ALGORITHM:?FU_ALGORITHM is required}"
: "${FEDBN_SOURCE_MODE:?FEDBN_SOURCE_MODE is required}"
: "${RUN_PREFIX:?RUN_PREFIX is required}"

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
DATASET_PATH="${PROJECT_DIR}/dataset/oral3"

cd "${PROJECT_DIR}"

mkdir -p \
  logs/fu_cli_domain \
  checkpoints/fu_cli_domain \
  outputs/fu_cli_domain \
  slurm_logs/class_imbalance_fu_cli

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Working directory:  $(pwd)"
echo "Python:             $(which python3)"
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Source FL run:      ${SOURCE_RUN}"
echo "Algorithm:          ${FU_ALGORITHM}"
echo "Forgotten hospital: ${FORGET_CLIENT}"
echo "FedBN source mode:  ${FEDBN_SOURCE_MODE}"
echo "Run prefix:         ${RUN_PREFIX}"
echo "============================================================"

nvidia-smi

if [[ ! -d "${DATASET_PATH}/Train" ]] || \
   [[ ! -d "${DATASET_PATH}/Test" ]]; then
    echo "ERROR: Dataset Train/ or Test/ folder is missing."
    exit 1
fi

SOURCE_CHECKPOINT="checkpoints/fl/${SOURCE_RUN}/best.pt"

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: Source checkpoint not found: ${SOURCE_CHECKPOINT}"
    exit 1
fi

if [[ "${FEDBN_SOURCE_MODE}" == "retained_bn_average" ]]; then
    FEDBN_DIR="checkpoints/fl/${SOURCE_RUN}/per_hospital"

    if [[ ! -d "${FEDBN_DIR}" ]]; then
        echo "ERROR: FedBN per-hospital folder not found: ${FEDBN_DIR}"
        exit 1
    fi

    FEDBN_COUNT="$(
        find "${FEDBN_DIR}" \
          -maxdepth 1 \
          -type f \
          -name '*.pt' \
          | wc -l
    )"

    if [[ "${FEDBN_COUNT}" -lt 3 ]]; then
        echo "ERROR: Expected at least 3 FedBN hospital checkpoints."
        echo "Found: ${FEDBN_COUNT}"
        exit 1
    fi
fi

echo ""
echo "Starting FUSED-CLI unlearning..."
echo ""

python3 scripts/run_fu_cli_domain.py \
  --source_run "${SOURCE_RUN}" \
  --forget_client "${FORGET_CLIENT}" \
  --algorithm "${FU_ALGORITHM}" \
  --fedbn_source_mode "${FEDBN_SOURCE_MODE}" \
  --global_epoch 50 \
  --local_epoch 3 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --num_unlearning_layers 4 \
  --adapter_sparsity 0.05 \
  --relearn_rounds 50 \
  --no_shadow_mia \
  --run_id "${RUN_PREFIX}_${SLURM_JOB_ID}"

echo ""
echo "Done."
echo "Run ID: ${RUN_PREFIX}_${SLURM_JOB_ID}"