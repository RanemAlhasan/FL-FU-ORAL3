#!/bin/bash -l

#SBATCH -J fu_cli_norm
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_cli_normalized/output_fu_cli_norm_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_cli_normalized/error_fu_cli_norm_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

set -euo pipefail


# =============================================================================
# Usage
# =============================================================================
#
# Canada:
#
#   sbatch scripts/research_run/domain_aware/02_fu_cli_normalized.sh
#
# India:
#
#   sbatch scripts/research_run/domain_aware/02_fu_cli_normalized.sh \
#       India_Dataset
#
# Spain:
#
#   sbatch scripts/research_run/domain_aware/02_fu_cli_normalized.sh \
#       Spain_Dataset
#
# Optional custom source:
#
#   sbatch scripts/research_run/domain_aware/02_fu_cli_normalized.sh \
#       Canada_Dataset \
#       fl_fedbn_oral_344913
#
# =============================================================================


# =============================================================================
# Experiment inputs
# =============================================================================

FORGET_CLIENT="${1:-Canada_Dataset}"
SOURCE_RUN="${2:-fl_fedbn_oral_344913}"


# =============================================================================
# Experimental configuration
# =============================================================================

ALGORITHM="fedbn"
FEDBN_SOURCE_MODE="retained_bn_average"

PHASE2_IMBALANCE_METHOD="standard_ce"

# IMPORTANT:
# This is the control experiment.
# It uses mean absolute CLI change but NO domain score.
CLI_SCORE_MODE="mean_abs"

NUM_UNLEARNING_LAYERS=4
ADAPTER_SPARSITY=0.05

GLOBAL_EPOCHS=50
LOCAL_EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=0.005

EXPECTED_TRAIN_SAMPLES=2538
EXPECTED_TEST_SAMPLES=637

SELECTION_LOG_TOP_K=15


# =============================================================================
# Project and environment
# =============================================================================

PROJECT_DIR="/export/home/ralhasan/fl-fu-oral3"

CONDA_SH="/export/home/ralhasan/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/export/home/ralhasan/anaconda3/envs/fl_fu"

cd "${PROJECT_DIR}"

# Load Conda exactly as in the successful domain-aware run.
if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: Conda initialization script not found:"
    echo "  ${CONDA_SH}"
    exit 1
fi

source "${CONDA_SH}"

conda activate "${CONDA_ENV}"

# Resolve Python AFTER activating the environment.
PYTHON="$(which python3)"

export PYTHONUNBUFFERED=1


# =============================================================================
# Verify Python environment
# =============================================================================

echo "Checking Python environment..."
echo "Python executable: ${PYTHON}"

"${PYTHON}" -c '
import sys
print("sys.executable:", sys.executable)

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
'

echo "Python environment OK."
echo ""


# =============================================================================
# Construct run name
# =============================================================================

HOSPITAL_SHORT="${FORGET_CLIENT/_Dataset/}"

HOSPITAL_SLUG="$(
    echo "${HOSPITAL_SHORT}" \
    | tr "[:upper:]" "[:lower:]"
)"

RUN_ID="fu_cli_normalized_${ALGORITHM}_${HOSPITAL_SLUG}_natural_${SLURM_JOB_ID}"


# =============================================================================
# Source paths
# =============================================================================

SOURCE_CONFIG_PATH="logs/fl/${SOURCE_RUN}/config.snapshot.yaml"

SOURCE_CHECKPOINT_DIR="checkpoints/fl/${SOURCE_RUN}/per_hospital"


# =============================================================================
# Input validation
# =============================================================================

if [ ! -f "scripts/run_fu_cli_domain.py" ]; then
    echo "ERROR: Main FU script was not found:"
    echo "  ${PROJECT_DIR}/scripts/run_fu_cli_domain.py"
    exit 1
fi

if [ ! -f "${SOURCE_CONFIG_PATH}" ]; then
    echo "ERROR: Phase-1 configuration was not found:"
    echo "  ${SOURCE_CONFIG_PATH}"
    exit 1
fi

if [ ! -d "${SOURCE_CHECKPOINT_DIR}" ]; then
    echo "ERROR: FedBN personalized checkpoint directory was not found:"
    echo "  ${SOURCE_CHECKPOINT_DIR}"
    exit 1
fi

for hospital in \
    Spain_Dataset \
    Canada_Dataset \
    India_Dataset
do
    checkpoint="${SOURCE_CHECKPOINT_DIR}/${hospital}.pt"

    if [ ! -f "${checkpoint}" ]; then
        echo "ERROR: Required FedBN checkpoint was not found:"
        echo "  ${checkpoint}"
        exit 1
    fi
done

# =============================================================================
# Job information
# =============================================================================

echo "================================================================================"
echo "NORMALIZED FUSED-CLI CONTROL EXPERIMENT"
echo "================================================================================"
echo "SLURM job ID:          ${SLURM_JOB_ID}"
echo "Run ID:                ${RUN_ID}"
echo "Source run:            ${SOURCE_RUN}"
echo "Forgotten hospital:    ${FORGET_CLIENT}"
echo "Algorithm:             ${ALGORITHM}"
echo "FedBN source mode:     ${FEDBN_SOURCE_MODE}"
echo "Phase-2 imbalance:     ${PHASE2_IMBALANCE_METHOD}"
echo "CLI score mode:        ${CLI_SCORE_MODE}"
echo "Domain sensitivity:    DISABLED"
echo "Critical layers K:     ${NUM_UNLEARNING_LAYERS}"
echo "Adapter sparsity:      ${ADAPTER_SPARSITY}"
echo "Global iterations:     ${GLOBAL_EPOCHS}"
echo "Local epochs:          ${LOCAL_EPOCHS}"
echo "Batch size:            ${BATCH_SIZE}"
echo "Learning rate:         ${LEARNING_RATE}"
echo "Expected train size:   ${EXPECTED_TRAIN_SAMPLES}"
echo "Expected test size:    ${EXPECTED_TEST_SAMPLES}"
echo "Python:                ${PYTHON}"
echo "PyTorch version:       $("${PYTHON}" -c 'import torch; print(torch.__version__)')"
echo "Node:                  ${SLURMD_NODENAME:-unknown}"
echo "Started:               $(date --iso-8601=seconds)"
echo "================================================================================"
echo ""

nvidia-smi


# =============================================================================
# Run normalized FUSED-CLI
# =============================================================================

"${PYTHON}" -u scripts/run_fu_cli_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client "${FORGET_CLIENT}" \
    --algorithm "${ALGORITHM}" \
    --fedbn_source_mode "${FEDBN_SOURCE_MODE}" \
    --phase2_imbalance_method "${PHASE2_IMBALANCE_METHOD}" \
    --cli_score_mode "${CLI_SCORE_MODE}" \
    --num_unlearning_layers "${NUM_UNLEARNING_LAYERS}" \
    --adapter_sparsity "${ADAPTER_SPARSITY}" \
    --global_epoch "${GLOBAL_EPOCHS}" \
    --local_epoch "${LOCAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --expected_train_samples "${EXPECTED_TRAIN_SAMPLES}" \
    --expected_test_samples "${EXPECTED_TEST_SAMPLES}" \
    --selection_log_top_k "${SELECTION_LOG_TOP_K}" \
    --no_shadow_mia \
    --run_id "${RUN_ID}"


# =============================================================================
# Completion
# =============================================================================

echo ""
echo "================================================================================"
echo "NORMALIZED FUSED-CLI COMPLETED"
echo "================================================================================"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Run ID:       ${RUN_ID}"
echo "Hospital:     ${FORGET_CLIENT}"
echo "Finished:     $(date --iso-8601=seconds)"
echo ""
echo "Artifacts:"
echo "  logs/fu_cli_normalized/${RUN_ID}"
echo "  checkpoints/fu_cli_normalized/${RUN_ID}"
echo "  outputs/fu_cli_normalized/${RUN_ID}"
echo "================================================================================"