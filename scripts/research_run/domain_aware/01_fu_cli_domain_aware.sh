#!/bin/bash -l

#SBATCH -J fu_cli_da_canada
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_cli_domain_aware/output_fu_cli_da_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_cli_domain_aware/error_fu_cli_da_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

set -euo pipefail


# =============================================================================
# Usage
# =============================================================================
#
# Canada with default settings:
#
#   sbatch scripts/research_run/domain_aware/01_fu_cli_domain_aware.sh
#
# Another hospital:
#
#   sbatch scripts/research_run/domain_aware/01_fu_cli_domain_aware.sh \
#       India_Dataset
#
# Full arguments:
#
#   sbatch scripts/research_run/domain_aware/01_fu_cli_domain_aware.sh \
#       <FORGET_CLIENT> \
#       <DOMAIN_LAMBDA> \
#       <SOURCE_RUN> \
#       <DOMAIN_SOURCE_RUN>
#
# Example:
#
#   sbatch scripts/research_run/domain_aware/01_fu_cli_domain_aware.sh \
#       Canada_Dataset \
#       0.5 \
#       fl_fedbn_oral_344913 \
#       fl_fedbn_oral_344913
#
# =============================================================================


# =============================================================================
# Experiment inputs
# =============================================================================

FORGET_CLIENT="${1:-Canada_Dataset}"
DOMAIN_LAMBDA="${2:-0.5}"
SOURCE_RUN="${3:-fl_fedbn_oral_344913}"
DOMAIN_SOURCE_RUN="${4:-fl_fedbn_oral_344913}"


# =============================================================================
# Selected experimental configuration
# =============================================================================

ALGORITHM="fedbn"
FEDBN_SOURCE_MODE="retained_bn_average"

# Matched with Phase-1:
# natural sampling + standard cross-entropy.
PHASE2_IMBALANCE_METHOD="standard_ce"

CLI_SCORE_MODE="domain_aware"
DOMAIN_POOLING="equal_hospital"

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
# Paths
# =============================================================================

PROJECT_DIR="/export/home/ralhasan/fl-fu-oral3"

PYTHON="/export/home/ralhasan/anaconda3/envs/fl_fu/bin/python3"

SLURM_LOG_DIR="${PROJECT_DIR}/slurm_logs/fu_cli_domain_aware"


# =============================================================================
# Environment
# =============================================================================

cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python interpreter was not found:"
    echo "  ${PYTHON}"
    exit 1
fi

if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: PyTorch cannot be imported using:"
    echo "  ${PYTHON}"
    exit 1
fi

mkdir -p "${SLURM_LOG_DIR}"


# =============================================================================
# Construct experiment names and paths
# =============================================================================

HOSPITAL_SHORT="${FORGET_CLIENT/_Dataset/}"

HOSPITAL_SLUG="$(
    echo "${HOSPITAL_SHORT}" \
    | tr '[:upper:]' '[:lower:]'
)"

# Examples:
#   0.5  -> 05
#   0.25 -> 025
#   1.0  -> 10
LAMBDA_SLUG="$(
    echo "${DOMAIN_LAMBDA}" \
    | tr -d '.'
)"

RUN_ID="fu_cli_domain_aware_${ALGORITHM}_${HOSPITAL_SLUG}_natural_l${LAMBDA_SLUG}_${SLURM_JOB_ID}"

DOMAIN_SCORES_PATH="outputs/domain_sensitivity/${DOMAIN_POOLING}/${DOMAIN_SOURCE_RUN}/${HOSPITAL_SHORT}/domain_sensitivity.json"

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
    echo "ERROR: FedBN personalized checkpoints were not found:"
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

if [ ! -f "${DOMAIN_SCORES_PATH}" ]; then
    echo "ERROR: Domain-sensitivity score file was not found:"
    echo "  ${DOMAIN_SCORES_PATH}"
    exit 1
fi

# =============================================================================
# Job information
# =============================================================================

echo "================================================================================"
echo "DOMAIN-AWARE FUSED-CLI EXPERIMENT"
echo "================================================================================"
echo "SLURM job ID:          ${SLURM_JOB_ID}"
echo "Run ID:                ${RUN_ID}"
echo "Source run:            ${SOURCE_RUN}"
echo "Domain source run:     ${DOMAIN_SOURCE_RUN}"
echo "Forgotten hospital:    ${FORGET_CLIENT}"
echo "Algorithm:             ${ALGORITHM}"
echo "FedBN source mode:     ${FEDBN_SOURCE_MODE}"
echo "Phase-2 imbalance:     ${PHASE2_IMBALANCE_METHOD}"
echo "CLI score mode:        ${CLI_SCORE_MODE}"
echo "Domain pooling:        ${DOMAIN_POOLING}"
echo "Domain lambda:         ${DOMAIN_LAMBDA}"
echo "Critical layers K:     ${NUM_UNLEARNING_LAYERS}"
echo "Adapter sparsity:      ${ADAPTER_SPARSITY}"
echo "Global iterations:     ${GLOBAL_EPOCHS}"
echo "Local epochs:          ${LOCAL_EPOCHS}"
echo "Batch size:            ${BATCH_SIZE}"
echo "Learning rate:         ${LEARNING_RATE}"
echo "Expected train size:   ${EXPECTED_TRAIN_SAMPLES}"
echo "Expected test size:    ${EXPECTED_TEST_SAMPLES}"
echo "Domain scores:         ${DOMAIN_SCORES_PATH}"
echo "Python:                ${PYTHON}"
echo "PyTorch version:       $("${PYTHON}" -c 'import torch; print(torch.__version__)')"
echo "Node:                  ${SLURMD_NODENAME:-unknown}"
echo "Started:               $(date --iso-8601=seconds)"
echo "================================================================================"

nvidia-smi


# =============================================================================
# Run full domain-aware FUSED-CLI experiment
# =============================================================================

"${PYTHON}" -u scripts/run_fu_cli_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client "${FORGET_CLIENT}" \
    --algorithm "${ALGORITHM}" \
    --fedbn_source_mode "${FEDBN_SOURCE_MODE}" \
    --phase2_imbalance_method "${PHASE2_IMBALANCE_METHOD}" \
    --cli_score_mode "${CLI_SCORE_MODE}" \
    --domain_source_run "${DOMAIN_SOURCE_RUN}" \
    --domain_pooling "${DOMAIN_POOLING}" \
    --domain_scores_path "${DOMAIN_SCORES_PATH}" \
    --domain_lambda "${DOMAIN_LAMBDA}" \
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
# Completion information
# =============================================================================

echo ""
echo "================================================================================"
echo "DOMAIN-AWARE FUSED-CLI COMPLETED"
echo "================================================================================"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Run ID:       ${RUN_ID}"
echo "Hospital:     ${FORGET_CLIENT}"
echo "Finished:     $(date --iso-8601=seconds)"
echo ""
echo "Artifacts:"
echo "  logs/fu_cli_domain_aware/${RUN_ID}"
echo "  checkpoints/fu_cli_domain_aware/${RUN_ID}"
echo "  outputs/fu_cli_domain_aware/${RUN_ID}"
echo "================================================================================"