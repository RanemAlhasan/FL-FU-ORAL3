#!/bin/bash -l

#SBATCH --job-name=bn_domain
#SBATCH --partition=gpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Six independent jobs: array indexes 0 through 5
#SBATCH --array=0-5

#SBATCH --output=slurm_logs/domain_sensitivity/output_bn_domain_%A_%a.txt
#SBATCH --error=slurm_logs/domain_sensitivity/error_bn_domain_%A_%a.txt

set -euo pipefail


# =============================================================================
# Usage
# =============================================================================
#
# mkdir -p slurm_logs/domain_sensitivity
#
# sbatch \
#   scripts/research_run/domain_sensitivity/02_bn_domain_sensitivity_array.sh \
#   fl_fedbn_oral_<JOB_ID>
#
# %A = parent SLURM array job ID
# %a = array task ID from 0 to 5
# =============================================================================


# =============================================================================
# Input
# =============================================================================

SOURCE_RUN="${1:-}"

if [ -z "${SOURCE_RUN}" ]; then
    echo "ERROR: Missing FedBN source run ID."
    echo ""
    echo "Usage:"
    echo "  sbatch $0 fl_fedbn_oral_<JOB_ID>"
    exit 1
fi


# =============================================================================
# Environment
# =============================================================================

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh

conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR="/export/home/ralhasan/fl-fu-oral3"

cd "${PROJECT_DIR}"


# =============================================================================
# Experiment combinations
# =============================================================================

HOSPITALS=(
    "Canada_Dataset"
    "India_Dataset"
    "Spain_Dataset"
    "Canada_Dataset"
    "India_Dataset"
    "Spain_Dataset"
)

POOLING_METHODS=(
    "equal_hospital"
    "equal_hospital"
    "equal_hospital"
    "sample_weighted"
    "sample_weighted"
    "sample_weighted"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"

FORGET_CLIENT="${HOSPITALS[$TASK_ID]}"
POOLING="${POOLING_METHODS[$TASK_ID]}"

METRIC="sym_kl"
TOP_K=15
DEVICE="cpu"

ANALYSIS_SCRIPT="scripts/analyze_bn_domain_sensitivity.py"

OUTPUT_ROOT="outputs/domain_sensitivity/${POOLING}"


# =============================================================================
# Validation
# =============================================================================

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "ERROR: Analysis script not found:"
    echo "  ${PROJECT_DIR}/${ANALYSIS_SCRIPT}"
    exit 1
fi

SOURCE_CONFIG="logs/fl/${SOURCE_RUN}/config.snapshot.yaml"

if [ ! -f "${SOURCE_CONFIG}" ]; then
    echo "ERROR: Source-run configuration not found:"
    echo "  ${SOURCE_CONFIG}"
    exit 1
fi

SOURCE_CHECKPOINTS="checkpoints/fl/${SOURCE_RUN}/per_hospital"

if [ ! -d "${SOURCE_CHECKPOINTS}" ]; then
    echo "ERROR: FedBN per-hospital checkpoints not found:"
    echo "  ${SOURCE_CHECKPOINTS}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"


# =============================================================================
# Job information
# =============================================================================

echo "================================================================================"
echo "FEDBN DOMAIN-SENSITIVITY ARRAY JOB"
echo "================================================================================"
echo "Array job ID:       ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:      ${SLURM_ARRAY_TASK_ID}"
echo "SLURM job ID:       ${SLURM_JOB_ID}"
echo "Source run:         ${SOURCE_RUN}"
echo "Forgotten hospital: ${FORGET_CLIENT}"
echo "Pooling method:     ${POOLING}"
echo "Metric:             ${METRIC}"
echo "Top K:              ${TOP_K}"
echo "Device:             ${DEVICE}"
echo "Output root:        ${OUTPUT_ROOT}"
echo "Node:               ${SLURMD_NODENAME:-unknown}"
echo "Python:             $(which python3)"
echo "Started:            $(date --iso-8601=seconds)"
echo "================================================================================"
echo ""


# =============================================================================
# Run one array-task experiment
# =============================================================================

python3 -u "${ANALYSIS_SCRIPT}" \
    --source_run "${SOURCE_RUN}" \
    --forget_client "${FORGET_CLIENT}" \
    --metric "${METRIC}" \
    --pooling "${POOLING}" \
    --top_k "${TOP_K}" \
    --output_root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"


# =============================================================================
# Completion
# =============================================================================

HOSPITAL_SHORT="${FORGET_CLIENT/_Dataset/}"

RESULT_DIR="${OUTPUT_ROOT}/${SOURCE_RUN}/${HOSPITAL_SHORT}"

echo ""
echo "================================================================================"
echo "DOMAIN-SENSITIVITY JOB COMPLETED"
echo "================================================================================"
echo "Forgotten hospital: ${FORGET_CLIENT}"
echo "Pooling method:     ${POOLING}"
echo "Finished:           $(date --iso-8601=seconds)"
echo ""
echo "Results:"
echo "  ${RESULT_DIR}/domain_sensitivity.csv"
echo "  ${RESULT_DIR}/domain_sensitivity.json"
echo "================================================================================"