#!/bin/bash -l

#SBATCH -J bn_domain_all
#SBATCH -o output_bn_domain_all_%j.txt
#SBATCH -e error_bn_domain_all_%j.txt

#SBATCH -p gpu-all
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail


# =============================================================================
# Usage:
#
# sbatch scripts/research_run/domain_sensitivity/01_run_all_bn_domain_sensitivity.sh \
#     fl_fedbn_oral_<JOB_ID>
#
# Example:
#
# sbatch scripts/research_run/domain_sensitivity/01_run_all_bn_domain_sensitivity.sh \
#     fl_fedbn_oral_344933
# =============================================================================


SOURCE_RUN="${1:-}"

if [ -z "${SOURCE_RUN}" ]; then
    echo "ERROR: A FedBN source run ID is required."
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

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3

cd "${PROJECT_DIR}"


# =============================================================================
# Validation
# =============================================================================

ANALYSIS_SCRIPT="scripts/analyze_bn_domain_sensitivity.py"

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "ERROR: Analysis script not found:"
    echo "  ${PROJECT_DIR}/${ANALYSIS_SCRIPT}"
    exit 1
fi

SOURCE_CONFIG="logs/fl/${SOURCE_RUN}/config.yaml"
SOURCE_CHECKPOINTS="checkpoints/${SOURCE_RUN}/per_hospital"

if [ ! -f "${SOURCE_CONFIG}" ]; then
    echo "ERROR: Source-run config not found:"
    echo "  ${SOURCE_CONFIG}"
    exit 1
fi

if [ ! -d "${SOURCE_CHECKPOINTS}" ]; then
    echo "ERROR: FedBN per-hospital checkpoints not found:"
    echo "  ${SOURCE_CHECKPOINTS}"
    exit 1
fi


# =============================================================================
# Settings
# =============================================================================

HOSPITALS=(
    "Canada_Dataset"
    "India_Dataset"
    "Spain_Dataset"
)

POOLING_METHODS=(
    "equal_hospital"
    "sample_weighted"
)

METRIC="sym_kl"
TOP_K=15
DEVICE="cpu"


# =============================================================================
# Job information
# =============================================================================

echo "================================================================================"
echo "FEDBN DOMAIN-SENSITIVITY ANALYSIS"
echo "================================================================================"
echo "SLURM job ID:       ${SLURM_JOB_ID:-not_available}"
echo "Source run:         ${SOURCE_RUN}"
echo "Project directory:  ${PROJECT_DIR}"
echo "Python:             $(which python3)"
echo "Metric:             ${METRIC}"
echo "Device:             ${DEVICE}"
echo "Top K:              ${TOP_K}"
echo "Hospitals:          ${HOSPITALS[*]}"
echo "Pooling methods:    ${POOLING_METHODS[*]}"
echo "Started:            $(date --iso-8601=seconds)"
echo "================================================================================"
echo ""


# =============================================================================
# Run all hospital × pooling combinations
# =============================================================================

TOTAL_RUNS=$(( ${#HOSPITALS[@]} * ${#POOLING_METHODS[@]} ))
CURRENT_RUN=0

for POOLING in "${POOLING_METHODS[@]}"; do

    OUTPUT_ROOT="outputs/domain_sensitivity/${POOLING}"

    mkdir -p "${OUTPUT_ROOT}"

    for HOSPITAL in "${HOSPITALS[@]}"; do

        CURRENT_RUN=$((CURRENT_RUN + 1))

        echo ""
        echo "================================================================================"
        echo "Run ${CURRENT_RUN}/${TOTAL_RUNS}"
        echo "Forgotten hospital: ${HOSPITAL}"
        echo "Pooling method:     ${POOLING}"
        echo "Output root:        ${OUTPUT_ROOT}"
        echo "Started:            $(date --iso-8601=seconds)"
        echo "================================================================================"
        echo ""

        python3 -u "${ANALYSIS_SCRIPT}" \
            --source_run "${SOURCE_RUN}" \
            --forget_client "${HOSPITAL}" \
            --metric "${METRIC}" \
            --pooling "${POOLING}" \
            --top_k "${TOP_K}" \
            --output_root "${OUTPUT_ROOT}" \
            --device "${DEVICE}"

        echo ""
        echo "Completed:"
        echo "  Hospital: ${HOSPITAL}"
        echo "  Pooling:  ${POOLING}"
        echo "  Time:     $(date --iso-8601=seconds)"
        echo ""

    done
done


# =============================================================================
# Final result summary
# =============================================================================

echo ""
echo "================================================================================"
echo "ALL DOMAIN-SENSITIVITY ANALYSES COMPLETED"
echo "================================================================================"
echo "Source run: ${SOURCE_RUN}"
echo "Finished:   $(date --iso-8601=seconds)"
echo ""
echo "Generated result directories:"
echo ""

for POOLING in "${POOLING_METHODS[@]}"; do
    for HOSPITAL in "${HOSPITALS[@]}"; do

        HOSPITAL_SHORT="${HOSPITAL/_Dataset/}"

        RESULT_DIR="outputs/domain_sensitivity/${POOLING}/${SOURCE_RUN}/${HOSPITAL_SHORT}"

        echo "  ${RESULT_DIR}"
        echo "    ├── domain_sensitivity.csv"
        echo "    └── domain_sensitivity.json"

    done
done

echo ""
echo "SLURM output:"
echo "  output_bn_domain_all_${SLURM_JOB_ID:-unknown}.txt"
echo ""
echo "SLURM errors:"
echo "  error_bn_domain_all_${SLURM_JOB_ID:-unknown}.txt"
echo "================================================================================"
