#!/bin/bash -l
#SBATCH -J fucli_favg_ca
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/class_imbalance_fu_cli/output_fucli_favg_ca_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/class_imbalance_fu_cli/error_fucli_favg_ca_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
COMMON_SCRIPT="${PROJECT_DIR}/scripts/research_run/class_imbalance_fu_cli/common.sh"

SOURCE_RUN="fl_fedavg_oral_343638"
FORGET_CLIENT="Canada_Dataset"
FU_ALGORITHM="fedavg"
FEDBN_SOURCE_MODE="global"
RUN_PREFIX="fu_cli_canada_fedavg_wrs_oral"

if [[ ! -f "${COMMON_SCRIPT}" ]]; then
    echo "ERROR: Common runner not found: ${COMMON_SCRIPT}"
    exit 1
fi

source "${COMMON_SCRIPT}"