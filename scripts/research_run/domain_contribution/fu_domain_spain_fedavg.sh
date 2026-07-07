#!/bin/bash -l

#SBATCH -J fu_domain_spain_fedavg
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_domain/output_fu_domain_spain_fedavg_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_domain/error_fu_domain_spain_fedavg_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# CONTRIBUTION: FUSED unlearning, algorithm=fedavg, forgetting Spain_Dataset.
# Source checkpoint is the plain-FedAvg Phase-1 run (clean ablation).
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
SOURCE_RUN="fl_fedavg_oral_329811"

cd "$PROJECT_DIR"

mkdir -p logs/fu_domain checkpoints/fu_domain outputs/fu_domain slurm_logs/fu_domain

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

if [ ! -f "checkpoints/fl/${SOURCE_RUN}/best.pt" ]; then
    echo "ERROR: checkpoints/fl/${SOURCE_RUN}/best.pt not found. Has that FL run completed?"
    exit 1
fi

echo ""
echo "=== FUSED unlearning, algorithm=fedavg, forget=Spain_Dataset ==="

python3 scripts/run_fu_lora_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client Spain_Dataset \
    --algorithm fedavg \
    --fedprox_mu 0.01 \
    --fedmoon_mu 1.0 \
    --fedmoon_temperature 0.5 \
    --global_epoch 50 \
    --local_epoch 3 \
    --batch_size 16 \
    --run_id "fu_domain_fedavg_spain_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fu_domain_fedavg_spain_oral_${SLURM_JOB_ID}"
