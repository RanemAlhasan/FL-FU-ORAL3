#!/bin/bash -l

#SBATCH -J fu_cli_spain_fedprox
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fused_cli/output_fu_cli_spain_fedprox_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fused_cli/error_fu_cli_spain_fedprox_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# FUSED oral3 via the REAL paper method (CLI + sparse adapters), algorithm=fedprox,
# forgetting Spain_Dataset. Source checkpoint is the plain-FedAvg Phase-1 run
# (clean ablation, matches fu_domain_*'s design).
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
SOURCE_RUN="fl_fedavg_oral_329811"

cd "$PROJECT_DIR"

mkdir -p logs/fu_cli_domain checkpoints/fu_cli_domain outputs/fu_cli_domain slurm_logs/fused_cli

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

if [ ! -f "checkpoints/fl/${SOURCE_RUN}/best.pt" ]; then
    echo "ERROR: checkpoints/fl/${SOURCE_RUN}/best.pt not found."
    exit 1
fi

echo ""
echo "=== FUSED-CLI oral3, algorithm=fedprox, forget=Spain_Dataset ==="

python3 scripts/run_fu_cli_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client Spain_Dataset \
    --algorithm fedprox \
    --fedprox_mu 0.01 \
    --fedmoon_mu 1.0 \
    --fedmoon_temperature 0.5 \
    --num_unlearning_layers 4 \
    --adapter_sparsity 0.05 \
    --global_epoch 50 \
    --local_epoch 3 \
    --batch_size 16 \
    --run_id "fu_cli_fedprox_spain_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fu_cli_fedprox_spain_oral_${SLURM_JOB_ID}"
