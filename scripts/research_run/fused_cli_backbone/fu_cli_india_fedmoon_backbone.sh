#!/bin/bash -l

#SBATCH -J fu_cli_india_fedmoon_bb
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fused_cli_backbone/output_fu_cli_india_fedmoon_bb_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fused_cli_backbone/error_fu_cli_india_fedmoon_bb_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# NEW (backbone) ablation, additive -- does NOT replace or overwrite
# scripts/research_run/fused_cli/fu_cli_india_fedmoon.sh.
#
# The original fu_cli_india_fedmoon.sh always forks from the plain-FedAvg
# Phase-1 checkpoint (fl_fedavg_oral_329811) by design (isolates the effect
# of Phase-2's aggregation algorithm alone -- see run_fu_cli_domain.py
# docstring). That means it never tests unlearning starting from an
# ACTUALLY fedmoon-personalized backbone, which is what you need for a fair,
# direct comparison against retrain_india_fedmoon (which DOES retrain from a
# real fedmoon model). This script fixes that: it forks from the real
# fedmoon-trained Phase-1 checkpoint (fl_fedmoon_oral_330530).
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
SOURCE_RUN="fl_fedmoon_oral_330530"

cd "$PROJECT_DIR"

mkdir -p logs/fu_cli_domain_backbone checkpoints/fu_cli_domain_backbone outputs/fu_cli_domain_backbone slurm_logs/fused_cli_backbone

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN} (real fedmoon-personalized backbone)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

if [ ! -f "checkpoints/fl/${SOURCE_RUN}/best.pt" ]; then
    echo "ERROR: checkpoints/fl/${SOURCE_RUN}/best.pt not found. Has that FL run completed?"
    exit 1
fi

echo ""
echo "=== FUSED-CLI oral3 (BACKBONE ablation), algorithm=fedmoon, forget=India_Dataset ==="

python3 scripts/run_fu_cli_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client India_Dataset \
    --algorithm fedmoon \
    --fedprox_mu 0.01 \
    --fedmoon_mu 1.0 \
    --fedmoon_temperature 0.5 \
    --num_unlearning_layers 4 \
    --adapter_sparsity 0.05 \
    --global_epoch 50 \
    --local_epoch 3 \
    --batch_size 16 \
    --run_id "fu_cli_fedmoon_india_backbone_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fu_cli_fedmoon_india_backbone_oral_${SLURM_JOB_ID}"
