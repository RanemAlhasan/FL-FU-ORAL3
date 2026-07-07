#!/bin/bash -l

#SBATCH -J fu_domain_india_fedbn_bb
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/domain_contribution_backbone/output_fu_domain_india_fedbn_bb_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/domain_contribution_backbone/error_fu_domain_india_fedbn_bb_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# NEW (backbone) ablation, additive -- does NOT replace or overwrite
# scripts/research_run/domain_contribution/fu_domain_india_fedbn.sh.
#
# The original fu_domain_india_fedbn.sh always forks from the plain-FedAvg
# Phase-1 checkpoint (fl_fedavg_oral_329811) by design (isolates the effect
# of Phase-2's aggregation algorithm alone -- see run_fu_lora_domain.py
# docstring). That means it never tests unlearning starting from an
# ACTUALLY fedbn-personalized backbone, which is what you need for a fair,
# direct comparison against retrain_india_fedbn (which DOES retrain from a
# real fedbn model). This script fixes that: it forks from the real
# fedbn-trained Phase-1 checkpoint (fl_fedbn_oral_330775).
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
SOURCE_RUN="fl_fedbn_oral_330775"

cd "$PROJECT_DIR"

mkdir -p logs/fu_domain_backbone checkpoints/fu_domain_backbone outputs/fu_domain_backbone slurm_logs/domain_contribution_backbone

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN} (real fedbn-personalized backbone)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

if [ ! -f "checkpoints/fl/${SOURCE_RUN}/best.pt" ]; then
    echo "ERROR: checkpoints/fl/${SOURCE_RUN}/best.pt not found. Has that FL run completed?"
    exit 1
fi

echo ""
echo "=== FUSED oral3 (BACKBONE ablation, LoRA), algorithm=fedbn, forget=India_Dataset ==="

python3 scripts/run_fu_lora_domain.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client India_Dataset \
    --algorithm fedbn \
    --fedprox_mu 0.01 \
    --fedmoon_mu 1.0 \
    --fedmoon_temperature 0.5 \
    --global_epoch 50 \
    --local_epoch 3 \
    --batch_size 16 \
    --run_id "fu_domain_fedbn_india_backbone_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fu_domain_fedbn_india_backbone_oral_${SLURM_JOB_ID}"
