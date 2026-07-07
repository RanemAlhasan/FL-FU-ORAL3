#!/bin/bash -l

#SBATCH -J fu_lora_india_oral
#SBATCH -o /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_lora/output_fu_lora_india_%j.txt
#SBATCH -e /export/home/ralhasan/fl-fu-oral3/slurm_logs/fu_lora/error_fu_lora_india_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32G

# =============================================================================
# Phase 2, Run 3/3: FUSED LoRA unlearning, forgetting India_Dataset.
# =============================================================================

set -euo pipefail

source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
SOURCE_RUN="fl_fedavg_oral_329811"

cd "$PROJECT_DIR"

mkdir -p logs/fu_lora checkpoints/fu_lora outputs/fu_lora slurm_logs/fu_lora

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Source run: ${SOURCE_RUN}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
nvidia-smi

echo "Cleaning old Ray state..."
ray stop --force || true
unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_REDIS_ADDRESS
export RAY_ADDRESS=""
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

if [ ! -f "checkpoints/fl/${SOURCE_RUN}/best.pt" ]; then
    echo "ERROR: checkpoints/fl/${SOURCE_RUN}/best.pt not found. Has that FL run completed?"
    exit 1
fi

if [ ! -f "logs/fl/${SOURCE_RUN}/config.snapshot.yaml" ]; then
    echo "ERROR: logs/fl/${SOURCE_RUN}/config.snapshot.yaml not found. run_fu_lora.py needs this."
    exit 1
fi

echo ""
echo "=== Phase 2: FUSED LoRA unlearning (forget=India_Dataset) ==="

python3 scripts/run_fu_lora.py \
    --source_run "${SOURCE_RUN}" \
    --forget_client India_Dataset \
    --global_epoch 50 \
    --local_epoch 3 \
    --batch_size 16 \
    --run_id "fu_lora_india_oral_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fu_lora_india_oral_${SLURM_JOB_ID} (forked from ${SOURCE_RUN})"