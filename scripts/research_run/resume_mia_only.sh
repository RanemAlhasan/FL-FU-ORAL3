#!/bin/bash -l
#SBATCH -J resume_mia_only
#SBATCH -o output_resume_mia_only_%j.txt
#SBATCH -e error_resume_mia_only_%j.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH --time 96:00:00
#SBATCH --exclusive
# =============================================================================
# Resume ONLY the MIA step for fused_cifar10_full_327966, using the FIXED
# membership_inference_attack (src/eval/mia.py): shadow models reuse the
# already-trained Phase-A checkpoint instead of retraining from scratch,
# train on BYZANTINE-ATTACKED proxy loaders, and both real/proxy evaluation
# pools combine train+test loader logits per client.
# Loads existing phase_a_model.pt / fused_model.pt / retrain_model.pt
# checkpoints, skips Phase A, Phase B, and ReA entirely. ReA is handled
# separately by resume_rea_only.sh (also fixed).
# =============================================================================
set -euo pipefail
source /export/home/ralhasan/anaconda3/etc/profile.d/conda.sh
conda activate /export/home/ralhasan/anaconda3/envs/fl_fu

PROJECT_DIR=/export/home/ralhasan/fl-fu-oral3
cd "$PROJECT_DIR"

DATASET_ROOT=/export/home/ralhasan/fl-fu-oral3/dataset/cifar10
SOURCE_RUN=fused_cifar10_full_327966

mkdir -p logs/cifar10 checkpoints/cifar10 outputs/cifar10 slurm_logs/cifar10

echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
echo "Dataset root: ${DATASET_ROOT}"
echo "Source run (checkpoints to load): ${SOURCE_RUN}"
nvidia-smi
echo ""
echo "=== Resuming MIA only (fixed: attacked proxy loaders, cheap shadow reuse, train+test pooling) ==="
python3 scripts/resume_mia_only.py \
    --config configs/cifar10_client_unlearning.yaml \
    --source_run "${SOURCE_RUN}" \
    --set dataset_root="${DATASET_ROOT}" \
    --set logs_root=logs/cifar10 \
    --set checkpoints_root=checkpoints/cifar10 \
    --set outputs_root=outputs/cifar10 \
    --run_id "fused_cifar10_resume_mia_${SLURM_JOB_ID}"

echo ""
echo "Done. run_id = fused_cifar10_resume_mia_${SLURM_JOB_ID}"
echo "Check logs/cifar10/fused_cifar10_resume_mia_${SLURM_JOB_ID}/train.log for MIA numbers."
echo "Compare against original run's RA/FA at logs/cifar10/${SOURCE_RUN}/train.log"
echo "(ReA numbers from the original run are now STALE -- resubmit resume_rea_only.sh too.)"