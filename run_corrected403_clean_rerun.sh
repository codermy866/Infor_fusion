#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHON_BIN="${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAGES="${STAGES:-all}"
EPOCHS="${EPOCHS:-50}"
SEEDS="${SEEDS:-42,123,456}"
GPU="${GPU:-0}"
LOG="paper_revision/results/real_50epoch_5center_corrected/logs/clean_rerun_master_$(date +%Y%m%d_%H%M%S).log"

mkdir -p paper_revision/results/real_50epoch_5center_corrected/logs

echo "Starting corrected 403 clean rerun: stages=${STAGES} epochs=${EPOCHS} seeds=${SEEDS}" | tee -a "$LOG"

"$PYTHON_BIN" paper_revision/scripts/run_corrected403_clean_rerun.py \
  --stages "$STAGES" \
  --epochs "$EPOCHS" \
  --seeds "$SEEDS" \
  --gpu "$GPU" \
  "$@" 2>&1 | tee -a "$LOG"

echo "Master log: $LOG"
