#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-pretrained1}"
EPOCHS="${EPOCHS:-50}"
METHOD="${METHOD:-HyDRA_Full_Pretrained}"
PYTHON_BIN="${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-paper_revision/configs/full_pretrained_config.py}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/data2/hmy_pri/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/data2/hmy_pri/.cache/torch}"

mkdir -p logs checkpoints results paper_revision/results/predictions paper_revision/results/run_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="paper_revision/results/run_logs/formal_${METHOD}_run${RUN_ID}_seed${SEED}_${STAMP}.log"

echo "============================================================" | tee -a "$WRAPPER_LOG"
echo "Formal pretrained training start: method=${METHOD}, run=${RUN_ID}, seed=${SEED}, epochs=${EPOCHS}" | tee -a "$WRAPPER_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$WRAPPER_LOG"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "$WRAPPER_LOG"
echo "CONFIG_PATH=${CONFIG_PATH}" | tee -a "$WRAPPER_LOG"
echo "============================================================" | tee -a "$WRAPPER_LOG"

"$PYTHON_BIN" training/train_bio_cot_v3.2.py \
  --config "$CONFIG_PATH" \
  --epochs "$EPOCHS" \
  --seed "$SEED" 2>&1 | tee -a "$WRAPPER_LOG"

CHECKPOINT="$(find paper_revision/results/full_pretrained/checkpoints -name 'best_model_v3_*.pth' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
echo "Best checkpoint: ${CHECKPOINT}" | tee -a "$WRAPPER_LOG"

for SPLIT in internal_validation external_test; do
  "$PYTHON_BIN" paper_revision/scripts/evaluate_checkpoint_predictions.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --method "$METHOD" \
    --run_id "$RUN_ID" \
    --seed "$SEED" 2>&1 | tee -a "$WRAPPER_LOG"
done

for SETTING in remove_colposcopy remove_oct remove_clinical_prior random_one random_two; do
  "$PYTHON_BIN" paper_revision/scripts/evaluate_checkpoint_predictions.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT" \
    --split external_test \
    --method "${METHOD}_${SETTING}" \
    --run_id "$RUN_ID" \
    --seed "$SEED" \
    --missing-modality "$SETTING" 2>&1 | tee -a "$WRAPPER_LOG"
done

"$PYTHON_BIN" paper_revision/scripts/build_centerwise_calibration.py 2>&1 | tee -a "$WRAPPER_LOG"

echo "============================================================" | tee -a "$WRAPPER_LOG"
echo "Formal pretrained training and output generation finished." | tee -a "$WRAPPER_LOG"
echo "Wrapper log: ${WRAPPER_LOG}" | tee -a "$WRAPPER_LOG"
echo "============================================================" | tee -a "$WRAPPER_LOG"
