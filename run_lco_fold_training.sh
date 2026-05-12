#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

HELD_OUT_CENTER="${HELD_OUT_CENTER:?Set HELD_OUT_CENTER, e.g. Enshi, Wuda, Xiangyang, Jingzhou, or Shiyan}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-1}"
EPOCHS="${EPOCHS:-50}"
METHOD="${METHOD:-HyDRA_Full_LCO_${HELD_OUT_CENTER}}"
PYTHON_BIN="${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data2/hmy_pri/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p logs checkpoints results paper_revision/results/predictions paper_revision/results/run_logs

"$PYTHON_BIN" paper_revision/scripts/build_lco_splits.py

FOLD_DIR="paper_revision/splits/leave_one_center_out/${HELD_OUT_CENTER}"
if [[ ! -d "$FOLD_DIR" ]]; then
  echo "Unknown held-out center: ${HELD_OUT_CENTER}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="paper_revision/results/run_logs/lco_${HELD_OUT_CENTER}_run${RUN_ID}_seed${SEED}_${STAMP}.log"

echo "============================================================" | tee -a "$WRAPPER_LOG"
echo "LCO training start: held_out_center=${HELD_OUT_CENTER}, method=${METHOD}, run=${RUN_ID}, seed=${SEED}, epochs=${EPOCHS}" | tee -a "$WRAPPER_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$WRAPPER_LOG"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "$WRAPPER_LOG"
echo "DATA_ROOT=${DATA_ROOT}" | tee -a "$WRAPPER_LOG"
echo "============================================================" | tee -a "$WRAPPER_LOG"

"$PYTHON_BIN" training/train_bio_cot_v3.2.py \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --data-root "$DATA_ROOT" \
  --train-csv "${FOLD_DIR}/train_labels.csv" \
  --val-csv "${FOLD_DIR}/val_labels.csv" 2>&1 | tee -a "$WRAPPER_LOG"

CHECKPOINT="$(find checkpoints -name 'best_model_v3_*.pth' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
echo "Best checkpoint: ${CHECKPOINT}" | tee -a "$WRAPPER_LOG"

"$PYTHON_BIN" paper_revision/scripts/evaluate_checkpoint_predictions.py \
  --checkpoint "$CHECKPOINT" \
  --split internal_validation \
  --csv "${FOLD_DIR}/val_labels.csv" \
  --data-root "$DATA_ROOT" \
  --method "$METHOD" \
  --run_id "$RUN_ID" \
  --seed "$SEED" 2>&1 | tee -a "$WRAPPER_LOG"

"$PYTHON_BIN" paper_revision/scripts/evaluate_checkpoint_predictions.py \
  --checkpoint "$CHECKPOINT" \
  --split external_test \
  --csv "${FOLD_DIR}/external_test_labels.csv" \
  --data-root "$DATA_ROOT" \
  --method "$METHOD" \
  --run_id "$RUN_ID" \
  --seed "$SEED" 2>&1 | tee -a "$WRAPPER_LOG"

"$PYTHON_BIN" paper_revision/scripts/build_centerwise_calibration.py 2>&1 | tee -a "$WRAPPER_LOG"

echo "============================================================" | tee -a "$WRAPPER_LOG"
echo "LCO training and output generation finished." | tee -a "$WRAPPER_LOG"
echo "Wrapper log: ${WRAPPER_LOG}" | tee -a "$WRAPPER_LOG"
echo "============================================================" | tee -a "$WRAPPER_LOG"
