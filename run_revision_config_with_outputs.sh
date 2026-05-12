#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH to a paper_revision/configs/*.py file}"
METHOD="${METHOD:?Set METHOD, e.g. HyDRA_Variational}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-1}"
EPOCHS="${EPOCHS:-50}"
PYTHON_BIN="${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/data2/hmy_pri/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/data2/hmy_pri/.cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p logs checkpoints results paper_revision/results/predictions paper_revision/results/run_logs

STAMP="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="paper_revision/results/run_logs/formal_${METHOD}_run${RUN_ID}_seed${SEED}_${STAMP}.log"

echo "============================================================" | tee -a "$WRAPPER_LOG"
echo "Revision run start: method=${METHOD}, run=${RUN_ID}, seed=${SEED}, epochs=${EPOCHS}" | tee -a "$WRAPPER_LOG"
echo "CONFIG_PATH=${CONFIG_PATH}" | tee -a "$WRAPPER_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$WRAPPER_LOG"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "$WRAPPER_LOG"
echo "============================================================" | tee -a "$WRAPPER_LOG"

"$PYTHON_BIN" training/train_bio_cot_v3.2.py \
  --config "$CONFIG_PATH" \
  --epochs "$EPOCHS" \
  --seed "$SEED" 2>&1 | tee -a "$WRAPPER_LOG"

CHECKPOINT_DIR="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import importlib.util, sys
import contextlib
import io
from pathlib import Path
spec = importlib.util.spec_from_file_location("cfg", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
classes = [
    v for v in m.__dict__.values()
    if isinstance(v, type) and "Config" in v.__name__ and getattr(v, "__module__", None) == m.__name__
]
with contextlib.redirect_stdout(io.StringIO()):
    cfg = classes[0]()
print(cfg.checkpoint_dir)
PY
)"
CHECKPOINT="$(find "$CHECKPOINT_DIR" -name 'best_model_v3_*.pth' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
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
echo "Revision run finished." | tee -a "$WRAPPER_LOG"
echo "Wrapper log: ${WRAPPER_LOG}" | tee -a "$WRAPPER_LOG"
echo "============================================================" | tee -a "$WRAPPER_LOG"
