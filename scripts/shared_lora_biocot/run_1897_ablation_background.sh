#!/usr/bin/env bash
# Full 1897 module ablation → summarize → auto-prune ineffective components.
set -euo pipefail

EXP_ROOT="/data2/hmy_pri/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026"
PYTHON="/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
GPU="${1:-1}"
LOG_DIR="${EXP_ROOT}/outputs/publishable_v2/shared_lora_biocot/improved_1897/logs"
mkdir -p "${LOG_DIR}"

cd "${EXP_ROOT}"

echo "[$(date -Iseconds)] Full module ablation suite on GPU ${GPU}" | tee -a "${LOG_DIR}/ablation_suite.log"

"${PYTHON}" scripts/shared_lora_biocot/run_1897_improved_loco.py \
  --skip-complete \
  --gpu "${GPU}" \
  --epochs 20 \
  --batch-size 8 \
  2>&1 | tee -a "${LOG_DIR}/ablation_suite.log"

"${PYTHON}" scripts/shared_lora_biocot/summarize_1897_ablation.py \
  2>&1 | tee -a "${LOG_DIR}/ablation_suite.log"

if "${PYTHON}" scripts/shared_lora_biocot/apply_ablation_pruning.py 2>&1 | tee -a "${LOG_DIR}/ablation_suite.log"; then
  "${PYTHON}" scripts/shared_lora_biocot/apply_ablation_pruning.py --apply \
    2>&1 | tee -a "${LOG_DIR}/ablation_suite.log"
else
  echo "[$(date -Iseconds)] Pruning skipped (ablation incomplete)" | tee -a "${LOG_DIR}/ablation_suite.log"
fi

echo "[$(date -Iseconds)] Done" | tee -a "${LOG_DIR}/ablation_suite.log"
