#!/usr/bin/env bash
# LOCO split diagnostics + g2 manifest vs hard-center val comparison (20ep default).
set -euo pipefail

EXP_ROOT="/data2/hmy_pri/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026"
PYTHON="/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
GPU="${1:-1}"
EPOCHS="${2:-20}"
LOG_DIR="${EXP_ROOT}/outputs/publishable_v2/shared_lora_biocot/improved_1897/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/hardcenter_val_comparison.nohup.log"

cd "${EXP_ROOT}"

echo "[$(date -Iseconds)] hard-center comparison started (GPU ${GPU}, epochs ${EPOCHS})" | tee -a "${LOG}"

"${PYTHON}" scripts/shared_lora_biocot/diagnose_loco_splits.py 2>&1 | tee -a "${LOG}"

"${PYTHON}" scripts/shared_lora_biocot/run_1897_hardcenter_val_comparison.py \
  --gpu "${GPU}" \
  --epochs "${EPOCHS}" \
  --batch-size 8 \
  --skip-diagnose \
  --reuse-manifest-g2 \
  2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] hard-center comparison COMPLETE" | tee -a "${LOG}"
echo "Report: outputs/publishable_v2/shared_lora_biocot/improved_1897/hardcenter_val_comparison/tables/Report_HardCenter_Val_Comparison.md" | tee -a "${LOG}"
