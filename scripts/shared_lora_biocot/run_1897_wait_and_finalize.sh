#!/usr/bin/env bash
# Wait for hard-center + 100ep jobs, then emit comparison and final tables.
set -euo pipefail

EXP_ROOT="/data2/hmy_pri/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026"
PYTHON="/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
LOG_DIR="${EXP_ROOT}/outputs/publishable_v2/shared_lora_biocot/improved_1897/logs"
IMPROVED="${EXP_ROOT}/outputs/publishable_v2/shared_lora_biocot/improved_1897"
LOG="${LOG_DIR}/wait_and_finalize.nohup.log"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

cd "${EXP_ROOT}"

log "wait_and_finalize started"

log "Phase A: waiting for hard-center g2 training..."
while pgrep -f "hardcenter_val_comparison.py" >/dev/null 2>&1; do
  pending=$("${PYTHON}" - <<'PY' 2>/dev/null || true
from pathlib import Path
from scripts.shared_lora_biocot.run_1897_improved_loco import group_is_complete
root = Path("outputs/publishable_v2/shared_lora_biocot/improved_1897/hardcenter_val_comparison/g2_hardcenter_val/ablations/g2")
print("complete" if group_is_complete(root) else "running")
PY
)
  log "  hard-center status: ${pending}"
  sleep 300
done

HARD_METRICS="${IMPROVED}/hardcenter_val_comparison/g2_hardcenter_val/ablations/g2/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
if [[ ! -f "${HARD_METRICS}" ]]; then
  log "ERROR: hard-center metrics missing: ${HARD_METRICS}"
  exit 1
fi

log "Phase B: build hard-center vs manifest comparison table"
"${PYTHON}" scripts/shared_lora_biocot/run_1897_hardcenter_val_comparison.py --compare-only 2>&1 | tee -a "${LOG}"

log "Phase C: waiting for 100-epoch production..."
while pgrep -f "run_1897_final_production.py" >/dev/null 2>&1; do
  sleep 300
  log "  final_production still running..."
done

FINAL_METRICS="${IMPROVED}/final_production_100ep/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
if [[ ! -f "${FINAL_METRICS}" ]]; then
  log "WARN: final production metrics missing; attempting build anyway"
fi

log "Phase D: build final experiment tables"
"${PYTHON}" scripts/shared_lora_biocot/build_final_experiment_tables.py 2>&1 | tee -a "${LOG}"

log "wait_and_finalize COMPLETE"
log "Hard-center comparison: ${IMPROVED}/hardcenter_val_comparison/tables/Report_HardCenter_Val_Comparison.md"
log "Final summary: ${IMPROVED}/tables/Table_Final_1897_LOCO_Summary.csv"
