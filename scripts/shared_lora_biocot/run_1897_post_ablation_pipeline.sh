#!/usr/bin/env bash
# Wait for ablation → prune → 100-epoch production → final tables.
set -euo pipefail

EXP_ROOT="/data2/hmy_pri/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026"
PYTHON="/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
GPU="${1:-1}"
LOG_DIR="${EXP_ROOT}/outputs/publishable_v2/shared_lora_biocot/improved_1897/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/post_ablation_pipeline.log"

cd "${EXP_ROOT}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

log "Post-ablation pipeline started (GPU ${GPU})"

log "Phase 1: waiting for ablation training to finish..."
while pgrep -f "run_1897_improved_loco.py" >/dev/null 2>&1; do
  pending=$("${PYTHON}" - <<'PY' 2>/dev/null || true
from scripts.shared_lora_biocot.run_1897_improved_loco import pending_ablation_groups
p = pending_ablation_groups()
print(",".join(p) if p else "none")
PY
)
  log "  ablation running; pending groups: ${pending:-unknown}"
  sleep 600
done

log "Phase 2: waiting for all ablation groups to have metrics..."
while true; do
  if "${PYTHON}" - <<'PY'
from scripts.shared_lora_biocot.run_1897_improved_loco import all_ablation_groups_complete, pending_ablation_groups
import sys
if all_ablation_groups_complete():
    sys.exit(0)
print("pending:", pending_ablation_groups())
sys.exit(1)
PY
  then
    break
  fi
  sleep 600
done

log "Phase 3: summarize ablation + prune ineffective modules"
"${PYTHON}" scripts/shared_lora_biocot/summarize_1897_ablation.py 2>&1 | tee -a "${LOG}"
"${PYTHON}" scripts/shared_lora_biocot/apply_ablation_pruning.py 2>&1 | tee -a "${LOG}"
"${PYTHON}" scripts/shared_lora_biocot/apply_ablation_pruning.py --apply 2>&1 | tee -a "${LOG}"

log "Phase 3b: LOCO split diagnostics (hard-center val recommendation)"
"${PYTHON}" scripts/shared_lora_biocot/diagnose_loco_splits.py 2>&1 | tee -a "${LOG}"

log "Phase 4: 100-epoch production LOCO with pruned stack"
"${PYTHON}" scripts/shared_lora_biocot/run_1897_final_production.py \
  --gpu "${GPU}" \
  --epochs 100 \
  --batch-size 8 \
  2>&1 | tee -a "${LOG}"

log "Phase 5: build final experiment tables"
"${PYTHON}" scripts/shared_lora_biocot/build_final_experiment_tables.py 2>&1 | tee -a "${LOG}"

log "Post-ablation pipeline COMPLETE"
log "Final tables: outputs/publishable_v2/shared_lora_biocot/improved_1897/tables/Table_Final_1897_LOCO_Summary.csv"
log "Final report: outputs/publishable_v2/shared_lora_biocot/improved_1897/reports/Report_Final_1897_Experiment.md"
