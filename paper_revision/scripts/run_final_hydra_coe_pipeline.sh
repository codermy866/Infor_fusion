#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
SEEDS=(${SEEDS:-42 123 456})
EPOCHS="${EPOCHS:-50}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="paper_revision/results/final_pipeline_manifest.csv"
mkdir -p "$(dirname "$MANIFEST")"
echo "stage,method,seed,config_path,checkpoint_path,internal_prediction_csv,external_prediction_csv,status,start_time,end_time,error_message" > "$MANIFEST"

record() {
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11}" >> "$MANIFEST"
}

run_stage() {
  local stage="$1"; shift
  local start end status err
  start="$(date -Is)"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $stage: $*"
    status="dry_run"
    err=""
  elif "$@"; then
    status="ok"
    err=""
  else
    status="failed"
    err="command_failed"
  fi
  end="$(date -Is)"
  record "$stage" "" "" "" "" "" "" "$status" "$start" "$end" "$err"
}

"$PYTHON_BIN" - <<'PY'
import pandas as pd
from pathlib import Path
idx=Path("paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv")
assert idx.exists(), idx
assert len(pd.read_csv(idx, encoding="utf-8-sig")) == 1897
assert Path("paper_revision/tables/image_volume_summary.csv").exists()
PY

run_stage "1_alignment_audit" "$PYTHON_BIN" paper_revision/scripts/build_data_alignment_audit.py --dry-run
run_stage "2_split_rebuild" "$PYTHON_BIN" paper_revision/scripts/build_target_adapted_validation_splits.py
run_stage "3_config_sanity" "$PYTHON_BIN" paper_revision/scripts/build_config_sanity.py
run_stage "4_stage1_adapter_dry_run" "$PYTHON_BIN" paper_revision/scripts/train_stage1_clinical_semantic_adapter.py --config paper_revision/configs/stage1_clinical_semantic_adapter_config.py --dry-run
run_stage "5_guideline_prototypes" "$PYTHON_BIN" paper_revision/scripts/check_guideline_prototypes.py
run_stage "6_locked_threshold_tables" "$PYTHON_BIN" paper_revision/scripts/build_locked_threshold_tables.py --toy-test
run_stage "7_missing_modality_table" "$PYTHON_BIN" paper_revision/scripts/build_missing_modality_table.py
run_stage "8_corruption_table" "$PYTHON_BIN" paper_revision/scripts/build_corruption_robustness_table.py
run_stage "9_label_noise_splits" "$PYTHON_BIN" paper_revision/scripts/create_label_noise_splits.py
run_stage "10_label_noise_table" "$PYTHON_BIN" paper_revision/scripts/build_label_noise_table.py
run_stage "11_loco_tables" "$PYTHON_BIN" paper_revision/scripts/build_loco_tables.py
run_stage "12_centerwise_calibration" "$PYTHON_BIN" paper_revision/scripts/build_centerwise_calibration_tables.py
run_stage "13_coe_faithfulness" "$PYTHON_BIN" paper_revision/scripts/run_coe_faithfulness.py
run_stage "14_coe_tables" "$PYTHON_BIN" paper_revision/scripts/build_coe_faithfulness_tables.py
run_stage "15_clinical_decision" "$PYTHON_BIN" paper_revision/scripts/build_clinical_decision_quality.py

echo "Pipeline manifest written to $MANIFEST"
