#!/bin/bash
# Refresh R0-R13 completion table and print pipeline heartbeat.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}"
TABLE_MD="paper_revision/results/real_50epoch_5center_corrected/status/R0_R13_completion_table.md"

echo "=== $(date -Iseconds) ==="
if pgrep -f "run_corrected403_clean_rerun" >/dev/null 2>&1; then
  echo "Pipeline: RUNNING"
  ps aux | grep -E "train_bio_cot_v3.2|evaluate_checkpoint" | grep -v grep | head -3 || true
else
  echo "Pipeline: STOPPED (check nohup log for completion or errors)"
fi

# Latest epoch hint from recent train logs
for log in $(find paper_revision/results -name "train_seed*.log" -mmin -120 2>/dev/null | head -5); do
  epoch=$(grep -oP 'Epoch \K[0-9]+(?=/50 - 训练)' "$log" 2>/dev/null | tail -1 || true)
  if [[ -n "${epoch:-}" ]]; then
    echo "Log $(basename "$log"): last train epoch ${epoch}/50"
  fi
done

"$PYTHON" paper_revision/scripts/build_r0_r13_completion_table.py
echo ""
echo "Table: $TABLE_MD"
