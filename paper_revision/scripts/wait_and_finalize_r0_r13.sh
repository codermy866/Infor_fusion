#!/bin/bash
# Wait for clean-rerun pipeline to exit, then write final R0-R13 table.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}"
LOG="paper_revision/results/real_50epoch_5center_corrected/status/wait_finalize.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date -Iseconds)] Waiting for run_corrected403_clean_rerun..." | tee -a "$LOG"
while pgrep -f "run_corrected403_clean_rerun.py" >/dev/null 2>&1; do
  "$PYTHON" paper_revision/scripts/build_r0_r13_completion_table.py >> "$LOG" 2>&1 || true
  sleep 600
done

echo "[$(date -Iseconds)] Pipeline stopped. Final table:" | tee -a "$LOG"
"$PYTHON" paper_revision/scripts/build_r0_r13_completion_table.py | tee -a "$LOG"
echo "[$(date -Iseconds)] Done." | tee -a "$LOG"
