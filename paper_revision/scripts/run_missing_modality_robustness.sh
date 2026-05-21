#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
CHECKPOINT="${CHECKPOINT:-}"
CONFIG="${CONFIG:-paper_revision/configs/all_center_elbo_structured_prior_config.py}"
SEED="${SEED:-42}"
METHOD="${METHOD:-HyDRA-CoE Full}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/best_model.pth before running missing-modality robustness." >&2
  exit 2
fi
for setting in none remove_oct remove_colposcopy remove_clinical_text random_one random_two; do
  python paper_revision/scripts/evaluate_checkpoint_predictions.py \
    --checkpoint "$CHECKPOINT" --config "$CONFIG" --split external_test \
    --method "$METHOD" --run_id "missing_${setting}" --seed "$SEED" \
    --missing-modality "$setting" || true
done
python paper_revision/scripts/build_missing_modality_table.py
