#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
CHECKPOINT="${CHECKPOINT:-}"
CONFIG="${CONFIG:-paper_revision/configs/all_center_elbo_structured_prior_config.py}"
SEED="${SEED:-42}"
METHOD="${METHOD:-HyDRA-CoE Full}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/best_model.pth before running corruption robustness." >&2
  exit 2
fi
for corruption in colpo_blur colpo_brightness colpo_occlusion oct_speckle oct_stripe oct_intensity oct_bscan_dropout; do
  for severity in 1.0 2.0 3.0; do
    python paper_revision/scripts/evaluate_checkpoint_predictions.py \
      --checkpoint "$CHECKPOINT" --config "$CONFIG" --split external_test \
      --method "$METHOD" --run_id "corrupt_${corruption}_${severity}" --seed "$SEED" \
      --input-corruption "$corruption" --corruption-severity "$severity" || true
  done
done
python paper_revision/scripts/build_corruption_robustness_table.py
