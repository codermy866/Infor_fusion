#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
SEEDS=(42 123 456)
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
CONFIG="${CONFIG:-paper_revision/configs/all_center_elbo_structured_prior_config.py}"
for center_dir in paper_revision/splits/loco/*; do
  [[ -d "$center_dir" ]] || continue
  center="$(basename "$center_dir")"
  for seed in "${SEEDS[@]}"; do
    python training/train_bio_cot_v3.2.py --config "$CONFIG" --seed "$seed" --epochs "$EPOCHS" --gpu "$GPU" \
      --train-csv "$center_dir/train_labels.csv" --val-csv "$center_dir/val_labels.csv" || true
    echo "LOCO $center seed $seed training command completed or failed; evaluation requires checkpoint selection."
  done
done
python paper_revision/scripts/build_loco_tables.py || true
python paper_revision/scripts/build_centerwise_calibration_tables.py || true
