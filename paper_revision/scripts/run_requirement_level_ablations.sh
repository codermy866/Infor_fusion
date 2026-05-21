#!/usr/bin/env bash
set -u

SEEDS=(42 123 456)
EPOCHS="${EPOCHS:-50}"
GPU="${GPU:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/paper_revision/results/tables/ablation_run_manifest.csv"
mkdir -p "$(dirname "$MANIFEST")"
echo "config_path,seed,train_status,internal_eval_status,external_eval_status,start_time,end_time,error_message" > "$MANIFEST"

CONFIGS=(
  paper_revision/configs/abl_no_clinical_semantic_adapter_config.py
  paper_revision/configs/abl_no_clinical_structured_prior_config.py
  paper_revision/configs/abl_no_hpv_config.py
  paper_revision/configs/abl_no_tct_config.py
  paper_revision/configs/abl_no_age_config.py
  paper_revision/configs/abl_image_only_config.py
  paper_revision/configs/abl_clinical_only_config.py
  paper_revision/configs/abl_no_variational_reliability_config.py
  paper_revision/configs/abl_no_center_aware_reliability_config.py
  paper_revision/configs/abl_no_posterior_refinement_config.py
  paper_revision/configs/abl_no_guideline_prototype_config.py
  paper_revision/configs/abl_no_counterfactual_robustness_config.py
)

cd "$ROOT" || exit 1
for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    start_time="$(date -Is)"
    train_status="pending"
    internal_status="pending"
    external_status="pending"
    error_message=""
    if python training/train_bio_cot_v3.2.py --config "$config" --seed "$seed" --epochs "$EPOCHS" --gpu "$GPU"; then
      train_status="ok"
    else
      train_status="failed"
      error_message="training_failed"
    fi
    if [[ "$train_status" == "ok" ]]; then
      checkpoint="$(find paper_revision/results -path '*checkpoints*' -name '*.pth' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
      method="$(basename "$config" .py)"
      if python paper_revision/scripts/evaluate_checkpoint_predictions.py --checkpoint "$checkpoint" --config "$config" --split internal_validation --method "$method" --run_id "abl_seed${seed}" --seed "$seed"; then
        internal_status="ok"
      else
        internal_status="failed"
      fi
      if python paper_revision/scripts/evaluate_checkpoint_predictions.py --checkpoint "$checkpoint" --config "$config" --split external_test --method "$method" --run_id "abl_seed${seed}" --seed "$seed"; then
        external_status="ok"
      else
        external_status="failed"
      fi
    fi
    end_time="$(date -Is)"
    echo "$config,$seed,$train_status,$internal_status,$external_status,$start_time,$end_time,$error_message" >> "$MANIFEST"
  done
done

python paper_revision/scripts/build_locked_threshold_tables.py \
  --pred-dir paper_revision/results/predictions \
  --output-dir paper_revision/results/tables \
  --threshold-rule max_specificity_at_sensitivity:0.95 || true
