# Shared LoRA BioCOT Route Validation

This note records the current implementation route and the completed local
validation for the lightweight tri-modal adaptation path.

## Route

1. Pretrain the expert base with OCT and structured clinical text only.
   `f_colpo=None` is now a first-class model path.
2. Downstream centre adaptation treats colposcopy as an auxiliary modality.
   The OCT/Text expert base is frozen and colposcopy is injected through an
   independent visual encoder, a projection head, and Shared LoRA residuals.
3. Shared LoRA residuals are applied around cross-modal attention and final
   fusion, while `L_colpo_bridge_ot` and `L_colpo_bridge_align` explicitly
   align colposcopy evidence to semantic and causal manifolds.
4. LOCO should train on four source centres, choose checkpoints on an
   inner-source validation centre, and evaluate the held-out centre only after
   selection.

## Implemented Files

- `models/bio_cot_v3_2.py`
  - modality masks for missing colposcopy,
  - independent `colpo_encoder`,
  - `CrossModalSharedLoRABridge`,
  - `freeze_expert_base()`,
  - trainable parameter summary,
  - colpo-aware posterior refinement masking,
  - bridge OT/alignment losses.
- `training/train_bio_cot_v3.2.py`
  - `pretrain_without_colpo`,
  - `pass_raw_colpo_to_model`,
  - optimizer restricted to trainable parameters,
  - Shared LoRA bridge losses in logs and history.
- `config.py`
  - Shared LoRA and missing-colpo configuration switches.
- `configs/shared_lora_smoke_config.py`
  - minimal real-data smoke configuration.
- `configs/shared_lora_loco_template.py`
  - LOCO adaptation template.
- `scripts/sanity/sanity_biocot_shared_lora_dummy.py`
  - dummy forward/backward check.

## Validation Outputs

- Dummy sanity:
  `outputs/publishable_v2/shared_lora_biocot/audit/biocot_shared_lora_dummy_sanity.json`
- Dummy rerun log:
  `outputs/publishable_v2/shared_lora_biocot/audit/biocot_shared_lora_dummy_sanity_rerun.log`
- Real-data GPU smoke log:
  `outputs/publishable_v2/shared_lora_biocot/smoke/shared_lora_training_smoke_lora_frozen_with_bridge_loss.log`
- Smoke history:
  `outputs/publishable_v2/shared_lora_biocot/smoke/logs/training_history_20260526_170922.json`
- Smoke training figure:
  `outputs/publishable_v2/shared_lora_biocot/smoke/logs/training_curves_20260526_170922.png`
- Smoke checkpoint:
  `outputs/publishable_v2/shared_lora_biocot/smoke/checkpoints/best_model_v3_20260526_170922.pth`

## Sanity Results

- Missing-colpo path: `f_colpo=None` forward/backward passed.
- Missing-colpo mask: colpo mask is all zero.
- Dummy-colpo path: `L_colpo_bridge_ot` and `L_colpo_bridge_align` are present.
- Frozen-base trainable ratio in dummy check: 0.925%.
- Real-data smoke trainable ratio: 927,362 / 124,117,154 = 0.7472%.
- Real-data smoke bridge losses:
  - `L_colpo_bridge_ot`: 1.967947
  - `L_colpo_bridge_align`: 0.997131

The real-data smoke uses only 8 training cases and 4 validation cases. Its
classification metrics are only a wiring check and must not be interpreted as a
paper result.
