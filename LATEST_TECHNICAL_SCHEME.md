# Latest Technical Scheme

This repository is locked to the current Information Fusion supplementary
experiment scheme.

## Use These Files For The Current Method

- Executable pipeline:
  - `scripts/if_supplementary/run_all_if_supplementary_experiments.py`
  - `scripts/if_supplementary/complete_p06_random_dropout_predictions.py`
  - `scripts/if_supplementary/complete_p10_perturbed_reliability_export.py`
  - `scripts/if_supplementary/complete_p11_coe_intervention_logits.py`
  - `scripts/if_supplementary/refresh_partial_completion_package.py`
- Reusable implementation:
  - `src/if_supplementary/common.py`
  - `src/if_supplementary/random_dropout.py`
  - `src/if_supplementary/reliability_perturbations.py`
  - `src/if_supplementary/coe_interventions.py`
  - `src/if_supplementary/saliency_or_occlusion.py`
- Configuration:
  - `configs/if_supplementary_same_backbone_baselines.yaml`
- Current result package:
  - `paper_revisions/if_supplementary_experiments/`

## Current Claim Boundary

The current manuscript-facing scheme is the locked n=1897 multicenter LOCO
Information Fusion supplementary package. It supports:

- locked LOCO benchmark reporting;
- validation-locked CIN3+ safety/referral analyses;
- patient-level random modality dropout stress testing from the locked
  prediction registry;
- feature-level proxy reliability perturbation analysis;
- conservative CoE audit outputs.

It does not support:

- raw-image perturbation reliability claims for P10;
- CoE faithfulness, causal explanation, or saliency-grounded claims for P11;
- clinical deployment safety claims.

For exact allowed and forbidden claims, use:

- `paper_revisions/if_supplementary_experiments/13_submission_audit/IF_FINAL_EXPERIMENT_CLAIM_LOCK.md`
- `paper_revisions/if_supplementary_experiments/partial_completion/PARTIAL_PREFLIGHT_AUDIT.md`

Older revision-route files have been removed from the tracked repository so
method drafting should use the files listed above as the source of truth.

