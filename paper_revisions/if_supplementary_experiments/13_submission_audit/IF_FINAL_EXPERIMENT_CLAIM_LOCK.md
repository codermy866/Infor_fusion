# IF Final Experiment Claim Lock

Updated at: `2026-06-03 10:18:40` after P06/P10/P11 partial-completion execution.

## Protocol Invariants

1. Analytic cohort remains locked at n=1897.
2. Primary protocol remains strict five-fold LOCO.
3. Validation-locked thresholds are reused; held-out test labels are not used for model selection or threshold selection.
4. New patient-level outputs omit raw patient and case identifiers.

## P06/P10/P11 Status

| Step | Status | Evidence | Claim boundary |
|---|---|---|---|
| P06 | PASS_FEATURE_CACHE_PROXY | `05_modality_ablation_and_missingness/random_dropout_patient_level_predictions.csv` | Patient-level random modality dropout stress test from locked prediction registry; not raw-image checkpoint re-inference. |
| P10 | PASS_FEATURE_LEVEL_PROXY | `09_reliability_validation/reliability_weights_clean_and_perturbed_patient_level.csv` | Clean-vs-perturbed reliability response under feature-level proxy perturbations; not raw-image corruption reliability validation. |
| P11 | PARTIAL_NOT_ESTABLISHED | `10_coe_faithfulness/coe_intervention_logits_patient_level.csv` | Clean CoE proxy states audited; no faithfulness, causal explanation, or saliency claim. |

## Allowed Claims

- Locked multicenter LOCO benchmark results.
- Validation-locked CIN3+ safety/referral trade-off.
- Patient-level missing-modality stress testing using locked prediction-registry proxies.
- Feature-level reliability-weight response as an internal diagnostic.
- CoE clean proxy trajectories as transparency aids only.

## Claims To Remove Or Soften

- Clinical deployment safety.
- Raw-image perturbation reliability validation.
- CoE faithfulness, causal explanation, saliency-grounded explanation, or counterfactual intervention claims.
- Any statement that P06/P10/P11 completion used target labels for model selection.
