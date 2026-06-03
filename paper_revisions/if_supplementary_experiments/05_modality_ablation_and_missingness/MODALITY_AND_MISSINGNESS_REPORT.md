# Modality Contribution and Missingness Report

## P06 Completion Status

Status: `PASS_FEATURE_CACHE_PROXY`.

Patient-level random modality dropout predictions were generated for 10%, 30%, and 50% dropout with five deterministic repeats.
The completion uses locked single-modality, dual-modality, and full-modality patient-level predictions plus validation-locked thresholds.
It does not reload raw-image checkpoints; therefore it should be described as a feature-cache/prediction-registry stress test.

Patient-level rows: `28455`.
Unique patients: `1286`.

## Summary

                 condition  dropout_rate    auroc      npv  cin3_sensitivity  cin3_false_negatives  referral_rate
random_modality_dropout_10           0.1 0.631474 0.816479          0.294241                 134.8       0.137691
random_modality_dropout_30           0.3 0.605673 0.811791          0.265969                 140.2       0.142963
random_modality_dropout_50           0.5 0.591061 0.810418          0.282723                 137.0       0.168582

## Claim Boundary

- Allowed: random missing-modality stress testing at patient level under the locked LOCO prediction registry.
- Not allowed: claiming this is a raw-image checkpoint re-inference experiment.
