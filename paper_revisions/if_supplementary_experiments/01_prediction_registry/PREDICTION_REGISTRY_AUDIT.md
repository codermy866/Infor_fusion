# Prediction Registry Audit

Standardized prediction files were created for `25` main models across validation and test splits.

- All later analyses in this package use patient-level seed-averaged predictions from `standardized_patient_mean_*_predictions.csv`.
- Patient and case identifiers are hashed in standardized files.
- Undefined metrics are returned as blank/NaN for one-class groups.

## Registry Summary
| protocol | usable_for_main_table | n_files |
| --- | --- | --- |
| strict_loco | False | 9 |
| strict_loco | True | 9 |
| strict_loco_shared_lora_supplement | False | 16 |
