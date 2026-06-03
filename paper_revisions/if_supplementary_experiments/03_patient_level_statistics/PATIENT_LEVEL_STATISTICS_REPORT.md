# Patient-Level Statistics Report

Pooled bootstrap confidence intervals and paired tests are reused from the locked Step2 output generated with `2000` patient-level resamples.
Center-level rows contain point estimates because the existing locked Step2 file does not store center-level bootstrap CIs.

Recommended Table 2 replacement: `pooled_metrics_with_ci.csv` plus center-level details from `main_metrics_with_ci_by_center.csv`.

Statistical superiority should only be claimed for comparisons whose paired bootstrap CI excludes zero and p-value supports the direction.

HyDRA pooled metrics:
| model_name | protocol | n | aggregation | operating_point | auroc | auroc_ci_low | auroc_ci_high | auprc | auprc_ci_low | auprc_ci_high | sensitivity | sensitivity_ci_low | sensitivity_ci_high | specificity | specificity_ci_low | specificity_ci_high | ppv | ppv_ci_low | ppv_ci_high | npv | npv_ci_low | npv_ci_high | f1 | f1_ci_low | f1_ci_high | referral_rate | referral_rate_ci_low | referral_rate_ci_high | balanced_accuracy | brier | ece | ci_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HyDRA_CoE_Full | strict_loco | 1897 | pooled_micro | t_safety95 | 0.6362 | 0.6044 | 0.6669 | 0.3247 | 0.2864 | 0.3714 | 0.9289 | 0.9028 | 0.9519 | 0.1291 | 0.1129 | 0.1463 | 0.2185 | 0.1995 | 0.2374 | 0.8739 | 0.8285 | 0.9143 | 0.3538 | 0.3285 | 0.3785 | 0.883 | 0.8682 | 0.8967 | 0.5713 | 0.2853 | 0.2797 | outputs/publishable_v2/step2_main_loco/statistics/bootstrap_ci_all_metrics.csv |
