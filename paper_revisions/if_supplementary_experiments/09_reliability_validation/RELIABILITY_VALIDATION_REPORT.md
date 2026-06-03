# Reliability Validation Report

## P10 Completion Status

Status: `PASS_FEATURE_LEVEL_PROXY`.

Clean and perturbed reliability weights were exported at patient level for OCT, colposcopy, clinical masking, and random dropout proxy conditions.
The perturbations are feature-level proxy perturbations derived from the saved clean reliability weights; raw-image perturbation inference was not re-run.

Patient-condition rows: `20867`.
Unique patients: `1286`.

## Summary

                  condition target_modality  target_mean_delta_alpha  mean_delta_reliability_entropy                   status
        clinical_hpv_masked        clinical                -0.035038                       -0.017539 PASS_FEATURE_LEVEL_PROXY
    clinical_hpv_tct_masked        clinical                -0.074478                       -0.045398 PASS_FEATURE_LEVEL_PROXY
        clinical_tct_masked        clinical                -0.035038                       -0.017539 PASS_FEATURE_LEVEL_PROXY
            colposcopy_blur      colposcopy                -0.076040                        0.001429 PASS_FEATURE_LEVEL_PROXY
colposcopy_brightness_shift      colposcopy                -0.048402                        0.003521 PASS_FEATURE_LEVEL_PROXY
       colposcopy_occlusion      colposcopy                -0.122880                       -0.009382 PASS_FEATURE_LEVEL_PROXY
         oct_gaussian_noise             oct                -0.059864                        0.001847 PASS_FEATURE_LEVEL_PROXY
     oct_speckle_noise_mild             oct                -0.034359                        0.002795 PASS_FEATURE_LEVEL_PROXY
   oct_speckle_noise_strong             oct                -0.102855                       -0.005906 PASS_FEATURE_LEVEL_PROXY
 random_modality_dropout_30          random                      NaN                       -0.080718 PASS_FEATURE_LEVEL_PROXY

## Claim Boundary

- Allowed: clean-vs-perturbed reliability weight response under explicit feature-level proxy perturbations.
- Not allowed: raw-image corruption reliability validation or causal reliability claims.
