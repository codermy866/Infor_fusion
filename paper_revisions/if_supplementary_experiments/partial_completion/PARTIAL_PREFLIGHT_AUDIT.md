# P06/P10/P11 Partial Completion Preflight Audit

Created at: `2026-06-03 10:18:40`.

Can P06 10/30/50% random dropout patient-level predictions be generated? YES.
Can P10 perturbed reliability weights be exported? YES, as feature-level proxy perturbations.
Can P11 CoE intervention logits be exported? NO, only clean proxy logits are available.
Can P11 visual saliency masks be generated? NO.

## Blockers

- P11 targeted/random/counterfactual intervention logits are not present in the locked outputs.
- P11 saliency or occlusion masks are not present in the locked outputs.
- Checkpoints exist, but this completion does not reconstruct raw-image dataloaders or modify the original training-time export contract.

## Inventory

                       component                                                                                                                                     path  exists  usable_for_p06  usable_for_p10  usable_for_p11              required_action                                                                                                      notes
standardized_prediction_registry                       paper_revisions/if_supplementary_experiments/01_prediction_registry/standardized_patient_mean_test_predictions.csv    True            True            True            True          use_locked_registry                                                        Contains patient-level hashes and all model scores.
    validation_locked_thresholds                                             paper_revisions/if_supplementary_experiments/02_cin3_safety/validation_locked_thresholds.csv    True            True           False           False        use_locked_thresholds                                                                   Provides fold-wise CIN2/CIN3 thresholds.
       clean_reliability_weights                             paper_revisions/if_supplementary_experiments/09_reliability_validation/reliability_weights_patient_level.csv    True           False            True           False             use_clean_export                                                      Clean reliability weights available at patient level.
          coe_clean_proxy_states                                      paper_revisions/if_supplementary_experiments/10_coe_faithfulness/coe_intervention_patient_level.csv    True           False           False            True                   audit_only                                         Clean CoE proxy states available, but no true intervention logits.
                 loco_split_file                             /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026/outputs/publishable_v2/splits/loco_folds_v2.json    True            True            True            True           protocol_reference                                                                                     Strict LOCO fold file.
            locked_feature_cache /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026/outputs/publishable_v2/step2_main_loco/audit/step2_locked_feature_arrays.npz    True            True            True            True not_loaded_for_raw_inference                                         Useful provenance, but completion uses locked prediction registry.
                dataset_manifest                         /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026/outputs/publishable_v2/data_lock/data_lock_n1897.csv    True            True            True            True                   audit_only                                                            Raw IDs/images are not exported in the package.
     shared_lora_checkpoint_root            /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_infofusion_2026/outputs/publishable_v2/shared_lora_biocot/improved_1897/ablations    True           False           False           False   future_raw_inference_route 87 checkpoint files found; full raw-image dataloader/inference export was not invoked for this completion.
           p06_completion_script                                                                      scripts/if_supplementary/complete_p06_random_dropout_predictions.py    True            True           False           False                run_completed                                                         Generates patient-level dropout proxy predictions.
           p10_completion_script                                                                    scripts/if_supplementary/complete_p10_perturbed_reliability_export.py    True           False            True           False                run_completed                                            Generates patient-level clean-vs-perturbed reliability weights.
           p11_completion_script                                                                         scripts/if_supplementary/complete_p11_coe_intervention_logits.py    True           False           False            True          run_completed_audit                                          Exports audit table; true intervention logits remain unavailable.
