# CoE Faithfulness Report

## P11 Completion Status

Status: `PARTIAL_NOT_ESTABLISHED`.

A patient-level CoE intervention-logit table was exported, but only the clean reference logits are available.
Targeted, random-control, counterfactual, and visual saliency/occlusion intervention logits are not present in the locked outputs.
Therefore CoE faithfulness and causal explanation claims remain unsupported.

Patient-condition rows: `22764`.
Unique patients: `1286`.

## Visual Intervention Availability

  visual_intervention_type  saliency_available  occlusion_prediction_available  raw_image_intervention_executed         status                                                                                     blocker
      colposcopy_occlusion               False                           False                            False NOT_EXECUTABLE No saved saliency masks or occlusion-inference logits were available in the locked outputs.
             oct_occlusion               False                           False                            False NOT_EXECUTABLE No saved saliency masks or occlusion-inference logits were available in the locked outputs.
random_visual_mask_control               False                           False                            False NOT_EXECUTABLE           Random visual controls require the same unavailable occlusion-inference pipeline.

## Claim Boundary

- Allowed: clean CoE proxy state visualization and explicit audit of missing intervention evidence.
- Not allowed: CoE faithfulness, causal explanation, or saliency-grounded claims.
