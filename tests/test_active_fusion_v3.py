import unittest

import torch

from models.active_fusion_v3 import (
    ActiveFusionV3,
    ActiveFusionV3Config,
    gradient_reverse,
)
from scripts.active_fusion_v3.run_v3 import select_by_retention_and_safety


class ActiveFusionV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = ActiveFusionV3Config(
            patch_dim=16,
            vlm_dim=8,
            case_vlm_dim=20,
            clinical_dim=6,
            hidden_dim=32,
            private_dim=16,
            num_heads=4,
            dropout=0.0,
            use_bicer=False,
        )
        tokens = {
            "colposcopy": torch.randn(4, 8),
            "oct": torch.randn(4, 8),
        }
        self.model = ActiveFusionV3(config, tokens)
        self.clinical = torch.randn(5, 6)
        self.col = torch.randn(5, 6, 16)
        self.oct = torch.randn(5, 6, 16)
        self.semantic = torch.randn(5, 20)

    def test_gradient_reversal_changes_only_backward_sign(self):
        value = torch.tensor([1.0, -2.0], requires_grad=True)
        output = gradient_reverse(value, 0.25)
        self.assertTrue(torch.equal(value, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(value.grad, torch.full_like(value, -0.25)))

    def test_vlm_anchor_does_not_change_clinical_only_risk(self):
        self.model.eval()
        first = self.model.all_subset_outputs(
            self.clinical, self.col, self.oct, self.semantic
        )
        second = self.model.all_subset_outputs(
            self.clinical, self.col, self.oct, self.semantic.flip(0)
        )
        self.assertTrue(
            torch.allclose(
                first["logit_clinical"],
                second["logit_clinical"],
                atol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                first["precision_weight_all"],
                second["precision_weight_all"],
                atol=1e-6,
            )
        )

    def test_multitask_losses_are_finite(self):
        y2 = torch.tensor([0, 1, 0, 1, 0])
        y3 = torch.tensor([0, 0, 0, 1, 0])
        centre = torch.tensor([0, 1, 2, 3, 4])
        loss, pieces = self.model.training_losses(
            self.clinical,
            self.col,
            self.oct,
            y2,
            self.semantic,
            cin3_labels=y3,
            centre_labels=centre,
            lambda_bicer=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        for key in ("anchor", "domain", "reliability", "cin3", "safety_error"):
            self.assertTrue(torch.isfinite(pieces[key]))

    def test_constant_vlm_expert_is_conditional_on_colposcopy(self):
        config = ActiveFusionV3Config(
            patch_dim=16,
            vlm_dim=8,
            case_vlm_dim=20,
            clinical_dim=6,
            hidden_dim=32,
            private_dim=16,
            num_heads=4,
            dropout=0.0,
            use_bicer=False,
            use_semantic_anchor=False,
            use_concept_risk_expert=True,
            concept_expert_gate_mode="constant",
        )
        model = ActiveFusionV3(
            config,
            {
                "colposcopy": torch.randn(4, 8),
                "oct": torch.randn(4, 8),
            },
        ).eval()
        first = model.all_subset_outputs(
            self.clinical, self.col, self.oct, self.semantic
        )
        second = model.all_subset_outputs(
            self.clinical, self.col, self.oct, self.semantic.flip(0)
        )
        self.assertTrue(
            torch.allclose(
                first["logit_clinical"],
                second["logit_clinical"],
                atol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                first["logit_clinical_colposcopy"],
                second["logit_clinical_colposcopy"],
                atol=1e-6,
            )
        )

    def test_policy_never_acquires_one_modality_twice(self):
        self.model.eval()
        output = self.model.run_policy(
            self.clinical,
            self.col,
            self.oct,
            self.semantic,
            cost_weight=-10.0,
            safety_threshold=0.5,
        )
        actions = output["actions"]
        for row in actions:
            acquired = [int(value) for value in row if int(value) != 0]
            self.assertEqual(len(acquired), len(set(acquired)))
        self.assertTrue((output["acquisition_count"] <= 2).all())

    def test_source_selection_enforces_performance_and_safety(self):
        def candidate(auprc, sensitivity, count, brier=0.2):
            return {
                "metrics": {
                    "cin2_auprc": auprc,
                    "cin3_sensitivity": sensitivity,
                    "mean_acquisition_count": count,
                    "cin2_brier": brier,
                }
            }

        selected, satisfied = select_by_retention_and_safety(
            [
                candidate(0.50, 0.90, 0.1),
                candidate(0.49, 0.96, 0.7),
                candidate(0.48, 0.97, 0.5),
            ],
            static_auprc=0.50,
            retention=0.95,
            cin3_sensitivity_floor=0.95,
        )
        self.assertTrue(satisfied)
        self.assertEqual(selected["metrics"]["mean_acquisition_count"], 0.5)

    def test_failed_source_safety_constraint_is_explicit(self):
        selected, satisfied = select_by_retention_and_safety(
            [
                {
                    "metrics": {
                        "cin2_auprc": 0.49,
                        "cin3_sensitivity": 0.90,
                        "mean_acquisition_count": 0.2,
                        "cin2_brier": 0.2,
                    }
                }
            ],
            static_auprc=0.50,
            retention=0.95,
            cin3_sensitivity_floor=0.95,
        )
        self.assertFalse(satisfied)
        self.assertEqual(selected["metrics"]["cin3_sensitivity"], 0.90)


if __name__ == "__main__":
    unittest.main()
