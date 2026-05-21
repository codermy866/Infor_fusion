#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance-oriented ELBO HyDRA with explicit structured clinical prior."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_patient_holdout_config import AllCenterPatientHoldoutHyDRAConfig


@dataclass
class AllCenterELBOStructuredPriorConfig(AllCenterPatientHoldoutHyDRAConfig):
    experiment_name: str = "HyDRA_ELBO_StructuredPrior_AllCenter"
    experiment_description: str = (
        "ELBO HyDRA using cached patch features, explicit structured clinical prior, "
        "variational reliability, posterior refinement, ASCCP prior matching, and CoE supervision."
    )

    expected_aligned_n: int = 1897
    raw_registry_n: int = 3010
    expected_image_volume_approx: int = 130000
    input_modalities: list = None
    no_report_mode: bool = True
    use_vlm_retriever: bool = False
    vlm_json_path: str = None
    clinical_feature_dim: int = 14
    data_root: str = str(ROOT / "paper_revision" / "splits" / "target_adapted_validation" / "all_center_patient_holdout_70_10_20")
    feature_cache_path: str = str(ROOT / "paper_revision" / "cache" / "patch_features_final_1897.pt")
    guideline_prototype_path: str = str(ROOT / "paper_revision" / "configs" / "guideline_clinical_prototypes.json")
    asccp_prototype_path: str = str(ROOT / "paper_revision" / "configs" / "guideline_clinical_prototypes.json")
    primary_endpoint: str = "CIN2+"
    locked_sensitivity_target: float = 0.95
    threshold_rule: str = "max_specificity_at_sensitivity:0.95"
    load_clinical_semantic_adapter_path: str = None
    use_visual_notes: bool = False
    use_adaptive_gating: bool = False
    fusion_strategy: str = "variational"
    dropout_rate: float = 0.30
    lambda_reliability_kl: float = 0.005
    lambda_modality_likelihood: float = 0.03
    lambda_asccp_ot: float = 0.05
    lambda_coe: float = 0.05
    lambda_align: float = 0.25
    lambda_ot: float = 0.35
    checkpoint_metric: str = "auc_minus_ece"
    ece_penalty: float = 0.15

    output_dir: str = "paper_revision/results/elbo_structured_prior/results"
    checkpoint_dir: str = "paper_revision/results/elbo_structured_prior/checkpoints"
    log_dir: str = "paper_revision/results/elbo_structured_prior/logs"

    def __post_init__(self):
        if self.input_modalities is None:
            self.input_modalities = ["colposcopy", "oct", "clinical_text_hpv_tct_age"]
        super().__post_init__()
        self.use_vlm_retriever = False
        self.vlm_json_path = None
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
