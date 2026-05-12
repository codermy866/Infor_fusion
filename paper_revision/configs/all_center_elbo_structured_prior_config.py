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

    use_vlm_retriever: bool = False
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
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
