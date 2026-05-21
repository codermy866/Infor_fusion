#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Corrected five-centre HyDRA-CoE retraining config.

This config is identical in experimental scope to the main no-report ELBO
structured-prior setting, but writes all retraining outputs to an isolated
folder so corrected five-hospital results cannot be confused with legacy runs.
"""

from dataclasses import dataclass
from pathlib import Path

from paper_revision.configs.all_center_elbo_structured_prior_config import (
    AllCenterELBOStructuredPriorConfig,
)


@dataclass
class CorrectedFiveCenterELBOStructuredPriorConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE_Corrected5Center_ELBO_StructuredPrior"
    experiment_description: str = (
        "Corrected five-hospital no-report retraining using canonical centre mapping: "
        "C01_WUHAN_RENMIN, C02_ENSHI, C03_XIANGYANG, C04_SHIYAN, C05_JINGZHOU."
    )

    num_centers: int = 5
    # Evaluation and training use cached patch features in this corrected run,
    # so initializing the unused ViT path should not contact external services.
    vit_pretrained: bool = False
    output_dir: str = "paper_revision/results/corrected_5center_retrain/elbo_structured_prior/results"
    checkpoint_dir: str = "paper_revision/results/corrected_5center_retrain/elbo_structured_prior/checkpoints"
    log_dir: str = "paper_revision/results/corrected_5center_retrain/elbo_structured_prior/logs"

    def __post_init__(self):
        super().__post_init__()
        self.num_centers = 5
        self.no_report_mode = True
        self.use_vlm_retriever = False
        self.vlm_json_path = None
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
