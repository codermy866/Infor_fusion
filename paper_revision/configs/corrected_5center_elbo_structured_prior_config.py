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
from paper_revision.configs.corrected403_base_config import CORRECTED_ROOT, STAGE1_ADAPTER


@dataclass
class CorrectedFiveCenterELBOStructuredPriorConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE Full"
    experiment_description: str = (
        "Corrected five-hospital no-report retraining using canonical centre mapping: "
        "C01_WUHAN_RENMIN, C02_ENSHI, C03_XIANGYANG, C04_SHIYAN, C05_JINGZHOU."
    )

    num_centers: int = 5
    expected_external_n: int = 403
    vit_pretrained: bool = False
    load_clinical_semantic_adapter_path: str = str(STAGE1_ADAPTER)
    freeze_clinical_semantic_adapter_at_start: bool = True
    unfreeze_clinical_semantic_adapter_epoch: int = 5
    adapter_lr_multiplier: float = 0.1
    output_dir: str = str(CORRECTED_ROOT / "full_hydra_coe" / "results")
    checkpoint_dir: str = str(CORRECTED_ROOT / "full_hydra_coe" / "checkpoints")
    log_dir: str = str(CORRECTED_ROOT / "full_hydra_coe" / "logs")
    prediction_output_dir: str = str(CORRECTED_ROOT / "full_hydra_coe" / "predictions")

    def __post_init__(self):
        super().__post_init__()
        self.num_centers = 5
        self.no_report_mode = True
        self.use_vlm_retriever = False
        self.vlm_json_path = None
        for dir_name in [
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.prediction_output_dir,
        ]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
