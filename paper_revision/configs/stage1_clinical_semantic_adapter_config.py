#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-1 no-report clinical semantic adapter configuration."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class Stage1ClinicalSemanticAdapterConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE_Stage1_ClinicalSemanticAdapter"
    experiment_description: str = (
        "Stage-1 no-report contrastive alignment between OCT/colposcopy visual evidence "
        "and HPV/TCT/Age clinical semantic variables. No labels, reports, or VLM evidence are used as targets."
    )

    stage1_epochs: int = 20
    stage1_batch_size: int = 32
    stage1_learning_rate: float = 1e-4
    lambda_align: float = 1.0
    lambda_recon: float = 0.25
    lambda_center: float = 0.0

    output_dir: str = "paper_revision/results/stage1_clinical_semantic_adapter/results"
    checkpoint_dir: str = "paper_revision/results/stage1_clinical_semantic_adapter/checkpoints"
    log_dir: str = "paper_revision/results/stage1_clinical_semantic_adapter/logs"

    def __post_init__(self):
        super().__post_init__()
        self.use_vlm_retriever = False
        self.vlm_json_path = None
        self.no_report_mode = True
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
