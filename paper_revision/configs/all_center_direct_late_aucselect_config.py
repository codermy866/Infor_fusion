#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""All-center DirectLate baseline selected by internal validation AUC."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_direct_late_config import AllCenterDirectLateConfig


@dataclass
class AllCenterDirectLateAucSelectConfig(AllCenterDirectLateConfig):
    experiment_name: str = "DirectLateAUCSelect_AllCenterPatientHoldout"
    experiment_description: str = "Direct late-fusion baseline using internal validation AUC as checkpoint selection."
    checkpoint_metric: str = "auc"
    output_dir: str = "paper_revision/results/all_center_direct_late_aucselect/results"
    checkpoint_dir: str = "paper_revision/results/all_center_direct_late_aucselect/checkpoints"
    log_dir: str = "paper_revision/results/all_center_direct_late_aucselect/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
