#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Requirement ablation: remove trajectory-conditioned CoE readout entirely."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_patient_holdout_config import AllCenterPatientHoldoutHyDRAConfig


@dataclass
class AblationNoTrajectoryCoEConfig(AllCenterPatientHoldoutHyDRAConfig):
    experiment_name: str = "Ablation_NoTrajectoryCoE"
    experiment_description: str = "HyDRA variational model without trajectory-conditioned CoE readout."

    use_coe_readout: bool = False
    use_coe_supervision: bool = False
    lambda_coe: float = 0.0

    output_dir: str = "paper_revision/results/ablation_no_trajectory_coe/results"
    checkpoint_dir: str = "paper_revision/results/ablation_no_trajectory_coe/checkpoints"
    log_dir: str = "paper_revision/results/ablation_no_trajectory_coe/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
