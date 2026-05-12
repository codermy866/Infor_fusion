#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""All-center patient holdout direct concat-fusion baseline."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_patient_holdout_config import AllCenterPatientHoldoutHyDRAConfig


@dataclass
class AllCenterDirectConcatConfig(AllCenterPatientHoldoutHyDRAConfig):
    experiment_name: str = "DirectConcat_AllCenterPatientHoldout"
    experiment_description: str = "Same cached patch features with direct concatenation fusion and no posterior refinement."
    use_variational_reliability: bool = False
    fusion_strategy: str = "concat"
    direct_fusion_only: bool = True
    use_dual: bool = False
    use_ot: bool = False
    use_adaptive_gating: bool = False
    train_modality_dropout_prob: float = 0.08
    train_two_modality_dropout_prob: float = 0.02
    learning_rate: float = 0.0001
    output_dir: str = "paper_revision/results/all_center_direct_concat/results"
    checkpoint_dir: str = "paper_revision/results/all_center_direct_concat/checkpoints"
    log_dir: str = "paper_revision/results/all_center_direct_concat/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
