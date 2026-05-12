#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full HyDRA with variational modality reliability inference."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.full_pretrained_config import FullPretrainedHyDRAConfig


@dataclass
class HyDRAVariationalConfig(FullPretrainedHyDRAConfig):
    experiment_name: str = "HyDRA_Variational"
    experiment_description: str = "HyDRA with modality-specific variational reliability posterior inference."

    use_variational_reliability: bool = True
    use_center_aware_reliability: bool = True
    use_adaptive_gating: bool = False
    fusion_strategy: str = "variational"
    lambda_reliability_kl: float = 0.005

    output_dir: str = "paper_revision/results/hydra_variational/results"
    checkpoint_dir: str = "paper_revision/results/hydra_variational/checkpoints"
    log_dir: str = "paper_revision/results/hydra_variational/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
