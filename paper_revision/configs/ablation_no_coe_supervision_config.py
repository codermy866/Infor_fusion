#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Requirement ablation: remove weak supervision from trajectory CoE readout."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.hydra_variational_config import HyDRAVariationalConfig


@dataclass
class AblationNoCoESupervisionConfig(HyDRAVariationalConfig):
    experiment_name: str = "Ablation_NoCoESupervision"
    experiment_description: str = "HyDRA variational model without weak trajectory CoE supervision."

    use_coe_readout: bool = True
    use_coe_supervision: bool = False
    lambda_coe: float = 0.0

    output_dir: str = "paper_revision/results/ablation_no_coe_supervision/results"
    checkpoint_dir: str = "paper_revision/results/ablation_no_coe_supervision/checkpoints"
    log_dir: str = "paper_revision/results/ablation_no_coe_supervision/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
