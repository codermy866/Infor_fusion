#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Requirement ablation: replace text-derived ASCCP prototypes with random anchors."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.hydra_variational_config import HyDRAVariationalConfig


@dataclass
class AblationRandomASCCPPriorConfig(HyDRAVariationalConfig):
    experiment_name: str = "Ablation_RandomASCCPPrototypePrior"
    experiment_description: str = "HyDRA variational model with random trainable ASCCP prior anchors."

    use_asccp_prior: bool = True
    use_text_derived_asccp: bool = False

    output_dir: str = "paper_revision/results/ablation_random_asccp_prior/results"
    checkpoint_dir: str = "paper_revision/results/ablation_random_asccp_prior/checkpoints"
    log_dir: str = "paper_revision/results/ablation_random_asccp_prior/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
