#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CoE supervision weight sweep: lambda_coe=0.02."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class CoELambda002Config(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "CoE_Lambda_0p02"
    experiment_description: str = "ELBO HyDRA with weak trajectory CoE supervision weight lambda_coe=0.02."

    use_coe_readout: bool = True
    use_coe_supervision: bool = True
    lambda_coe: float = 0.02

    output_dir: str = "paper_revision/results/coe_lambda_0p02/results"
    checkpoint_dir: str = "paper_revision/results/coe_lambda_0p02/checkpoints"
    log_dir: str = "paper_revision/results/coe_lambda_0p02/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
