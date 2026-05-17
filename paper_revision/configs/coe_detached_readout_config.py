#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CoE readout with detached diagnosis features.

The CoE head still receives weak supervision, but its inputs are detached so
the CoE loss updates only the readout head and does not back-propagate into the
diagnosis branch.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class CoEDetachedReadoutConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "CoE_DetachedReadout"
    experiment_description: str = "ELBO HyDRA with detached CoE readout supervision."

    use_coe_readout: bool = True
    use_coe_supervision: bool = True
    detach_coe_readout_inputs: bool = True
    lambda_coe: float = 0.05

    output_dir: str = "paper_revision/results/coe_detached_readout/results"
    checkpoint_dir: str = "paper_revision/results/coe_detached_readout/checkpoints"
    log_dir: str = "paper_revision/results/coe_detached_readout/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
