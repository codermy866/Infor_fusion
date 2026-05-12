#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HyDRA config for the rebuilt full-multimodal Enshi-external split."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.full_pretrained_config import FullPretrainedHyDRAConfig


@dataclass
class FullMultimodalEnshiExternalConfig(FullPretrainedHyDRAConfig):
    experiment_name: str = "HyDRA_Resplit_EnshiExternal"
    experiment_description: str = "Full-multimodal resplit with Enshi held out as a larger external center."

    data_root: str = str(ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "recommended_enshi_external")
    num_centers: int = 5

    output_dir: str = "paper_revision/results/resplit_enshi_external/results"
    checkpoint_dir: str = "paper_revision/results/resplit_enshi_external/checkpoints"
    log_dir: str = "paper_revision/results/resplit_enshi_external/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
