#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full intended HyDRA configuration for reruns after the offline-safe check."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import BioCOT_v3_2_Config


@dataclass
class FullPretrainedHyDRAConfig(BioCOT_v3_2_Config):
    experiment_name: str = "HyDRA_Full_Pretrained"
    experiment_description: str = "Full HyDRA with pretrained ViT feature extractor and VLM text retriever."

    vit_pretrained: bool = True
    use_vlm_retriever: bool = True
    text_model_name: str = str(ROOT / "paper_revision" / "cache" / "pubmedbert_safetensors")

    output_dir: str = "paper_revision/results/full_pretrained/results"
    checkpoint_dir: str = "paper_revision/results/full_pretrained/checkpoints"
    log_dir: str = "paper_revision/results/full_pretrained/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
