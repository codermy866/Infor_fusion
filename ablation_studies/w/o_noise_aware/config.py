#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Noise-Aware MHC (Bio-COT 3.2 + 5.0)
移除5.0的噪声感知流形超连接
"""
from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

@dataclass
class NoNoiseAwareConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_noise_aware"
    experiment_description: str = "Ablation: 移除5.0的噪声感知流形超连接"
    
    num_epochs: int = 20
    use_noise_aware: bool = False  # 🔥 禁用噪声感知融合
    
    # 保留其他5.0优势
    use_hierarchical: bool = True
    use_clinical_evolver: bool = True
    use_text_adapter: bool = True
    dropout_rate: float = 0.4
    drop_path_rate: float = 0.2
    
    output_dir: str = 'ablation_studies/w/o_noise_aware/results'
    checkpoint_dir: str = 'ablation_studies/w/o_noise_aware/checkpoints'
    log_dir: str = 'ablation_studies/w/o_noise_aware/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

