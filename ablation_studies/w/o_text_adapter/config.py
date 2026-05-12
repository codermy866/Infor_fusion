#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Text Adapter (Bio-COT 3.2 + 5.0)
移除5.0的Text Adapter（VLM集成增强）
"""
from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

@dataclass
class NoTextAdapterConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_text_adapter"
    experiment_description: str = "Ablation: 移除5.0的Text Adapter（VLM集成增强）"
    
    num_epochs: int = 20
    use_text_adapter: bool = False  # 🔥 禁用Text Adapter
    
    # 保留其他5.0优势
    use_hierarchical: bool = True
    use_noise_aware: bool = True
    use_clinical_evolver: bool = True
    dropout_rate: float = 0.4
    drop_path_rate: float = 0.2
    
    output_dir: str = 'ablation_studies/w/o_text_adapter/results'
    checkpoint_dir: str = 'ablation_studies/w/o_text_adapter/checkpoints'
    log_dir: str = 'ablation_studies/w/o_text_adapter/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

