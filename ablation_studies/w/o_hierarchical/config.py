#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Hierarchical Multi-Scale (Bio-COT 3.2 + 5.0)
移除5.0的分层多尺度特征提取
"""
from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

@dataclass
class NoHierarchicalConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_hierarchical"
    experiment_description: str = "Ablation: 移除5.0的分层多尺度特征提取"
    
    num_epochs: int = 20
    use_hierarchical: bool = False  # 🔥 禁用分层特征提取
    
    # 保留其他5.0优势
    use_noise_aware: bool = True
    use_clinical_evolver: bool = True
    use_text_adapter: bool = True
    dropout_rate: float = 0.4
    drop_path_rate: float = 0.2
    
    output_dir: str = 'ablation_studies/w/o_hierarchical/results'
    checkpoint_dir: str = 'ablation_studies/w/o_hierarchical/checkpoints'
    log_dir: str = 'ablation_studies/w/o_hierarchical/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

