#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Clinical Evolver (Bio-COT 3.2 + 5.0)
移除5.0的动态临床查询演化
"""
from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

@dataclass
class NoClinicalEvolverConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_clinical_evolver"
    experiment_description: str = "Ablation: 移除5.0的动态临床查询演化"
    
    num_epochs: int = 20
    use_clinical_evolver: bool = False  # 🔥 禁用临床查询演化
    
    # 保留其他5.0优势
    use_hierarchical: bool = True
    use_noise_aware: bool = True
    use_text_adapter: bool = True
    dropout_rate: float = 0.4
    drop_path_rate: float = 0.2
    
    output_dir: str = 'ablation_studies/w/o_clinical_evolver/results'
    checkpoint_dir: str = 'ablation_studies/w/o_clinical_evolver/checkpoints'
    log_dir: str = 'ablation_studies/w/o_clinical_evolver/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

