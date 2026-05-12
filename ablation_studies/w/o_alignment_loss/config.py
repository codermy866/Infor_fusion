#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: w/o Alignment Loss (Bio-COT 3.2)
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config


@dataclass
class NoAlignmentLossConfig(BioCOT_v3_2_Config):
    experiment_name: str = "w/o_alignment_loss"
    experiment_description: str = "Ablation: 移除深度对齐损失"
    
    num_epochs: int = 20
    lambda_align: float = 0.0
    
    output_dir: str = 'ablation_studies/w/o_alignment_loss/results'
    checkpoint_dir: str = 'ablation_studies/w/o_alignment_loss/checkpoints'
    log_dir: str = 'ablation_studies/w/o_alignment_loss/logs'
    
    
    # 🔥 5.0优势：默认启用所有5.0特性
    use_hierarchical: bool = True  # 分层多尺度特征提取
    use_noise_aware: bool = True  # 噪声感知融合
    use_clinical_evolver: bool = True  # 动态临床查询演化
    use_text_adapter: bool = True  # Text Adapter
    dropout_rate: float = 0.4  # 激进正则化
    drop_path_rate: float = 0.2  # DropPath
    lambda_ortho: float = 0.5  # 正交损失权重
    lambda_noise: float = 0.1  # 噪声正则化损失权重

    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

