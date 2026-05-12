#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: Baseline (Bio-COT 3.2)
移除所有高级模块，仅保留基础分类功能
"""

from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config


@dataclass
class BaselineConfig(BioCOT_v3_2_Config):
    """
    Baseline 配置：保留4.0和5.0的核心优势，移除其他高级模块
    
    4.0的核心优势（保留）：
    - ✅ 动态知识更新（VLMAugmentedRetriever）
    - ✅ VLM资源利用（VLM缓存）
    - ✅ 参数效率（Frozen Text Encoder + Trainable Adapter）
    - ✅ 丰富语义信息（VLM描述）
    
    5.0的核心优势（保留）：
    - ✅ 分层多尺度特征提取（HierarchicalViT）
    - ✅ 噪声感知流形超连接（NA-mHC）
    - ✅ 动态临床查询演化（ClinicalEvolver）
    - ✅ 激进正则化策略（Dropout 0.4, DropPath 0.2）
    - ✅ 正交损失（解耦方式）
    - ✅ Text Adapter（VLM集成增强）
    
    其他高级模块（移除）：
    - ❌ Visual Notes（3.1的优势）
    - ❌ OT Loss（3.1的优势）
    - ❌ Dual Head（3.1的优势）
    - ❌ Cross Attention（3.1的优势）
    - ❌ Adaptive Gating（3.1的优势）
    """
    
    experiment_name: str = "baseline"
    experiment_description: str = "Baseline: 保留4.0和5.0的核心优势，移除其他高级模块"
    
    num_epochs: int = 20
    
    # 移除其他高级模块（3.1的优势）
    use_visual_notes: bool = False
    use_ot: bool = False
    use_dual: bool = False
    use_cross_attn: bool = False
    use_adaptive_gating: bool = False
    
    # 关闭对应的损失权重
    lambda_align: float = 0.0
    lambda_ot: float = 0.0
    lambda_consist: float = 0.0
    lambda_adv: float = 0.0
    lambda_sparse: float = 0.0
    
    # 🔥 5.0优势：默认启用所有5.0特性
    use_hierarchical: bool = True  # 分层多尺度特征提取
    use_noise_aware: bool = True  # 噪声感知融合
    use_clinical_evolver: bool = True  # 动态临床查询演化
    use_text_adapter: bool = True  # Text Adapter
    dropout_rate: float = 0.4  # 激进正则化
    drop_path_rate: float = 0.2  # DropPath
    lambda_ortho: float = 0.5  # 正交损失权重
    lambda_noise: float = 0.1  # 噪声正则化损失权重
    
    output_dir: str = 'ablation_studies/baseline/results'
    checkpoint_dir: str = 'ablation_studies/baseline/checkpoints'
    log_dir: str = 'ablation_studies/baseline/logs'
    
    def __post_init__(self):
        self.num_epochs = 20
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

