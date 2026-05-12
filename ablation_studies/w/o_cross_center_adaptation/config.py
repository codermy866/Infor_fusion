#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验2：w/o Cross-Center Adaptation
移除：NA-mHC中的中心差异建模
替代：标准流形超连接（无中心差异建模）
验证：显式建模中心差异对泛化能力的重要性
"""

from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

class AblationConfig_CrossCenterAdaptation(BioCOT_v3_2_Config):
    """消融实验：移除跨中心适应机制"""
    
    # ============================================================
    # 消融设置：移除中心差异建模
    # ============================================================
    use_noise_aware: bool = True  # 保留NA-mHC结构
    use_center_specific_noise: bool = False  # 移除中心特定噪声建模
    
    # 替代方案：标准MHC（无中心差异建模）
    # 注意：这需要在模型代码中实现标准MHC（不使用中心差异）
    mhc_use_center_noise: bool = False  # 禁用中心噪声建模
    
    # 保留其他所有组件
    use_adaptive_gating: bool = True
    use_hierarchical: bool = True
    use_clinical_evolver: bool = True
    use_text_adapter: bool = True
    use_visual_notes: bool = True
    use_vlm_retriever: bool = True
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_cross_center_adaptation/results'
    checkpoint_dir: str = 'ablation_studies/w/o_cross_center_adaptation/checkpoints'
    log_dir: str = 'ablation_studies/w/o_cross_center_adaptation/logs'

