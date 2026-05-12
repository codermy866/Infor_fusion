#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验1：w/o Multi-Modal Fusion
移除：自适应模态门控（AMCG）+ 噪声感知流形超连接（NA-mHC）
替代：简单特征拼接（Concatenation）
验证：自适应融合机制的必要性
"""

from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

class AblationConfig_MultiModalFusion(BioCOT_v3_2_Config):
    """消融实验：移除多模态融合机制"""
    
    # ============================================================
    # 消融设置：移除自适应多模态融合机制
    # ============================================================
    use_adaptive_gating: bool = False  # 移除自适应模态门控（AMCG）
    use_noise_aware: bool = False  # 移除噪声感知流形超连接（NA-mHC）
    
    # 替代方案：简单特征拼接
    # 注意：这需要在模型代码中实现简单拼接逻辑
    fusion_method: str = 'concatenation'  # 使用简单拼接
    
    # 保留其他5.0优势
    use_hierarchical: bool = True
    use_clinical_evolver: bool = True
    use_text_adapter: bool = True
    
    # 保留VLM知识增强
    use_visual_notes: bool = True
    use_vlm_retriever: bool = True
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_multi_modal_fusion/results'
    checkpoint_dir: str = 'ablation_studies/w/o_multi_modal_fusion/checkpoints'
    log_dir: str = 'ablation_studies/w/o_multi_modal_fusion/logs'

