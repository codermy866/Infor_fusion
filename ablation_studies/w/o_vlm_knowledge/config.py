#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验3：w/o VLM Knowledge
移除：Frozen VLM + Trainable Adapter + Visual Notes
替代：仅使用视觉特征（无VLM知识）
验证：医学知识整合对诊断性能的提升
"""

from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BioCOT_v3_2_Config

class AblationConfig_VLMKnowledge(BioCOT_v3_2_Config):
    """消融实验：移除VLM知识增强"""
    
    # ============================================================
    # 消融设置：移除VLM相关组件
    # ============================================================
    use_vlm_retriever: bool = False  # 移除VLM知识检索器
    use_text_adapter: bool = False  # 移除Text Adapter
    use_visual_notes: bool = False  # 移除Visual Notes模块
    
    # 替代方案：仅使用视觉特征
    # 注意：模型将不使用VLM知识，仅依赖视觉特征和临床信息
    
    # 保留其他所有组件
    use_adaptive_gating: bool = True
    use_noise_aware: bool = True
    use_hierarchical: bool = True
    use_clinical_evolver: bool = True
    
    # 输出目录
    output_dir: str = 'ablation_studies/w/o_vlm_knowledge/results'
    checkpoint_dir: str = 'ablation_studies/w/o_vlm_knowledge/checkpoints'
    log_dir: str = 'ablation_studies/w/o_vlm_knowledge/logs'

