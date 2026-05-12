#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 消融实验配置：w/o VLM Retriever
禁用VLM增强知识检索器，使用可学习的静态嵌入（类似3.1的note_projector）
"""

from dataclasses import dataclass
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # experiments/exp_bio3.2
sys.path.insert(0, str(ROOT))

from config import BioCOT_v3_2_Config


@dataclass
class BioCOT_v3_2_Config_WO_VLMRetriever(BioCOT_v3_2_Config):
    """禁用VLM Retriever的配置"""
    
    # 禁用VLM Retriever
    use_vlm_retriever: bool = False
    
    # 其他配置保持不变
    num_epochs: int = 20  # 消融实验使用20个epoch
    
    def __post_init__(self):
        super().__post_init__()
        # 确保VLM Retriever被禁用
        self.use_vlm_retriever = False
        print(f"⚠️ 消融实验：VLM Retriever已禁用，使用可学习的静态嵌入")

