#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0: 数据集加载器
支持加载预计算的Knowledge Note Embeddings（.npy格式）
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import json
from typing import Optional, Dict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.exp_5centers.train_bio_cot_5centers_multimodal import FiveCentersMultimodalDataset


class FiveCentersMultimodalDatasetV3_2(FiveCentersMultimodalDataset):
    """
    5中心多模态数据集加载器（Bio-COT 3.2版本）
    关键改进：返回图像文件名（用于VLM检索），不再需要预计算嵌入
    """
    
    def __init__(
        self,
        csv_path: str,
        transform=None,
        oct_num_frames: int = 20,
        max_col_images: int = 3,
        balance_negative_frames: bool = True,
    ):
        """
        Args:
            csv_path: CSV文件路径（labels.csv）
            knowledge_embed_path: Knowledge Note Embeddings文件路径（.npy格式）
            transform: 图像变换
            oct_num_frames: 每个样本使用的OCT帧数
            max_col_images: 每个样本使用的Colposcopy图像数
            balance_negative_frames: 如果True，阴性病人使用与阳性病人相同的帧数
        """
        # 调用父类初始化（加载图像和基础数据）
        super().__init__(
            csv_path=csv_path,
            transform=transform,
            oct_num_frames=oct_num_frames,
            max_col_images=max_col_images,
            balance_negative_frames=balance_negative_frames
        )
        print(f"✅ Bio-COT 3.2数据集初始化完成（使用VLM动态检索）")
    
    def __getitem__(self, idx):
        """
        获取一个样本（Bio-COT 3.2版本）
        
        关键改进：返回图像文件名列表，用于VLM检索
        
        Returns:
            dict: 包含图像、文件名、临床数据、标签等
        """
        # 调用父类方法获取基础数据
        sample = super().__getitem__(idx)
        
        # ⚠️ 关键改动：提取图像文件名列表（用于VLM检索）
        row = self.df.iloc[idx]
        
        # 1. 提取OCT图像文件名（使用第一帧作为代表）
        oct_paths_str = row['oct_paths']
        if pd.notna(oct_paths_str):
            oct_paths = [p.strip() for p in str(oct_paths_str).split(',')]
            if hasattr(self, 'oct_num_frames'):
                oct_paths = oct_paths[:self.oct_num_frames]
            oct_image_names = [Path(p).name for p in oct_paths if p]
            sample['oct_image_name'] = oct_image_names[0] if oct_image_names else "unknown.jpg"
        else:
            sample['oct_image_name'] = "unknown.jpg"
        
        # 2. 提取Colposcopy图像文件名（使用第一张作为代表）
        col_paths_str = row['col_paths']
        if pd.notna(col_paths_str):
            col_paths = [p.strip() for p in str(col_paths_str).split(',')]
            col_paths = col_paths[:self.max_col_images]
            col_image_names = [Path(p).name for p in col_paths if p]
            sample['col_image_name'] = col_image_names[0] if col_image_names else "unknown.jpg"
        else:
            sample['col_image_name'] = "unknown.jpg"
        
        # 3. 构造临床信息字符串（用于VLM增强）
        clinical_data = sample.get('clinical_data', {})
        hpv = clinical_data.get('hpv', 0)
        tct = clinical_data.get('tct', 'NILM')
        age = clinical_data.get('age', 50)
        
        clinical_info_str = f"HPV: {'positive' if hpv > 0 else 'negative'}, TCT: {tct}, Age: {int(age)}"
        sample['clinical_info_str'] = clinical_info_str
        
        # 4. 为了兼容，返回主要的图像文件名（OCT优先，因为OCT是主要模态）
        sample['image_name'] = sample['oct_image_name']
        
        return sample

