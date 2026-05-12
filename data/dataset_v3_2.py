#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: 数据集加载器
关键改进：返回图像文件名（用于VLM检索）- 从4.0引入
保留3.1的基础功能
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

# 从exp_bio3.2本地的parent_dataset目录导入父类
import sys
local_parent_path = Path(__file__).resolve().parent / 'parent_dataset'
if local_parent_path.exists():
    sys.path.insert(0, str(local_parent_path))
    from train_bio_cot_5centers_multimodal import FiveCentersMultimodalDataset
else:
    raise ImportError(f"无法找到FiveCentersMultimodalDataset父类，请检查路径: {local_parent_path}")


class FiveCentersMultimodalDatasetV3_2(FiveCentersMultimodalDataset):
    """
    5中心多模态数据集加载器（Bio-COT 3.2版本）
    
    关键改进（从4.0引入）：
    1. 返回图像文件名（用于VLM检索）
    2. 返回临床信息字符串（用于VLM增强）
    
    保留3.1的基础功能：
    - 多模态图像加载（OCT + Colposcopy）
    - 标签和中心信息
    """
    
    def __init__(
        self,
        csv_path: str,
        transform=None,
        oct_num_frames: int = 20,
        max_col_images: int = 3,
        balance_negative_frames: bool = True,
        data_root: str = None,
    ):
        """
        Args:
            csv_path: CSV文件路径（labels.csv）
            transform: 图像变换
            oct_num_frames: 每个样本使用的OCT帧数
            max_col_images: 每个样本使用的Colposcopy图像数
            balance_negative_frames: 如果True，阴性病人使用与阳性病人相同的帧数
            data_root: 数据根目录（用于新格式数据集）
        """
        self.data_root = Path(data_root) if data_root else Path(csv_path).parent
        self.csv_path = Path(csv_path)
        
        # 读取CSV并检查格式
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        # 检查是否是新格式（有OCT列但没有oct_paths列）
        if 'OCT' in df.columns and 'oct_paths' not in df.columns:
            # 新格式：需要构建路径
            print(f"📊 检测到新格式CSV，正在构建图像路径...")
            df = self._convert_new_format_to_old(df, csv_path)
            # 保存临时CSV供父类使用
            temp_csv = Path(csv_path).parent / f"temp_{Path(csv_path).name}"
            df.to_csv(temp_csv, index=False, encoding='utf-8')
            csv_path = str(temp_csv)
        
        # 调用父类初始化（加载图像和基础数据）
        super().__init__(
            csv_path=csv_path,
            transform=transform,
            oct_num_frames=oct_num_frames,
            max_col_images=max_col_images,
            balance_negative_frames=balance_negative_frames
        )
    
    def __getitem__(self, idx):
        """
        获取一个样本（Bio-COT 3.2版本）
        
        关键改进：返回图像文件名列表，用于VLM检索
        
        Returns:
            dict: 包含图像、文件名、临床数据、标签等
        """
        # 调用父类方法获取基础数据
        sample = super().__getitem__(idx)
        
        # ⚠️ 关键改动：提取图像文件名列表（从4.0引入）
        row = self.df.iloc[idx]
        
        # 1. 提取OCT图像文件名
        oct_paths_str = row['oct_paths']
        if pd.notna(oct_paths_str):
            # 支持分号和逗号分隔
            if ';' in str(oct_paths_str):
                oct_paths = [p.strip() for p in str(oct_paths_str).split(';') if p.strip()]
            else:
                oct_paths = [p.strip() for p in str(oct_paths_str).split(',') if p.strip()]
            # 只取实际使用的帧数
            if hasattr(self, 'oct_num_frames'):
                oct_paths = oct_paths[:self.oct_num_frames]
            
            # 提取文件名（不含路径）
            oct_image_names = [Path(p).name for p in oct_paths if p]
            # 返回第一帧的文件名作为代表（用于VLM检索）
            sample['oct_image_name'] = oct_image_names[0] if oct_image_names else "unknown.jpg"
        else:
            sample['oct_image_name'] = "unknown.jpg"
        
        # 2. 提取Colposcopy图像文件名
        col_paths_str = row['col_paths']
        if pd.notna(col_paths_str):
            # 支持分号和逗号分隔
            if ';' in str(col_paths_str):
                col_paths = [p.strip() for p in str(col_paths_str).split(';') if p.strip()]
            else:
                col_paths = [p.strip() for p in str(col_paths_str).split(',') if p.strip()]
            col_paths = col_paths[:self.max_col_images]
            
            # 提取文件名
            col_image_names = [Path(p).name for p in col_paths if p]
            # 返回第一张图像的文件名
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
        
        # 4. 为了兼容，返回主要的图像文件名（OCT优先）
        # 如果模型需要单个文件名，使用oct_image_name
        sample['image_name'] = sample['oct_image_name']
        
        # 5. 为了兼容训练脚本，添加image_names和clinical_info
        # ⚠️ 注意：DataLoader的默认collate_fn会将列表合并，所以这里直接返回字符串
        # 训练脚本需要手动处理batch合并
        sample['image_names'] = sample['oct_image_name']  # 字符串，不是列表
        sample['clinical_info'] = clinical_info_str  # 字符串，不是列表
        
        return sample
    
    def _convert_new_format_to_old(self, df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
        """将新格式CSV转换为父类期望的格式"""
        import glob
        
        # 重命名列
        df = df.rename(columns={
            'OCT': 'oct_id',
            'AGE': 'age',
            'HPV清洗': 'hpv',
            'TCT清洗': 'tct',
            'ID': 'patient_id'
        })
        
        # 确定优先数据目录（根据CSV路径判断是训练集、验证集还是外部测试集）。
        # 同时保留全局候选目录，支持后续leave-one-center-out重划分后的混合CSV。
        csv_path_str = str(self.csv_path) if hasattr(self, 'csv_path') else ''
        csv_path_lower = csv_path_str.lower()
        all_oct_bases = [
            self.data_root / 'internal_train' / 'train' / 'oct',
            self.data_root / 'internal_train' / 'val' / 'oct',
            self.data_root / 'external_validation' / 'oct',
        ]
        all_col_bases = [
            self.data_root / 'internal_train' / 'train' / 'col',
            self.data_root / 'internal_train' / 'val' / 'col',
            self.data_root / 'external_validation' / 'col',
        ]
        if 'external' in csv_path_lower or 'test' in csv_path_lower:
            oct_base = self.data_root / 'external_validation' / 'oct'
            col_base = self.data_root / 'external_validation' / 'col'
        elif 'train' in csv_path_lower or 'train_labels' in csv_path_lower:
            oct_base = self.data_root / 'internal_train' / 'train' / 'oct'
            col_base = self.data_root / 'internal_train' / 'train' / 'col'
        elif 'val' in csv_path_lower or 'val_labels' in csv_path_lower:
            oct_base = self.data_root / 'internal_train' / 'val' / 'oct'
            col_base = self.data_root / 'internal_train' / 'val' / 'col'
        else:
            # 尝试自动检测
            oct_base = self.data_root / 'internal_train' / 'train' / 'oct'
            col_base = self.data_root / 'internal_train' / 'train' / 'col'
            if not oct_base.exists():
                oct_base = self.data_root / 'internal_train' / 'val' / 'oct'
                col_base = self.data_root / 'internal_train' / 'val' / 'col'
        
        # 构建oct_paths和col_paths
        oct_paths_list = []
        col_paths_list = []
        
        for idx, row in df.iterrows():
            oct_id = str(row['oct_id'])
            patient_id = str(row['patient_id'])
            
            # 构建OCT路径
            oct_dir = oct_base / oct_id
            if not oct_dir.exists():
                for base in all_oct_bases:
                    candidate = base / oct_id
                    if candidate.exists():
                        oct_dir = candidate
                        break
            if oct_dir.exists():
                oct_files = sorted(glob.glob(str(oct_dir / '*.png')))
                oct_paths_list.append(';'.join(oct_files))
            else:
                oct_paths_list.append('')
            
            # 构建Colposcopy路径
            col_dir = col_base / patient_id
            if not col_dir.exists():
                for base in all_col_bases:
                    candidate = base / patient_id
                    if candidate.exists():
                        col_dir = candidate
                        break
            if col_dir.exists():
                col_files = sorted(glob.glob(str(col_dir / '*.jpg')) + glob.glob(str(col_dir / '*.png')))
                col_paths_list.append(';'.join(col_files))
            else:
                col_paths_list.append('')
        
        df['oct_paths'] = oct_paths_list
        df['col_paths'] = col_paths_list
        df['oct_count'] = [len(p.split(';')) if p else 0 for p in oct_paths_list]
        df['col_count'] = [len(p.split(';')) if p else 0 for p in col_paths_list]
        df['is_positive_patient'] = df['label'] == 1
        
        return df
