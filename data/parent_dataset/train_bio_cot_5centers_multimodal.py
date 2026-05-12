#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT模型在5中心多模态数据集上的训练脚本
使用OCT + Colposcopy + Clinical数据（多模态）
支持多中心训练（5个医院）
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("⚠️ seaborn未安装，混淆矩阵将使用matplotlib绘制")
from datetime import datetime
import json
from PIL import Image
from typing import Dict, List, Tuple

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
# 添加exp_bio3.2本地的src目录到sys.path
local_exp_src = Path(__file__).resolve().parent.parent.parent / 'src'
if local_exp_src.exists():
    sys.path.insert(0, str(local_exp_src.parent))

from src.models.bida.bio_cot_model import BioCOTModel
from src.models.bida.prior_net import build_clinical_vector
from src.utils.anti_overfitting import FocalLoss


class FiveCentersMultimodalDataset(Dataset):
    """5中心多模态数据集加载器"""
    
    def __init__(
        self,
        csv_path: str,
        transform=None,
        oct_num_frames: int = 60,
        max_col_images: int = 3,
        balance_negative_frames: bool = True
    ):
        """
        Args:
            csv_path: CSV文件路径（labels.csv）
            transform: 图像变换
            oct_num_frames: 每个样本使用的OCT帧数（仅用于阳性病人，如果balance_negative_frames=True）
            max_col_images: 每个样本使用的Colposcopy图像数（最多3张）
            balance_negative_frames: 如果True，阴性病人使用与阳性病人相同的帧数
        """
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.oct_num_frames = oct_num_frames
        self.max_col_images = max_col_images
        self.balance_negative_frames = balance_negative_frames
        
        # 读取CSV文件
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='gbk')
            except Exception as e:
                raise ValueError(f"无法读取CSV文件: {e}")
        
        # 提取中心ID。若重划分CSV提供了稳定的5中心group id，则优先使用它，
        # 避免M20105/M20203或M0008/M22101被当成不同中心导致索引漂移。
        if 'center_group_id' in self.df.columns:
            self.df['center_id'] = self.df.get('center_name', self.df['center_group_id']).astype(str)
            self.df['center_idx'] = pd.to_numeric(self.df['center_group_id'], errors='coerce').fillna(0).astype(int)
            unique_centers = sorted(self.df['center_id'].unique())
            self.center_to_idx = {
                center: int(self.df.loc[self.df['center_id'] == center, 'center_idx'].iloc[0])
                for center in unique_centers
            }
        else:
            # 从oct_id中提取，例如M22105 -> 22105
            self.df['center_id'] = self.df['oct_id'].apply(self._extract_center_id)
            
            # 创建中心ID到数字的映射（旧格式兼容）
            unique_centers = sorted(self.df['center_id'].unique())
            self.center_to_idx = {center: idx for idx, center in enumerate(unique_centers)}
            self.df['center_idx'] = self.df['center_id'].map(self.center_to_idx)
        
        # 计算阳性病人的平均帧数（用于平衡阴性病人）
        if self.balance_negative_frames and 'is_positive_patient' in self.df.columns:
            pos_df = self.df[self.df['is_positive_patient'] == True]
            if len(pos_df) > 0:
                # 使用中位数更稳健，但至少20帧
                self.positive_median_frames = max(20, int(pos_df['oct_count'].median()))
                self.positive_mean_frames = max(20, int(pos_df['oct_count'].mean()))
                # 为了提高预测准确率，优先使用oct_num_frames（如果设置得合理）
                # 如果oct_num_frames在合理范围内（不超过中位数的2倍），使用oct_num_frames
                if self.oct_num_frames >= self.positive_median_frames and self.oct_num_frames <= self.positive_median_frames * 2.5:
                    self.unified_frames = self.oct_num_frames
                else:
                    # 否则使用中位数，但至少使用oct_num_frames（如果中位数太小）
                    self.unified_frames = max(self.positive_median_frames, min(self.oct_num_frames, self.positive_median_frames * 2))
                print(f"📊 帧数平衡策略:")
                print(f"   阳性病人中位数帧数: {self.positive_median_frames}")
                print(f"   设置的目标帧数: {self.oct_num_frames}")
                print(f"   统一使用帧数: {self.unified_frames} (阴性和阳性病人保持一致，提高预测准确率)")
            else:
                self.unified_frames = self.oct_num_frames
        else:
            self.unified_frames = self.oct_num_frames
        
        print(f"✅ 从 {csv_path} 加载了 {len(self.df)} 个样本")
        print(f"   标签分布: 阴性={sum(self.df['label']==0)}, 阳性={sum(self.df['label']==1)}")
        print(f"   中心分布: {self.df['center_idx'].value_counts().sort_index().to_dict()}")
    
    def _extract_center_id(self, oct_id: str) -> str:
        """从oct_id提取中心ID（例如M22105 -> 22105, M0008 -> 0008）"""
        import re
        if pd.isna(oct_id):
            return 'unknown'
        oct_id_str = str(oct_id)
        if oct_id_str.startswith('M'):
            # 使用正则表达式提取M后面的所有数字（例如M22105 -> 22105, M0008 -> 0008）
            match = re.match(r'M(\d+)', oct_id_str)
            if match:
                return match.group(1)
            # 如果正则匹配失败，使用原来的方法
            return oct_id_str[1:6] if len(oct_id_str) > 6 else oct_id_str[1:]
        return 'unknown'
    
    def __len__(self):
        return len(self.df)
    
    def _load_oct_frames(self, oct_paths_str: str, is_positive_patient: bool = False) -> torch.Tensor:
        """
        加载OCT帧序列
        Args:
            oct_paths_str: OCT路径字符串（分号分隔）
            is_positive_patient: 是否为阳性病人（用于决定使用的帧数）
        """
        # 如果启用平衡，所有样本使用统一帧数；否则使用oct_num_frames
        if self.balance_negative_frames:
            target_frames = self.unified_frames  # 阴性和阳性都使用统一帧数
        else:
            target_frames = self.oct_num_frames
        
        if pd.isna(oct_paths_str) or oct_paths_str == '':
            return torch.zeros(target_frames, 3, 224, 224)
        
        # 解析路径（分号分隔）
        oct_paths_raw = [p.strip() for p in str(oct_paths_str).split(';') if p.strip()]
        oct_paths = []
        for p in oct_paths_raw:
            path = Path(p)
            if path.exists():
                oct_paths.append(path)
            else:
                # 尝试查找文件
                filename = path.name
                # 在5centers_multi目录中查找
                base_dirs = [
                    Path('/data2/hmy/5Center_datas/5centers_multi/train/oct'),
                    Path('/data2/hmy/5Center_datas/5centers_multi/val/oct'),
                    Path('/data2/hmy/5Center_datas/5centers_multi/test/oct'),
                ]
                found = False
                for base_dir in base_dirs:
                    candidate = base_dir / filename
                    if candidate.exists():
                        oct_paths.append(candidate)
                        found = True
                        break
                if not found:
                    # 尝试递归查找
                    for base_dir in base_dirs:
                        candidates = list(base_dir.rglob(filename))
                        if candidates:
                            oct_paths.append(candidates[0])
                            break
        
        if not oct_paths:
            return torch.zeros(target_frames, 3, 224, 224)
        
        # 均匀采样到指定帧数
        if len(oct_paths) >= target_frames:
            indices = np.linspace(0, len(oct_paths) - 1, target_frames, dtype=int)
            oct_paths = [oct_paths[i] for i in indices]
        else:
            # 循环填充
            while len(oct_paths) < target_frames:
                oct_paths.extend(oct_paths)
            oct_paths = oct_paths[:target_frames]
        
        # 加载图像
        frames = []
        for oct_path in oct_paths:
            try:
                img = Image.open(oct_path).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)
                frames.append(img)
            except Exception as e:
                frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames) if frames else torch.zeros(target_frames, 3, 224, 224)
    
    def _load_colposcopy_images(self, col_paths_str: str) -> torch.Tensor:
        """加载Colposcopy图像"""
        if pd.isna(col_paths_str) or col_paths_str == '':
            return torch.zeros(self.max_col_images, 3, 224, 224)
        
        col_paths_raw = [p.strip() for p in str(col_paths_str).split(';') if p.strip()]
        col_paths = []
        for p in col_paths_raw:
            path = Path(p)
            if path.exists():
                col_paths.append(path)
            else:
                # 尝试查找文件
                filename = path.name
                base_dirs = [
                    Path('/data2/hmy/5Center_datas/5centers_multi/train/col'),
                    Path('/data2/hmy/5Center_datas/5centers_multi/val/col'),
                    Path('/data2/hmy/5Center_datas/5centers_multi/test/col'),
                ]
                for base_dir in base_dirs:
                    candidate = base_dir / filename
                    if candidate.exists():
                        col_paths.append(candidate)
                        break
                    else:
                        candidates = list(base_dir.rglob(filename))
                        if candidates:
                            col_paths.append(candidates[0])
                            break
        
        # 限制最多max_col_images张
        col_paths = col_paths[:self.max_col_images]
        
        images = []
        for col_path in col_paths:
            try:
                img = Image.open(col_path).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)
                images.append(img)
            except Exception as e:
                images.append(torch.zeros(3, 224, 224))
        
        # 填充到max_col_images张
        while len(images) < self.max_col_images:
            images.append(torch.zeros(3, 224, 224))
        
        return torch.stack(images[:self.max_col_images])
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 判断是否为阳性病人
        is_positive_patient = False
        if 'is_positive_patient' in row:
            is_positive_patient = bool(row['is_positive_patient'])
        
        # 加载OCT帧序列（根据是否为阳性病人使用不同的帧数）
        oct_images = self._load_oct_frames(row['oct_paths'], is_positive_patient=is_positive_patient)
        
        # 加载Colposcopy图像
        colposcopy_images = self._load_colposcopy_images(row['col_paths'])
        
        # 解析临床数据
        age = float(row['age']) if pd.notna(row['age']) else 50.0
        hpv_val = row['hpv']
        if pd.isna(hpv_val):
            hpv = 0.0
        elif isinstance(hpv_val, (int, float)):
            hpv = 1.0 if float(hpv_val) > 0 else 0.0
        else:
            hpv_str = str(hpv_val).lower()
            hpv = 1.0 if any(k in hpv_str for k in ['16', '18', 'positive', '阳性', '高危', '1']) else 0.0
        
        tct_str = str(row['tct']).upper() if pd.notna(row['tct']) else ''
        tct_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
        if 'ASC-US' in tct_str:
            tct_onehot[0] = 1.0
        elif 'ASC-H' in tct_str:
            tct_onehot[1] = 1.0
        elif 'LSIL' in tct_str:
            tct_onehot[2] = 1.0
        elif 'HSIL' in tct_str:
            tct_onehot[3] = 1.0
        elif 'SCC' in tct_str or '癌' in tct_str:
            tct_onehot[4] = 1.0
        
        clinical_features = torch.tensor([
            age / 100.0,  # 归一化年龄
            hpv,
            *tct_onehot
        ], dtype=torch.float32)
        
        clinical_data = {
            'hpv': int(hpv),
            'tct': tct_str,
            'age': age
        }
        
        label = int(row['label'])
        center_idx = int(row['center_idx'])
        oct_id = str(row['oct_id'])
        
        return {
            'oct_images': oct_images,
            'colposcopy_images': colposcopy_images,
            'clinical_features': clinical_features,
            'clinical_data': clinical_data,
            'label': torch.tensor(label, dtype=torch.long),
            'center_idx': torch.tensor(center_idx, dtype=torch.long),
            'oct_id': oct_id
        }


# 全局ViT模型实例（避免重复创建）
_vit_model = None
_vit_device = None

def extract_features_with_vit(images: torch.Tensor, device: torch.device, log_func=None) -> torch.Tensor:
    """
    使用Vision Transformer (ViT)提取图像特征（完全在GPU上运行）
    Args:
        images: [B, F, C, H, W] 或 [B, C, H, W] 或 [B, N, C, H, W]
        device: GPU设备
        log_func: 日志输出函数（可选）
    Returns:
        features: [B, 768] (ViT-Base的feature维度)
    """
    global _vit_model, _vit_device
    
    try:
        import timm
    except ImportError:
        raise ImportError("timm未安装，请先安装: pip install timm")
    
    # 只在第一次调用时创建模型，并确保在正确的GPU上
    if _vit_model is None or _vit_device != device:
        if log_func:
            log_func(f"📊 正在初始化ViT模型到GPU {device}...")
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # 创建ViT模型（使用CLS token作为特征）
            _vit_model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0,  # 去掉分类头
                global_pool='token',  # 使用CLS token（'token'对应CLS token）
            )
            
            # 立即移动到GPU
            _vit_model = _vit_model.to(device)
            _vit_model.eval()
            _vit_device = device
            
            if log_func:
                log_func(f"✅ ViT模型已加载到GPU {device}")
                # 检查GPU内存使用
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    log_func(f"   GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
            
            # 预热模型（避免第一次推理慢）
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224, device=device)
                _ = _vit_model(dummy_input)
    
    # 确保输入在GPU上（如果不在，则移动）
    if not images.is_cuda:
        images = images.to(device, non_blocking=True)
        if log_func:
            log_func(f"⚠️ 输入数据不在GPU上，已移动到 {device}")
    elif images.device != device:
        images = images.to(device, non_blocking=True)
        if log_func:
            log_func(f"⚠️ 输入数据在错误的GPU上，已移动到 {device}")
    
    # 特征提取（完全在GPU上）
    with torch.no_grad():
        if len(images.shape) == 5:  # [B, F, C, H, W] 或 [B, N, C, H, W]
            B, num_frames, C, H, W = images.shape
            images = images.view(B * num_frames, C, H, W)
            features = _vit_model(images)  # [B*num_frames, 768] - 已在GPU上
            features = features.view(B, num_frames, -1)
            features = features.mean(dim=1)  # [B, 768] 平均池化
        elif len(images.shape) == 4:  # [B, C, H, W]
            features = _vit_model(images)  # [B, 768] - 已在GPU上
        else:
            raise ValueError(f"Unsupported image shape: {images.shape}")
    
    return features


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    cls_losses = []
    ot_losses = []
    consist_losses = []
    adv_losses = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # 使用non_blocking异步传输到GPU，减少CPU等待时间
        oct_images = batch['oct_images'].to(device, non_blocking=True)
        colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
        clinical_features = batch['clinical_features'].to(device, non_blocking=True)
        clinical_data = [item for item in batch['clinical_data']]
        labels = batch['label'].to(device, non_blocking=True)
        center_labels = batch['center_idx'].to(device, non_blocking=True)
        
        # 每10个batch输出一次GPU使用情况（不在这里设置，避免类型错误）
        # GPU内存信息会在损失信息中一起显示
        
        optimizer.zero_grad()
        
        # 提取特征（使用ViT，完全在GPU上运行）
        oct_features = extract_features_with_vit(oct_images, device)
        colpo_features = extract_features_with_vit(colposcopy_images, device)
        
        # 准备clinical_data（batch级别字典）
        batch_clinical_data = {
            'hpv': [cd.get('hpv', 0) for cd in clinical_data],
            'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
            'age': [cd.get('age', 50.0) for cd in clinical_data]
        }
        
        # 前向传播
        outputs = model(
            oct_features=oct_features,
            colpo_features=colpo_features,
            clinical_features=clinical_features,
            clinical_data=batch_clinical_data,
            center_labels=center_labels,
            return_loss_components=True,
            use_counterfactual=True
        )
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        
        # 添加损失组件
        if 'loss_components' in outputs:
            loss_components = outputs['loss_components']
            loss = loss + loss_components.get('L_ot', 0) + loss_components.get('L_consist', 0) + loss_components.get('L_adv', 0)
            
            cls_losses.append(loss.item() - loss_components.get('L_ot', 0).item() - loss_components.get('L_consist', 0).item() - loss_components.get('L_adv', 0).item())
            ot_losses.append(loss_components.get('L_ot', 0).item())
            consist_losses.append(loss_components.get('L_consist', 0).item())
            adv_losses.append(loss_components.get('L_adv', 0).item())
        else:
            cls_losses.append(loss.item())
            ot_losses.append(0.0)
            consist_losses.append(0.0)
            adv_losses.append(0.0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        pos_pred = sum(all_preds)
        pos_prob = np.mean(all_probs) if all_probs else 0.0
        
        # 准备显示信息（确保所有值都是标量）
        loss_components = outputs.get('loss_components', {})
        if loss_components:
            # 确保损失组件是tensor，然后转换为标量
            ot_loss_tensor = loss_components.get('L_ot', torch.tensor(0.0, device=device))
            consist_loss_tensor = loss_components.get('L_consist', torch.tensor(0.0, device=device))
            adv_loss_tensor = loss_components.get('L_adv', torch.tensor(0.0, device=device))
            
            ot_loss_val = ot_loss_tensor.item() if torch.is_tensor(ot_loss_tensor) else float(ot_loss_tensor)
            consist_loss_val = consist_loss_tensor.item() if torch.is_tensor(consist_loss_tensor) else float(consist_loss_tensor)
            adv_loss_val = adv_loss_tensor.item() if torch.is_tensor(adv_loss_tensor) else float(adv_loss_tensor)
            
            cls_loss_val = loss.item() - ot_loss_val - consist_loss_val - adv_loss_val
        else:
            cls_loss_val = loss.item()
            ot_loss_val = 0.0
            consist_loss_val = 0.0
            adv_loss_val = 0.0
        
        # 获取GPU内存信息（每10个batch显示一次）
        gpu_mem_str = ''
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            gpu_mem_str = f', GPU={memory_allocated:.1f}GB/{memory_reserved:.1f}GB'
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss_val:.4f}',
            'ot': f'{ot_loss_val:.4f}',
            'consist': f'{consist_loss_val:.4f}',
            'adv': f'{adv_loss_val:.4f}',
            'acc': f'{acc:.4f}',
            'pos': f'{pos_pred}/{len(all_preds)}',
            'prob': f'{pos_prob:.3f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'cls_loss': np.mean(cls_losses) if cls_losses else 0.0,
        'ot_loss': np.mean(ot_losses) if ot_losses else 0.0,
        'consist_loss': np.mean(consist_losses) if consist_losses else 0.0,
        'adv_loss': np.mean(adv_losses) if adv_losses else 0.0
    }


def plot_feature_heatmap_vit(model, dataloader, device, output_dir, timestamp, num_samples=20):
    """绘制特征热图（z_causal, z_noise, z_sem的相似度热图）- 使用ViT特征提取"""
    model.eval()
    
    all_z_causal = []
    all_z_noise = []
    all_z_sem = []
    all_labels = []
    
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            oct_images = batch['oct_images'].to(device, non_blocking=True)
            colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
            clinical_features = batch['clinical_features'].to(device, non_blocking=True)
            clinical_data = batch['clinical_data']
            labels = batch['label'].to(device)
            
            # 使用ViT提取特征
            oct_features = extract_features_with_vit(oct_images, device)
            colpo_features = extract_features_with_vit(colposcopy_images, device)
            
            # 构建临床向量
            batch_clinical_data = {
                'hpv': [cd.get('hpv', 0) for cd in clinical_data],
                'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
                'age': [cd.get('age', 50.0) for cd in clinical_data]
            }
            
            center_labels = batch.get('center_idx', torch.zeros(len(labels), dtype=torch.long, device=device))
            if center_labels.device != device:
                center_labels = center_labels.to(device)
            
            outputs = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_features=clinical_features,
                clinical_data=batch_clinical_data,
                center_labels=center_labels,
                return_loss_components=False,
                use_counterfactual=False
            )
            
            all_z_causal.append(outputs['z_causal'].cpu())
            all_z_noise.append(outputs['z_noise'].cpu())
            all_z_sem.append(outputs['z_sem'].cpu())
            all_labels.append(labels.cpu())
            
            count += len(labels)
            if count >= num_samples:
                break
    
    # 拼接所有特征
    z_causal = torch.cat(all_z_causal, dim=0)[:num_samples]  # [N, 768]
    z_noise = torch.cat(all_z_noise, dim=0)[:num_samples]  # [N, 768]
    z_sem = torch.cat(all_z_sem, dim=0)[:num_samples]  # [N, 768]
    labels = torch.cat(all_labels, dim=0)[:num_samples]  # [N]
    
    # 计算相似度矩阵（余弦相似度）
    def cosine_similarity_matrix(x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        return torch.mm(x_norm, x_norm.t())
    
    sim_causal = cosine_similarity_matrix(z_causal).numpy()
    sim_noise = cosine_similarity_matrix(z_noise).numpy()
    sim_sem = cosine_similarity_matrix(z_sem).numpy()
    
    # 绘制热图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # z_causal相似度热图
    im1 = axes[0].imshow(sim_causal, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_title('z_causal Similarity Heatmap', fontsize=12, weight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')
    
    # z_noise相似度热图
    im2 = axes[1].imshow(sim_noise, cmap='plasma', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_title('z_noise Similarity Heatmap', fontsize=12, weight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[1], label='Cosine Similarity')
    
    # z_sem相似度热图
    im3 = axes[2].imshow(sim_sem, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[2].set_title('z_sem Similarity Heatmap', fontsize=12, weight='bold')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Sample Index')
    plt.colorbar(im3, ax=axes[2], label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_heatmap_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征热图已保存")


def validate(model, dataloader, criterion, device, epoch, args):
    """验证"""
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Val]')
        
        for batch in pbar:
            # 使用non_blocking异步传输到GPU
            oct_images = batch['oct_images'].to(device, non_blocking=True)
            colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
            clinical_features = batch['clinical_features'].to(device, non_blocking=True)
            clinical_data = [item for item in batch['clinical_data']]
            labels = batch['label'].to(device, non_blocking=True)
            center_labels = batch['center_idx'].to(device, non_blocking=True)
            
            # 提取特征（完全在GPU上运行）
            oct_features = extract_features_with_vit(oct_images, device)
            colpo_features = extract_features_with_vit(colposcopy_images, device)
            
            # 准备clinical_data
            batch_clinical_data = {
                'hpv': [cd.get('hpv', 0) for cd in clinical_data],
                'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
                'age': [cd.get('age', 50.0) for cd in clinical_data]
            }
            
            # 前向传播（验证时不需要计算损失组件）
            outputs = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_features=clinical_features,
                clinical_data=batch_clinical_data,
                center_labels=center_labels,
                return_loss_components=False,
                use_counterfactual=False
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算AUC和F1
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    f1 = f1_score(all_labels, all_preds)
    
    # 确保返回的是numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


# 导入可视化函数（从本地模块导入，确保exp_5centers文件夹完全独立）
from visualization_utils_5centers import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_loss_heatmap,
    plot_roc_curve,
    plot_prediction_distribution,
    plot_advanced_violin_analysis,
    plot_loss_boxplot,
    plot_metrics_comparison,
    plot_loss_component_analysis
)
# 注意：plot_feature_heatmap使用ResNet50，我们使用自定义的plot_feature_heatmap_vit（已在上面定义）


class BioCOT5CentersArgs:
    """5中心Bio-COT训练参数"""
    def __init__(self):
        self.data_root = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
        self.batch_size = 64  # 增加到64，充分利用GPU显存（A6000有48GB显存，可以支持更大batch）
        self.num_epochs = 1000  # 增加到1000个epoch，使训练曲线更平滑，适合论文展示
        self.learning_rate = 0.00012
        self.num_workers = 4  # 增加到4，加快数据加载速度（多进程并行加载数据）
        self.pin_memory = True  # 使用pin_memory加速CPU到GPU传输
        self.oct_frames = 20  # 使用20帧（基于阳性病人中位数），平衡准确率和训练速度
        self.colposcopy_images = 3
        
        # 输出目录（相对于当前脚本位置）
        script_dir = Path(__file__).resolve().parent
        self.output_dir = script_dir / 'results_multimodal'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = script_dir / 'checkpoints_multimodal'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = script_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)


def main():
    """主训练函数"""
    args = BioCOT5CentersArgs()
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f'train_bio_cot_5centers_multimodal_{timestamp}.log'
    
    # 重定向stdout和stderr到日志文件
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_f = open(log_file, 'w', encoding='utf-8', buffering=1)  # 行缓冲
    sys.stdout = log_f
    sys.stderr = log_f
    
    print(f"📝 日志文件: {log_file}")
    log_f.flush()  # 立即刷新
    print("=" * 80)
    print("=" * 80)
    print("Bio-COT 多模态训练（5中心数据集）")
    print("=" * 80)
    print(f"数据路径: {args.data_root}")
    
    # 强制使用GPU（优先使用cuda:1，如果显存不足则使用cuda:0）
    if torch.cuda.is_available():
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        free_memories = [int(x) for x in result.stdout.strip().split('\n')]
        
        # 优先使用cuda:1（显存更多），如果显存不足则使用cuda:0
        if len(free_memories) > 1 and free_memories[1] > 10000:  # cuda:1有超过10GB显存
            gpu_id = 1
        elif len(free_memories) > 0 and free_memories[0] > 10000:  # cuda:0有超过10GB显存
            gpu_id = 0
        else:
            # 选择显存最多的GPU
            gpu_id = int(np.argmax(free_memories))
        
        device = torch.device(f'cuda:{gpu_id}')
        print(f"✅ 使用GPU设备: {device} (GPU {gpu_id}, 可用显存: {free_memories[gpu_id]}MB)")
        print(f"   强制使用GPU，不使用CPU！")
        
        # 输出GPU详细信息
        print(f"\n📊 GPU详细信息:")
        print(f"   GPU名称: {torch.cuda.get_device_name(gpu_id)}")
        print(f"   总显存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f}GB")
        print(f"   当前已分配: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB")
        print(f"   当前已保留: {torch.cuda.memory_reserved(device) / 1024**3:.2f}GB")
        log_f.flush()
    else:
        raise RuntimeError("❌ CUDA不可用！请检查GPU环境！")
    
    print(f"Batch Size: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"OCT帧数: {args.oct_frames}")
    print(f"Colposcopy图像数: {args.colposcopy_images}")
    print("=" * 80)
    
    # 加载数据集
    train_csv = Path(args.data_root) / 'internal_train' / 'labels.csv'
    val_csv = Path(args.data_root) / 'internal_val' / 'labels.csv'
    test_csv = Path(args.data_root) / 'external_test' / 'labels.csv'
    
    train_dataset = FiveCentersMultimodalDataset(
        csv_path=str(train_csv),
        oct_num_frames=args.oct_frames,
        max_col_images=args.colposcopy_images,
        balance_negative_frames=True  # 启用帧数平衡
    )
    
    val_dataset = FiveCentersMultimodalDataset(
        csv_path=str(val_csv),
        oct_num_frames=args.oct_frames,
        max_col_images=args.colposcopy_images,
        balance_negative_frames=True  # 启用帧数平衡
    )
    
    # 获取中心数量
    num_centers = len(train_dataset.center_to_idx)
    print(f"\n📊 检测到 {num_centers} 个中心")
    log_f.flush()
    
    # 使用加权采样平衡类别
    print(f"📊 计算类别权重...")
    log_f.flush()
    # 直接从DataFrame获取标签，避免逐个加载样本
    train_labels = train_dataset.df['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(label)] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"✅ 加权采样器创建完成 (类别权重: {dict(zip(range(len(class_weights)), class_weights))})")
    log_f.flush()
    
    def collate_fn(batch):
        """自定义collate函数"""
        oct_images = torch.stack([item['oct_images'] for item in batch])
        colposcopy_images = torch.stack([item['colposcopy_images'] for item in batch])
        clinical_features = torch.stack([item['clinical_features'] for item in batch])
        clinical_data = [item['clinical_data'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        center_indices = torch.stack([item['center_idx'] for item in batch])
        oct_ids = [item['oct_id'] for item in batch]
        
        return {
            'oct_images': oct_images,
            'colposcopy_images': colposcopy_images,
            'clinical_features': clinical_features,
            'clinical_data': clinical_data,
            'label': labels,
            'center_idx': center_indices,
            'oct_id': oct_ids
        }
    
    print(f"\n📊 创建DataLoader...")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Num Workers: {args.num_workers}")
    print(f"   Pin Memory: {args.pin_memory}")
    log_f.flush()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,  # 保持worker进程，避免重复创建
        prefetch_factor=4 if args.num_workers > 0 else 2,  # 预取更多数据，减少等待时间
        drop_last=True  # 丢弃最后一个不完整的batch，保持训练稳定
    )
    
    print(f"✅ 训练DataLoader创建完成")
    print(f"   📊 DataLoader优化配置:")
    print(f"      Batch Size: {args.batch_size} (提升训练速度)")
    print(f"      Num Workers: {args.num_workers} (并行数据加载)")
    print(f"      Pin Memory: {args.pin_memory} (加速CPU到GPU传输)")
    print(f"      Persistent Workers: {True if args.num_workers > 0 else False} (保持worker进程)")
    print(f"      Prefetch Factor: {4 if args.num_workers > 0 else 2} (预取数据)")
    log_f.flush()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else 2
    )
    
    print(f"✅ 验证DataLoader创建完成")
    log_f.flush()
    
    # 检测特征维度
    print(f"\n📊 正在加载第一个batch以检测特征维度...")
    print(f"   这可能需要一些时间（加载图像和提取特征）...")
    log_f.flush()
    
    print(f"   开始迭代DataLoader...")
    log_f.flush()
    try:
        sample_batch = next(iter(train_loader))
        print(f"✅ Batch加载完成，OCT图像形状: {sample_batch['oct_images'].shape}")
        print(f"   Colposcopy图像形状: {sample_batch['colposcopy_images'].shape}")
        print(f"   数据设备: OCT={sample_batch['oct_images'].device}, Colpo={sample_batch['colposcopy_images'].device}")
        log_f.flush()
    except Exception as e:
        print(f"❌ 加载batch时出错: {e}")
        import traceback
        traceback.print_exc()
        log_f.flush()
        raise
    
    print(f"📊 正在提取ViT特征（GPU: {device}）...")
    log_f.flush()
    
    # 确保数据在GPU上
    sample_images = sample_batch['oct_images'][:1].to(device, non_blocking=True)
    print(f"   输入数据设备: {sample_images.device}")
    log_f.flush()
    
    # 检查GPU内存
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated(device) / 1024**3
        print(f"   GPU内存（提取前）: {memory_before:.2f}GB")
        log_f.flush()
    
    sample_oct_features = extract_features_with_vit(sample_images, device, log_func=lambda msg: (print(f"   {msg}"), log_f.flush()))
    input_dim = sample_oct_features.shape[1]
    
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated(device) / 1024**3
        print(f"   GPU内存（提取后）: {memory_after:.2f}GB")
        log_f.flush()
    
    print(f"\n✅ 检测到实际特征维度: {input_dim}")
    print(f"   特征数据设备: {sample_oct_features.device}")
    log_f.flush()
    
    # 创建模型
    print("\n📊 使用传统MLP图像编码器")
    model = BioCOTModel(
        embed_dim=768,
        num_classes=2,
        num_centers=num_centers,  # 使用实际中心数量
        input_dim=input_dim
    ).to(device)
    
    # 验证模型在GPU上
    next_param = next(model.parameters())
    if next_param.device.type != 'cuda':
        raise RuntimeError(f"❌ 模型未在GPU上！当前设备: {next_param.device}")
    print(f"✅ 模型已移动到GPU: {next_param.device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ 模型创建完成")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 使用更大的学习率以适应更大的batch size（线性缩放：batch_size从8到32，学习率也相应增加）
    scaled_lr = args.learning_rate * (args.batch_size / 8)  # 线性缩放学习率
    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=1e-5, betas=(0.9, 0.999))
    print(f"\n📊 优化器配置:")
    print(f"   基础学习率: {args.learning_rate}")
    print(f"   缩放后学习率: {scaled_lr:.6f} (batch_size={args.batch_size}, 线性缩放)")
    print(f"   权重衰减: 1e-5")
    log_f.flush()
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'train_cls_loss': [],
        'train_ot_loss': [],
        'train_consist_loss': [],
        'train_adv_loss': []
    }
    
    best_auc = 0.0
    best_epoch = 0
    
    print(f"\n🚀 开始训练...")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, args)
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_ot_loss'].append(train_metrics['ot_loss'])
        history['train_consist_loss'].append(train_metrics['consist_loss'])
        history['train_adv_loss'].append(train_metrics['adv_loss'])
        
        print(f"\nEpoch {epoch}/{args.num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Loss Components - CLS: {train_metrics['cls_loss']:.4f}, OT: {train_metrics['ot_loss']:.4f}, Consist: {train_metrics['consist_loss']:.4f}, Adv: {train_metrics['adv_loss']:.4f}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'history': history,
                'num_centers': num_centers
            }
            torch.save(checkpoint, args.checkpoint_dir / f'best_model_5centers_{timestamp}.pth')
            print(f"  ✅ 保存最佳模型 (AUC: {best_auc:.4f})")
    
    # 最终评估
    print(f"\n{'='*80}")
    print(f"训练完成！最佳AUC: {best_auc:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}")
    
    # 绘制训练曲线
    plot_training_curves(history, args.output_dir, timestamp)
    
    # 绘制损失热图
    plot_loss_heatmap(history, args.output_dir, timestamp)
    
    # 绘制损失箱线图
    plot_loss_boxplot(history, args.output_dir, timestamp)
    
    # 绘制指标对比图
    plot_metrics_comparison(history, args.output_dir, timestamp)
    
    # 绘制混淆矩阵（使用最佳模型的验证集结果）
    best_checkpoint = torch.load(args.checkpoint_dir / f'best_model_5centers_{timestamp}.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    final_val_metrics = validate(model, val_loader, criterion, device, args.num_epochs, args)
    cm = confusion_matrix(final_val_metrics['labels'], final_val_metrics['preds'])
    plot_confusion_matrix(cm, args.output_dir, timestamp)
    
    # 绘制ROC曲线
    plot_roc_curve(final_val_metrics['labels'], final_val_metrics['probs'], args.output_dir, timestamp)
    
    # 绘制预测概率分布（包含多种图表）
    plot_prediction_distribution(final_val_metrics['labels'], final_val_metrics['probs'], args.output_dir, timestamp)
    
    # 绘制高级小提琴图分析
    print("\n📊 生成高级小提琴图分析...")
    plot_advanced_violin_analysis(final_val_metrics['labels'], final_val_metrics['probs'], args.output_dir, timestamp)
    
    # 绘制损失组件详细分析
    print("\n📊 生成损失组件详细分析...")
    plot_loss_component_analysis(history, args.output_dir, timestamp)
    
    # 绘制特征热图（使用验证集样本，使用ViT特征提取）
    print("\n📊 生成特征热图...")
    try:
        plot_feature_heatmap_vit(model, val_loader, device, args.output_dir, timestamp, num_samples=min(20, len(val_dataset)))
    except Exception as e:
        print(f"⚠️ 特征热图生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存结果（包含完整数据，便于后续重新绘制）
    results = {
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'final_val_acc': final_val_metrics['acc'],
        'final_val_f1': final_val_metrics['f1'],
        'final_val_auc': final_val_metrics['auc'],
        'num_centers': num_centers,
        'history': history,
        'confusion_matrix': cm.tolist(),
        # 保存预测结果，便于后续重新绘制
        'final_val_metrics': {
            'labels': final_val_metrics['labels'].tolist() if isinstance(final_val_metrics['labels'], np.ndarray) else final_val_metrics['labels'],
            'probs': final_val_metrics['probs'].tolist() if isinstance(final_val_metrics['probs'], np.ndarray) else final_val_metrics['probs'],
            'preds': final_val_metrics['preds'].tolist() if isinstance(final_val_metrics['preds'], np.ndarray) else final_val_metrics['preds'],
            'acc': final_val_metrics['acc'],
            'auc': final_val_metrics['auc'],
            'f1': final_val_metrics['f1']
        },
        # 训练配置信息
        'training_config': {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'oct_frames': args.oct_frames,
            'colposcopy_images': args.colposcopy_images,
            'num_centers': num_centers
        }
    }
    
    results_file = args.output_dir / f'results_bio_cot_5centers_multimodal_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完整结果已保存到: {results_file}")
    print(f"💡 提示：使用 visualization_standalone.py 脚本可以重新绘制图表并自定义色调")
    print(f"   命令示例: python visualization_standalone.py --json_path {results_file}")
    
    print(f"\n✅ 结果已保存到: {args.output_dir}")
    
    # 恢复stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()
    
    print(f"\n✅ 训练完成！日志文件: {log_file}")


if __name__ == '__main__':
    main()
