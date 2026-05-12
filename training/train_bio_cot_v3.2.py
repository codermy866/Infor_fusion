#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.0: 训练脚本
实现动态Beta策略和稀疏性损失
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 在导入torch之前设置CUDA_VISIBLE_DEVICES和显存优化环境变量
import os
# 🔧 PyTorch显存优化：避免碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 🔧 消除tokenizers多进程警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Respect an explicit CUDA_VISIBLE_DEVICES value. If none is provided, choose
# the GPU with the lowest utilization before importing torch.
if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ.get('CUDA_VISIBLE_DEVICES', '').strip() == '':
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_usage = {}
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 2:
                    gpu_id = int(parts[0])
                    usage = int(parts[1])
                    gpu_usage[gpu_id] = usage
            if gpu_usage:
                best_gpu = min(gpu_usage.items(), key=lambda x: x[1])[0]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
                print(f"✅ 自动选择GPU {best_gpu} (使用率: {gpu_usage[best_gpu]}%)")
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    except Exception:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, 
    matthews_corrcoef, average_precision_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bio_cot_v3_2 import BioCOT_v3_2, create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from data.cached_patch_dataset import CachedPatchFeatureDataset
from config import BioCOT_v3_2_Config
# 修复漏洞1：使用新的函数提取Patch特征（丢弃[CLS]）
try:
    from .extract_vit_patches import extract_patch_features_with_vit
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from training.extract_vit_patches import extract_patch_features_with_vit
# 从exp_bio3.2本地的src目录导入
import sys
from pathlib import Path
# 添加exp_bio3.2的src目录到sys.path
local_src_path = Path(__file__).resolve().parent.parent / 'src'
if local_src_path.exists():
    sys.path.insert(0, str(local_src_path.parent))
from src.utils.anti_overfitting import FocalLoss


def visualize_training(history: Dict, log_dir: Path, timestamp: str, best_auc: float):
    """
    生成训练过程可视化图表
    
    Args:
        history: 训练历史字典
        log_dir: 日志目录
        timestamp: 时间戳
        best_auc: 最佳AUC值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. AUC曲线
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=best_auc, color='r', linestyle='--', linewidth=2, label=f'Best AUC: {best_auc:.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('AUC', fontsize=12)
    ax3.set_title('Validation AUC', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. F1-Score曲线
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, history['val_f1'], 'm-', label='Val F1', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('F1-Score', fontsize=12)
    ax4.set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Loss组件分解
    ax5 = plt.subplot(2, 3, 5)
    if history.get('cls_loss') and len(history['cls_loss']) > 0:
        ax5.plot(epochs, history['cls_loss'], 'b-', label='Classification Loss', linewidth=2)
    if history.get('ot_loss') and len(history['ot_loss']) > 0 and any(v > 0 for v in history['ot_loss']):
        ax5.plot(epochs, history['ot_loss'], 'g-', label='OT Loss', linewidth=2)
    if history.get('align_loss') and len(history['align_loss']) > 0 and any(v > 0 for v in history['align_loss']):
        ax5.plot(epochs, history['align_loss'], 'cyan', label='🔥 Alignment Loss', linewidth=2)  # 🔥 [NEW]
    if history.get('sparse_loss') and len(history['sparse_loss']) > 0 and any(v > 0 for v in history['sparse_loss']):
        ax5.plot(epochs, history['sparse_loss'], 'orange', label='Sparse Loss', linewidth=2)
    if history.get('consist_loss') and len(history['consist_loss']) > 0 and any(v > 0 for v in history['consist_loss']):
        ax5.plot(epochs, history['consist_loss'], 'purple', label='Consistency Loss', linewidth=2)
    if history.get('adv_loss') and len(history['adv_loss']) > 0 and any(v > 0 for v in history['adv_loss']):
        ax5.plot(epochs, history['adv_loss'], 'r-', label='Adversarial Loss', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')  # 使用对数刻度
    
    # 6. 综合性能指标
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, history['val_auc'], 'g-', label='AUC', linewidth=2, marker='o', markersize=4)
    ax6.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2, marker='s', markersize=4)
    ax6.plot(epochs, history['val_f1'], 'm-', label='F1-Score', linewidth=2, marker='^', markersize=4)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Comprehensive Performance Metrics', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # 添加总标题
    fig.suptitle(f'Bio-COT 3.1 Logic Loop Training Results (Best AUC: {best_auc:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存图表
    output_path = log_dir / f"training_curves_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 可视化图表已保存: {output_path}")


def get_dynamic_beta(epoch: int, max_epochs: int = 100) -> float:
    """
    动态Beta策略（Warm-up）
    
    Args:
        epoch: 当前epoch
        max_epochs: 总epoch数
    
    Returns:
        beta: 背景抑制系数
    """
    # 前5个Epoch不进行过滤 (beta=1.0)，让模型先学全局
    if epoch < 5:
        current_beta = 1.0
    elif epoch < 20:
        # 线性衰减: 1.0 -> 0.1
        current_beta = 1.0 - (0.9 * (epoch - 5) / 15)
    else:
        current_beta = 0.1  # 强过滤模式
    
    return current_beta


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Compute ECE for validation-time checkpoint selection."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y_true_arr.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_arr >= left) & (y_prob_arr <= right)
        else:
            mask = (y_prob_arr >= left) & (y_prob_arr < right)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(y_true_arr[mask].mean() - y_prob_arr[mask].mean()))
    return ece


def checkpoint_score(val_results: Dict[str, float], config) -> float:
    """Validation score used for model selection."""
    metric = getattr(config, "checkpoint_metric", "auc")
    auc = float(val_results.get("auc", 0.0))
    if metric == "auc_minus_ece":
        penalty = float(getattr(config, "ece_penalty", 0.25))
        return auc - penalty * float(val_results.get("ece", 0.0))
    if metric == "auc_plus_auprc":
        return auc + 0.25 * float(val_results.get("pr_auc", 0.0))
    return auc


def corrupt_images(images: torch.Tensor, config, modality: str) -> torch.Tensor:
    """Light training-time corruption for domain robustness."""
    prob = float(getattr(config, "train_image_corruption_prob", 0.0))
    if prob <= 0.0 or torch.rand((), device=images.device).item() >= prob:
        return images
    out = images.clone()
    if modality == "oct":
        if torch.rand((), device=out.device).item() < 0.5:
            out = out + torch.randn_like(out) * float(getattr(config, "oct_speckle_std", 0.06))
        if torch.rand((), device=out.device).item() < 0.5:
            stripe_width = max(1, int(out.shape[-1] * 0.03))
            x0 = int(torch.randint(0, max(1, out.shape[-1] - stripe_width), (1,), device=out.device).item())
            out[..., x0 : x0 + stripe_width] = out[..., x0 : x0 + stripe_width] * 0.55
    else:
        if torch.rand((), device=out.device).item() < 0.5:
            scale = 0.75 + torch.rand((), device=out.device).item() * 0.5
            out = out * scale
        if torch.rand((), device=out.device).item() < 0.5:
            h, w = out.shape[-2:]
            occ_h, occ_w = max(1, h // 5), max(1, w // 5)
            y0 = int(torch.randint(0, max(1, h - occ_h), (1,), device=out.device).item())
            x0 = int(torch.randint(0, max(1, w - occ_w), (1,), device=out.device).item())
            out[..., y0 : y0 + occ_h, x0 : x0 + occ_w] = 0.0
    return out.clamp(-3.0, 3.0)


def apply_modality_dropout(
    oct_features: torch.Tensor,
    colpo_features: torch.Tensor,
    clinical_features: torch.Tensor | None,
    clinical_info,
    config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, object]:
    """Drop modalities during training to reduce reliance on center-specific cues."""
    p_one = float(getattr(config, "train_modality_dropout_prob", 0.0))
    p_two = float(getattr(config, "train_two_modality_dropout_prob", 0.0))
    if p_one <= 0.0 and p_two <= 0.0:
        return oct_features, colpo_features, clinical_features, clinical_info

    batch_size = oct_features.shape[0]
    device = oct_features.device
    drop_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
    for i in range(batch_size):
        draw = torch.rand((), device=device).item()
        if draw < p_two:
            selected = torch.randperm(3, device=device)[:2]
            drop_mask[i, selected] = True
        elif draw < p_two + p_one:
            selected = torch.randint(0, 3, (1,), device=device)
            drop_mask[i, selected] = True

    if drop_mask[:, 0].any():
        oct_features = oct_features.clone()
        oct_features[drop_mask[:, 0]] = 0.0
    if drop_mask[:, 1].any():
        colpo_features = colpo_features.clone()
        colpo_features[drop_mask[:, 1]] = 0.0
    if drop_mask[:, 2].any():
        if clinical_features is not None:
            clinical_features = clinical_features.clone()
            clinical_features[drop_mask[:, 2]] = 0.0
        if isinstance(clinical_info, list):
            clinical_info = ["" if drop_mask[i, 2].item() else clinical_info[i] for i in range(batch_size)]

    return oct_features, colpo_features, clinical_features, clinical_info


def clinical_features_from_batch(batch: Dict[str, object], batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Build the 7-D clinical feature vector used by the clinical evolver."""
    if "clinical_features" in batch and batch["clinical_features"] is not None:
        return batch["clinical_features"].to(device, non_blocking=True).float()
    if "clinical_data" not in batch:
        return None
    clinical_data = batch["clinical_data"]
    if not isinstance(clinical_data, dict):
        return None
    hpv = clinical_data.get("hpv", torch.zeros(batch_size, device=device))
    age = clinical_data.get("age", torch.zeros(batch_size, device=device))
    tct = clinical_data.get("tct", torch.zeros(batch_size, device=device))
    hpv = hpv.to(device).float() if isinstance(hpv, torch.Tensor) else torch.as_tensor(hpv, device=device).float()
    age = age.to(device).float() if isinstance(age, torch.Tensor) else torch.as_tensor(age, device=device).float()
    if isinstance(tct, torch.Tensor):
        tct = tct.to(device).long()
    else:
        tct = torch.zeros(batch_size, dtype=torch.long, device=device)
    tct = torch.clamp(tct, 0, 4)
    tct_onehot = F.one_hot(tct, num_classes=5).float()
    return torch.cat([hpv.unsqueeze(1), age.unsqueeze(1), tct_onehot], dim=1)


def normalize_string_batch(values) -> list[str]:
    if isinstance(values, list):
        return [str(v) for v in values]
    if isinstance(values, tuple):
        return [str(v) for v in values]
    return [str(values)]


def maybe_perturb_cached_features(features: torch.Tensor, config) -> torch.Tensor:
    prob = float(getattr(config, "train_feature_noise_prob", 0.0))
    if prob <= 0.0 or torch.rand((), device=features.device).item() >= prob:
        return features
    std = float(getattr(config, "train_feature_noise_std", 0.02))
    return features + torch.randn_like(features) * std


def get_patch_features_from_batch(batch, device, config, training: bool):
    """Return patient-level OCT/colposcopy patch features from cache or images."""
    if "oct_patch_features" in batch and "colpo_patch_features" in batch:
        oct_features_patch = batch["oct_patch_features"].to(device, non_blocking=True).float()
        colpo_features_patch = batch["colpo_patch_features"].to(device, non_blocking=True).float()
        if training:
            oct_features_patch = maybe_perturb_cached_features(oct_features_patch, config)
            colpo_features_patch = maybe_perturb_cached_features(colpo_features_patch, config)
        return oct_features_patch, colpo_features_patch

    oct_images = batch['oct_images'].to(device, non_blocking=True)
    colposcopy_images = batch['colposcopy_images'].to(device, non_blocking=True)
    if training:
        oct_images = corrupt_images(oct_images, config, "oct")
        colposcopy_images = corrupt_images(colposcopy_images, config, "colposcopy")

    B_oct = oct_images.shape[0]
    if len(oct_images.shape) == 5:
        F_oct = oct_images.shape[1]
        oct_images_flat = oct_images.view(B_oct * F_oct, *oct_images.shape[2:])
        oct_features_patch = extract_patch_features_with_vit(
            oct_images_flat,
            device,
            batch_size=config.vit_batch_size,
            pretrained=getattr(config, 'vit_pretrained', False),
        )
        oct_features_patch = oct_features_patch.view(B_oct, F_oct, 196, 768).mean(dim=1)
    else:
        oct_features_patch = extract_patch_features_with_vit(
            oct_images,
            device,
            batch_size=config.vit_batch_size,
            pretrained=getattr(config, 'vit_pretrained', False),
        )

    B_colpo = colposcopy_images.shape[0]
    if len(colposcopy_images.shape) == 5:
        N_colpo = colposcopy_images.shape[1]
        colpo_images_flat = colposcopy_images.view(B_colpo * N_colpo, *colposcopy_images.shape[2:])
        colpo_features_patch = extract_patch_features_with_vit(
            colpo_images_flat,
            device,
            batch_size=config.vit_batch_size,
            pretrained=getattr(config, 'vit_pretrained', False),
        )
        colpo_features_patch = colpo_features_patch.view(B_colpo, N_colpo, 196, 768).mean(dim=1)
    else:
        colpo_features_patch = extract_patch_features_with_vit(
            colposcopy_images,
            device,
            batch_size=config.vit_batch_size,
            pretrained=getattr(config, 'vit_pretrained', False),
        )

    return oct_features_patch, colpo_features_patch


def train_epoch(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    device, 
    epoch, 
    config, 
    log_print=None
):
    """训练一个epoch"""
    if log_print is None:
        log_print = print
    
    model.train()
    model.set_epoch(epoch)  # 设置epoch（用于Warm-up）
    
    # 获取动态Beta
    current_beta = get_dynamic_beta(epoch, config.num_epochs)
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    cls_losses = []
    ot_losses = []
    align_losses = []  # 🔥 [NEW] 对齐损失
    align_recalls = []  # 🔥 [NEW] 对齐召回率 Recall@1（越高越好）
    sparse_losses = []
    consist_losses = []
    adv_losses = []
    
    log_print(f"\n{'='*80}")
    log_print(f"📊 Epoch {epoch}/{config.num_epochs} - 训练阶段")
    log_print(f"{'='*80}")
    log_print(f"当前Beta: {current_beta:.3f} (Warm-up策略)")
    log_print(f"总batch数: {len(dataloader)}")
    
    # 修复BrokenPipeError：设置tqdm的file参数，避免管道错误
    import sys
    try:
        # 尝试使用标准输出，如果失败则禁用tqdm
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', 
                   file=sys.stdout if sys.stdout.isatty() else None,
                   disable=not sys.stdout.isatty())
    except (BrokenPipeError, OSError):
        # 如果管道错误，禁用tqdm
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Train]', disable=True)
    
    for batch_idx, batch in enumerate(pbar):
        labels = batch['label'].to(device, non_blocking=True)
        center_labels = batch['center_idx'].to(device, non_blocking=True)
        # 🔥 3.2改动：提取image_names和clinical_info（用于VLM检索）
        # DataLoader的默认collate_fn会将字符串列表合并，所以这里直接使用
        image_names = normalize_string_batch(batch['image_names'])
        clinical_info = normalize_string_batch(batch.get('clinical_info', ""))
        
        optimizer.zero_grad()
        
        oct_features_patch, colpo_features_patch = get_patch_features_from_batch(
            batch,
            device,
            config,
            training=True,
        )
        B_oct = oct_features_patch.shape[0]

        # 验证image_names长度与batch大小匹配
        if len(image_names) != B_oct:
            raise ValueError(
                f"image_names长度({len(image_names)})与batch大小({B_oct})不匹配！\n"
                f"image_names类型: {type(image_names)}, 值: {image_names[:3] if len(image_names) > 3 else image_names}"
            )
        
        clinical_features = clinical_features_from_batch(batch, B_oct, device)
        oct_features_patch, colpo_features_patch, clinical_features, clinical_info = apply_modality_dropout(
            oct_features_patch,
            colpo_features_patch,
            clinical_features,
            clinical_info,
            config,
        )
        
        # 前向传播（整合5.0优势）
        outputs = model(
            f_oct=oct_features_patch,
            f_colpo=colpo_features_patch,
            image_names=image_names,
            clinical_info=clinical_info,
            center_labels=center_labels,
            clinical_features=clinical_features,  # 🔥 5.0新增
            return_loss_components=True,
            current_beta=current_beta
        )
        
        logits = outputs['pred']
        
        # 计算分类损失
        L_cls = criterion(logits, labels)
        
        # 计算总损失
        total_loss_batch = L_cls
        
        # 添加OT损失
        if config.use_ot and 'L_ot' in outputs.get('loss_components', {}):
            L_ot = outputs['loss_components']['L_ot']
            total_loss_batch = total_loss_batch + config.lambda_ot * L_ot
            ot_losses.append(L_ot.item())
        
        # 🔥 [NEW] 添加对齐损失 (Alignment Loss - 逻辑闭环关键)
        if 'L_align' in outputs.get('loss_components', {}):
            L_align = outputs['loss_components']['L_align']
            total_loss_batch = total_loss_batch + config.lambda_align * L_align
            align_losses.append(L_align.item())
        else:
            align_losses.append(0.0)

        # 🔥 [NEW] 记录对齐召回率（不参与反传）
        if 'Recall_Align' in outputs.get('loss_components', {}):
            try:
                align_recalls.append(float(outputs['loss_components']['Recall_Align'].detach().cpu().item()))
            except Exception:
                align_recalls.append(0.0)
        else:
            align_recalls.append(0.0)
        
        # 添加稀疏性损失（修复漏洞3：已添加下界保护）
        if config.use_visual_notes and 'L_sparse' in outputs.get('loss_components', {}):
            L_sparse = outputs['loss_components']['L_sparse']
            total_loss_batch = total_loss_batch + config.lambda_sparse * L_sparse
            sparse_losses.append(L_sparse.item())
            
            # 监控注意力值（用于调试）
            if 'attn_mean_oct' in outputs.get('loss_components', {}):
                attn_mean_oct = outputs['loss_components']['attn_mean_oct']
                attn_mean_colpo = outputs['loss_components'].get('attn_mean_colpo', 0.0)
                if batch_idx == 0:  # 只打印第一个batch
                    log_print(f"     注意力均值: OCT={attn_mean_oct:.4f}, Colpo={attn_mean_colpo:.4f}")
        else:
            sparse_losses.append(0.0)
        
        # 🔥 5.0优势：正交损失（解耦方式）
        if 'L_ortho' in outputs.get('loss_components', {}):
            L_ortho = outputs['loss_components']['L_ortho']
            lambda_ortho = getattr(config, 'lambda_ortho', 0.5)
            total_loss_batch = total_loss_batch + lambda_ortho * L_ortho
        
        # 🔥 5.0优势：噪声正则化损失
        if 'L_noise' in outputs.get('loss_components', {}):
            L_noise = outputs['loss_components']['L_noise']
            lambda_noise = getattr(config, 'lambda_noise', 0.1)
            total_loss_batch = total_loss_batch + lambda_noise * L_noise

        if 'L_reliability_kl' in outputs.get('loss_components', {}):
            L_reliability_kl = outputs['loss_components']['L_reliability_kl']
            lambda_reliability_kl = getattr(config, 'lambda_reliability_kl', 0.01)
            total_loss_batch = total_loss_batch + lambda_reliability_kl * L_reliability_kl

        if 'L_posterior_smooth' in outputs.get('loss_components', {}):
            L_posterior_smooth = outputs['loss_components']['L_posterior_smooth']
            lambda_posterior_smooth = getattr(config, 'lambda_posterior_smooth', 0.01)
            total_loss_batch = total_loss_batch + lambda_posterior_smooth * L_posterior_smooth

        if 'L_asccp_ot' in outputs.get('loss_components', {}):
            L_asccp_ot = outputs['loss_components']['L_asccp_ot']
            lambda_asccp_ot = getattr(config, 'lambda_asccp_ot', 0.05)
            total_loss_batch = total_loss_batch + lambda_asccp_ot * L_asccp_ot

        if 'L_modality_likelihood' in outputs.get('loss_components', {}):
            L_modality_likelihood = outputs['loss_components']['L_modality_likelihood']
            lambda_modality_likelihood = getattr(config, 'lambda_modality_likelihood', 0.05)
            total_loss_batch = total_loss_batch + lambda_modality_likelihood * L_modality_likelihood
        
        # 添加一致性损失
        if config.use_dual and 'L_consist' in outputs.get('loss_components', {}):
            L_consist = outputs['loss_components']['L_consist']
            total_loss_batch = total_loss_batch + config.lambda_consist * L_consist
            consist_losses.append(L_consist.item())
        else:
            consist_losses.append(0.0)
        
        # 添加对抗损失
        if config.use_dual and 'L_adv' in outputs.get('loss_components', {}):
            L_adv = outputs['loss_components']['L_adv']
            total_loss_batch = total_loss_batch + config.lambda_adv * L_adv
            adv_losses.append(L_adv.item())
        else:
            adv_losses.append(0.0)
        
        # 反向传播
        total_loss_batch.backward()
        optimizer.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        cls_losses.append(L_cls.item())
        
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        # 更新进度条（捕获可能的管道错误）
        try:
            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'cls': f"{L_cls.item():.4f}",
                'beta': f"{current_beta:.3f}",
                'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
            })
        except (BrokenPipeError, OSError):
            # 如果管道错误，只更新日志，不更新进度条
            if batch_idx % 10 == 0:  # 每10个batch打印一次
                log_print(f"      Batch {batch_idx}/{len(dataloader)}: loss={total_loss_batch.item():.4f}, acc={accuracy_score(all_labels, all_preds):.4f}")
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 🔥 [NEW] 计算训练集的全面分类指标
    try:
        precision_train = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_train = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        balanced_acc_train = balanced_accuracy_score(all_labels, all_preds)
        mcc_train = matthews_corrcoef(all_labels, all_preds)
        
        # 对于二分类，计算阳性类别的指标
        if len(np.unique(all_labels)) == 2:
            precision_pos_train = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            recall_pos_train = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            sensitivity_train = recall_pos_train
            
            # 计算混淆矩阵
            cm_train = confusion_matrix(all_labels, all_preds)
            if cm_train.shape == (2, 2):
                TN_train, FP_train, FN_train, TP_train = cm_train.ravel()
                specificity_train = TN_train / (TN_train + FP_train) if (TN_train + FP_train) > 0 else 0.0
                npv_train = TN_train / (TN_train + FN_train) if (TN_train + FN_train) > 0 else 0.0
            else:
                specificity_train = 0.0
                npv_train = 0.0
        else:
            precision_pos_train = precision_train
            recall_pos_train = recall_train
            sensitivity_train = recall_train
            specificity_train = 0.0
            npv_train = 0.0
    except Exception as e:
        precision_train = 0.0
        recall_train = 0.0
        precision_pos_train = 0.0
        recall_pos_train = 0.0
        sensitivity_train = 0.0
        specificity_train = 0.0
        npv_train = 0.0
        balanced_acc_train = 0.0
        mcc_train = 0.0
    
    log_print(f"\n  📊 Epoch {epoch} 训练统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率 (Accuracy): {acc:.4f}")
    log_print(f"     - 平衡准确率 (Balanced Accuracy): {balanced_acc_train:.4f}")  # 🔥 [NEW]
    log_print(f"     - F1-Score: {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    log_print(f"     - MCC (Matthews): {mcc_train:.4f}")  # 🔥 [NEW]
    if len(np.unique(all_labels)) == 2:
        log_print(f"\n     📋 二分类详细指标:")
        log_print(f"     - Precision (阳性, PPV): {precision_pos_train:.4f}")
        log_print(f"     - Recall/Sensitivity (敏感度): {recall_pos_train:.4f}")
        log_print(f"     - Specificity (特异性): {specificity_train:.4f}")  # 🔥 [NEW]
        log_print(f"     - NPV (阴性预测值): {npv_train:.4f}")  # 🔥 [NEW]
    log_print(f"     - Precision (加权平均): {precision_train:.4f}")
    log_print(f"     - Recall (加权平均): {recall_train:.4f}")
    log_print(f"     - 分类损失: {np.mean(cls_losses):.6f}")
    if config.use_ot:
        log_print(f"     - OT损失: {np.mean(ot_losses):.6f}")
    if align_losses and any(v > 0 for v in align_losses):
        log_print(f"     - 🔥 对齐损失 (Alignment): {np.mean(align_losses):.6f}")
        log_print(f"     - ✅ 对齐召回率 (Recall@1): {np.mean(align_recalls):.4f}")
    if config.use_visual_notes:
        log_print(f"     - 稀疏性损失: {np.mean(sparse_losses):.6f}")
    if config.use_dual:
        log_print(f"     - 一致性损失: {np.mean(consist_losses):.6f}")
        log_print(f"     - 对抗损失: {np.mean(adv_losses):.6f}")
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'balanced_acc': balanced_acc_train,  # 🔥 [NEW] 训练集平衡准确率
        'precision': precision_train,  # 训练集Precision
        'recall': recall_train,  # 训练集Recall
        'precision_pos': precision_pos_train,  # 训练集阳性Precision
        'recall_pos': recall_pos_train,  # 训练集阳性Recall
        'sensitivity': sensitivity_train,  # 🔥 [NEW] 训练集敏感度
        'specificity': specificity_train,  # 🔥 [NEW] 训练集特异性
        'npv': npv_train,  # 🔥 [NEW] 训练集阴性预测值
        'mcc': mcc_train,  # 🔥 [NEW] 训练集MCC
        'cls_loss': np.mean(cls_losses),
        'ot_loss': np.mean(ot_losses) if ot_losses else 0.0,
        'align_loss': np.mean(align_losses) if align_losses else 0.0,
        'align_recall': float(np.mean(align_recalls)) if align_recalls else 0.0,
        'sparse_loss': np.mean(sparse_losses) if sparse_losses else 0.0,
        'consist_loss': np.mean(consist_losses) if consist_losses else 0.0,
        'adv_loss': np.mean(adv_losses) if adv_losses else 0.0,
    }


def validate(model, dataloader, criterion, device, epoch, config, log_print=None):
    """验证"""
    if log_print is None:
        log_print = print
    
    model.eval()
    model.set_epoch(epoch)
    
    # 使用固定的Beta（强过滤模式）
    current_beta = 0.1
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    align_recalls = []  # 🔥 [NEW] 验证阶段的对齐召回率
    
    log_print(f"\n  📊 Epoch {epoch}/{config.num_epochs} - 验证阶段")
    log_print(f"     当前Beta: {current_beta:.3f}")
    
    with torch.no_grad():
        # 修复BrokenPipeError：设置tqdm的file参数
        import sys
        try:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]',
                       file=sys.stdout if sys.stdout.isatty() else None,
                       disable=not sys.stdout.isatty())
        except (BrokenPipeError, OSError):
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs} [Val]', disable=True)
        
        for batch_idx, batch in enumerate(pbar):
            labels = batch['label'].to(device, non_blocking=True)
            center_labels = batch['center_idx'].to(device, non_blocking=True)
            # 🔥 3.2改动：提取image_names和clinical_info
            image_names = normalize_string_batch(batch['image_names'])
            clinical_info = normalize_string_batch(batch.get('clinical_info', ""))
            
            # 验证长度匹配
            oct_features_patch, colpo_features_patch = get_patch_features_from_batch(
                batch,
                device,
                config,
                training=False,
            )
            B_oct_val = oct_features_patch.shape[0]
            if len(image_names) != B_oct_val:
                raise ValueError(
                    f"验证阶段: image_names长度({len(image_names)})与batch大小({B_oct_val})不匹配！"
                )

            clinical_features = clinical_features_from_batch(batch, B_oct_val, device)
            
            # 前向传播（3.2改动：使用image_names和clinical_info）
            # [关键修改] 验证时也设置 return_loss_components=True 以获取 Recall
            outputs = model(
                f_oct=oct_features_patch,
                f_colpo=colpo_features_patch,
                image_names=image_names,  # 🔥 3.2改动
                clinical_info=clinical_info,  # 🔥 3.2改动
                center_labels=center_labels,
                clinical_features=clinical_features,
                return_loss_components=True,  # 🔥 [FIX] 改为 True 以获取 Recall
                current_beta=current_beta
            )
            
            logits = outputs['pred']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 🔥 [NEW] 提取并记录对齐召回率
            if 'loss_components' in outputs and 'Recall' in outputs['loss_components']:
                try:
                    recall_val = float(outputs['loss_components']['Recall'].detach().cpu().item())
                    align_recalls.append(recall_val)
                except Exception:
                    align_recalls.append(0.0)
            elif 'loss_components' in outputs and 'Recall_Align' in outputs['loss_components']:
                try:
                    recall_val = float(outputs['loss_components']['Recall_Align'].detach().cpu().item())
                    align_recalls.append(recall_val)
                except Exception:
                    align_recalls.append(0.0)
            else:
                align_recalls.append(0.0)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            # 调试：打印第一个batch的预测分布
            if batch_idx == 0:
                log_print(f"     第一个batch预测分布: {torch.bincount(preds)}")
                log_print(f"     第一个batch标签分布: {torch.bincount(labels)}")
                log_print(f"     第一个batch预测概率范围: [{probs[:, 1].min():.4f}, {probs[:, 1].max():.4f}]")
            
            # 更新进度条（捕获可能的管道错误）
            try:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy_score(all_labels, all_preds):.4f}"
                })
            except (BrokenPipeError, OSError):
                pass  # 忽略管道错误，继续训练
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = 0.0
        log_print(f"     ⚠️ AUC计算失败: {e}")
    
    # 🔧 修复F1计算：使用weighted F1，即使某个类别没有预测到也能计算
    # 原因：早期训练时，模型可能只预测一个类别，导致F1=0
    # 解决方案：使用weighted F1，按类别样本数加权，更公平
    try:
        # 检查是否有两个类别
        unique_labels = np.unique(all_labels)
        unique_preds = np.unique(all_preds)
        
        if len(unique_labels) < 2:
            log_print(f"     ⚠️ 标签只有{len(unique_labels)}个类别，无法计算F1")
            f1 = 0.0
        elif len(unique_preds) < 2:
            # 🔧 修复：即使预测只有1个类别，也使用weighted F1计算
            # weighted F1会按类别样本数加权，即使某个类别没有预测到，也能给出有意义的值
            f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_weighted  # 使用weighted F1作为主要指标
            
            # 只在第一个epoch或前几个epoch打印警告
            if epoch <= 3:
                log_print(f"     ⚠️ 预测只有{len(unique_preds)}个类别（早期训练正常现象）")
                log_print(f"     使用weighted F1: {f1_weighted:.4f}, macro F1: {f1_macro:.4f}")
        else:
            # 正常情况：使用weighted F1（更公平，考虑类别不平衡）
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # 额外检查：如果F1很低，打印混淆矩阵用于调试
            if f1 < 0.1 and epoch <= 5:
                cm = confusion_matrix(all_labels, all_preds)
                log_print(f"     ⚠️ F1较低（{f1:.4f}），混淆矩阵:\n{cm}")
                log_print(f"     标签分布: {np.bincount(all_labels)}")
                log_print(f"     预测分布: {np.bincount(all_preds)}")
    except Exception as e:
        f1 = 0.0
        log_print(f"     ⚠️ F1计算失败: {e}")
    
    # 🔥 [NEW] 计算平均对齐召回率
    avg_recall = np.mean(align_recalls) if align_recalls else 0.0
    
    # 🔥 [NEW] 计算全面的分类任务指标
    try:
        # 基础指标
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 二分类特有指标
        if len(np.unique(all_labels)) == 2:
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
            else:
                # 处理边界情况
                if len(cm.ravel()) == 1:
                    TN = cm[0, 0] if cm.shape == (1, 1) else 0
                    FP, FN, TP = 0, 0, 0
                else:
                    TN, FP, FN, TP = 0, 0, 0, 0
            
            # 阳性类别指标（Sensitivity/Recall）
            precision_pos = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            recall_pos = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            sensitivity = recall_pos  # Sensitivity = Recall for positive class
            
            # Specificity (特异性) = TN / (TN + FP)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            
            # NPV (Negative Predictive Value, 阴性预测值) = TN / (TN + FN)
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
            
            # Balanced Accuracy = (Sensitivity + Specificity) / 2
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            
            # Matthews Correlation Coefficient (MCC)
            mcc = matthews_corrcoef(all_labels, all_preds)
            
            # PR-AUC (Precision-Recall AUC) - 处理类别不平衡
            try:
                pr_auc = average_precision_score(all_labels, all_probs)
            except:
                pr_auc = 0.0
        else:
            precision_pos = precision
            recall_pos = recall
            sensitivity = recall
            specificity = 0.0
            npv = 0.0
            balanced_acc = acc
            mcc = 0.0
            pr_auc = 0.0
    except Exception as e:
        precision = 0.0
        recall = 0.0
        precision_pos = 0.0
        recall_pos = 0.0
        sensitivity = 0.0
        specificity = 0.0
        npv = 0.0
        balanced_acc = 0.0
        mcc = 0.0
        pr_auc = 0.0
        log_print(f"     ⚠️ 分类指标计算失败: {e}")
    
    log_print(f"\n  📊 Epoch {epoch} 验证统计:")
    log_print(f"     - 平均损失: {avg_loss:.6f}")
    log_print(f"     - 准确率 (Accuracy): {acc:.4f}")
    log_print(f"     - 平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")  # 🔥 [NEW]
    log_print(f"     - AUC (ROC): {auc:.4f}")
    log_print(f"     - PR-AUC: {pr_auc:.4f}")  # 🔥 [NEW] Precision-Recall AUC
    log_print(f"     - F1-Score: {f1:.4f}")
    log_print(f"     - MCC (Matthews): {mcc:.4f}")  # 🔥 [NEW] Matthews相关系数
    if len(np.unique(all_labels)) == 2:
        log_print(f"\n     📋 二分类详细指标:")
        log_print(f"     - Precision (阳性, PPV): {precision_pos:.4f}")
        log_print(f"     - Recall/Sensitivity (敏感度, TPR): {recall_pos:.4f}")
        log_print(f"     - Specificity (特异性, TNR): {specificity:.4f}")  # 🔥 [NEW]
        log_print(f"     - NPV (阴性预测值): {npv:.4f}")  # 🔥 [NEW]
    log_print(f"     - Precision (加权平均): {precision:.4f}")
    log_print(f"     - Recall (加权平均): {recall:.4f}")
    ece = expected_calibration_error(all_labels, all_probs, n_bins=getattr(config, "calibration_bins", 10))
    log_print(f"     - ECE: {ece:.4f}")
    log_print(f"     - 🔗 对齐召回率 (Recall@1): {avg_recall:.4f}")  # 对齐任务的Recall
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'balanced_acc': balanced_acc,  # 🔥 [NEW] 平衡准确率
        'auc': auc,
        'pr_auc': pr_auc,  # 🔥 [NEW] PR-AUC
        'precision': precision,
        'recall': recall,
        'precision_pos': precision_pos,  # 阳性类别精确率 (PPV)
        'recall_pos': recall_pos,  # 阳性类别召回率（敏感度, Sensitivity）
        'sensitivity': sensitivity,  # 🔥 [NEW] 敏感度
        'specificity': specificity,  # 🔥 [NEW] 特异性
        'npv': npv,  # 🔥 [NEW] 阴性预测值
        'f1': f1,
        'mcc': mcc,  # 🔥 [NEW] Matthews相关系数
        'ece': ece,
        'align_recall': avg_recall,  # 对齐任务的Recall
    }


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Bio-COT 3.1 训练脚本')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选，默认使用 config.py）')
    parser.add_argument('--gpu', type=int, default=None,
                       help='指定GPU ID（可选，默认自动选择）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='覆盖配置中的训练epoch数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--data-root', type=str, default=None,
                       help='图像数据根目录；默认使用配置文件中的 data_root')
    parser.add_argument('--train-csv', type=str, default=None,
                       help='训练CSV路径；用于leave-one-center-out等补充实验')
    parser.add_argument('--val-csv', type=str, default=None,
                       help='内部验证CSV路径；用于leave-one-center-out等补充实验')
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        # 从指定路径加载配置
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # 查找“在该配置文件中定义的”配置类（避免把import进来的BioCOT_v3_Config误选为第一个）
        config_classes = [
            cls for cls in config_module.__dict__.values()
            if isinstance(cls, type)
            and 'Config' in cls.__name__
            and getattr(cls, "__module__", None) == getattr(config_module, "__name__", None)
        ]
        if not config_classes:
            raise ValueError(f"在 {args.config} 中未找到配置类（请确保文件内定义了 *Config 类）")

        # 若存在多个，优先选择"非BioCOT_v3_2_Config"的那个（一般是 BaselineConfig/NoXXXConfig）
        config_class = None
        for cls in config_classes:
            if cls.__name__ != "BioCOT_v3_2_Config":
                config_class = cls
                break
        if config_class is None:
            config_class = config_classes[0]

        config = config_class()
    else:
        # 使用默认配置
        from config import BioCOT_v3_2_Config
        config = BioCOT_v3_2_Config()
    
    # 🔥 强制确保消融实验使用20个epoch
    if args.config and 'ablation_studies' in str(args.config):
        config.num_epochs = 20
        print(f"✅ 检测到消融实验，强制设置 num_epochs = 20")
    if args.epochs is not None:
        config.num_epochs = args.epochs
        print(f"✅ 命令行覆盖 num_epochs = {config.num_epochs}")
    if args.data_root is not None:
        config.data_root = args.data_root
        print(f"✅ 命令行覆盖 data_root = {config.data_root}")

    # 设置随机种子，保证正式实验可复现
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.log_dir) / f"train_bio_cot_v3_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_f = open(log_file, 'w', encoding='utf-8')
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_f)
        log_f.flush()
    
    log_print("=" * 80)
    log_print("Bio-COT 3.2 Enhanced Logic Loop: 融合3.1和4.0的优势")
    log_print("核心升级:")
    log_print("  1. Adaptive Modality Gating (自适应模态门控) - 3.1")
    log_print("  2. Enhanced Visual Notes (Cross-Attention机制) - 3.1")
    log_print("  3. Semantic-Visual Alignment Loop (语义-视觉对齐闭环) - 3.1")
    log_print("  4. Frozen VLM + Trainable Adapter (计算效率) - 4.0")
    log_print("  5. Dynamic Knowledge Generation (知识复杂度) - 4.0")
    log_print("=" * 80)
    log_print(f"数据路径: {config.data_root}")
    log_print(f"VLM缓存路径: {config.vlm_json_path}")
    log_print(f"Text Encoder: {config.text_model_name}")
    log_print(f"随机种子: {args.seed}")
    log_print(f"模块配置:")
    log_print(f"  - use_visual_notes: {config.use_visual_notes}")
    log_print(f"  - use_ot: {config.use_ot}")
    log_print(f"  - use_dual: {config.use_dual}")
    log_print(f"  - use_cross_attn: {config.use_cross_attn}")
    log_print(f"  - use_adaptive_gating: {getattr(config, 'use_adaptive_gating', True)}")
    log_print("=" * 80)
    
    # 选择GPU
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            # CUDA_VISIBLE_DEVICES 已在导入torch之前设置；此处使用可见设备0即可。
            device = torch.device('cuda:0')
            log_print(f"✅ 使用可见GPU设备: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
        log_print(f"✅ 使用GPU设备: {device}")
    else:
        raise RuntimeError("❌ CUDA不可用！")
    
    # 加载数据集
    log_print("\n📊 开始加载数据集...")
    train_csv = Path(args.train_csv) if args.train_csv else Path(config.data_root) / 'train_labels.csv'
    val_csv = Path(args.val_csv) if args.val_csv else Path(config.data_root) / 'val_labels.csv'
    log_print(f"训练CSV: {train_csv}")
    log_print(f"验证CSV: {val_csv}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if getattr(config, "use_cached_patch_features", False):
        feature_cache_path = getattr(config, "feature_cache_path", None)
        if not feature_cache_path:
            raise ValueError("use_cached_patch_features=True 但未设置 feature_cache_path")
        log_print(f"  📦 使用缓存patch特征: {feature_cache_path}")
        train_dataset = CachedPatchFeatureDataset(train_csv, feature_cache_path)
    else:
        train_dataset = FiveCentersMultimodalDatasetV3_2(
            csv_path=str(train_csv),
            transform=transform,
            oct_num_frames=config.oct_frames,
            max_col_images=config.colposcopy_images,
            balance_negative_frames=False,  # 🔧 禁用帧数平衡以避免显存溢出
            data_root=str(config.data_root)
        )
    log_print(f"  ✅ 训练集加载完成: {len(train_dataset)} 个样本")
    
    if getattr(config, "use_cached_patch_features", False):
        val_dataset = CachedPatchFeatureDataset(val_csv, getattr(config, "feature_cache_path"))
    else:
        val_dataset = FiveCentersMultimodalDatasetV3_2(
            csv_path=str(val_csv),
            transform=transform,
            oct_num_frames=config.oct_frames,
            max_col_images=config.colposcopy_images,
            balance_negative_frames=False,  # 🔧 禁用帧数平衡以避免显存溢出
            data_root=str(config.data_root)
        )
    log_print(f"  ✅ 验证集加载完成: {len(val_dataset)} 个样本")
    
    # 创建加权采样器
    # ⚠️ 性能修复：不要通过 train_dataset[i] 逐个触发图像加载（会非常慢）
    if hasattr(train_dataset, "df") and "label" in getattr(train_dataset, "df").columns:
        train_labels = train_dataset.df["label"].astype(int).values
    else:
        # fallback（不推荐）：只读取label字段，避免加载图片
        train_labels = [int(train_dataset.df.iloc[i]["label"]) for i in range(len(train_dataset))] if hasattr(train_dataset, "df") else [train_dataset[i]["label"].item() for i in range(len(train_dataset))]

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.asarray([class_weights[int(label)] for label in train_labels], dtype=np.float64)
    if getattr(config, "center_balanced_sampling", False) and hasattr(train_dataset, "df"):
        center_col = "center_group_id" if "center_group_id" in train_dataset.df.columns else "center_idx"
        if center_col not in train_dataset.df.columns:
            center_col = None
    else:
        center_col = None
    if center_col is not None:
        center_labels_np = train_dataset.df[center_col].astype(int).values
        center_counts = np.bincount(center_labels_np)
        center_weights = 1.0 / np.maximum(center_counts, 1)
        sample_weights = sample_weights * np.asarray([center_weights[int(center)] for center in center_labels_np], dtype=np.float64)
        sample_weights = sample_weights / np.maximum(sample_weights.mean(), 1e-12)
        log_print(f"  ✅ 启用中心均衡采样: center_col={center_col}, center_counts={center_counts.tolist()}")
    sample_weights = sample_weights.tolist()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # 创建模型
    log_print("\n📊 正在创建Bio-COT 3.2模型...")
    try:
        model = create_bio_cot_v3_2(config)
    except Exception as e:
        log_print(f"   ❌ 模型创建失败: {e}")
        import traceback
        log_print(traceback.format_exc())
        raise
    
    # 🔥 紧急修复：确保learnable_knowledge_base存在（仿照exp_bio3.0_improved方案）
    if not hasattr(model, 'learnable_knowledge_base'):
        import torch.nn as nn
        model.learnable_knowledge_base = nn.Parameter(torch.randn(1, config.embed_dim) * 0.02)
        log_print(f"   ⚠️ 检测到learnable_knowledge_base缺失，已手动创建")
    
    model = model.to(device)
    
    # 🔥 调试：检查模型属性
    log_print(f"   模型 use_vlm_retriever: {getattr(model, 'use_vlm_retriever', 'N/A')}")
    log_print(f"   模型 has learnable_knowledge_base: {hasattr(model, 'learnable_knowledge_base')}")
    log_print(f"   模型 has note_projector: {hasattr(model, 'note_projector')}")
    if hasattr(model, 'learnable_knowledge_base'):
        log_print(f"   learnable_knowledge_base 形状: {model.learnable_knowledge_base.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"✅ 模型创建完成")
    log_print(f"   总参数量: {total_params:,}")
    log_print(f"   可训练参数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 训练循环
    log_print("\n" + "=" * 80)
    log_print("🚀 开始训练...")
    log_print("=" * 80)
    
    best_auc = 0.0
    history = {
        # 训练集指标
        'train_loss': [], 'train_acc': [], 'train_balanced_acc': [], 'train_precision': [], 'train_recall': [],
        'train_precision_pos': [], 'train_recall_pos': [], 'train_sensitivity': [], 'train_specificity': [],
        'train_npv': [], 'train_mcc': [],
        # 验证集指标
        'val_loss': [], 'val_acc': [], 'val_balanced_acc': [], 'val_auc': [], 'val_pr_auc': [], 'val_f1': [], 'val_ece': [], 'val_selection_score': [],
        'val_precision': [], 'val_recall': [], 'val_precision_pos': [], 'val_recall_pos': [],
        'val_sensitivity': [], 'val_specificity': [], 'val_npv': [], 'val_mcc': [],
        # 损失指标
        'cls_loss': [], 'ot_loss': [], 'align_loss': [], 'align_recall': [], 'sparse_loss': [], 'consist_loss': [], 'adv_loss': []
    }
    
    for epoch in range(1, config.num_epochs + 1):
        try:
            # 训练
            train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, log_print=log_print)
            
            # 验证
            val_results = validate(model, val_loader, criterion, device, epoch, config, log_print=log_print)
        except Exception as e:
            log_print(f"\n❌ Epoch {epoch} 训练失败: {e}")
            import traceback
            log_print(f"详细错误信息:\n{traceback.format_exc()}")
            log_print("训练已停止，请检查错误信息")
            break
        
        # 记录历史（训练集）
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['acc'])
        history['train_balanced_acc'].append(train_results.get('balanced_acc', 0.0))  # 🔥 [NEW]
        history['train_precision'].append(train_results.get('precision', 0.0))
        history['train_recall'].append(train_results.get('recall', 0.0))
        history['train_precision_pos'].append(train_results.get('precision_pos', 0.0))
        history['train_recall_pos'].append(train_results.get('recall_pos', 0.0))
        history['train_sensitivity'].append(train_results.get('sensitivity', 0.0))  # 🔥 [NEW]
        history['train_specificity'].append(train_results.get('specificity', 0.0))  # 🔥 [NEW]
        history['train_npv'].append(train_results.get('npv', 0.0))  # 🔥 [NEW]
        history['train_mcc'].append(train_results.get('mcc', 0.0))  # 🔥 [NEW]
        # 记录历史（验证集）
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['acc'])
        history['val_balanced_acc'].append(val_results.get('balanced_acc', 0.0))  # 🔥 [NEW]
        history['val_auc'].append(val_results['auc'])
        history['val_pr_auc'].append(val_results.get('pr_auc', 0.0))  # 🔥 [NEW]
        history['val_f1'].append(val_results['f1'])
        history['val_ece'].append(val_results.get('ece', 0.0))
        history['val_precision'].append(val_results.get('precision', 0.0))
        history['val_recall'].append(val_results.get('recall', 0.0))
        history['val_precision_pos'].append(val_results.get('precision_pos', 0.0))
        history['val_recall_pos'].append(val_results.get('recall_pos', 0.0))
        history['val_sensitivity'].append(val_results.get('sensitivity', 0.0))  # 🔥 [NEW]
        history['val_specificity'].append(val_results.get('specificity', 0.0))  # 🔥 [NEW]
        history['val_npv'].append(val_results.get('npv', 0.0))  # 🔥 [NEW]
        history['val_mcc'].append(val_results.get('mcc', 0.0))  # 🔥 [NEW]
        # 记录损失
        history['cls_loss'].append(train_results['cls_loss'])
        history['ot_loss'].append(train_results['ot_loss'])
        history['align_loss'].append(train_results.get('align_loss', 0.0))
        history['align_recall'].append(train_results.get('align_recall', 0.0))
        history['sparse_loss'].append(train_results['sparse_loss'])
        history['consist_loss'].append(train_results['consist_loss'])
        history['adv_loss'].append(train_results['adv_loss'])
        
        selection_score = checkpoint_score(val_results, config)
        history['val_selection_score'].append(selection_score)
        log_print(
            f"  📌 Checkpoint selection score ({getattr(config, 'checkpoint_metric', 'auc')}): "
            f"{selection_score:.4f}"
        )

        # 保存最佳模型
        if selection_score > best_auc:
            best_auc = selection_score
            checkpoint_path = Path(config.checkpoint_dir) / f"best_model_v3_{timestamp}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'best_selection_score': best_auc,
                'best_val_auc': val_results['auc'],
                'best_val_ece': val_results.get('ece', 0.0),
                'config': config.__dict__,
                'history': history
            }, checkpoint_path)
            log_print(
                f"  ✅ 保存最佳模型 (selection={best_auc:.4f}, "
                f"AUC={val_results['auc']:.4f}, ECE={val_results.get('ece', 0.0):.4f})"
            )
    
    log_print(f"\n✅ 训练完成！最佳checkpoint选择分数: {best_auc:.4f}")
    
    # 保存训练历史
    history_file = Path(config.log_dir) / f"training_history_{timestamp}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    log_print(f"📁 训练历史已保存到: {history_file}")
    
    # 生成可视化图表
    log_print(f"\n📊 正在生成可视化图表...")
    visualize_training(history, Path(config.log_dir), timestamp, best_auc)
    log_print(f"✅ 可视化完成！")
    
    log_f.close()


if __name__ == '__main__':
    main()
