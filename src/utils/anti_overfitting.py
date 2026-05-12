#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
防过拟合工具模块
包含MixUp、CutMix、Focal Loss等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: 处理类别不平衡，关注难分类样本
    FL = -α(1-p_t)^γ * log(p_t)
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        # 应用标签平滑
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            smooth_targets = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / num_classes
        else:
            smooth_targets = targets_one_hot
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算p_t (预测概率)
        p = F.softmax(inputs, dim=1)
        p_t = (p * smooth_targets).sum(dim=1)
        
        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用alpha权重
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]
        
        # 计算focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp数据增强
    
    Args:
        x: [B, ...] 输入数据
        y: [B] 标签
        alpha: Beta分布参数
    
    Returns:
        mixed_x: 混合后的数据
        y_a: 原始标签
        y_b: 混合标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    MixUp损失计算
    
    Args:
        criterion: 损失函数
        pred: [B, num_classes] 预测
        y_a: [B] 标签A
        y_b: [B] 标签B
        lam: 混合系数
    
    Returns:
        loss: 混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix数据增强
    
    Args:
        x: [B, C, H, W] 输入图像
        y: [B] 标签
        alpha: Beta分布参数
    
    Returns:
        mixed_x: 混合后的图像
        y_a: 原始标签
        y_b: 混合标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # 计算裁剪区域
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机选择裁剪中心
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 计算裁剪边界
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行CutMix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 调整lambda以匹配实际裁剪面积
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def cutmix_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    CutMix损失计算
    
    Args:
        criterion: 损失函数
        pred: [B, num_classes] 预测
        y_a: [B] 标签A
        y_b: [B] 标签B
        lam: 混合系数
    
    Returns:
        loss: 混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

