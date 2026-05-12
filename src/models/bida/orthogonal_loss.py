#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orthogonal Loss for Causal-Noise Disentanglement
核心：强制 z_causal 和 z_noise 正交，确保因果特征不包含噪声信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalLoss(nn.Module):
    """
    正交损失：确保 z_causal ⊥ z_noise
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        z_causal: torch.Tensor,
        z_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        计算正交损失
        
        Args:
            z_causal: [B, D] 因果特征
            z_noise: [B, D'] 噪声特征
        
        Returns:
            loss: 标量
        """
        # 如果维度不同，需要投影到相同维度
        if z_causal.size(-1) != z_noise.size(-1):
            # 投影到较小的维度
            min_dim = min(z_causal.size(-1), z_noise.size(-1))
            z_causal = z_causal[:, :min_dim]
            z_noise = z_noise[:, :min_dim]
        
        # 计算内积矩阵：z_causal^T @ z_noise
        # 对于每个样本，计算内积
        inner_product = (z_causal * z_noise).sum(dim=-1)  # [B]
        
        # 正交损失：内积应该为0
        # 使用L2损失而不是平方，数值更稳定
        loss = inner_product.abs()  # 使用绝对值，数值范围更小
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DistributionMatchingLoss(nn.Module):
    """
    分布匹配损失：KL散度匹配
    L_dist = D_KL(Q(z_causal) || P_bio(μ_bio, σ_bio))
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        z_causal: torch.Tensor,
        μ_bio: torch.Tensor,
        σ_bio: torch.Tensor
    ) -> torch.Tensor:
        """
        计算KL散度损失
        
        Args:
            z_causal: [B, D] 因果特征（假设为点估计，需要转换为分布）
            μ_bio: [B, D] 生物流形分布的均值
            σ_bio: [B, D] 生物流形分布的标准差
        
        Returns:
            kl_loss: 标量
        """
        # 将 z_causal 视为从 Q 分布中采样
        # 为了计算KL散度，我们需要估计 Q 的参数
        # 简化：假设 Q 是单位方差的高斯分布，均值为 z_causal
        μ_q = z_causal
        σ_q = torch.ones_like(μ_q)  # 单位方差
        
        # KL散度：D_KL(Q || P)
        # KL = 0.5 * [log(σ_p^2/σ_q^2) + (σ_q^2 + (μ_q - μ_p)^2)/σ_p^2 - 1]
        # 注意：σ_bio可能很小，需要添加数值稳定性
        σ_bio_safe = σ_bio + 1e-6  # 确保数值稳定
        σ_q_safe = σ_q + 1e-6
        
        kl = 0.5 * (
            torch.log(σ_bio_safe.pow(2) / σ_q_safe.pow(2)) +
            (σ_q_safe.pow(2) + (μ_q - μ_bio).pow(2)) / σ_bio_safe.pow(2) -
            1
        )
        
        # 对所有维度求平均（而不是求和），然后对所有样本求平均
        # 这样loss的数值范围更合理（每个维度的KL散度通常在0-10之间）
        kl_loss = kl.mean(dim=-1)  # [B] - 改为mean而不是sum
        
        if self.reduction == 'mean':
            return kl_loss.mean()
        elif self.reduction == 'sum':
            return kl_loss.sum()
        else:
            return kl_loss


class NoiseSupervisionLoss(nn.Module):
    """
    噪声监督损失：用 z_noise 预测医院ID
    L_noise = CrossEntropy(CenterPredictor(z_noise), CenterLabel)
    """
    
    def __init__(self, num_centers: int = 5):
        super().__init__()
        self.num_centers = num_centers
        self.center_predictor = nn.Sequential(
            nn.Linear(768, 256),  # 假设 z_noise 维度为 768
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_centers)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        z_noise: torch.Tensor,
        center_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算噪声监督损失
        
        Args:
            z_noise: [B, D'] 噪声特征
            center_labels: [B] 中心标签（0-4）
        
        Returns:
            loss: 标量
        """
        # 预测中心ID
        center_logits = self.center_predictor(z_noise)  # [B, num_centers]
        
        # 交叉熵损失
        loss = self.criterion(center_logits, center_labels)
        
        return loss

