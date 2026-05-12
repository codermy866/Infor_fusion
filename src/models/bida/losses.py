#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Bio-COT核心损失函数
包含：Sinkhorn最优传输距离、反事实一致性损失等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDistance(nn.Module):
    """
    语义引导最优传输 (Semantic-Guided Optimal Transport, SG-OT)
    用于替代KL散度，实现更灵活的分布对齐
    """
    
    def __init__(self, eps=0.1, max_iter=100, reduction='mean'):
        """
        Args:
            eps: 熵正则化系数（越小越接近精确OT，但数值越不稳定）
            max_iter: Sinkhorn迭代次数
            reduction: 'mean' or 'sum'
        """
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
    
    def forward(self, x, y):
        """
        计算Sinkhorn最优传输距离（数值稳定版本）
        
        Args:
            x: [B, D] 图像因果特征 z_causal
            y: [B, D] 语义先验特征 z_sem (来自Student Prior)
        
        Returns:
            cost: 标量，最优传输距离
        """
        # 检查输入有效性
        if torch.isnan(x).any() or torch.isnan(y).any():
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        if torch.isinf(x).any() or torch.isinf(y).any():
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        B, D = x.shape
        
        # 归一化特征，提高数值稳定性
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        
        # 1. 计算代价矩阵 (Cost Matrix)
        # 使用欧氏距离平方，但限制范围避免数值溢出
        x_col = x.unsqueeze(1)  # [B, 1, D]
        y_lin = y.unsqueeze(0)  # [1, B, D]
        C = torch.sum((x_col - y_lin) ** 2, dim=2)  # [B, B]
        
        # 限制代价矩阵的范围，避免exp溢出
        C = torch.clamp(C, min=0, max=10.0)
        
        # 2. Sinkhorn迭代（对数域版本，数值更稳定）
        # 初始化：u = 0, v = 0
        u = torch.zeros(B, device=x.device, dtype=x.dtype)
        v = torch.zeros(B, device=x.device, dtype=x.dtype)
        
        # 计算 K = exp(-C / eps)，添加数值稳定性
        C_scaled = C / self.eps
        C_scaled = torch.clamp(C_scaled, min=-10.0, max=10.0)  # 限制范围
        K = torch.exp(-C_scaled)
        K = torch.clamp(K, min=1e-10, max=1e10)  # 防止K过小或过大
        
        # Sinkhorn迭代
        for _ in range(self.max_iter):
            # u = 1 / (K @ v + epsilon)
            Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)
            u = 1.0 / (Kv + 1e-8)
            u = torch.clamp(u, min=1e-8, max=1e8)  # 限制u的范围
            
            # v = 1 / (K^T @ u + epsilon)
            KTu = torch.matmul(K.t(), u.unsqueeze(-1)).squeeze(-1)
            v = 1.0 / (KTu + 1e-8)
            v = torch.clamp(v, min=1e-8, max=1e8)  # 限制v的范围
        
        # 3. 计算最优传输距离
        # Transport Plan: gamma = diag(u) * K * diag(v)
        gamma = u.unsqueeze(-1) * K * v.unsqueeze(0)  # [B, B]
        
        # 检查gamma是否有nan或inf
        if torch.isnan(gamma).any() or torch.isinf(gamma).any():
            # 如果出现数值问题，使用简化的MSE损失作为fallback
            cost = torch.mean((x - y) ** 2)
        else:
            cost = torch.sum(gamma * C)
            if self.reduction == 'mean':
                cost = cost / B  # 归一化到每个样本
        
        # 最终检查
        if torch.isnan(cost) or torch.isinf(cost):
            cost = torch.mean((x - y) ** 2)  # Fallback to MSE
        
        return cost


class CounterfactualConsistencyLoss(nn.Module):
    """
    反事实一致性损失
    核心思想：对因果特征添加不同中心的噪声后，预测结果应该保持不变
    这证明了模型学会了"以不变（因果）应万变（噪声）"
    """
    
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean' or 'sum'
        """
        super(CounterfactualConsistencyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, logits_orig, logits_cf):
        """
        计算反事实一致性损失
        
        Args:
            logits_orig: [B, num_classes] 原始预测（z_causal）
            logits_cf: [B, num_classes] 反事实预测（z_causal + z_noise_cf）
        
        Returns:
            loss: 标量，一致性损失
        """
        # 使用MSE损失或KL散度
        # MSE更简单直接
        loss = F.mse_loss(logits_orig, logits_cf, reduction=self.reduction)
        
        return loss


class AdversarialLoss(nn.Module):
    """
    对抗损失：确保z_noise包含域信息
    使用中心判别器预测中心ID
    """
    
    def __init__(self, num_centers=5, reduction='mean'):
        """
        Args:
            num_centers: 中心数量
            reduction: 'mean' or 'sum'
        """
        super(AdversarialLoss, self).__init__()
        self.num_centers = num_centers
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, center_logits, center_labels):
        """
        计算对抗损失
        
        Args:
            center_logits: [B, num_centers] 中心判别器输出
            center_labels: [B] 中心标签
        
        Returns:
            loss: 标量，对抗损失
        """
        # 检查判别器输出维度
        num_classes = center_logits.size(1)
        
        # 如果判别器输出维度为2（说明是单中心情况，但判别器被设置为2维），使用熵损失
        # 或者如果num_centers为1或2，也使用熵损失（鼓励预测不确定性）
        if num_classes == 2 or self.num_centers <= 2:
            # 使用熵损失：鼓励预测接近均匀分布（高熵），即无法区分中心
            probs = torch.softmax(center_logits, dim=1)  # [B, num_classes]
            # 计算熵：H(p) = -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # [B]
            # 最大熵 = log(num_classes)，当num_classes=2时，max_entropy = log(2) ≈ 0.693
            max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32, device=center_logits.device))
            # 对抗损失 = 最大熵 - 当前熵（鼓励高熵，即不确定性）
            # 当预测完全确定时（熵=0），损失=max_entropy；当预测完全不确定时（熵=max_entropy），损失=0
            loss = torch.clamp(max_entropy - entropy.mean(), min=0.0)
            
            # 添加调试信息（仅在第一个batch）
            if not hasattr(self, '_entropy_debug_printed'):
                print(f"🔍 Adv Loss Debug: num_classes={num_classes}, num_centers={self.num_centers}, "
                      f"entropy={entropy.mean().item():.4f}, max_entropy={max_entropy.item():.4f}, "
                      f"loss={loss.item():.4f}")
                self._entropy_debug_printed = True
            
            return loss
        else:
            # 多个中心时，使用交叉熵损失
            return self.criterion(center_logits, center_labels)


# 保留原有的损失函数以兼容
class OrthogonalLoss(nn.Module):
    """正交损失（可选，用于对比实验）"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, z_causal, z_noise):
        inner_product = (z_causal * z_noise).sum(dim=-1)
        loss = inner_product.abs()
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class DistributionMatchingLoss(nn.Module):
    """KL散度损失（可选，用于对比实验）"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, z_causal, μ_bio, σ_bio):
        μ_q = z_causal
        σ_q = torch.ones_like(μ_q)
        σ_bio_safe = σ_bio + 1e-6
        σ_q_safe = σ_q + 1e-6
        
        kl = 0.5 * (
            torch.log(σ_bio_safe.pow(2) / σ_q_safe.pow(2)) +
            (σ_q_safe.pow(2) + (μ_q - μ_bio).pow(2)) / σ_bio_safe.pow(2) -
            1
        )
        kl_loss = kl.mean(dim=-1)
        
        if self.reduction == 'mean':
            return kl_loss.mean()
        return kl_loss.sum()


