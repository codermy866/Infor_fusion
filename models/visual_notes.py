#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.1: 增强型视觉笔记模块
升级：使用 Cross-Attention 机制进行文本引导的图像特征增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EnhancedVisualNoteLayer(nn.Module):
    """
    增强型视觉笔记层：基于 Cross-Attention 的特征调制
    Text (Query) -> 指导 -> Image (Key/Value)
    """
    
    def __init__(self, img_dim: int = 768, text_dim: int = 768, hidden_dim: int = 768, num_heads: int = 4):
        super().__init__()
        
        # 1. 投影层
        self.q_proj = nn.Linear(text_dim, hidden_dim)  # Text acts as Query
        self.k_proj = nn.Linear(img_dim, hidden_dim)   # Image acts as Key
        
        # 2. 缩放因子
        self.scale = hidden_dim ** -0.5
        
        # 3. 门控网络 (用于生成最终的 Soft Mask)
        # 将 Attention Score 映射为 0-1 的门控值
        self.gate_net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        
        # 4. 特征精炼 (可选，用于增强被激活的特征)
        self.feat_refine = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, img_dim),
            nn.GELU()
        )

    def forward(
        self, 
        img_feats: torch.Tensor,  # [B, N, D]
        text_feats: torch.Tensor,  # [B, D]
        beta: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_feats: 图像 Patch 特征
            text_feats: 文本 Knowledge 特征
            beta: 背景抑制系数 (0.1 表示背景保留 10%)
        """
        B, N, D = img_feats.shape
        
        # --- 1. Cross-Attention Score Calculation ---
        # Query: Text [B, 1, H]
        q = self.q_proj(text_feats).unsqueeze(1)
        # Key: Image [B, N, H]
        k = self.k_proj(img_feats)
        
        # Attention Logits: [B, 1, H] @ [B, H, N] -> [B, 1, N]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Transpose to [B, N, 1] for masking
        attn_logits = attn_logits.transpose(1, 2)
        
        # --- 2. Adaptive Gating (Soft Mask Generation) ---
        # 使用门控网络生成 mask (0~1)
        # 相比简单的 sigmoid，这里允许网络学习阈值偏移
        raw_mask = self.gate_net(attn_logits) # [B, N, 1]
        
        # 保护下界，防止梯度消失 (Safety Clip)
        # 训练初期允许更多信息通过，防止冷启动失败
        mask = torch.clamp(raw_mask, min=0.05, max=1.0)
        
        # --- 3. Feature Modulation (Visual Note Logic) ---
        # High response regions: Keep original (x 1.0)
        # Low response regions: Suppress (x beta)
        modulation_weight = mask + (1 - mask) * beta
        
        # Apply modulation
        img_focused = img_feats * modulation_weight
        
        # Optional: Refine the focused features
        img_focused = self.feat_refine(img_focused) + img_focused # Residual
        
        return img_focused, mask

class VisualNotesModule(nn.Module):
    """
    视觉笔记模块封装（支持 Warm-up 策略）
    """
    def __init__(self, img_dim=768, text_dim=768, hidden_dim=768, warmup_epochs=10):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # 使用增强版 Layer
        self.layer = EnhancedVisualNoteLayer(img_dim, text_dim, hidden_dim)
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        
    def get_beta(self) -> float:
        """动态 Beta 策略：Warm-up 期间不抑制，之后逐渐增强抑制"""
        if self.current_epoch < self.warmup_epochs:
            return 1.0 # 不抑制
        elif self.current_epoch < self.warmup_epochs * 2:
            # 线性衰减 1.0 -> 0.3
            ratio = (self.current_epoch - self.warmup_epochs) / self.warmup_epochs
            return 1.0 - 0.7 * ratio
        else:
            return 0.3 # 稳定在 0.3

    def forward(self, img, text, beta=None):
        if beta is None:
            beta = self.get_beta()
        return self.layer(img, text, beta=beta)

