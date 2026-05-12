#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-Invariant Distributional Anchoring (BIDA) Model
核心框架：流形投影与正交解耦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import torchvision.models as tvm

from .distributional_anchor import DistributionalAnchor
from .orthogonal_loss import OrthogonalLoss, DistributionMatchingLoss, NoiseSupervisionLoss


class DualHeadImageEncoder(nn.Module):
    """
    双头图像编码器
    Head 1: z_causal (用于分类)
    Head 2: z_noise (用于对抗预测医院ID)
    
    注意：输入是已经提取好的图像特征（512维），不是原始图像
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # 输入特征维度（oct_features和colpo_features的维度）
        embed_dim: int = 768,  # 输出嵌入维度
        backbone: str = 'resnet50'  # 保留参数以兼容，但实际不使用
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        
        # 特征投影层：将输入特征（512维）投影到embed_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 双头网络
        # Head 1: z_causal (因果特征，用于分类)
        self.causal_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Head 2: z_noise (噪声特征，用于对抗预测医院ID)
        self.noise_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            image_features: [B, embed_dim] 图像特征
        
        Returns:
            z_causal: [B, embed_dim] 因果特征
            z_noise: [B, embed_dim] 噪声特征
        """
        # 投影特征
        proj_feat = self.feature_proj(image_features)  # [B, embed_dim]
        
        # 双头输出
        z_causal = self.causal_head(proj_feat)  # [B, embed_dim]
        z_noise = self.noise_head(proj_feat)  # [B, embed_dim]
        
        return z_causal, z_noise


class BIDAModel(nn.Module):
    """
    Bio-Invariant Distributional Anchoring (BIDA) Model
    
    架构：
    - Branch A: Clinical Text -> VLM -> Distributional Anchor (μ_bio, σ_bio)
    - Branch B: Image -> ResNet50 -> Dual Head (z_causal, z_noise)
    - Constraint: z_causal 必须在 N(μ_bio, σ_bio) 内
    - Orthogonality: z_causal ⊥ z_noise
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        backbone: str = 'resnet50',
        vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_centers = num_centers
        
        # Branch A: Distributional Anchor
        self.distributional_anchor = DistributionalAnchor(
            embed_dim=embed_dim,
            vlm_model=vlm_model,
            freeze_vlm=True
        )
        
        # Branch B: Dual Head Image Encoder
        # 注意：输入是已经提取好的特征（512维），不是原始图像
        self.image_encoder = DualHeadImageEncoder(
            input_dim=512,  # oct_features和colpo_features的维度
            embed_dim=embed_dim
        )
        
        # 分类器（使用 z_causal）- 增强网络容量
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # 第一层保持全维度
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # 降低dropout，从0.3到0.2
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 损失函数
        self.orthogonal_loss_fn = OrthogonalLoss(reduction='mean')
        self.distribution_loss_fn = DistributionMatchingLoss(reduction='mean')
        self.noise_loss_fn = NoiseSupervisionLoss(num_centers=num_centers)
    
    def forward(
        self,
        oct_features: torch.Tensor,
        colpo_features: torch.Tensor,
        clinical_features: torch.Tensor,
        clinical_data: Optional[Dict] = None,
        center_labels: Optional[torch.Tensor] = None,
        oct_images: Optional[torch.Tensor] = None,  # 原始OCT图像，用于VLM
        colposcopy_images: Optional[torch.Tensor] = None,  # 原始阴道镜图像，用于VLM
        return_loss_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            oct_features: [B, embed_dim] OCT特征
            colpo_features: [B, embed_dim] Colposcopy特征
            clinical_features: [B, D] 临床特征（用于fallback）
            clinical_data: dict 临床数据（用于VLM）
            center_labels: [B] 中心标签（用于L_noise）
            return_loss_components: 是否返回损失组件
        
        Returns:
            output dict with keys: 'logits', 'z_causal', 'z_noise', 'mu_bio', 'sigma_bio', ...
        """
        B = oct_features.size(0)
        
        # Branch A: 生成生物流形分布
        # VLM处理图像+文本的联合理解：OCT图像 + 临床文本描述
        μ_bio, σ_bio = self.distributional_anchor(
            clinical_data=clinical_data,
            clinical_features=clinical_features,
            oct_images=oct_images,  # 原始OCT图像，用于VLM图像+文本联合理解
            colposcopy_images=colposcopy_images  # 原始阴道镜图像（可选）
        )  # [B, embed_dim], [B, embed_dim]
        
        # Branch B: 图像特征提取（使用OCT和Colposcopy的平均）
        # oct_features和colpo_features应该是512维
        if oct_features.size(-1) != colpo_features.size(-1):
            # 如果维度不匹配，取最小维度
            min_dim = min(oct_features.size(-1), colpo_features.size(-1))
            image_feat = (oct_features[..., :min_dim] + colpo_features[..., :min_dim]) / 2
        else:
            image_feat = (oct_features + colpo_features) / 2  # [B, 512]
        
        # 双头输出
        z_causal, z_noise = self.image_encoder(image_feat)  # [B, embed_dim], [B, embed_dim]
        
        # 分类（使用 z_causal）
        logits = self.classifier(z_causal)  # [B, num_classes]
        
        # 构建输出
        output = {
            'logits': logits,
            'z_causal': z_causal,
            'z_noise': z_noise,
            'mu_bio': μ_bio,
            'sigma_bio': σ_bio
        }
        
        # 如果需要计算损失组件
        if return_loss_components and center_labels is not None:
            # L_dist: 分布匹配损失
            L_dist = self.distribution_loss_fn(z_causal, μ_bio, σ_bio)
            
            # L_orth: 正交损失
            L_orth = self.orthogonal_loss_fn(z_causal, z_noise)
            
            # L_noise: 噪声监督损失
            L_noise = self.noise_loss_fn(z_noise, center_labels)
            
            output['loss_components'] = {
                'L_dist': L_dist,
                'L_orth': L_orth,
                'L_noise': L_noise
            }
        
        return output

