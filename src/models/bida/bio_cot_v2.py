#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 2.0: 基于LLM语义锚点与因果最优传输的多模态分类框架
核心升级：
1. Text Encoder: MLP → Frozen Med-LLM (Offline Embeddings)
2. Baseline Fusion: Concat → Cross-Attention Fusion
3. Alignment: MSE/KL → Sinkhorn Optimal Transport (OT)
4. Modular Design: 支持消融实验（use_ot, use_dual, use_llm）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .prior_net import StudentPriorNet, build_clinical_vector
from .memory_bank import NoiseMemoryBank, CenterDiscriminator
from .losses import SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss


class TextProjector(nn.Module):
    """
    临床语义映射器
    将离线提取的LLM嵌入（4096维）映射到与图像特征对齐的维度（768维）
    """
    
    def __init__(self, input_dim: int = 4096, embed_dim: int = 768):
        """
        Args:
            input_dim: LLM嵌入维度（默认4096，根据实际LLM调整）
            embed_dim: 输出维度（与图像特征对齐，默认768）
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, embed_dim)  # Output: z_sem
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # kaiming_normal_只支持'relu'和'leaky_relu'，对于GELU使用xavier_normal_
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] LLM嵌入向量
        
        Returns:
            z_sem: [B, embed_dim] 语义锚点特征
        """
        return self.net(x)


class CrossModalFusion(nn.Module):
    """
    跨模态融合模块（Strong Baseline）
    使用Cross-Attention融合图像和文本特征
    Query: Image, Key/Value: Text (Semantic Anchor)
    """
    
    def __init__(self, dim: int = 768, num_heads: int = 8):
        """
        Args:
            dim: 特征维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-Attention: Query=Image, Key/Value=Text
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: [B, embed_dim] 图像特征
            text_feat: [B, embed_dim] 文本特征（语义锚点）
        
        Returns:
            fused_feat: [B, embed_dim] 融合后的特征
        """
        # 添加序列维度: [B, embed_dim] -> [B, 1, embed_dim]
        img_feat = img_feat.unsqueeze(1)  # [B, 1, embed_dim]
        text_feat = text_feat.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Cross-Attention: Query=Image, Key/Value=Text
        attn_out, attn_weights = self.cross_attn(
            query=img_feat,
            key=text_feat,
            value=text_feat
        )
        
        # Residual connection + LayerNorm
        x = self.norm1(img_feat + attn_out)  # [B, 1, embed_dim]
        
        # Feed-Forward Network
        ffn_out = self.ffn(x)
        
        # Residual connection + LayerNorm
        x = self.norm2(x + ffn_out)  # [B, 1, embed_dim]
        
        # 移除序列维度: [B, 1, embed_dim] -> [B, embed_dim]
        return x.squeeze(1)


class DualHeadImageEncoder(nn.Module):
    """
    双头图像编码器（用于消融实验）
    Head 1: z_causal (用于分类)
    Head 2: z_noise (用于对抗预测医院ID)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        
        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 双头网络
        self.causal_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: [B, input_dim] 图像特征
        
        Returns:
            z_causal: [B, embed_dim] 因果特征
            z_noise: [B, embed_dim] 噪声特征
        """
        proj_feat = self.feature_proj(image_features)  # [B, embed_dim]
        z_causal = self.causal_head(proj_feat)  # [B, embed_dim]
        z_noise = self.noise_head(proj_feat)  # [B, embed_dim]
        return z_causal, z_noise


class BioCOT_v2(nn.Module):
    """
    Bio-COT 2.0模型
    模块化设计，支持消融实验
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 768,  # ViT输出维度
        llm_embed_dim: int = 4096,  # LLM嵌入维度
        use_ot: bool = True,  # 是否使用Sinkhorn OT
        use_dual: bool = True,  # 是否使用Dual-Head
        use_llm: bool = True,  # 是否使用LLM嵌入（否则使用传统MLP）
        use_cross_attn: bool = True,  # 是否使用Cross-Attention融合（否则使用Concat）
    ):
        """
        Args:
            embed_dim: 特征嵌入维度（默认768，与ViT对齐）
            num_classes: 分类类别数
            num_centers: 中心数量
            input_dim: 图像特征输入维度（ViT输出，默认768）
            llm_embed_dim: LLM嵌入维度（默认4096，根据实际LLM调整）
            use_ot: 是否使用Sinkhorn OT损失
            use_dual: 是否使用Dual-Head结构
            use_llm: 是否使用LLM嵌入（True=LLM, False=传统MLP）
            use_cross_attn: 是否使用Cross-Attention融合（True=Cross-Attn, False=Concat）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.use_ot = use_ot
        self.use_dual = use_dual
        self.use_llm = use_llm
        self.use_cross_attn = use_cross_attn
        
        # 1. 文本编码器（二选一）
        if use_llm:
            # LLM路径：TextProjector
            self.text_projector = TextProjector(
                input_dim=llm_embed_dim,
                embed_dim=embed_dim
            )
            # print(f"✅ 使用LLM文本编码器 (输入维度: {llm_embed_dim} → 输出维度: {embed_dim})")
        else:
            # 传统路径：Student Prior网络
            self.student_prior = StudentPriorNet(
                input_dim=7,  # HPV(1) + TCT(5) + Age(1)
                output_dim=embed_dim
            )
            # print(f"✅ 使用传统MLP文本编码器 (输入维度: 7 → 输出维度: {embed_dim})")
        
        # 2. 图像编码器
        if use_dual:
            # Dual-Head结构
            self.image_encoder = DualHeadImageEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim
            )
            # print(f"✅ 使用Dual-Head图像编码器")
        else:
            # 单头结构（仅Causal Head）
            self.image_encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            # print(f"✅ 使用单头图像编码器")
        
        # 3. 融合模块（二选一）
        if use_cross_attn:
            # Cross-Attention融合
            self.fusion_module = CrossModalFusion(dim=embed_dim, num_heads=8)
            # print(f"✅ 使用Cross-Attention融合")
        else:
            # 简单拼接融合
            self.fusion_module = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            )
            # print(f"✅ 使用Concat融合")
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 5. 损失函数模块（用于消融实验）
        if use_ot:
            self.ot_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        
        if use_dual:
            self.memory_bank = NoiseMemoryBank(
                num_centers=num_centers,
                feat_dim=embed_dim,
                capacity=100
            )
            self.consistency_loss = CounterfactualConsistencyLoss()
            self.adversarial_loss = AdversarialLoss(num_centers=max(num_centers, 2))
            self.center_discriminator = CenterDiscriminator(
                feat_dim=embed_dim,
                num_centers=max(num_centers, 2)
            )
    
    def forward(
        self,
        oct_features: torch.Tensor,
        colpo_features: torch.Tensor,
        clinical_embeddings: Optional[torch.Tensor] = None,  # [B, llm_embed_dim] LLM嵌入
        clinical_data: Optional[Dict] = None,  # 传统路径的临床数据
        center_labels: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        use_counterfactual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            oct_features: [B, input_dim] OCT特征
            colpo_features: [B, input_dim] Colposcopy特征
            clinical_embeddings: [B, llm_embed_dim] LLM嵌入（use_llm=True时使用）
            clinical_data: dict 传统临床数据（use_llm=False时使用）
            center_labels: [B] 中心标签
            return_loss_components: 是否返回损失组件
            use_counterfactual: 是否使用反事实干预
        
        Returns:
            output: dict包含logits, z_causal, z_sem, z_noise等
        """
        B = oct_features.shape[0]
        device = oct_features.device
        
        # 1. 图像特征融合（OCT + Colposcopy）
        # 简单加权融合
        img_feat = 0.6 * oct_features + 0.4 * colpo_features  # [B, input_dim]
        
        # 2. 图像编码
        if self.use_dual:
            z_causal, z_noise = self.image_encoder(img_feat)  # [B, embed_dim], [B, embed_dim]
        else:
            z_causal = self.image_encoder(img_feat)  # [B, embed_dim]
            z_noise = None
        
        # 3. 文本编码（语义锚点生成）
        if self.use_llm:
            # LLM路径
            if clinical_embeddings is None:
                raise ValueError("use_llm=True时，必须提供clinical_embeddings")
            z_sem = self.text_projector(clinical_embeddings)  # [B, embed_dim]
        else:
            # 传统路径
            if clinical_data is None:
                raise ValueError("use_llm=False时，必须提供clinical_data")
            clinical_vec = build_clinical_vector(clinical_data, device=device)  # [B, 7]
            z_sem = self.student_prior(clinical_vec)  # [B, embed_dim]
        
        # 4. 跨模态融合
        if self.use_cross_attn:
            # Cross-Attention融合
            fused_feat = self.fusion_module(z_causal, z_sem)  # [B, embed_dim]
        else:
            # 简单拼接融合
            multimodal_feat = torch.cat([z_causal, z_sem], dim=-1)  # [B, embed_dim * 2]
            fused_feat = self.fusion_module(multimodal_feat)  # [B, embed_dim]
        
        # 5. 分类
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        # 6. 构建输出
        output = {
            'logits': logits,
            'z_causal': z_causal,
            'z_sem': z_sem,
        }
        
        if z_noise is not None:
            output['z_noise'] = z_noise
        
        # 7. 计算损失组件（如果需要）
        if return_loss_components:
            loss_dict = {}
            
            # 7.1 分类损失（始终计算）
            # 注意：分类损失在训练循环中计算，这里不计算
            
            # 7.2 Sinkhorn OT损失
            if self.use_ot:
                L_ot = self.ot_loss(z_causal, z_sem)
                loss_dict['L_ot'] = L_ot
            
            # 7.3 反事实一致性损失（需要Dual-Head）
            if self.use_dual and use_counterfactual and center_labels is not None:
                L_consist = torch.tensor(0.0, device=device, requires_grad=True)
                try:
                    # 更新Memory Bank
                    self.memory_bank.update(z_noise, center_labels)
                    
                    # 检查Memory Bank是否有样本
                    total_samples = self.memory_bank.count.sum().item()
                    centers_with_samples = (self.memory_bank.count > 0).sum().item()
                    
                    if total_samples > 0 and centers_with_samples >= 1:
                        # 生成反事实噪声
                        if self.num_centers > 1 and centers_with_samples > 1:
                            available_centers = torch.where(self.memory_bank.count > 0)[0].cpu().numpy()
                            if len(available_centers) > 1:
                                fake_center_ids_list = []
                                for i in range(B):
                                    current_center = center_labels[i].item()
                                    other_centers = [c for c in available_centers if c != current_center]
                                    if len(other_centers) > 0:
                                        fake_center_ids_list.append(np.random.choice(other_centers))
                                    else:
                                        fake_center_ids_list.append(current_center)
                                fake_center_ids = torch.tensor(fake_center_ids_list, device=device, dtype=center_labels.dtype)
                            else:
                                fake_center_ids = center_labels
                        else:
                            fake_center_ids = center_labels
                        
                        # 从Memory Bank采样反事实噪声
                        z_noise_cf = self.memory_bank.get_counterfactual_noise(fake_center_ids, strategy='random')
                        
                        if z_noise_cf is not None and z_noise_cf.shape[0] == B and z_noise_cf.shape[1] == z_causal.shape[1]:
                            alpha = 0.3
                            z_mix = z_causal + alpha * z_noise_cf
                            logits_cf = self.classifier(z_mix)
                            if logits.shape[0] == logits_cf.shape[0]:
                                L_consist = self.consistency_loss(logits, logits_cf)
                            else:
                                L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                        else:
                            # Fallback: 使用当前batch的z_noise
                            if z_noise is not None:
                                z_noise_cf_fallback = z_noise.detach().clone()
                                if z_noise_cf_fallback.shape[0] == z_causal.shape[0]:
                                    alpha = 0.3
                                    z_mix = z_causal + alpha * z_noise_cf_fallback
                                    logits_cf = self.classifier(z_mix)
                                    L_consist = self.consistency_loss(logits, logits_cf)
                                else:
                                    L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                            else:
                                L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                    else:
                        # Memory Bank未填充，使用fallback
                        if z_noise is not None:
                            z_noise_cf_fallback = z_noise.detach().clone()
                            if z_noise_cf_fallback.shape[0] == z_causal.shape[0]:
                                alpha = 0.3
                                z_mix = z_causal + alpha * z_noise_cf_fallback
                                logits_cf = self.classifier(z_mix)
                                L_consist = self.consistency_loss(logits, logits_cf)
                            else:
                                L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                        else:
                            L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                except Exception as e:
                    # 异常处理，使用fallback
                    if z_noise is not None:
                        z_noise_cf_fallback = z_noise.detach().clone()
                        if z_noise_cf_fallback.shape[0] == z_causal.shape[0]:
                            alpha = 0.3
                            z_mix = z_causal + alpha * z_noise_cf_fallback
                            logits_cf = self.classifier(z_mix)
                            L_consist = self.consistency_loss(logits, logits_cf)
                        else:
                            L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                    else:
                        L_consist = torch.tensor(0.01, device=device, requires_grad=True)
                
                L_consist = torch.clamp(L_consist, min=0.001, max=10.0)
                loss_dict['L_consist'] = L_consist
            
            # 7.4 对抗损失（需要Dual-Head）
            if self.use_dual and center_labels is not None:
                center_logits = self.center_discriminator(z_noise)
                L_adv = self.adversarial_loss(center_logits, center_labels)
                loss_dict['L_adv'] = L_adv
            
            output['loss_components'] = loss_dict
        
        return output

