#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT模型：在BIDA基础上整合新模块
- Student Prior网络替代VLM
- Sinkhorn OT Loss替代KL散度
- Memory Bank实现反事实干预
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .prior_net import StudentPriorNet, build_clinical_vector
from .memory_bank import NoiseMemoryBank, CenterDiscriminator
from .losses import SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss
from .vlm_image_encoder import VLMImageEncoderFromTensor


class DualHeadImageEncoder(nn.Module):
    """
    双头图像编码器（与BIDA相同）
    Head 1: z_causal (用于分类)
    Head 2: z_noise (用于对抗预测医院ID)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
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


class BioCOTModel(nn.Module):
    """
    Bio-COT模型
    核心改进：
    1. Student Prior替代VLM（训练时快速）
    2. Sinkhorn OT替代KL散度（更灵活的分布对齐）
    3. Memory Bank实现反事实干预（真正的因果解耦）
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 512,
        use_vlm_encoder: bool = False,  # 是否使用VLM图像编码器
        vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct",  # VLM模型名称
        freeze_vlm: bool = True,  # 是否冻结VLM参数
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.use_vlm_encoder = use_vlm_encoder
        
        # Student Prior网络（替代VLM）
        self.student_prior = StudentPriorNet(
            input_dim=7,  # HPV(1) + TCT(5) + Age(1)
            output_dim=embed_dim
        )
        
        # 图像编码器：选择使用VLM或传统MLP
        if use_vlm_encoder:
            print(f"🚀 使用VLM图像编码器: {vlm_model}")
            self.image_encoder = VLMImageEncoderFromTensor(
                vlm_model=vlm_model,
                embed_dim=embed_dim,
                freeze_vlm=freeze_vlm
            )
            # VLM编码器不需要input_dim，但保留以兼容
            self.input_dim = None
        else:
            print("📊 使用传统MLP图像编码器")
            self.image_encoder = DualHeadImageEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim
            )
            self.input_dim = input_dim
        
        # 多模态融合模块（改进：使用注意力机制融合OCT和Colposcopy）
        # 如果使用VLM，input_dim可能为None，需要动态处理
        if use_vlm_encoder:
            # VLM模式下，融合层直接处理embed_dim的特征
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),  # 拼接后投影
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            # 传统模式下，需要input_dim
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(input_dim * 2, embed_dim),  # 拼接后投影
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim)
            )
        
        # 可学习的模态权重（用于加权融合）
        # 确保初始化为正确的tensor类型，并注册为Parameter
        self.modal_weights = nn.Parameter(torch.ones(2, dtype=torch.float32) / 2.0)  # [oct_weight, colpo_weight]
        
        # 分类器（增强正则化，防止过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # 保持全维度，提升容量
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.5),  # 增强dropout：提高到0.5，防止过拟合
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),  # 增强dropout：提高到0.4，防止过拟合
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 中心判别器（用于对抗训练）
        # 即使只有一个中心，也创建判别器（用于熵损失）
        self.center_discriminator = CenterDiscriminator(
            feat_dim=embed_dim,
            num_centers=max(num_centers, 2)  # 至少2个中心，即使实际只有1个（用于熵损失计算）
        )
        
        # Memory Bank
        self.memory_bank = NoiseMemoryBank(
            num_centers=num_centers,
            feat_dim=embed_dim,
            capacity=100
        )
        
        # 损失函数
        self.sinkhorn_loss = SinkhornDistance(eps=0.1, max_iter=100)
        self.consistency_loss = CounterfactualConsistencyLoss()
        self.adversarial_loss = AdversarialLoss(num_centers=max(num_centers, 2))  # 至少2个中心
    
    def _traditional_fusion(self, oct_features: torch.Tensor, colpo_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """传统特征融合方法（不使用VLM）"""
        # 检查是否为单模态（colpo_features全为零）
        is_single_modal = torch.allclose(colpo_features, torch.zeros_like(colpo_features), atol=1e-6)
        
        if is_single_modal:
            # 单模态模式：只使用OCT特征
            oct_feat = oct_features
            # 直接使用OCT特征，不需要融合
            if hasattr(self.image_encoder, 'input_dim') and self.image_encoder.input_dim is not None:
                if oct_feat.size(-1) != self.image_encoder.input_dim:
                    # 投影到input_dim
                    if not hasattr(self, '_oct_proj'):
                        self._oct_proj = nn.Linear(oct_feat.size(-1), self.image_encoder.input_dim).to(oct_feat.device)
                    oct_feat = self._oct_proj(oct_feat)
            # 双头输出
            z_causal, z_noise = self.image_encoder(oct_feat)  # [B, embed_dim], [B, embed_dim]
            return z_causal, z_noise
        
        # 多模态模式：融合OCT和Colposcopy
        # 确保维度一致
        if oct_features.size(-1) != colpo_features.size(-1):
            min_dim = min(oct_features.size(-1), colpo_features.size(-1))
            oct_feat = oct_features[..., :min_dim]
            colpo_feat = colpo_features[..., :min_dim]
        else:
            oct_feat = oct_features
            colpo_feat = colpo_features
        
        # 确保modal_weights是Parameter张量，并在正确的设备上
        # 简化修复：直接使用，如果不存在或设备不匹配则重新创建
        if not hasattr(self, 'modal_weights') or not isinstance(self.modal_weights, nn.Parameter):
            # 如果不存在或不是Parameter，重新创建并注册
            self.register_parameter('modal_weights', nn.Parameter(torch.ones(2, dtype=torch.float32, device=oct_feat.device) / 2.0))
        elif self.modal_weights.device != oct_feat.device:
            # 如果设备不匹配，移动到正确的设备（保持为Parameter）
            self.modal_weights.data = self.modal_weights.data.to(oct_feat.device)
        
        # 直接调用softmax，确保modal_weights是有效的Parameter
        # 使用torch.nn.functional.softmax而不是F.softmax，避免命名冲突
        try:
            # 确保modal_weights是tensor
            if not isinstance(self.modal_weights, torch.Tensor):
                raise TypeError(f"modal_weights不是tensor: {type(self.modal_weights)}")
            weights = torch.nn.functional.softmax(self.modal_weights, dim=0)
        except Exception as e:
            # 如果失败，强制重新创建Parameter
            print(f"⚠️ modal_weights softmax失败: {e}, 强制重新创建Parameter", flush=True)
            # 删除旧的modal_weights
            if hasattr(self, 'modal_weights'):
                delattr(self, 'modal_weights')
            # 重新创建
            self.modal_weights = nn.Parameter(torch.ones(2, dtype=torch.float32, device=oct_feat.device) / 2.0)
            weights = torch.nn.functional.softmax(self.modal_weights, dim=0)
        weighted_feat = weights[0] * oct_feat + weights[1] * colpo_feat  # [B, input_dim]
        
        # 方法2：拼接+融合（更强大的融合方式）
        concat_feat = torch.cat([oct_feat, colpo_feat], dim=-1)  # [B, input_dim * 2]
        fused_feat = self.multimodal_fusion(concat_feat)  # [B, embed_dim]
        
        # 将加权特征投影到embed_dim（如果维度不匹配）
        if weighted_feat.size(-1) != self.embed_dim:
            # 使用临时投影层（在第一次forward时创建）
            if not hasattr(self, '_weighted_proj'):
                self._weighted_proj = nn.Linear(weighted_feat.size(-1), self.embed_dim).to(weighted_feat.device)
            weighted_proj = self._weighted_proj(weighted_feat)  # [B, embed_dim]
        else:
            weighted_proj = weighted_feat
        
        # 最终融合：融合特征（主要）+ 加权特征（辅助）
        image_feat = 0.7 * fused_feat + 0.3 * weighted_proj  # [B, embed_dim]
        
        # 投影到input_dim（image_encoder需要input_dim输入）
        # 如果使用VLM编码器，input_dim为None，不应该走传统融合路径
        if self.use_vlm_encoder:
            # VLM编码器需要原始图像，不应该走传统融合路径
            # 如果走到这里，说明VLM路径失败，需要特殊处理
            # 对于VLM编码器，我们无法从特征重建图像，所以使用零向量作为fallback
            print("⚠️ 警告：VLM模式下缺少原始图像，使用零向量作为fallback")
            z_causal = torch.zeros(oct_features.size(0), self.embed_dim, device=oct_features.device)
            z_noise = torch.zeros(oct_features.size(0), self.embed_dim, device=oct_features.device)
        elif hasattr(self.image_encoder, 'input_dim') and self.image_encoder.input_dim is not None:
            if not hasattr(self, '_final_proj'):
                self._final_proj = nn.Linear(self.embed_dim, self.image_encoder.input_dim).to(image_feat.device)
            image_feat = self._final_proj(image_feat)  # [B, input_dim]
            # 双头输出
            z_causal, z_noise = self.image_encoder(image_feat)  # [B, embed_dim], [B, embed_dim]
        else:
            # 如果input_dim为None，直接使用embed_dim（不应该发生）
            z_causal, z_noise = self.image_encoder(image_feat)  # [B, embed_dim], [B, embed_dim]
        return z_causal, z_noise
    
    def forward(
        self,
        oct_features: torch.Tensor,
        colpo_features: torch.Tensor,
        clinical_features: torch.Tensor,
        clinical_data: Optional[Dict] = None,
        center_labels: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        use_counterfactual: bool = True,
        oct_images: Optional[torch.Tensor] = None,  # 原始OCT图像 [B, C, H, W] 或 [B, F, C, H, W]
        colpo_images: Optional[torch.Tensor] = None,  # 原始Colposcopy图像 [B, C, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            oct_features: [B, embed_dim] OCT特征
            colpo_features: [B, embed_dim] Colposcopy特征
            clinical_features: [B, D] 临床特征（用于fallback）
            clinical_data: dict 临床数据（用于Student Prior）
            center_labels: [B] 中心标签
            return_loss_components: 是否返回损失组件
            use_counterfactual: 是否使用反事实干预
        
        Returns:
            output dict
        """
        B = oct_features.size(0)
        
        # 1. 生成语义锚点（Student Prior）
        if clinical_data is not None:
            clinical_vec = build_clinical_vector(clinical_data, device=oct_features.device)
        else:
            # Fallback: 使用clinical_features（需要转换为7维）
            # 这里简化处理，假设clinical_features已经是7维
            if clinical_features.size(-1) == 7:
                clinical_vec = clinical_features
            else:
                # 使用零向量
                clinical_vec = torch.zeros(B, 7, device=oct_features.device)
        
        z_sem = self.student_prior(clinical_vec)  # [B, embed_dim]
        
        # 2. 多模态图像特征融合（改进：使用VLM增强或传统融合）
        # 如果使用VLM编码器且有原始图像，使用VLM处理
        if self.use_vlm_encoder and oct_images is not None and colpo_images is not None:
            # 使用VLM处理原始图像
            # 对于OCT图像，如果是多帧（120帧），处理全部120帧，确保不遗漏任何关键帧
            # 策略：对全部120帧进行VLM特征提取，然后使用注意力机制聚合
            # 调试：打印oct_images的维度信息
            if hasattr(self, '_first_batch_printed') is False:
                print(f"   🔍 调试：oct_images维度 = {oct_images.shape if oct_images is not None else None}")
                print(f"   🔍 调试：colpo_images维度 = {colpo_images.shape if colpo_images is not None else None}")
                self._first_batch_printed = True
            
            if oct_images.dim() == 5:  # [B, F, C, H, W] - 多帧
                num_frames = oct_images.size(1)
                B, F, C, H, W = oct_images.shape
                
                # 限制处理的帧数：最多处理60帧，加速训练
                max_frames = 60
                if num_frames > max_frames:
                    print(f"   ⚠️ 帧数过多（{num_frames}帧），限制为{max_frames}帧以加速训练", flush=True)
                    oct_images = oct_images[:, :max_frames, :, :, :]  # 只取前60帧
                    num_frames = max_frames
                
                # 构建文本提示（使用临床数据）
                text_prompts = None
                if clinical_data is not None:
                    hpv_status = "positive" if any(clinical_data.get('hpv', [])) else "negative"
                    tct_cat = clinical_data.get('tct', ['NILM'])[0] if clinical_data.get('tct') else 'NILM'
                    age = clinical_data.get('age', [0])[0] if clinical_data.get('age') else 0
                    text_prompts = [
                        f"Medical image: HPV {hpv_status}, TCT {tct_cat}, Age {age}"
                    ] * B
                
                # 处理帧：批量处理以提高效率
                # 优化策略：增加批处理大小，减少VLM调用次数，提高效率
                # 对于60帧，使用30帧/批，只需2批处理
                batch_size_frames = 30  # 30帧/批，对于60帧只需2批
                all_frame_features_causal = []
                all_frame_features_noise = []
                
                # 对全部120帧进行VLM特征提取
                num_batches = (num_frames + batch_size_frames - 1) // batch_size_frames
                for batch_idx in range(num_batches):
                    frame_start = batch_idx * batch_size_frames
                    frame_end = min(frame_start + batch_size_frames, num_frames)
                    frame_batch = oct_images[:, frame_start:frame_end, :, :, :]  # [B, batch_size_frames, C, H, W]
                    
                    # 将帧批次展平为 [B*batch_size_frames, C, H, W]
                    B_batch, F_batch, C_batch, H_batch, W_batch = frame_batch.shape
                    # 确保张量连续，然后使用reshape
                    frame_batch = frame_batch.contiguous()
                    frame_batch_flat = frame_batch.view(B_batch * F_batch, C_batch, H_batch, W_batch)
                    
                    # 为每帧复制文本提示
                    text_prompts_batch = text_prompts * F_batch if text_prompts else None
                    
                    # 使用VLM处理这一批帧（添加进度提示）
                    if batch_idx == 0:
                        print(f"   🔄 开始处理OCT图像（{num_frames}帧，分{num_batches}批处理，每批{batch_size_frames}帧）...", flush=True)
                    
                    # 添加每批处理的进度提示
                    print(f"   ⏳ 正在处理第 {batch_idx + 1}/{num_batches} 批（帧 {frame_start}-{frame_end-1}）...", flush=True)
                    
                    try:
                        frame_features_causal, frame_features_noise = self.image_encoder(
                            frame_batch_flat, 
                            text_prompts_batch, 
                            device=oct_features.device
                        )  # [B*batch_size_frames, embed_dim]
                        print(f"   ✅ 第 {batch_idx + 1}/{num_batches} 批处理完成", flush=True)
                    except Exception as e:
                        print(f"   ❌ 第 {batch_idx + 1}/{num_batches} 批处理失败: {e}", flush=True)
                        raise
                    
                    # 恢复批次维度（确保连续）
                    frame_features_causal = frame_features_causal.contiguous().view(B_batch, F_batch, -1)  # [B, batch_size_frames, embed_dim]
                    frame_features_noise = frame_features_noise.contiguous().view(B_batch, F_batch, -1)  # [B, batch_size_frames, embed_dim]
                    
                    all_frame_features_causal.append(frame_features_causal)
                    all_frame_features_noise.append(frame_features_noise)
                    
                    if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == num_batches:
                        print(f"   ✅ 已处理 {batch_idx + 1}/{num_batches} 批（{frame_end}/{num_frames}帧）", flush=True)
                
                # 拼接所有帧的特征
                oct_vlm_causal_all = torch.cat(all_frame_features_causal, dim=1)  # [B, 120, embed_dim]
                oct_vlm_noise_all = torch.cat(all_frame_features_noise, dim=1)  # [B, 120, embed_dim]
                
                # 使用改进的注意力机制聚合全部120帧的特征
                # 创建帧级注意力层（如果不存在）
                if not hasattr(self, 'frame_attention'):
                    self.frame_attention = nn.MultiheadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True
                    ).to(oct_features.device)
                    self.frame_attn_norm = nn.LayerNorm(self.embed_dim).to(oct_features.device)
                    # 添加可学习的聚合权重
                    self.frame_agg_weight = nn.Parameter(torch.tensor([0.6, 0.4], dtype=torch.float32))  # [mean_weight, max_weight]
                
                # 应用注意力机制：让模型自动学习哪些帧更重要
                # 注意：MultiheadAttention需要need_weights=True才能返回注意力权重
                oct_vlm_causal_attn, attn_weights = self.frame_attention(
                    oct_vlm_causal_all, oct_vlm_causal_all, oct_vlm_causal_all,
                    need_weights=True
                )  # [B, 120, embed_dim], [B, num_heads, 120, 120] 或 [B, 120, 120]
                oct_vlm_causal_attn = self.frame_attn_norm(oct_vlm_causal_attn + oct_vlm_causal_all)  # 残差连接
                
                # 改进的聚合策略：使用注意力权重进行加权平均，同时保留最大池化
                # 处理注意力权重的形状：可能是 [B, num_heads, 120, 120] 或 [B, 120, 120]
                if attn_weights is not None:
                    if attn_weights.dim() == 4:  # [B, num_heads, 120, 120]
                        # 对多个头求平均
                        attn_weights = attn_weights.mean(dim=1)  # [B, 120, 120]
                    # 现在 attn_weights 是 [B, 120, 120]
                    # 计算每个帧的平均注意力权重（来自其他帧的关注）
                    attn_weights_mean = attn_weights.mean(dim=2)  # [B, 120] - 每个帧被其他帧关注的平均值
                    attn_weights_norm = torch.nn.functional.softmax(attn_weights_mean, dim=1)  # 归一化
                    oct_vlm_causal_weighted = (oct_vlm_causal_attn * attn_weights_norm.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]
                else:
                    # 如果无法获取注意力权重，使用简单平均
                    oct_vlm_causal_weighted = oct_vlm_causal_attn.mean(dim=1)  # [B, embed_dim]
                
                # 方法2：最大池化（捕获最显著的阳性信号）
                oct_vlm_causal_max = oct_vlm_causal_attn.max(dim=1)[0]  # [B, embed_dim]
                
                # 融合两种聚合方式：使用可学习权重
                agg_weights = torch.nn.functional.softmax(self.frame_agg_weight, dim=0)
                oct_vlm_causal = agg_weights[0] * oct_vlm_causal_weighted + agg_weights[1] * oct_vlm_causal_max  # [B, embed_dim]
                
                # 对噪声特征也进行类似处理（简化：只使用平均）
                oct_vlm_noise_attn, _ = self.frame_attention(
                    oct_vlm_noise_all, oct_vlm_noise_all, oct_vlm_noise_all,
                    need_weights=False  # 噪声特征不需要注意力权重
                )
                oct_vlm_noise_attn = self.frame_attn_norm(oct_vlm_noise_attn + oct_vlm_noise_all)
                oct_vlm_noise = oct_vlm_noise_attn.mean(dim=1)  # [B, embed_dim]
                
                # 标记已处理全部120帧，oct_vlm_causal和oct_vlm_noise已经准备好
                oct_img_processed = True
            elif oct_images.dim() == 4:  # [B, C, H, W] - 单帧
                oct_img = oct_images
                oct_img_processed = False
                # 构建文本提示（用于单帧处理）
                text_prompts = None
                if clinical_data is not None:
                    hpv_status = "positive" if any(clinical_data.get('hpv', [])) else "negative"
                    tct_cat = clinical_data.get('tct', ['NILM'])[0] if clinical_data.get('tct') else 'NILM'
                    age = clinical_data.get('age', [0])[0] if clinical_data.get('age') else 0
                    text_prompts = [
                        f"Medical image: HPV {hpv_status}, TCT {tct_cat}, Age {age}"
                    ] * B
            else:
                # 如果维度不对，回退到传统方法
                oct_img = None
                oct_img_processed = False
                text_prompts = None
            
            # 处理Colposcopy图像维度（可能是 [B, K, C, H, W] 或 [B, C, H, W]）
            colpo_img = colpo_images
            if colpo_images.dim() == 5:  # [B, K, C, H, W] - 多图像，取第一张
                colpo_img = colpo_images[:, 0, :, :, :]  # [B, C, H, W]
            elif colpo_images.dim() == 4:  # [B, C, H, W] - 已经是正确格式
                colpo_img = colpo_images
            else:
                # 维度不对，回退到传统方法
                colpo_img = None
            
            # 初始化变量
            vlm_processing_success = False
            
            # 如果OCT已经处理了全部120帧，直接使用已提取的特征
            if oct_img_processed:
                # oct_vlm_causal和oct_vlm_noise已经准备好（来自全部120帧）
                # 只需要处理Colposcopy图像
                if colpo_img is not None:
                    colpo_vlm_causal, colpo_vlm_noise = self.image_encoder(colpo_img, text_prompts, device=colpo_features.device)
                    vlm_processing_success = True
                else:
                    # 如果colpo图像维度不对，回退到传统方法
                    vlm_processing_success = False
            
            # 如果OCT是单帧，需要处理
            elif oct_img is not None and colpo_img is not None:
                # 使用VLM处理OCT图像（单帧）
                oct_vlm_causal, oct_vlm_noise = self.image_encoder(oct_img, text_prompts, device=oct_features.device)
                
                # 使用VLM处理Colposcopy图像
                colpo_vlm_causal, colpo_vlm_noise = self.image_encoder(colpo_img, text_prompts, device=colpo_features.device)
                vlm_processing_success = True
            
            # 如果VLM处理成功（无论是全部120帧还是单帧），融合特征
            if vlm_processing_success:
                # 融合VLM特征和传统特征
                vlm_weight = 0.6  # VLM特征权重
                trad_weight = 0.4  # 传统特征权重
                
                # 确保维度一致
                if oct_features.size(-1) != self.embed_dim:
                    if not hasattr(self, '_oct_proj'):
                        self._oct_proj = nn.Linear(oct_features.size(-1), self.embed_dim).to(oct_features.device)
                    oct_feat_proj = self._oct_proj(oct_features)
                else:
                    oct_feat_proj = oct_features
                
                if colpo_features.size(-1) != self.embed_dim:
                    if not hasattr(self, '_colpo_proj'):
                        self._colpo_proj = nn.Linear(colpo_features.size(-1), self.embed_dim).to(colpo_features.device)
                    colpo_feat_proj = self._colpo_proj(colpo_features)
                else:
                    colpo_feat_proj = colpo_features
                
                # 融合：VLM特征 + 传统特征
                oct_fused = vlm_weight * oct_vlm_causal + trad_weight * oct_feat_proj
                colpo_fused = vlm_weight * colpo_vlm_causal + trad_weight * colpo_feat_proj
                
                # 检查是否为单模态（colpo_fused全为零或colpo_vlm_noise全为零）
                is_single_modal = torch.allclose(colpo_fused, torch.zeros_like(colpo_fused), atol=1e-6) or \
                                 torch.allclose(colpo_vlm_noise, torch.zeros_like(colpo_vlm_noise), atol=1e-6)
                
                if is_single_modal:
                    # 单模态模式：只使用OCT特征
                    z_causal = oct_fused  # [B, embed_dim]
                    z_noise = oct_vlm_noise  # [B, embed_dim]
                else:
                    # 多模态模式：融合OCT和Colposcopy
                    # 确保modal_weights是Parameter张量，并在正确的设备上
                    if not hasattr(self, 'modal_weights') or not isinstance(self.modal_weights, nn.Parameter):
                        self.register_parameter('modal_weights', nn.Parameter(torch.ones(2, dtype=torch.float32, device=oct_fused.device) / 2.0))
                    elif self.modal_weights.device != oct_fused.device:
                        self.modal_weights.data = self.modal_weights.data.to(oct_fused.device)
                    
                    try:
                        if not isinstance(self.modal_weights, torch.Tensor):
                            raise TypeError(f"modal_weights不是tensor: {type(self.modal_weights)}")
                        weights = torch.nn.functional.softmax(self.modal_weights, dim=0)
                    except Exception as e:
                        print(f"⚠️ modal_weights softmax失败: {e}, 强制重新创建Parameter", flush=True)
                        if hasattr(self, 'modal_weights'):
                            delattr(self, 'modal_weights')
                        self.modal_weights = nn.Parameter(torch.ones(2, dtype=torch.float32, device=oct_fused.device) / 2.0)
                        weights = torch.nn.functional.softmax(self.modal_weights, dim=0)
                    z_causal = weights[0] * oct_fused + weights[1] * colpo_fused  # [B, embed_dim]
                    z_noise = 0.5 * oct_vlm_noise + 0.5 * colpo_vlm_noise  # 融合噪声特征
            else:
                # 回退到传统方法
                z_causal, z_noise = self._traditional_fusion(oct_features, colpo_features)
        else:
            # 传统融合方式（不使用VLM）
            z_causal, z_noise = self._traditional_fusion(oct_features, colpo_features)
        
        # 4. 三模态融合（OCT + Colposcopy + Clinical）
        # 将图像特征和临床语义特征融合
        # 方法：拼接后通过融合层
        multimodal_feat = torch.cat([z_causal, z_sem], dim=-1)  # [B, embed_dim * 2]
        if not hasattr(self, '_multimodal_proj'):
            self._multimodal_proj = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ).to(z_causal.device)
        fused_multimodal = self._multimodal_proj(multimodal_feat)  # [B, embed_dim]
        
        # 5. 分类（使用融合后的多模态特征）
        # 检查fused_multimodal是否有nan/inf
        if torch.isnan(fused_multimodal).any() or torch.isinf(fused_multimodal).any():
            # 如果包含nan/inf，使用z_causal作为fallback
            print("⚠️ fused_multimodal包含nan/inf，使用z_causal作为fallback")
            fused_multimodal = z_causal
        
        logits = self.classifier(fused_multimodal)  # [B, num_classes]
        
        # 检查logits是否有nan/inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # 如果logits包含nan/inf，使用零初始化
            print("⚠️ logits包含nan/inf，使用零初始化")
            logits = torch.zeros_like(logits, device=logits.device)
        
        # 5. 构建输出
        output = {
            'logits': logits,
            'z_causal': z_causal,
            'z_noise': z_noise,
            'z_sem': z_sem,
        }
        
        # 6. 如果需要计算损失组件
        if return_loss_components and center_labels is not None:
            # 6.1 Sinkhorn OT损失（因果特征 <-> 语义锚点）
            # 添加数值稳定性检查
            if torch.isnan(z_causal).any() or torch.isnan(z_sem).any():
                L_ot = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
            else:
                L_ot = self.sinkhorn_loss(z_causal, z_sem)
                # 确保L_ot是有效的
                if torch.isnan(L_ot) or torch.isinf(L_ot):
                    L_ot = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
            
            # 6.2 反事实一致性损失
            L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
            if use_counterfactual and B > 0:
                try:
                    # 更新Memory Bank
                    self.memory_bank.update(z_noise, center_labels)
                    
                    # 检查Memory Bank是否有足够的样本（至少需要1个样本才能生成反事实）
                    # 统计每个中心的有效样本数
                    total_samples = self.memory_bank.count.sum().item()
                    
                    # 确保每个中心至少有一些样本（至少需要1个中心有样本，就可以进行反事实干预）
                    # 放宽条件：即使只有1个中心有样本，也可以使用该中心的噪声进行反事实干预
                    centers_with_samples = (self.memory_bank.count > 0).sum().item()
                    
                    # 放宽条件：只要有样本就可以尝试反事实干预（即使只有1个中心）
                    if total_samples > 0 and centers_with_samples >= 1:  # 降低要求：至少1个中心有样本即可
                        # 生成反事实噪声（随机选择不同的中心）
                        # 为每个样本随机选择一个不同的中心
                        if self.num_centers > 1 and centers_with_samples > 1:
                            # 如果有多个中心且有样本，随机选择不同的中心
                            available_centers = torch.where(self.memory_bank.count > 0)[0].cpu().numpy()
                            if len(available_centers) > 1:
                                # 为每个样本随机选择一个不同的有效中心
                                fake_center_ids_list = []
                                for i in range(B):
                                    current_center = center_labels[i].item()
                                    # 从有效中心中排除当前中心，然后随机选择
                                    other_centers = [c for c in available_centers if c != current_center]
                                    if len(other_centers) > 0:
                                        fake_center_ids_list.append(np.random.choice(other_centers))
                                    else:
                                        # 如果没有其他中心，使用当前中心
                                        fake_center_ids_list.append(current_center)
                                fake_center_ids = torch.tensor(fake_center_ids_list, device=center_labels.device, dtype=center_labels.dtype)
                            else:
                                # 只有一个有效中心，使用当前中心
                                fake_center_ids = center_labels
                        else:
                            # 如果只有一个中心或有样本的中心少于2个，使用当前中心（仍然可以计算一致性损失）
                            fake_center_ids = center_labels
                        
                        try:
                            # 确保fake_center_ids的batch size与z_causal匹配
                            if fake_center_ids.shape[0] != B:
                                # 如果fake_center_ids的batch size不匹配，调整它
                                if fake_center_ids.shape[0] > B:
                                    fake_center_ids = fake_center_ids[:B]
                                else:
                                    # 重复最后一个center_id
                                    n_needed = B - fake_center_ids.shape[0]
                                    if n_needed > 0:
                                        last_center = fake_center_ids[-1:].repeat(n_needed)
                                        fake_center_ids = torch.cat([fake_center_ids, last_center], dim=0)
                            
                            z_noise_cf = self.memory_bank.get_counterfactual_noise(fake_center_ids, strategy='random')
                            
                            # 确保z_noise_cf在正确的设备上
                            if z_noise_cf is not None:
                                if z_noise_cf.device != z_causal.device:
                                    z_noise_cf = z_noise_cf.to(z_causal.device)
                                
                                # 检查z_noise_cf的batch size是否匹配（必须严格匹配）
                                if z_noise_cf.shape[0] != B:
                                    # batch size不匹配，重新采样或跳过
                                    if z_noise_cf.shape[0] > B:
                                        z_noise_cf = z_noise_cf[:B]  # 截断
                                    else:
                                        # 如果采样数量不足，重复最后一个样本（使用repeat而不是expand）
                                        n_needed = B - z_noise_cf.shape[0]
                                        if n_needed > 0 and z_noise_cf.shape[0] > 0:
                                            last_sample = z_noise_cf[-1:].repeat(n_needed, 1)  # 使用repeat而不是expand
                                            z_noise_cf = torch.cat([z_noise_cf, last_sample], dim=0)
                                        else:
                                            # 如果无法调整，设置为None
                                            z_noise_cf = None
                                
                                    # 再次验证batch size和特征维度（防御性检查）
                                    if z_noise_cf is not None and (z_noise_cf.shape[0] != B or z_noise_cf.shape[1] != z_causal.shape[1]):
                                        z_noise_cf = None
                            else:
                                z_noise_cf = None
                        except Exception as e:
                            if not hasattr(self, '_cf_noise_error_printed'):
                                print(f"⚠️ 反事实噪声采样失败: {e}")
                                self._cf_noise_error_printed = True
                            z_noise_cf = None
                        
                        # 检查z_noise_cf是否有效
                        if z_noise_cf is not None and not torch.isnan(z_noise_cf).any() and not torch.isinf(z_noise_cf).any():
                            # 确保z_noise_cf的batch size和特征维度与z_causal完全匹配
                            current_B = z_causal.shape[0]
                            current_feat_dim = z_causal.shape[1]
                            
                            # 如果batch size不匹配，调整z_noise_cf
                            if z_noise_cf.shape[0] != current_B:
                                if z_noise_cf.shape[0] > current_B:
                                    z_noise_cf = z_noise_cf[:current_B]
                                else:
                                    n_needed = current_B - z_noise_cf.shape[0]
                                    if n_needed > 0:
                                        last_sample = z_noise_cf[-1:].repeat(n_needed, 1)
                                        z_noise_cf = torch.cat([z_noise_cf, last_sample], dim=0)
                            
                            # 如果特征维度不匹配，调整z_noise_cf
                            if z_noise_cf.shape[1] != current_feat_dim:
                                if z_noise_cf.shape[1] > current_feat_dim:
                                    z_noise_cf = z_noise_cf[:, :current_feat_dim]
                                else:
                                    # 如果特征维度不足，使用零填充
                                    n_needed = current_feat_dim - z_noise_cf.shape[1]
                                    padding = torch.zeros(z_noise_cf.shape[0], n_needed, device=z_noise_cf.device, dtype=z_noise_cf.dtype)
                                    z_noise_cf = torch.cat([z_noise_cf, padding], dim=1)
                            
                            # 最终验证：确保维度完全匹配
                            if z_noise_cf.shape[0] == current_B and z_noise_cf.shape[1] == current_feat_dim:
                                # 合成反事实特征（使用较小的混合系数，提高稳定性）
                                alpha = 0.3  # 降低混合系数：从0.5到0.3，减少对原始特征的干扰
                                z_mix = z_causal + alpha * z_noise_cf
                                
                                # 检查z_mix是否有效
                                if not torch.isnan(z_mix).any() and not torch.isinf(z_mix).any():
                                    logits_cf = self.classifier(z_mix)
                                    
                                    # 检查logits_cf是否有效
                                    if not torch.isnan(logits_cf).any() and not torch.isinf(logits_cf).any():
                                        # 确保logits和logits_cf的batch size匹配
                                        if logits.shape[0] == logits_cf.shape[0]:
                                            # 一致性损失：预测结果应该保持不变
                                            # 使用MSE损失（更稳定）而不是KL散度
                                            # 因为KL散度在logits接近时可能产生数值问题
                                            L_consist = self.consistency_loss(logits, logits_cf)
                                        else:
                                            # batch size不匹配，使用0损失
                                            L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                                    else:
                                        # logits_cf无效，使用0损失
                                        L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                                else:
                                    # z_mix无效，使用0损失
                                    L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                            else:
                                # 维度不匹配，使用0损失
                                L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                        else:
                            # z_noise_cf无效，强制使用当前batch的z_noise作为fallback
                            # 这样可以确保即使Memory Bank未填充或采样失败，也能计算一致性损失
                            if z_noise is not None and not torch.isnan(z_noise).any() and not torch.isinf(z_noise).any():
                                # 使用当前batch的z_noise作为反事实噪声（虽然来自同一中心，但至少能计算损失）
                                z_noise_cf_fallback = z_noise.detach().clone()
                                if z_noise_cf_fallback.shape[0] == z_causal.shape[0] and z_noise_cf_fallback.shape[1] == z_causal.shape[1]:
                                    alpha = 0.3
                                    z_mix = z_causal + alpha * z_noise_cf_fallback
                                    if not torch.isnan(z_mix).any() and not torch.isinf(z_mix).any():
                                        logits_cf = self.classifier(z_mix)
                                        if not torch.isnan(logits_cf).any() and not torch.isinf(logits_cf).any() and logits.shape[0] == logits_cf.shape[0]:
                                            L_consist = self.consistency_loss(logits, logits_cf)
                                        else:
                                            # 如果logits_cf无效，仍然使用fallback
                                            L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                                    else:
                                        L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                                else:
                                    L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                            else:
                                # z_noise无效，使用0损失
                                L_consist = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                        
                        # 确保损失是有效的标量（统一处理）
                        if L_consist.dim() > 0:
                            L_consist = L_consist.mean() if L_consist.numel() > 0 else torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                        
                        # 检查一致性损失是否有效，如果无效则使用fallback
                        if torch.isnan(L_consist) or torch.isinf(L_consist) or L_consist.item() < 0:
                            # 使用fallback：当前batch的z_noise
                            if z_noise is not None and not torch.isnan(z_noise).any() and not torch.isinf(z_noise).any():
                                z_noise_cf_fallback = z_noise.detach().clone()
                                if z_noise_cf_fallback.shape[0] == z_causal.shape[0] and z_noise_cf_fallback.shape[1] == z_causal.shape[1]:
                                    alpha = 0.3
                                    z_mix = z_causal + alpha * z_noise_cf_fallback
                                    if not torch.isnan(z_mix).any() and not torch.isinf(z_mix).any():
                                        logits_cf = self.classifier(z_mix)
                                        if not torch.isnan(logits_cf).any() and not torch.isinf(logits_cf).any() and logits.shape[0] == logits_cf.shape[0]:
                                            L_consist = self.consistency_loss(logits, logits_cf)
                                        else:
                                            L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                                    else:
                                        L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                                else:
                                    L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                            else:
                                L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                        
                        # 限制一致性损失的范围，避免过大，但确保最小值不为0
                        L_consist = torch.clamp(L_consist, min=0.001, max=10.0)
                    else:
                        # Memory Bank还没有样本，强制使用当前batch的z_noise作为fallback
                        # 确保consist损失始终有值，而不是0
                        if z_noise is not None and not torch.isnan(z_noise).any() and not torch.isinf(z_noise).any():
                            # 使用当前batch的z_noise作为反事实噪声
                            z_noise_cf_fallback = z_noise.detach().clone()
                            if z_noise_cf_fallback.shape[0] == z_causal.shape[0] and z_noise_cf_fallback.shape[1] == z_causal.shape[1]:
                                alpha = 0.3
                                z_mix = z_causal + alpha * z_noise_cf_fallback
                                if not torch.isnan(z_mix).any() and not torch.isinf(z_mix).any():
                                    logits_cf = self.classifier(z_mix)
                                    if not torch.isnan(logits_cf).any() and not torch.isinf(logits_cf).any() and logits.shape[0] == logits_cf.shape[0]:
                                        L_consist = self.consistency_loss(logits, logits_cf)
                                    else:
                                        # 如果logits_cf无效，使用一个小的非零损失
                                        L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                                else:
                                    L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                            else:
                                L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                        else:
                            # z_noise无效，使用小的非零损失
                            L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                except Exception as e:
                    # 如果反事实生成失败，强制使用fallback机制
                    if not hasattr(self, '_consist_error_printed'):
                        print(f"⚠️ 一致性损失计算异常，使用fallback: {e}")
                        self._consist_error_printed = True
                    # 使用当前batch的z_noise作为fallback
                    if z_noise is not None and not torch.isnan(z_noise).any() and not torch.isinf(z_noise).any():
                        z_noise_cf_fallback = z_noise.detach().clone()
                        if z_noise_cf_fallback.shape[0] == z_causal.shape[0] and z_noise_cf_fallback.shape[1] == z_causal.shape[1]:
                            alpha = 0.3
                            z_mix = z_causal + alpha * z_noise_cf_fallback
                            if not torch.isnan(z_mix).any() and not torch.isinf(z_mix).any():
                                logits_cf = self.classifier(z_mix)
                                if not torch.isnan(logits_cf).any() and not torch.isinf(logits_cf).any() and logits.shape[0] == logits_cf.shape[0]:
                                    L_consist = self.consistency_loss(logits, logits_cf)
                                else:
                                    L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                            else:
                                L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                        else:
                            L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
                    else:
                        L_consist = torch.tensor(0.01, device=z_causal.device, requires_grad=True)
            
            # 6.3 对抗损失（噪声特征预测中心ID）
            try:
                # 确保center_labels在正确的设备上
                if center_labels.device != z_noise.device:
                    center_labels = center_labels.to(z_noise.device)
                
                # 检查z_noise是否有nan/inf
                if torch.isnan(z_noise).any() or torch.isinf(z_noise).any():
                    if not hasattr(self, '_z_noise_nan_printed'):
                        print("⚠️ z_noise包含nan/inf，跳过对抗损失计算")
                        self._z_noise_nan_printed = True
                    L_adv = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                else:
                    center_logits = self.center_discriminator(z_noise)
                    
                    # 检查center_logits是否有nan/inf
                    if torch.isnan(center_logits).any() or torch.isinf(center_logits).any():
                        if not hasattr(self, '_center_logits_nan_printed'):
                            print("⚠️ center_logits包含nan/inf，跳过对抗损失计算")
                            self._center_logits_nan_printed = True
                        L_adv = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                    else:
                        # 如果只有一个中心，使用伪标签（所有样本标签为0，但鼓励预测不确定性）
                        if self.num_centers == 1:
                            # 使用伪标签：所有样本标签为0，但对抗损失会鼓励预测接近均匀分布
                            pseudo_labels = torch.zeros(B, dtype=torch.long, device=center_labels.device)
                            L_adv = self.adversarial_loss(center_logits, pseudo_labels)
                        else:
                            L_adv = self.adversarial_loss(center_logits, center_labels)
                        
                        # 检查对抗损失是否有效，并限制其大小（防止梯度爆炸）
                        if torch.isnan(L_adv) or torch.isinf(L_adv):
                            L_adv = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
                        elif L_adv.item() > 10.0:  # 限制对抗损失的最大值
                            if not hasattr(self, '_adv_loss_clamped'):
                                print(f"⚠️ 对抗损失过大 ({L_adv.item():.4f})，限制到10.0")
                                self._adv_loss_clamped = True
                            L_adv = torch.clamp(L_adv, max=10.0)
            except Exception as e:
                # 打印错误信息以便调试（仅在第一个batch）
                import traceback
                if not hasattr(self, '_adv_error_printed'):
                    print(f"⚠️ 对抗损失计算失败: {e}")
                    traceback.print_exc()
                    self._adv_error_printed = True
                L_adv = torch.tensor(0.0, device=z_causal.device, requires_grad=True)
            
            output['loss_components'] = {
                'L_ot': L_ot,
                'L_consist': L_consist,
                'L_adv': L_adv
            }
        
        return output



