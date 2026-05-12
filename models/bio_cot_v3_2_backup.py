#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 (Enhanced Logic Loop Version)
融合3.1和4.0的优势：
1. 保留3.1的所有优点（显式对齐、自适应模态融合、增强Visual Notes）
2. 引入4.0的优势（Frozen VLM + Trainable Adapter、动态知识生成）
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List

# 导入基础组件
from src.models.bida.bio_cot_v2 import DualHeadImageEncoder
from src.models.bida.memory_bank import NoiseMemoryBank, CenterDiscriminator
from src.models.bida.losses import SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss

# 导入增强后的 Visual Notes（3.1的优势）
from .visual_notes import VisualNotesModule

# 🔥 导入4.0的VLMAugmentedRetriever（4.0的优势）
try:
    from ..knowledge_base.enhanced_knowledge_retriever import VLMAugmentedRetriever
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    knowledge_base_path = Path(__file__).parent.parent / "knowledge_base"
    sys.path.insert(0, str(knowledge_base_path.parent))
    from knowledge_base.enhanced_knowledge_retriever import VLMAugmentedRetriever


class AdaptiveModalityGating(nn.Module):
    """
    [Innovation 1] 自适应模态互补门控 (AMCG) - 3.1的优势
    模仿医生决策：根据两个模态的特征质量和置信度，动态分配权重。
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2)
        )
        self.temperature = 1.0

    def forward(self, f_oct: torch.Tensor, f_colpo: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        concat_feat = torch.cat([f_oct, f_colpo], dim=-1)
        logits = self.score_net(concat_feat)
        weights = F.softmax(logits / self.temperature, dim=-1)
        w_oct = weights[:, 0:1]
        w_colpo = weights[:, 1:2]
        f_fused = w_oct * f_oct + w_colpo * f_colpo
        return f_fused, (w_oct, w_colpo)


class BioCOT_v3_2(nn.Module):
    """
    Bio-COT 3.2 Enhanced Version
    融合3.1和4.0的优势：保留3.1的所有优点 + 引入4.0的计算效率和知识复杂度
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 768,
        vlm_json_path: str = None,  # ⚠️ 新增：VLM缓存路径（必需，从4.0引入）
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        use_visual_notes: bool = True,
        use_ot: bool = True,
        use_dual: bool = True,
        use_cross_attn: bool = True,
        use_adaptive_gating: bool = True,  # ⚠️ 新增：控制自适应模态门控
        warmup_epochs: int = 10,
        hidden_dim: int = 768
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_centers = num_centers
        self.use_visual_notes = use_visual_notes
        self.use_ot = use_ot
        self.use_dual = use_dual
        self.use_adaptive_gating = use_adaptive_gating
        self.current_epoch = 0
        
        # ============================================================
        # 🔥 关键改动1：替换为VLMAugmentedRetriever（4.0的优势）
        # ============================================================
        # 从3.1的"预计算嵌入 + 可训练投影"改为"Frozen VLM + Trainable Adapter"
        # 优势：
        # 1. 计算效率：冻结Text Encoder，仅训练Adapter
        # 2. 知识复杂度：动态生成知识，基于VLM描述
        # ============================================================
        if vlm_json_path is None:
            raise ValueError("vlm_json_path 是必需的！请提供VLM缓存JSON文件路径。")
        
        self.knowledge_retriever = VLMAugmentedRetriever(
            vlm_json_path=vlm_json_path,
            visual_dim=embed_dim,
            text_model_name=text_model_name
        )
        # 注意：不再需要note_projector，因为VLMAugmentedRetriever已经包含了Adapter

        # ============================================================
        # [PROFESSIONAL FIX] 深度对齐投影头 + 共享语义空间（3.1的优势）
        # ============================================================
        # 保留3.1的显式对齐机制，确保对齐精度
        # ============================================================
        self.align_dim = 256
        
        # 方案1: 深度投影头（带残差连接，防止梯度消失）
        self.align_proj_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim)
        )
        self.align_proj_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim)
        )
        
        # 方案2: 共享的语义空间投影
        self.shared_align_proj = nn.Sequential(
            nn.Linear(self.align_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim),
            nn.GELU()
        )
        
        # 方案3: 温度系数优化
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.1))
        
        # 2. 增强型视觉笔记模块 (Visual Notes - Cross-Attention)（3.1的优势）
        if use_visual_notes:
            self.visual_notes_module = VisualNotesModule(
                img_dim=input_dim,
                text_dim=embed_dim,
                hidden_dim=hidden_dim,
                warmup_epochs=warmup_epochs
            )
        
        # 3. 自适应模态门控 (Adaptive Fusion)（3.1的优势）
        if use_adaptive_gating:
            self.adaptive_fusion = AdaptiveModalityGating(dim=embed_dim)
        else:
            self.adaptive_fusion = None
        
        # 4. 双头因果编码器 (Dual-Head Causal Encoder)
        if use_dual:
            self.dual_head = DualHeadImageEncoder(input_dim=input_dim, embed_dim=embed_dim)
        else:
            self.dual_head = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
            
        # 5. 跨模态融合 (Final Decision)（3.1的优势）
        self.final_fusion = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True) if use_cross_attn else None
        
        # 6. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 7. Loss Modules
        if use_ot:
            self.ot_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        
        if use_dual:
            self.memory_bank = NoiseMemoryBank(num_centers, embed_dim)
            self.consistency_loss = CounterfactualConsistencyLoss()
            self.adversarial_loss = AdversarialLoss(max(num_centers, 2))
            self.center_discriminator = CenterDiscriminator(embed_dim, max(num_centers, 2))
            
        # [CRITICAL FIX] 统一初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """统一初始化：Xavier for Linear, Standard for LayerNorm"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_epoch(self, epoch: int):
        """设置当前epoch（用于Visual Notes的Warm-up）"""
        self.current_epoch = int(epoch)
        if self.use_visual_notes:
            self.visual_notes_module.set_epoch(epoch)

    def extract_features(self, feat_raw, z_sem, beta):
        """Helper to process features with Visual Notes"""
        if self.use_visual_notes and len(feat_raw.shape) == 3:
            feat_focused, attn_map = self.visual_notes_module(feat_raw, z_sem, beta=beta)
            feat_pooled = feat_focused.mean(dim=1)
            return feat_pooled, attn_map
        else:
            feat_pooled = feat_raw if len(feat_raw.shape) == 2 else feat_raw.mean(dim=1)
            return feat_pooled, None

    def forward(
        self,
        f_oct: torch.Tensor,     # [B, N, D]
        f_colpo: torch.Tensor,   # [B, N, D]
        image_names: List[str],  # ⚠️ 新增：图像文件名列表（用于VLM检索）
        clinical_info: Optional[List[str]] = None,  # ⚠️ 新增：临床信息（可选）
        center_labels: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        current_beta: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Logic Loop (Enhanced with VLM)
        
        Args:
            f_oct: OCT图像特征 [B, N, D]
            f_colpo: Colposcopy图像特征 [B, N, D]
            image_names: 图像文件名列表（必需，用于VLM检索）
            clinical_info: 临床信息列表（可选）
            center_labels: 中心标签（用于对抗损失）
            return_loss_components: 是否返回损失组件
            current_beta: 当前beta值（用于Visual Notes）
        """
        output = {}
        B = f_oct.shape[0]
        device = f_oct.device
        
        # 验证必需参数
        if image_names is None:
            raise ValueError("image_names不能为None！")
        
        # 确保image_names是列表
        if not isinstance(image_names, list):
            image_names = [image_names] if isinstance(image_names, str) else list(image_names)
        
        if len(image_names) != B:
            raise ValueError(
                f"image_names长度({len(image_names)})与batch大小({B})不匹配！\n"
                f"image_names类型: {type(image_names)}, 前3个值: {image_names[:3] if len(image_names) >= 3 else image_names}"
            )
        
        # --- Step 1: 语义锚点生成（使用VLMAugmentedRetriever）---
        # 🔥 关键改动：从预计算嵌入改为动态VLM检索
        z_sem = self.knowledge_retriever(
            image_names=image_names,
            clinical_info=clinical_info,
            device=str(device)
        )  # [B, embed_dim]
        output['z_sem'] = z_sem
        
        # --- Step 2: 视觉笔记引导的特征提取（3.1的优势）---
        f_oct_pooled, attn_oct = self.extract_features(f_oct, z_sem, current_beta)
        f_colpo_pooled, attn_colpo = self.extract_features(f_colpo, z_sem, current_beta)
        
        if attn_oct is not None:
            output['attn_maps'] = [attn_oct, attn_colpo]
            
        # --- Step 3: 自适应模态融合 [INNOVATION]（3.1的优势）---
        if self.adaptive_fusion is not None:
            f_fused, (w_oct, w_colpo) = self.adaptive_fusion(f_oct_pooled, f_colpo_pooled)
            output['fusion_weights'] = {'oct': w_oct, 'colpo': w_colpo}
        else:
            # 禁用时使用简单平均融合
            f_fused = (f_oct_pooled + f_colpo_pooled) / 2.0
            output['fusion_weights'] = {'oct': torch.ones(B, 1, device=device) * 0.5, 
                                        'colpo': torch.ones(B, 1, device=device) * 0.5}
        
        # --- Step 4: 双头因果解耦 ---
        if self.use_dual:
            z_causal, z_noise = self.dual_head(f_fused)
        else:
            z_causal = self.dual_head(f_fused)
            z_noise = None
        output['z_causal'] = z_causal
        output['z_noise'] = z_noise
        
        # --- Step 5: 最终诊断 (语义-因果特征融合)（3.1的优势）---
        if self.final_fusion:
            z_causal_expanded = z_causal.unsqueeze(1)
            z_sem_expanded = z_sem.unsqueeze(1)
            f_final, _ = self.final_fusion(z_causal_expanded, z_sem_expanded, z_sem_expanded)
            f_final = f_final.squeeze(1) + z_causal
        else:
            f_final = z_causal + z_sem
            
        pred = self.classifier(f_final)
        output['pred'] = pred
        output['logits'] = pred
        
        # --- Step 6: Loss Calculation (Logic Loop)（3.1的优势：显式对齐）---
        if return_loss_components:
            loss_dict = {}
            
            # 6.1 OT Loss
            if self.use_ot:
                loss_dict['L_ot'] = self.ot_loss(z_causal, z_sem)
            
            # ============================================================
            # [PROFESSIONAL FIX] 深度对齐模块（3.1的优势：显式对齐）
            # ============================================================
            z_img_embed = self.align_proj_img(z_causal)
            z_txt_embed = self.align_proj_text(z_sem)
            
            z_img_shared = self.shared_align_proj(z_img_embed)
            z_txt_shared = self.shared_align_proj(z_txt_embed)
            
            z_img_norm = F.normalize(z_img_shared, p=2, dim=-1)
            z_txt_norm = F.normalize(z_txt_shared, p=2, dim=-1)
            
            logit_scale = self.logit_scale.exp().clamp(min=0.1, max=50.0)
            
            logits_per_image = logit_scale * torch.matmul(z_img_norm, z_txt_norm.t())
            logits_per_text = logits_per_image.t()
            labels_align = torch.arange(B, device=device)
            
            loss_align_ce = (F.cross_entropy(logits_per_image, labels_align) + 
                            F.cross_entropy(logits_per_text, labels_align)) / 2.0
            
            diagonal_distances = torch.norm(z_img_norm - z_txt_norm, p=2, dim=-1)
            loss_align_l2 = diagonal_distances.mean()
            
            loss_align = loss_align_ce + 0.1 * loss_align_l2
            loss_dict['L_align'] = loss_align
            loss_dict['L_align_ce'] = loss_align_ce
            loss_dict['L_align_l2'] = loss_align_l2
            
            with torch.no_grad():
                pred_i2t = logits_per_image.argmax(dim=1)
                correct = (pred_i2t == labels_align).float().sum()
                recall_val = correct / B
                loss_dict['Recall'] = recall_val
                loss_dict['Recall_Align'] = recall_val
                
                diagonal_sim = (z_img_norm * z_txt_norm).sum(dim=-1).mean()
                loss_dict['align_cosine_sim'] = diagonal_sim

            # ============================================================
            
            # 6.3 Sparse Loss
            if attn_oct is not None:
                 eps = 1e-8
                 attn_norm = attn_oct / (attn_oct.sum(dim=1, keepdim=True) + eps)
                 entropy = -torch.sum(attn_norm * torch.log(attn_norm + eps), dim=1).mean()
                 loss_dict['L_sparse'] = entropy * 0.1
            
            # 6.4 Consistency Loss
            if self.use_dual and center_labels is not None:
                if z_noise is not None:
                    c_logits = self.center_discriminator(z_noise)
                    loss_dict['L_adv'] = self.adversarial_loss(c_logits, center_labels)

                    self.memory_bank.update(z_noise, center_labels)

                    if self.num_centers >= 2:
                        rand = torch.randint(0, max(self.num_centers - 1, 1), (B,), device=device)
                        target_centers = (center_labels + 1 + rand) % self.num_centers
                    else:
                        target_centers = center_labels

                    z_noise_cf = self.memory_bank.get_counterfactual_noise(target_centers, strategy="random")
                    z_causal_cf = z_causal + z_noise_cf

                    if self.final_fusion:
                        z_causal_cf_expanded = z_causal_cf.unsqueeze(1)
                        z_sem_expanded = z_sem.unsqueeze(1)
                        f_final_cf, _ = self.final_fusion(z_causal_cf_expanded, z_sem_expanded, z_sem_expanded)
                        f_final_cf = f_final_cf.squeeze(1) + z_causal_cf
                    else:
                        f_final_cf = z_causal_cf + z_sem

                    pred_cf = self.classifier(f_final_cf)
                    L_consist = self.consistency_loss(pred, pred_cf)
                    loss_dict["L_consist"] = torch.clamp(L_consist, min=1e-6, max=10.0)

            output['loss_components'] = loss_dict
            
        return output

def create_bio_cot_v3_2(config):
    """Factory function to create BioCOT_v3_2 model"""
    return BioCOT_v3_2(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        input_dim=config.input_dim,
        vlm_json_path=config.vlm_json_path,
        text_model_name=config.text_model_name,
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        use_adaptive_gating=getattr(config, 'use_adaptive_gating', True),
        warmup_epochs=config.warmup_epochs,
        hidden_dim=getattr(config, 'hidden_dim', 768)
    )

