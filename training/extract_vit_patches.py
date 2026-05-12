#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复漏洞1：从ViT提取Patch特征（丢弃[CLS] token）
"""

import torch
import torch.nn as nn
from typing import Optional

# 全局ViT模型实例（避免重复创建）
_vit_model = None
_vit_device = None


def extract_patch_features_with_vit(
    images: torch.Tensor, 
    device: torch.device, 
    log_func=None,
    batch_size: int = 16,  # 🔧 添加batch_size参数以支持分批处理
    pretrained: bool = False  # 离线训练默认关闭HF下载
) -> torch.Tensor:
    """
    修复漏洞1：从ViT提取Patch特征（丢弃[CLS] token）
    🔧 优化：增加分批处理以避免显存溢出
    
    Args:
        images: [B, C, H, W] 或 [B, F, C, H, W] 图像tensor
        device: GPU设备
        log_func: 日志输出函数（可选）
        batch_size: 分批处理的批次大小（默认16）
    
    Returns:
        patch_features: [B, N, D] 或 [B, F, N, D] Patch特征（已丢弃[CLS]）
        N=196 for ViT-Base (14x14 patches), D=768
    """
    global _vit_model, _vit_device
    
    try:
        import timm
    except ImportError:
        raise ImportError("timm未安装，请先安装: pip install timm")
    
    # 只在第一次调用时创建模型
    if _vit_model is None or _vit_device != device:
        if log_func:
            log_func(f"📥 正在加载ViT模型（修复：提取Patch特征，丢弃[CLS]）...")
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # 创建ViT模型（不使用global_pool，保留所有tokens）
            _vit_model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0,  # 去掉分类头
                global_pool='',  # 不进行全局池化，保留所有tokens
            )
            
            _vit_model = _vit_model.to(device)
            _vit_model.eval()
            _vit_device = device
            
            if log_func:
                log_func(f"✅ ViT模型已加载到GPU {device}")
            
            # 预热模型
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224, device=device)
                _ = _vit_model(dummy_input)
    
    # 确保输入在GPU上
    if not images.is_cuda or images.device != device:
        images = images.to(device, non_blocking=True)
    
    # 特征提取（完全在GPU上）- 🔧 增加分批处理以避免显存溢出
    with torch.no_grad():
        if len(images.shape) == 5:  # [B, F, C, H, W]
            B, num_frames, C, H, W = images.shape
            images_flat = images.view(B * num_frames, C, H, W)
            
            # 🔧 分批处理以避免显存溢出
            total_samples = B * num_frames
            all_patch_tokens = []
            
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_images = images_flat[i:end_idx]
            
                # 获取所有tokens [batch, N+1, D]
                all_tokens = _vit_model.forward_features(batch_images)
            
            # 修复漏洞1：丢弃[CLS] token (index 0)，只保留Patches
                patch_tokens = all_tokens[:, 1:, :]  # [batch, N, D] N=196
                
                all_patch_tokens.append(patch_tokens)
            
            # 拼接所有批次 [B*F, N, D]
            patch_tokens = torch.cat(all_patch_tokens, dim=0)
            
            # Reshape回 [B, F, N, D]
            patch_tokens = patch_tokens.view(B, num_frames, -1, patch_tokens.shape[-1])
            
            return patch_tokens
            
        elif len(images.shape) == 4:  # [B, C, H, W]
            # 🔧 分批处理以避免显存溢出
            B, C, H, W = images.shape
            all_patch_tokens = []
            
            for i in range(0, B, batch_size):
                end_idx = min(i + batch_size, B)
                batch_images = images[i:end_idx]
                
                # 获取所有tokens [batch, N+1, D]
                all_tokens = _vit_model.forward_features(batch_images)
            
            # 修复漏洞1：丢弃[CLS] token (index 0)，只保留Patches
                patch_tokens = all_tokens[:, 1:, :]  # [batch, N, D] N=196
                
                all_patch_tokens.append(patch_tokens)
            
            # 拼接所有批次 [B, N, D]
            patch_tokens = torch.cat(all_patch_tokens, dim=0)
            
            return patch_tokens
        else:
            raise ValueError(f"不支持的图像形状: {images.shape}")
    
    return patch_tokens
