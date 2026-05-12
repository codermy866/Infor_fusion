#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM图像编码器：使用千问VL-Instruct模型作为图像backbone
支持Qwen2-VL-3B-Instruct和Qwen2-VL-7B-Instruct
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from PIL import Image
import numpy as np


class VLMImageEncoder(nn.Module):
    """
    使用VLM（千问VL-Instruct）作为图像编码器的双头网络
    
    输入：原始图像（PIL Image或Tensor）
    输出：z_causal, z_noise (768维嵌入特征)
    """
    
    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct",  # 或 "Qwen/Qwen2-VL-7B-Instruct"
        embed_dim: int = 768,
        freeze_vlm: bool = True,  # 是否冻结VLM参数
        device: Optional[str] = None
    ):
        super().__init__()
        self.vlm_model = vlm_model
        self.embed_dim = embed_dim
        self.freeze_vlm = freeze_vlm
        self.device = device
        
        # 加载VLM模型
        self._load_vlm()
        
        # 获取VLM的隐藏层维度
        try:
            if hasattr(self.vlm.config, 'text_config'):
                vlm_hidden_size = self.vlm.config.text_config.hidden_size
            elif hasattr(self.vlm.config, 'hidden_size'):
                vlm_hidden_size = self.vlm.config.hidden_size
            else:
                vlm_hidden_size = 2048  # Qwen2-VL-3B/7B的默认隐藏层大小
        except:
            vlm_hidden_size = 2048
        
        self.vlm_hidden_size = vlm_hidden_size
        
        # 投影层：将VLM特征投影到embed_dim
        self.vlm_proj = nn.Linear(vlm_hidden_size, embed_dim)
        
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
    
    def _load_vlm(self):
        """加载VLM模型（带重试机制和超时设置）"""
        import os
        import time
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
        
        # 设置HuggingFace超时和重试
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10分钟超时
        os.environ['HF_HUB_DOWNLOAD_RETRY'] = '5'  # 重试5次
        
        print(f"🔄 加载VLM模型: {self.vlm_model}")
        print(f"   超时设置: 600秒, 重试次数: 5")
        
        max_retries = 3
        retry_delay = 5  # 秒
        
        for retry in range(max_retries):
            try:
                # 尝试加载配置并开启output_hidden_states
                try:
                    config = AutoConfig.from_pretrained(
                        self.vlm_model,
                        trust_remote_code=True
                    )
                    config.output_hidden_states = True
                except Exception as e:
                    print(f"⚠️ 无法加载配置，尝试使用默认配置: {e}")
                    # 使用默认配置
                    from transformers import Qwen2VLConfig
                    config = Qwen2VLConfig()
                    config.output_hidden_states = True
                
                # 加载模型（添加错误处理）
                try:
                    print(f"   尝试 {retry + 1}/{max_retries}: 从HuggingFace加载模型...")
                    self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.vlm_model,
                        config=config,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                        resume_download=True  # 支持断点续传
                    )
                    print(f"   ✅ 模型加载成功")
                except Exception as e:
                    print(f"   ⚠️ 使用config加载失败: {e}")
                    print(f"   尝试不使用config重新加载...")
                    # 尝试不使用config
                    self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.vlm_model,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                        resume_download=True
                    )
                    # 手动设置output_hidden_states
                    if hasattr(self.vlm.config, 'output_hidden_states'):
                        self.vlm.config.output_hidden_states = True
                    print(f"   ✅ 模型加载成功（无config）")
                
                # 加载processor
                try:
                    print(f"   加载processor...")
                    self.processor = AutoProcessor.from_pretrained(
                        self.vlm_model,
                        use_fast=False,
                        trust_remote_code=True
                    )
                except Exception as e:
                    print(f"   ⚠️ 使用use_fast=False加载processor失败: {e}")
                    try:
                        self.processor = AutoProcessor.from_pretrained(
                            self.vlm_model,
                            trust_remote_code=True
                        )
                    except Exception as e2:
                        print(f"   ⚠️ 加载processor失败: {e2}")
                        raise
                
                # 冻结VLM参数
                if self.freeze_vlm:
                    for param in self.vlm.parameters():
                        param.requires_grad = False
                    self.vlm.eval()
                
                # 移动到指定设备（如果提供）
                if self.device is not None:
                    print(f"   移动模型到设备: {self.device}")
                    self.vlm = self.vlm.to(self.device)
                
                self.vlm_device = self.device if self.device is not None else None  # 将在forward时确定
                
                print(f"✅ VLM模型加载完成: {self.vlm_model}")
                if self.freeze_vlm:
                    print("   VLM参数已冻结（仅作为特征提取器）")
                
                # 验证模型是否可用
                print(f"   🔍 验证模型可用性...")
                try:
                    # 简单测试：检查模型是否有forward方法
                    assert hasattr(self.vlm, 'forward'), "模型缺少forward方法"
                    assert hasattr(self.processor, 'apply_chat_template'), "processor缺少apply_chat_template方法"
                    print(f"   ✅ 模型验证通过")
                except Exception as e:
                    print(f"   ⚠️ 模型验证失败: {e}")
                    raise
                
                return  # 成功加载，退出重试循环
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ 尝试 {retry + 1}/{max_retries} 失败: {error_msg[:200]}")
                
                if retry < max_retries - 1:
                    print(f"   ⏳ 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    print(f"❌ 所有重试均失败，无法加载VLM模型 {self.vlm_model}")
                    raise RuntimeError(f"VLM模型加载失败（已重试{max_retries}次）: {error_msg}")
    
    def _extract_vlm_features(
        self,
        images: List[Image.Image],
        text_prompts: Optional[List[str]] = None,
        device: str = "cuda:0"
    ) -> torch.Tensor:
        """
        使用VLM提取图像特征
        
        Args:
            images: List[PIL.Image] - 图像列表
            text_prompts: Optional[List[str]] - 可选的文本提示（用于图像+文本理解）
            device: 设备
        
        Returns:
            features: [B, vlm_hidden_size] - VLM特征
        """
        # 确保VLM在正确的设备上
        if self.vlm_device != device:
            self.vlm = self.vlm.to(device)
            self.vlm_device = device
        
        # 如果没有提供文本提示，使用默认提示
        if text_prompts is None:
            text_prompts = ["请描述这张医学图像的关键特征"] * len(images)
        
        # 构建messages格式
        messages_list = []
        for img, text in zip(images, text_prompts):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text}
                ]
            }]
            messages_list.append(messages)
        
        # 处理每个样本
        all_features = []
        
        with torch.no_grad() if self.freeze_vlm else torch.enable_grad():
            for messages in messages_list:
                try:
                    # 使用apply_chat_template处理messages
                    processed_text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    # 提取图像
                    image = messages[0]["content"][0]["image"]
                    
                    # 使用processor处理
                    inputs = self.processor(
                        text=[processed_text],
                        images=[image],
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # 移动到设备
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # 前向传播
                    outputs = self.vlm(**inputs, output_hidden_states=True)
                    
                    # 提取特征（使用最后一层的[CLS] token或平均池化）
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        # 使用最后一层的隐藏状态
                        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
                        # 取第一个token（通常是[CLS]或图像token）
                        features = hidden_states[:, 0, :]  # [B, hidden_size]
                    elif hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                        features = hidden_states[:, 0, :]
                    else:
                        # Fallback: 使用logits的平均值
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                            features = logits.mean(dim=1)  # [B, hidden_size]
                        else:
                            raise ValueError("无法从VLM输出中提取特征")
                    
                    all_features.append(features.squeeze(0))  # [hidden_size]
                
                except Exception as e:
                    print(f"⚠️ VLM特征提取失败: {e}")
                    # 使用零向量作为fallback
                    fallback_feat = torch.zeros(self.vlm_hidden_size, device=device)
                    all_features.append(fallback_feat)
        
        # 堆叠所有特征
        features = torch.stack(all_features, dim=0)  # [B, vlm_hidden_size]
        
        return features
    
    def forward(
        self,
        images: List[Image.Image],
        text_prompts: Optional[List[str]] = None,
        device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: List[PIL.Image] - 图像列表
            text_prompts: Optional[List[str]] - 可选的文本提示
            device: 设备（如果提供，会覆盖初始化时的device）
        
        Returns:
            z_causal: [B, embed_dim] - 因果特征
            z_noise: [B, embed_dim] - 噪声特征
        """
        if device is None:
            device = self.device if self.device is not None else "cuda:0"
        
        # 提取VLM特征
        vlm_features = self._extract_vlm_features(images, text_prompts, device)  # [B, vlm_hidden_size]
        
        # 确保vlm_features是float32类型（VLM可能输出half精度）
        if vlm_features.dtype != torch.float32:
            vlm_features = vlm_features.float()
        
        # 投影到embed_dim
        proj_feat = self.vlm_proj(vlm_features)  # [B, embed_dim]
        
        # 双头输出
        z_causal = self.causal_head(proj_feat)  # [B, embed_dim]
        z_noise = self.noise_head(proj_feat)  # [B, embed_dim]
        
        return z_causal, z_noise


class VLMImageEncoderFromTensor(nn.Module):
    """
    VLM图像编码器的Tensor版本
    输入：Tensor格式的图像 [B, C, H, W]
    内部转换为PIL Image后使用VLM
    """
    
    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        embed_dim: int = 768,
        freeze_vlm: bool = True,
        device: Optional[str] = None
    ):
        super().__init__()
        self.vlm_encoder = VLMImageEncoder(
            vlm_model=vlm_model,
            embed_dim=embed_dim,
            freeze_vlm=freeze_vlm,
            device=device
        )
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """将Tensor转换为PIL Image列表
        
        支持多种输入格式：
        - [B, C, H, W]: 标准batch格式
        - [B, K, C, H, W]: 多图像batch格式（如colposcopy的3张图）
        - [C, H, W]: 单张图像
        """
        images = []
        
        # 处理不同的输入维度
        if tensor.dim() == 3:
            # [C, H, W] - 单张图像，添加batch维度
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 5:
            # [B, K, C, H, W] - 多图像batch，取第一张（或平均）
            # 对于colposcopy，我们取第一张图像
            tensor = tensor[:, 0, :, :, :]  # [B, C, H, W]
        elif tensor.dim() != 4:
            raise ValueError(f"不支持的tensor维度: {tensor.dim()}, shape: {tensor.shape}")
        
        # 现在tensor应该是 [B, C, H, W] 格式
        B = tensor.size(0)
        
        for i in range(B):
            img_tensor = tensor[i]  # [C, H, W]
            
            # 检查维度
            if img_tensor.dim() != 3:
                raise ValueError(f"图像tensor维度错误: {img_tensor.dim()}, shape: {img_tensor.shape}")
            
            C, H, W = img_tensor.shape
            
            # 如果通道数不是3，尝试处理
            if C != 3:
                if C == 1:
                    # 灰度图转RGB
                    img_tensor = img_tensor.repeat(3, 1, 1)
                elif C > 3:
                    # 取前3个通道
                    img_tensor = img_tensor[:3, :, :]
                else:
                    raise ValueError(f"不支持的通道数: {C}")
            
            # 反归一化（假设是ImageNet归一化，如果值已经在[0,1]范围则跳过）
            if img_tensor.max() <= 1.0 and img_tensor.min() >= 0.0:
                # 值已经在[0,1]范围，直接使用
                img_tensor = torch.clamp(img_tensor, 0, 1)
            else:
                # 假设是ImageNet归一化，需要反归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 转换为numpy并转置
            try:
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                img_np = (img_np * 255).astype(np.uint8)
                
                # 转换为PIL Image
                img = Image.fromarray(img_np)
                images.append(img)
            except Exception as e:
                print(f"⚠️ 转换图像失败 (shape: {img_tensor.shape}): {e}")
                # 创建占位符图像
                placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
                images.append(placeholder)
        
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
        device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: [B, C, H, W] - Tensor格式的图像
            text_prompts: Optional[List[str]] - 可选的文本提示
            device: 设备
        
        Returns:
            z_causal: [B, embed_dim] - 因果特征
            z_noise: [B, embed_dim] - 噪声特征
        """
        # 转换为PIL Image
        pil_images = self._tensor_to_pil(images)
        
        # 使用VLM编码器
        return self.vlm_encoder(pil_images, text_prompts, device)

