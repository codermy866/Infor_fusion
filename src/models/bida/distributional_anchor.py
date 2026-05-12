#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-Invariant Distributional Anchoring Module
核心：将VLM特征转化为生物流形上的分布 (μ_bio, σ_bio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


def identify_center(oct_id: str) -> int:
    """
    从OCT ID识别中心ID
    
    Returns:
        center_id: 0-4 (Enshi, Xiangyang, Shiyan, Jingzhou, Wuda)
    """
    oct_str = str(oct_id)
    if 'M22105' in oct_str:
        return 0  # Enshi
    elif 'M22102' in oct_str:
        return 1  # Xiangyang
    elif 'M22104' in oct_str or 'M22101' in oct_str:
        return 2  # Shiyan
    elif 'M0008' in oct_str:
        return 3  # Jingzhou
    elif 'M20203' in oct_str or 'M20105' in oct_str:
        return 4  # Wuda
    return -1  # Unknown


class DistributionalAnchor(nn.Module):
    """
    Bio-Invariant Distributional Anchor
    将临床数据转换为生物流形上的分布
    
    设计说明：
    - 原始设计：临床数据 → 文本 → VLM → 语义特征 → 分布参数
    - 实际使用：对于结构化数据（age, HPV, TCT），直接MLP编码更合适
    - VLM作为可选功能，主要用于自然语言文本处理
    
    Input: Clinical Data (Age, HPV, TCT) - 结构化数据或文本
    Output: μ_bio, σ_bio (Gaussian distribution parameters)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        freeze_vlm: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vlm_model = vlm_model
        self.freeze_vlm = freeze_vlm
        
        # VLM的正确用法：处理图像+文本的联合理解
        # 输入：OCT图像 + 临床文本描述
        # 输出：融合了图像和文本的语义特征
        self.vlm = None  # 先初始化为None
        
        # 启用VLM用于图像+文本联合理解
        use_vlm = True  # 设置为True以启用VLM（处理图像+文本）
        
        # 加载VLM（Frozen）- 仅在use_vlm=True时加载
        if use_vlm:
            try:
                # 尝试新版本Qwen2-VL
                try:
                    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
                    # 加载配置并开启output_hidden_states
                    config = AutoConfig.from_pretrained(vlm_model)
                    config.output_hidden_states = True  # 关键修改：强制开启hidden_states输出
                    # 不使用device_map="auto"，先加载到CPU，然后在forward时移动到正确设备
                    self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                        vlm_model,
                        config=config,  # 使用修改后的config
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    self.vlm_device = None  # 将在forward时确定并移动
                    try:
                        self.processor = AutoProcessor.from_pretrained(vlm_model, use_fast=False)
                    except:
                        self.processor = AutoProcessor.from_pretrained(vlm_model)
                    print(f"✅ 成功加载Qwen2-VL模型: {vlm_model}")
                except ImportError:
                    # 尝试旧版本Qwen-VL
                    try:
                        from transformers import QwenVLForConditionalGeneration, AutoProcessor, AutoConfig
                        # 使用旧版本模型名称
                        old_model_name = "Qwen/Qwen-VL" if "Qwen2" in vlm_model else vlm_model
                        # 加载配置并开启output_hidden_states
                        config = AutoConfig.from_pretrained(old_model_name)
                        config.output_hidden_states = True  # 关键修改：强制开启hidden_states输出
                        self.vlm = QwenVLForConditionalGeneration.from_pretrained(
                            old_model_name,
                            config=config,  # 使用修改后的config
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                        self.vlm_device = None  # 将在forward时确定
                        try:
                            self.processor = AutoProcessor.from_pretrained(old_model_name, use_fast=False)
                        except:
                            self.processor = AutoProcessor.from_pretrained(old_model_name)
                        print(f"✅ 成功加载Qwen-VL模型 (旧版本): {old_model_name}")
                    except Exception as e2:
                        raise e2
                
                if freeze_vlm and self.vlm is not None:
                    for param in self.vlm.parameters():
                        param.requires_grad = False
                    self.vlm.eval()
                    # 初始化设备为None，将在forward时确定
                    if not hasattr(self, 'vlm_device'):
                        self.vlm_device = None
            except Exception as e:
                print(f"⚠️ Warning: Cannot load VLM model {vlm_model}: {e}")
                print("⚠️ Falling back to MLP-based clinical encoder")
                self.vlm = None
        else:
            print("ℹ️ 使用MLP编码器处理结构化临床数据（更简单高效）")
        
        # 获取VLM的隐藏层维度
        if self.vlm is not None:
            try:
                vlm_hidden_size = self.vlm.config.text_config.hidden_size
            except:
                vlm_hidden_size = 2048  # 默认值
        else:
            vlm_hidden_size = 256  # MLP fallback
        
        # 分布参数生成网络
        # 输入：VLM特征 -> 输出：μ_bio, σ_bio
        self.distribution_head = nn.Sequential(
            nn.Linear(vlm_hidden_size, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim * 2)  # [μ, σ]
        )
        
        # Fallback: 如果VLM不可用，使用MLP编码临床特征
        # 注意：对于结构化数据（age, HPV, TCT），MLP编码更合适
        self.fallback_encoder = nn.Sequential(
            nn.Linear(7, 256),  # HPV, TCT, Age等7维特征
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, vlm_hidden_size)
        )
    
    def clinical_to_text(self, clinical_data: Dict) -> list:
        """
        将临床数据转换为医学文本描述
        
        Args:
            clinical_data: dict with keys ['hpv', 'tct', 'age', ...]
        
        Returns:
            text_prompts: list of text strings
        """
        text_prompts = []
        
        for i in range(len(clinical_data.get('hpv', []))):
            hpv_val = clinical_data.get('hpv', [0])[i] if isinstance(clinical_data.get('hpv', []), (list, np.ndarray)) else clinical_data.get('hpv', 0)
            tct_val = clinical_data.get('tct', ['NILM'])[i] if isinstance(clinical_data.get('tct', []), (list, np.ndarray)) else clinical_data.get('tct', 'NILM')
            age_val = clinical_data.get('age', [45])[i] if isinstance(clinical_data.get('age', []), (list, np.ndarray)) else clinical_data.get('age', 45)
            
            # 构建医学文本
            hpv_status = "阳性" if int(hpv_val) == 1 else "阴性"
            text = f"患者年龄{int(age_val)}岁，HPV检测结果{hpv_status}，TCT检查结果为{tct_val}。"
            text_prompts.append(text)
        
        return text_prompts
    
    def forward(
        self,
        clinical_data: Optional[Dict] = None,
        clinical_features: Optional[torch.Tensor] = None,
        oct_images: Optional[torch.Tensor] = None,  # OCT图像 [B, C, H, W]
        colposcopy_images: Optional[torch.Tensor] = None  # 阴道镜图像 [B, C, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成生物流形分布参数
        
        Args:
            clinical_data: dict with clinical information (for VLM)
            clinical_features: [B, D] tensor (fallback, if VLM unavailable)
        
        Returns:
            μ_bio: [B, embed_dim] mean of bio-invariant distribution
            σ_bio: [B, embed_dim] std of bio-invariant distribution
        """
        # 初始化text_features为None
        text_features = None
        
        # VLM的正确用法：处理图像+文本的联合理解
        # 如果有图像和文本，使用VLM；否则使用MLP fallback
        use_vlm = (self.vlm is not None and 
                  clinical_data is not None and 
                  oct_images is not None)  # 有图像时才使用VLM
        
        if use_vlm:
            # VLM的正确用法：处理图像+文本的联合理解
            # 输入：OCT图像 + 临床文本描述
            text_prompts = self.clinical_to_text(clinical_data)
            
            try:
                # 确定目标设备（从clinical_features获取，因为它是从训练脚本传入的）
                device = clinical_features.device if clinical_features is not None else torch.device('cuda:1')
                
                # 确保VLM模型在正确的设备上
                if not hasattr(self, 'vlm_device') or self.vlm_device is None:
                    self.vlm_device = None
                
                current_vlm_device = next(self.vlm.parameters()).device if self.vlm is not None else None
                if current_vlm_device != device:
                    print(f"🔄 移动VLM模型从 {current_vlm_device} 到 {device}")
                    self.vlm = self.vlm.to(device)
                    self.vlm_device = device
                
                B = oct_images.size(0)
                
                # 准备图像：将tensor转换为PIL Image或numpy格式
                # Qwen2-VL的processor需要PIL Image或numpy数组
                from PIL import Image
                import torchvision.transforms as transforms
                
                # 将tensor转换为PIL Image
                to_pil = transforms.ToPILImage()
                image_list = []
                for i in range(B):
                    # 处理图像栈：oct_images可能是 [B, T, C, H, W] 或 [B, C, H, W]
                    img_tensor = oct_images[i]  # [T, C, H, W] 或 [C, H, W]
                    
                    # 如果是图像栈（多帧），取第一帧或平均帧
                    if img_tensor.dim() == 4 and img_tensor.size(0) > 1:
                        # 多帧图像栈 [T, C, H, W]，取第一帧
                        img_tensor = img_tensor[0]  # [C, H, W]
                    elif img_tensor.dim() == 4 and img_tensor.size(0) == 1:
                        # 单帧但有多余维度
                        img_tensor = img_tensor[0]  # [C, H, W]
                    
                    # 确保是 [C, H, W] 格式
                    if img_tensor.dim() != 3:
                        raise ValueError(f"Unexpected image tensor shape: {img_tensor.shape}")
                    
                    # 归一化到[0, 1]范围
                    if img_tensor.max() > 1.0:
                        img_tensor = img_tensor / 255.0
                    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                    
                    # 转换为PIL Image
                    img_pil = to_pil(img_tensor.cpu())
                    image_list.append(img_pil)
                
                # 使用VLM处理图像+文本
                # Qwen2-VL的正确使用方式：使用apply_chat_template + processor
                try:
                    # 方法1: 使用apply_chat_template（Qwen2-VL推荐的方式）
                    # 构建messages格式
                    processed_texts = []
                    processed_images = []
                    
                    for text, img in zip(text_prompts, image_list):
                        # 为每个样本构建messages
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": text}
                            ]
                        }]
                        
                        # 应用chat template
                        processed_text = self.processor.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        processed_texts.append(processed_text)
                        processed_images.append(img)
                    
                    # 使用processor处理text和images
                    inputs = self.processor(
                        text=processed_texts,  # List[str] - 经过chat template处理的文本
                        images=processed_images,  # List[PIL.Image]
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # 确保所有tensor都在正确的设备上
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
                    # 调试：检查inputs的keys和形状
                    if 'pixel_values' in inputs:
                        print(f"✅ Processor成功处理图像: pixel_values shape = {inputs['pixel_values'].shape}")
                    if 'input_ids' in inputs:
                        print(f"✅ Processor成功处理文本: input_ids shape = {inputs['input_ids'].shape}")
                    
                except Exception as proc_e:
                    # 如果apply_chat_template失败，尝试直接传入text和images（简单格式）
                    print(f"⚠️ apply_chat_template failed: {proc_e}, trying direct format")
                    try:
                        # 直接使用文本和图像（不使用chat template）
                        inputs = self.processor(
                            text=text_prompts,  # List[str]
                            images=image_list,  # List[PIL.Image]
                            return_tensors="pt",
                            padding=True
                        )
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    except Exception as proc_e2:
                        print(f"⚠️ Direct format also failed: {proc_e2}, using fallback")
                        import traceback
                        traceback.print_exc()
                        raise proc_e2
                
                with torch.no_grad() if self.freeze_vlm else torch.enable_grad():
                    # 调用VLM模型，强制开启output_hidden_states
                    outputs = self.vlm(**inputs, output_hidden_states=True)
                    
                    # 提取融合了图像和文本的语义特征
                    # Qwen2-VL的输出格式是BaseModelOutputWithPast
                    if hasattr(outputs, 'last_hidden_state'):
                        # last_hidden_state: [B, seq_len, hidden_size]
                        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_size]
                        
                        # 检查维度
                        if hidden_states.dim() != 3:
                            raise ValueError(f"hidden_states维度错误: {hidden_states.shape}, 期望 [B, seq_len, hidden_size]")
                        
                        # 确保batch维度正确
                        if hidden_states.size(0) != B:
                            print(f"⚠️ Batch size不匹配: hidden_states.size(0)={hidden_states.size(0)}, B={B}")
                            if hidden_states.size(0) == 1 and B > 1:
                                # 重复到正确的batch size
                                hidden_states = hidden_states.repeat(B, 1, 1)
                            elif hidden_states.size(0) > B:
                                # 取前B个
                                hidden_states = hidden_states[:B]
                        
                        # 取所有token的平均（推荐，包含更多信息）
                        text_features = hidden_states.mean(dim=1)  # [B, hidden_size]
                        
                        print(f"🔍 VLM hidden_states: shape = {hidden_states.shape}, text_features: shape = {text_features.shape}")
                            
                    elif hasattr(outputs, 'logits'):
                        # logits: [B, seq_len, vocab_size]
                        # 注意：logits不适合直接提取特征，应该使用last_hidden_state
                        # 如果只有logits，尝试从hidden_states提取（如果存在）
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                            # 使用最后一层的hidden_states
                            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
                            text_features = hidden_states.mean(dim=1)  # [B, hidden_size]
                        else:
                            # 如果只有logits，使用logits（不推荐，但作为fallback）
                            print("⚠️ 警告：使用logits提取特征，可能不准确")
                            logits = outputs.logits  # [B, seq_len, vocab_size]
                            # 对vocab维度取平均，得到 [B, seq_len]
                            text_features = logits.mean(dim=-1)  # [B, seq_len]
                            # 需要投影到hidden_size
                            seq_len = text_features.size(-1)
                            if not hasattr(self, 'logits_proj') or self.logits_proj is None:
                                # 获取VLM的hidden_size
                                try:
                                    vlm_hidden_size = self.vlm.config.text_config.hidden_size
                                except:
                                    vlm_hidden_size = 2048  # 默认值
                                self.logits_proj = nn.Linear(seq_len, vlm_hidden_size).to(device)
                            # 先投影，再取平均（如果seq_len > 1）
                            if seq_len > 1:
                                # 对每个token投影，然后取平均
                                text_features = text_features.unsqueeze(-1)  # [B, seq_len, 1]
                                text_features = self.logits_proj(text_features.squeeze(-1))  # [B, vlm_hidden_size]
                            else:
                                text_features = self.logits_proj(text_features)  # [B, vlm_hidden_size]
                    else:
                        # 尝试从模型输出中提取
                        if isinstance(outputs, tuple) and len(outputs) > 0:
                            first_output = outputs[0]
                            if hasattr(first_output, 'shape'):
                                if first_output.dim() == 3:
                                    # [B, seq_len, hidden_size]
                                    text_features = first_output.mean(dim=1)  # [B, hidden_size]
                                elif first_output.dim() == 2:
                                    text_features = first_output  # [B, hidden_size]
                                else:
                                    raise ValueError(f"Unexpected output shape: {first_output.shape}")
                            else:
                                raise ValueError(f"Cannot extract features from tuple output")
                        else:
                            raise ValueError(f"无法从VLM输出中提取特征，outputs类型: {type(outputs)}")
                    
                    # 确保特征维度正确
                    original_shape = text_features.shape
                    
                    # 处理维度问题
                    if text_features.dim() > 2:
                        # [B, seq_len, hidden_size] -> [B, seq_len * hidden_size]
                        text_features = text_features.view(text_features.size(0), -1)
                    elif text_features.dim() == 1:
                        # [hidden_size] -> [1, hidden_size]
                        text_features = text_features.unsqueeze(0)
                    
                    # 确保batch维度正确（B应该等于输入的batch size）
                    if text_features.size(0) != B:
                        if text_features.size(0) == 1 and B > 1:
                            # 如果只有1个样本但batch size > 1，需要重复
                            text_features = text_features.repeat(B, 1)
                        elif text_features.size(0) > B:
                            # 如果样本数 > batch size，取前B个
                            text_features = text_features[:B]
                    
                    # 获取VLM的hidden_size（用于distribution_head）
                    try:
                        vlm_hidden_size = self.vlm.config.text_config.hidden_size
                    except:
                        vlm_hidden_size = 2048  # 默认值
                    
                    # 如果最后一个维度不匹配，需要投影到vlm_hidden_size
                    if text_features.size(-1) != vlm_hidden_size:
                        if not hasattr(self, 'vlm_feature_proj'):
                            self.vlm_feature_proj = nn.Linear(text_features.size(-1), vlm_hidden_size).to(device)
                        text_features = self.vlm_feature_proj(text_features)  # [B, vlm_hidden_size]
                    
                    # 最终检查
                    assert text_features.size(0) == B, f"Batch size mismatch: {text_features.size(0)} != {B}"
                    assert text_features.size(-1) == vlm_hidden_size, f"Hidden size mismatch: {text_features.size(-1)} != {vlm_hidden_size}"
                    
                    print(f"✅ VLM特征提取成功: original_shape = {original_shape}, final_shape = {text_features.shape}, expected = [{B}, {vlm_hidden_size}]")
                
                # 将特征移回原始设备
                if text_features.device != clinical_features.device:
                    text_features = text_features.to(clinical_features.device)
                    
                print(f"✅ VLM成功处理 {B} 个样本的图像+文本联合理解")
            except Exception as e:
                print(f"⚠️ VLM encoding failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
                text_features = None  # 标记VLM失败，使用fallback
        
        # Fallback: 使用MLP编码临床特征
        if text_features is None:
            if clinical_features is None:
                raise ValueError("Either clinical_data (for VLM) or clinical_features (for fallback) must be provided")
            
            # 使用MLP编码
            text_features = self.fallback_encoder(clinical_features)
        
        # 生成分布参数
        # 添加维度检查
        if text_features.dim() != 2:
            raise ValueError(f"text_features维度错误: {text_features.shape}, 期望 [B, hidden_size]")
        if text_features.size(0) == 0:
            raise ValueError(f"text_features batch size为0: {text_features.shape}")
        
        # 检查distribution_head的输入维度
        # 确保text_features是正确的形状 [B, hidden_size]
        if text_features.dim() != 2:
            raise ValueError(f"text_features维度错误: {text_features.shape}, 期望 [B, hidden_size]")
        
        # 打印调试信息
        print(f"🔍 text_features shape before distribution_head: {text_features.shape}")
        
        first_layer = list(self.distribution_head.children())[0]
        expected_input_dim = first_layer.in_features
        if text_features.size(-1) != expected_input_dim:
            print(f"⚠️ 维度不匹配: text_features.size(-1)={text_features.size(-1)}, expected_input_dim={expected_input_dim}")
            # 如果维度不匹配，添加投影层（自动修复）
            if not hasattr(self, 'feature_proj') or self.feature_proj is None:
                self.feature_proj = nn.Linear(text_features.size(-1), expected_input_dim).to(text_features.device)
                print(f"✅ 创建投影层: {text_features.size(-1)} -> {expected_input_dim}")
            text_features = self.feature_proj(text_features)
            print(f"✅ 已投影: text_features shape = {text_features.shape}")
        
        dist_params = self.distribution_head(text_features)  # [B, embed_dim * 2]
        
        # 分离μ和σ
        μ_bio = dist_params[:, :self.embed_dim]  # [B, embed_dim]
        σ_bio = F.softplus(dist_params[:, self.embed_dim:]) + 1e-6  # [B, embed_dim], 确保为正
        
        return μ_bio, σ_bio

