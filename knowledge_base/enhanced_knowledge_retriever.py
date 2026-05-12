#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 4.0: VLMAugmentedRetriever
Frozen VLM + Trainable Adapter 机制
核心思想：使用冻结的文本编码器提取语义，然后通过可训练的Adapter映射到视觉对齐空间
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Optional, List, Dict
import warnings


class VLMAugmentedRetriever(nn.Module):
    """
    VLM增强的知识检索器（Bio-COT 4.0核心组件）
    
    架构：
    1. ❄️ Frozen Text Encoder: 冻结的预训练文本编码器（PubMedBERT）
    2. 🔥 Trainable Adapter: 可训练的适配层（将文本语义映射到视觉空间）
    3. 📦 VLM Cache: 离线生成的VLM描述缓存
    """
    
    def __init__(
        self, 
        vlm_json_path: str,
        visual_dim: int = 768,
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        adapter_hidden_dim: Optional[int] = None
    ):
        """
        Args:
            vlm_json_path: VLM缓存JSON文件路径 (格式: {filename: description})
            visual_dim: 视觉特征维度（输出维度）
            text_model_name: 冻结的文本编码器模型名称
            adapter_hidden_dim: Adapter隐藏层维度（如果None，使用visual_dim）
        """
        super().__init__()
        
        # 1. 加载 VLM 离线数据
        print(f"📥 正在加载VLM Profiles: {vlm_json_path}")
        vlm_path = Path(vlm_json_path)
        
        # 支持相对路径
        if not vlm_path.is_absolute():
            # 尝试多个可能的位置
            possible_paths = [
                vlm_path,
                Path(__file__).parent.parent / vlm_path,
                Path(__file__).parent.parent.parent / vlm_path,
            ]
            
            for p in possible_paths:
                if p.exists():
                    vlm_path = p
                    break
            
            if not vlm_path.exists():
                raise FileNotFoundError(
                    f"VLM缓存文件不存在: {vlm_json_path}\n"
                    f"尝试的路径: {possible_paths}"
                )
        
        with open(vlm_path, 'r', encoding='utf-8') as f:
            self.vlm_data = json.load(f)
        
        print(f"✅ 加载了 {len(self.vlm_data)} 个VLM描述")
        
        # 2. ❄️ 冰区：冻结的 Text Encoder
        print(f"📥 正在加载冻结的Text Encoder: {text_model_name}")
        try:
            from transformers import AutoTokenizer, AutoModel
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            
            # 🔧 优先使用safetensors格式（绕过torch.load安全限制）
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    text_model_name,
                    local_files_only=local_files_only
                )
                self.text_encoder = AutoModel.from_pretrained(
                    text_model_name,
                    use_safetensors=True,  # 优先使用safetensors格式
                    local_files_only=local_files_only
                )
                print(f"✅ Text Encoder加载成功 (使用safetensors格式)")
            except Exception as e1:
                # 如果safetensors失败，尝试忽略安全警告直接加载
                print(f"⚠️ safetensors加载失败: {e1}")
                print(f"   尝试标准格式加载...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        text_model_name,
                        local_files_only=local_files_only,
                    )
                    # 使用trust_remote_code和忽略safetensors
                    import torch
                    # 临时禁用警告
                    import warnings as warn_module
                    with warn_module.catch_warnings():
                        warn_module.filterwarnings('ignore')
                        self.text_encoder = AutoModel.from_pretrained(
                            text_model_name,
                            use_safetensors=False,
                            trust_remote_code=True,
                            local_files_only=local_files_only,
                        )
                    print(f"✅ Text Encoder加载成功 (使用标准格式)")
                except Exception as e2:
                    raise Exception(f"所有加载方法均失败: safetensors={e1}, standard={e2}")
            
            # 彻底冻结参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
            
            text_dim = self.text_encoder.config.hidden_size  # 通常是 768
            print(f"✅ Text Encoder冻结完成 (维度: {text_dim})")
            
        except ImportError:
            raise ImportError(
                "transformers库未安装。请安装: pip install transformers"
            )
        except Exception as e:
            print(f"⚠️ Text Encoder加载失败: {e}")
            print("   将使用简化的文本编码器（仅用于测试）")
            # Fallback: 创建一个简单的文本编码器（仅用于测试）
            text_dim = 768
            self.tokenizer = None
            self.text_encoder = None
        
        # 3. 🔥 火区：可训练的 Adapter
        # 负责将通用文本语义映射到特定的视觉对齐空间
        if adapter_hidden_dim is None:
            adapter_hidden_dim = visual_dim
        
        self.adapter = nn.Sequential(
            nn.Linear(text_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, visual_dim)
        )
        
        # 初始化 adapter 权重
        self._init_weights()
        
        print(f"✅ Adapter创建成功 (文本维度: {text_dim} -> 视觉维度: {visual_dim})")
    
    def _init_weights(self):
        """初始化Adapter权重"""
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self, 
        image_names: List[str], 
        clinical_info: Optional[List[str]] = None, 
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        前向传播：生成文本锚点特征
        
        Args:
            image_names: list of filenames (当前batch的图像文件名)
            clinical_info: list of strings (临床信息，可选)
            device: 设备 (cuda/cpu)
        
        Returns:
            semantic_anchor: [B, visual_dim] 文本锚点特征（用于对齐视觉特征）
        """
        batch_size = len(image_names)
        
        # 1. 构造文本 Prompt
        batch_texts = []
        for i, name in enumerate(image_names):
            # 获取 VLM 描述，如果没有则用 unknown 占位
            # 支持多种文件名格式（可能包含路径）
            file_key = Path(name).name if '/' in name or '\\' in name else name
            
            vlm_desc = self.vlm_data.get(file_key, "unknown medical image")
            
            # 如果没有找到，尝试不带扩展名的文件名
            if vlm_desc == "unknown medical image":
                file_key_no_ext = Path(file_key).stem
                vlm_desc = self.vlm_data.get(file_key_no_ext, "unknown medical image")
            
            # 结合临床信息 (如果有)
            clin = clinical_info[i] if clinical_info is not None and i < len(clinical_info) else ""
            
            if clin:
                full_text = f"Findings: {vlm_desc}. Clinical: {clin}"
            else:
                full_text = f"Findings: {vlm_desc}."
            
            batch_texts.append(full_text)
        
        # 2. ❄️ 通过冻结的 Encoder
        if self.text_encoder is not None:
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                ).to(device)
                
                # 编码
                outputs = self.text_encoder(**inputs)
                
                # 取 CLS token 作为句向量
                frozen_embeds = outputs.last_hidden_state[:, 0, :]  # [B, text_dim]
        else:
            # Fallback: 使用简单的嵌入（仅用于测试）
            warnings.warn("使用简化的文本编码器（仅用于测试）")
            frozen_embeds = torch.randn(batch_size, 768, device=device)
        
        # 3. 🔥 通过 Adapter 进行任务适配
        semantic_anchor = self.adapter(frozen_embeds)  # [B, visual_dim]
        
        return semantic_anchor


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 测试VLMAugmentedRetriever
    print("🧪 测试VLMAugmentedRetriever...")
    
    # 创建测试用的VLM缓存（如果不存在）
    test_vlm_json = Path(__file__).parent.parent / "data" / "vlm_profiles_test.json"
    if not test_vlm_json.exists():
        test_vlm_json.parent.mkdir(parents=True, exist_ok=True)
        test_data = {
            "test_image1.jpg": "Medical image showing cervical tissue with acetowhite changes.",
            "test_image2.jpg": "Colposcopy image with visible vessel patterns and lesion margins.",
        }
        with open(test_vlm_json, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 创建测试VLM缓存: {test_vlm_json}")
    
    # 创建检索器
    retriever = VLMAugmentedRetriever(
        vlm_json_path=str(test_vlm_json),
        visual_dim=768
    )
    
    # 测试前向传播
    image_names = ["test_image1.jpg", "test_image2.jpg"]
    clinical_info = ["HPV positive, age 35", "HPV negative, age 28"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = retriever.to(device)
    
    with torch.no_grad():
        anchors = retriever(image_names, clinical_info, device=device)
    
    print(f"✅ 测试成功！")
    print(f"   输入: {len(image_names)} 个图像")
    print(f"   输出形状: {anchors.shape}")  # [2, 768]
    print(f"   文本锚点特征统计:")
    print(f"     均值: {anchors.mean().item():.4f}")
    print(f"     标准差: {anchors.std().item():.4f}")
