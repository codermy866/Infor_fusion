#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: 轻量级Student Prior网络
目标：替代在线VLM，使用轻量级网络从临床数据生成语义锚点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentPriorNet(nn.Module):
    """
    轻量级先验网络
    输入：临床数据向量 [HPV(1) + TCT(5) + Age(1)] = 7维
    输出：语义锚点特征 [768维]，对齐到图像特征维度
    """
    
    def __init__(self, input_dim=7, output_dim=768, hidden_dims=[256, 512]):
        """
        Args:
            input_dim: 临床数据维度 (HPV + TCT + Age = 7)
            output_dim: 输出特征维度 (默认768，对齐图像特征)
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm1d，避免batch_size=1时的错误
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_vec):
        """
        前向传播
        
        Args:
            clinical_vec: [B, input_dim] 临床数据向量
                - HPV: [B, 1] 二值
                - TCT: [B, 5] one-hot编码
                - Age: [B, 1] 连续值
        
        Returns:
            z_sem: [B, output_dim] 语义锚点特征
        """
        return self.net(clinical_vec)


def build_clinical_vector(clinical_data, device='cuda'):
    """
    从临床数据字典构建向量
    
    Args:
        clinical_data: dict with keys 'hpv', 'tct', 'age'
        device: 设备
    
    Returns:
        clinical_vec: [B, 7] 临床数据向量
    """
    B = len(clinical_data.get('hpv', []))
    
    # HPV: 二值 [B, 1]
    hpv = torch.tensor(clinical_data['hpv'], dtype=torch.float32).unsqueeze(-1)  # [B, 1]
    
    # TCT: one-hot编码 [B, 5]
    # TCT类别映射: NILM=0, ASC-US=1, LSIL=2, HSIL=3, 其他=4
    tct_mapping = {
        'NILM': 0, 'ASC-US': 1, 'LSIL': 2, 'HSIL': 3
    }
    tct_indices = []
    for tct_val in clinical_data.get('tct', ['NILM'] * B):
        if isinstance(tct_val, str):
            tct_idx = tct_mapping.get(tct_val, 4)
        else:
            tct_idx = int(tct_val) if tct_val < 5 else 4
        tct_indices.append(tct_idx)
    
    tct_onehot = F.one_hot(torch.tensor(tct_indices), num_classes=5).float()  # [B, 5]
    
    # Age: 连续值 [B, 1]，归一化到[0, 1]
    age = torch.tensor(clinical_data.get('age', [45] * B), dtype=torch.float32).unsqueeze(-1)  # [B, 1]
    age = age / 100.0  # 归一化（假设年龄范围0-100）
    
    # 拼接
    clinical_vec = torch.cat([hpv, tct_onehot, age], dim=-1)  # [B, 7]
    
    return clinical_vec.to(device)


def pretrain_student_prior(student_prior, vlm_features_cache, clinical_data_dict, 
                          num_epochs=50, lr=1e-3, device='cuda:1'):
    """
    预训练Student Prior网络，使其输出拟合VLM特征
    
    Args:
        student_prior: StudentPriorNet实例
        vlm_features_cache: dict {sample_id: vlm_feature [1536]}
        clinical_data_dict: dict {sample_id: clinical_data}
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
    """
    student_prior = student_prior.to(device)
    student_prior.train()
    
    optimizer = torch.optim.Adam(student_prior.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 投影层：将VLM特征从1536维投影到768维
    vlm_proj = nn.Linear(1536, 768).to(device)
    optimizer_proj = torch.optim.Adam(vlm_proj.parameters(), lr=lr)
    
    print(f"🔄 开始预训练Student Prior网络，共 {len(vlm_features_cache)} 个样本...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_samples = 0
        
        for sample_id, vlm_feat in vlm_features_cache.items():
            if sample_id not in clinical_data_dict:
                continue
            
            # 构建临床向量
            clinical_vec = build_clinical_vector(
                clinical_data_dict[sample_id],
                device=device
            )  # [1, 7]
            
            # 获取VLM特征并投影
            vlm_feat_tensor = torch.from_numpy(vlm_feat).unsqueeze(0).to(device)  # [1, 1536]
            vlm_feat_proj = vlm_proj(vlm_feat_tensor)  # [1, 768]
            
            # Student Prior输出
            z_sem = student_prior(clinical_vec)  # [1, 768]
            
            # 计算损失
            loss = criterion(z_sem, vlm_feat_proj)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer_proj.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_proj.step()
            
            total_loss += loss.item()
            n_samples += 1
        
        avg_loss = total_loss / n_samples if n_samples > 0 else 0
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print(f"✅ Student Prior预训练完成！")
    return student_prior, vlm_proj


if __name__ == '__main__':
    # 测试代码
    model = StudentPriorNet(input_dim=7, output_dim=768)
    x = torch.randn(32, 7)
    y = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")


