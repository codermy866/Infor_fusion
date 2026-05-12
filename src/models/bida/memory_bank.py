#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: 反事实插值与Memory Bank
用于存储和采样不同中心的噪声特征，实现反事实干预
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoiseMemoryBank(nn.Module):
    """
    噪声特征Memory Bank
    为每个中心维护一个噪声特征库，用于反事实干预
    """
    
    def __init__(self, num_centers=5, feat_dim=768, capacity=100):
        """
        Args:
            num_centers: 中心数量
            feat_dim: 特征维度
            capacity: 每个中心的容量（存储的噪声特征数量）
        """
        super().__init__()
        self.num_centers = num_centers
        self.feat_dim = feat_dim
        self.capacity = capacity
        
        # 注册为buffer，不参与梯度计算
        # bank: [num_centers, capacity, feat_dim]
        self.register_buffer("bank", torch.randn(num_centers, capacity, feat_dim))
        self.register_buffer("ptr", torch.zeros(num_centers, dtype=torch.long))
        self.register_buffer("count", torch.zeros(num_centers, dtype=torch.long))  # 实际存储数量
    
    def update(self, z_noise, center_ids):
        """
        更新Memory Bank
        
        Args:
            z_noise: [B, feat_dim] 噪声特征
            center_ids: [B] 中心标签
        """
        with torch.no_grad():
            batch_centers = torch.unique(center_ids)
            
            for c in batch_centers:
                c = c.item()
                # 获取该中心的所有特征
                mask = (center_ids == c)
                feats = z_noise[mask].detach()  # [n, feat_dim]
                n_feats = feats.shape[0]
                
                if n_feats == 0:
                    continue
                
                # 更新策略：FIFO（先进先出）
                curr_ptr = self.ptr[c].item()
                curr_count = self.count[c].item()
                
                if curr_count + n_feats <= self.capacity:
                    # 还有空间，直接添加
                    end_ptr = curr_ptr + n_feats
                    self.bank[c, curr_ptr:end_ptr] = feats
                    self.ptr[c] = (curr_ptr + n_feats) % self.capacity
                    self.count[c] = min(curr_count + n_feats, self.capacity)
                else:
                    # 空间不足，覆盖旧的特征
                    remaining = self.capacity - curr_ptr
                    if remaining > 0:
                        # ⚠️ 修复：当 feats 数量小于 remaining 时，原实现会发生维度不匹配
                        take = min(remaining, feats.shape[0])
                        self.bank[c, curr_ptr:curr_ptr + take] = feats[:take]
                        feats = feats[take:]
                    
                    # 从开头继续填充
                    if len(feats) > 0:
                        fill_len = min(len(feats), self.capacity)
                        self.bank[c, :fill_len] = feats[:fill_len]
                        self.ptr[c] = fill_len
                    else:
                        self.ptr[c] = 0
                    
                    self.count[c] = self.capacity
    
    def get_counterfactual_noise(self, target_center_ids, strategy='random'):
        """
        为每个样本采样一个反事实噪声（来自指定中心）
        
        Args:
            target_center_ids: [B] 目标中心ID（可以是当前中心，也可以是其他中心）
            strategy: 采样策略
                - 'random': 随机采样
                - 'mean': 使用该中心的平均特征
                - 'nearest': 使用最近的特征
        
        Returns:
            z_noise_cf: [B, feat_dim] 反事实噪声特征（在target_center_ids的设备上）
        """
        B = target_center_ids.shape[0]
        device = target_center_ids.device
        noise_samples = []
        
        for i in range(B):
            c = target_center_ids[i].item()
            count = self.count[c].item()
            
            if count == 0:
                # 如果该中心还没有特征，使用随机噪声
                noise_samples.append(torch.randn(self.feat_dim, device=device))
            else:
                if strategy == 'random':
                    # 随机采样
                    rand_idx = torch.randint(0, count, (1,), device=self.bank.device).item()
                    # 确保从bank中取出的特征转移到正确的设备
                    noise_feat = self.bank[c, rand_idx].to(device)
                    noise_samples.append(noise_feat)
                elif strategy == 'mean':
                    # 使用平均特征
                    mean_feat = self.bank[c, :count].mean(dim=0).to(device)
                    noise_samples.append(mean_feat)
                elif strategy == 'nearest':
                    # 使用最近的特征（需要额外的查询特征，这里简化处理）
                    rand_idx = torch.randint(0, count, (1,), device=self.bank.device).item()
                    noise_feat = self.bank[c, rand_idx].to(device)
                    noise_samples.append(noise_feat)
                else:
                    # 默认随机采样
                    rand_idx = torch.randint(0, count, (1,), device=self.bank.device).item()
                    noise_feat = self.bank[c, rand_idx].to(device)
                    noise_samples.append(noise_feat)
        
        return torch.stack(noise_samples)  # [B, feat_dim]
    
    def get_all_center_noises(self, center_id):
        """
        获取指定中心的所有噪声特征（用于可视化）
        
        Args:
            center_id: 中心ID
        
        Returns:
            noises: [count, feat_dim] 该中心的所有噪声特征
        """
        count = self.count[center_id].item()
        if count == 0:
            return torch.empty(0, self.feat_dim, device=self.bank.device)
        return self.bank[center_id, :count]
    
    def reset(self):
        """重置Memory Bank"""
        self.bank.zero_()
        self.ptr.zero_()
        self.count.zero_()


class CenterDiscriminator(nn.Module):
    """
    中心判别器：从噪声特征预测中心ID
    用于对抗训练，确保z_noise包含域信息
    """
    
    def __init__(self, feat_dim=768, num_centers=5):
        """
        Args:
            feat_dim: 输入特征维度
            num_centers: 中心数量
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_centers)
        )
    
    def forward(self, z_noise):
        """
        预测中心ID
        
        Args:
            z_noise: [B, feat_dim] 噪声特征
        
        Returns:
            center_logits: [B, num_centers] 中心预测logits
        """
        return self.net(z_noise)

