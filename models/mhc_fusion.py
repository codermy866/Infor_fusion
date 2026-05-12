from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseAwareMHC(nn.Module):
    """
    [Bio-COT 5.0 Core Module] 整合到 3.2
    Noise-Aware Manifold Hyper-Connection (NA-mHC).
    
    Function:
    1. Modeling center-specific noise using learnable embeddings.
    2. Purifying visual features via a noise gate.
    3. Projecting clean visual-clinical interactions onto the Birkhoff Polytope via Sinkhorn.
    """
    def __init__(
        self, 
        img_dim: int = 768, 
        clinical_dim: int = 256, 
        num_centers: int = 5, 
        latent_dim: int = 256, 
        sinkhorn_iters: int = 3, 
        epsilon: float = 0.05,
        dropout: float = 0.3
    ):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        self.latent_dim = latent_dim

        # --- A. Projection Heads (with Dropout) ---
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, latent_dim),
            nn.Dropout(dropout)
        )
        self.clin_proj = nn.Sequential(
            nn.Linear(clinical_dim, latent_dim),
            nn.Dropout(dropout)
        )
        
        # --- B. Noise Modeling (The "Awareness" Part) ---
        self.center_embedding = nn.Embedding(num_centers, latent_dim)
        
        # 噪声门控
        self.noise_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.Dropout(dropout * 0.67),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # --- C. Reconstruction ---
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, img_dim),
            nn.Dropout(dropout * 0.67)
        )
        self.norm = nn.LayerNorm(img_dim)

    def sinkhorn_log_space(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Solves Optimal Transport plan in log-space for numerical stability."""
        log_P = -cost_matrix / self.epsilon
        
        for _ in range(self.sinkhorn_iters):
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
        
        return torch.exp(log_P)

    def forward(
        self, 
        img_feat: torch.Tensor, 
        clin_state: torch.Tensor, 
        center_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_feat: [B, N, D_v] - 视觉特征序列
            clin_state: [B, D_c] - 当前层的临床意图向量
            center_ids: [B] - 样本所属的中心 ID (LongTensor)
        Returns:
            feat_fused: [B, N, D_v] - 融合后的特征
            noise_prob: [B, N, 1] - 噪声概率图
        """
        B, N, _ = img_feat.shape
        
        # 1. 投影到潜在流形空间
        H_v = self.img_proj(img_feat)      # [B, N, L]
        H_c = self.clin_proj(clin_state).unsqueeze(1) # [B, 1, L]
        
        # 2. Noise Purification Process
        if center_ids is None:
            Z_noise = torch.zeros(B, 1, self.latent_dim, device=img_feat.device, dtype=img_feat.dtype)
        else:
            Z_noise = self.center_embedding(center_ids).unsqueeze(1) # [B, 1, L]
        
        Z_noise_expanded = Z_noise.expand(-1, N, -1) # [B, N, L]
        noise_prob = self.noise_gate(torch.cat([H_v, Z_noise_expanded], dim=-1)) # [B, N, 1]
        H_v_clean = H_v * (1.0 - noise_prob)
        
        # 3. Manifold-Constrained Fusion (Sinkhorn)
        H_v_norm = F.normalize(H_v_clean, dim=-1)
        H_c_norm = F.normalize(H_c, dim=-1)
        cost = 1.0 - torch.matmul(H_v_norm, H_c_norm.transpose(1, 2))
        transport_plan = self.sinkhorn_log_space(cost) # [B, N, 1]
        
        # 4. Feature Transport
        context = transport_plan * H_c # [B, N, L]
        out = self.out_proj(context)
        
        return self.norm(img_feat + out), noise_prob

