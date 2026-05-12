from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class HierarchicalViT(nn.Module):
    """
    ViT 封装：返回指定 block 的 token 特征（去掉 CLS 后的 patch tokens）。
    从 Bio-COT 5.0 引入：分层多尺度特征提取

    输出：List[Tensor]，每个 Tensor 形状为 [B, N, D]。
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        drop_path_rate: float = 0.0,  # 🔥 新增：DropPath支持
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm未安装，请先安装: pip install timm") from e

        # 🔥 使用ViT并传递drop_path_rate；默认可离线运行，避免训练被HF下载阻塞。
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            drop_path_rate=drop_path_rate,  # 🔥 关键：防止过拟合
        )
        if not hasattr(self.vit, "blocks"):
            raise ValueError(f"{model_name} 不是 VisionTransformer-like 模型，找不到 .blocks")

        self.out_indices = tuple(out_indices)
        self.out_indices_set = set(out_indices)

        # 常见字段：patch_embed/pos_embed/pos_drop/blocks/norm/cls_token
        self.embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        if self.embed_dim is None:
            raise ValueError("无法从 timm ViT 推断 embed_dim")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # patchify
        x = self.vit.patch_embed(x)  # [B, N, D]

        # prepend CLS
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            cls_tokens = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]

        # add pos embed
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed
        if hasattr(self.vit, "pos_drop"):
            x = self.vit.pos_drop(x)

        feats: List[torch.Tensor] = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.out_indices_set:
                # 对每个 stage 做一次 norm（更稳）
                if hasattr(self.vit, "norm") and self.vit.norm is not None:
                    feats.append(self.vit.norm(x))
                else:
                    feats.append(x)

        # 去掉 CLS token，返回 patch tokens
        out = [f[:, 1:, :] if f.dim() == 3 and f.size(1) > 1 else f for f in feats]
        return out
