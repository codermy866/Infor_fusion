from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn


def _state_dict_from_checkpoint(checkpoint_path: str | None) -> dict[str, torch.Tensor]:
    if not checkpoint_path:
        return {}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Vision checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}")
    return payload


def load_matching_vision_weights(model: nn.Module, checkpoint_path: str | None) -> tuple[int, int]:
    """Load only shape-compatible timm/RETFound keys into a vision backbone."""
    raw_state = _state_dict_from_checkpoint(checkpoint_path)
    if not raw_state:
        return 0, 0

    model_state = model.state_dict()
    matched = {}
    skipped = 0
    prefixes = ("module.", "model.", "visual_encoder.", "visual_encoder.vit.", "backbone.", "encoder.")
    for key, value in raw_state.items():
        clean_key = str(key)
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if clean_key in model_state and tuple(model_state[clean_key].shape) == tuple(value.shape):
            matched[clean_key] = value
        else:
            skipped += 1
    if not matched:
        raise RuntimeError(f"No shape-compatible weights found in {checkpoint_path}")
    model.load_state_dict(matched, strict=False)
    return len(matched), skipped


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
        checkpoint_path: str | None = None,
        output_dim: int | None = None,
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
        self.loaded_checkpoint_path = checkpoint_path
        if checkpoint_path:
            matched, skipped = load_matching_vision_weights(self.vit, checkpoint_path)
            print(f"Loaded vision checkpoint {checkpoint_path}: matched={matched}, skipped={skipped}")
        if not hasattr(self.vit, "blocks"):
            raise ValueError(f"{model_name} 不是 VisionTransformer-like 模型，找不到 .blocks")

        self.out_indices = tuple(out_indices)
        self.out_indices_set = set(out_indices)

        # 常见字段：patch_embed/pos_embed/pos_drop/blocks/norm/cls_token
        self.embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        if self.embed_dim is None:
            raise ValueError("无法从 timm ViT 推断 embed_dim")
        self.proj = (
            nn.Linear(int(self.embed_dim), int(output_dim), bias=False)
            if output_dim is not None and int(output_dim) != int(self.embed_dim)
            else nn.Identity()
        )
        self.embed_dim = int(output_dim) if output_dim is not None else int(self.embed_dim)

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
        out = [self.proj(f) for f in out]
        return out
