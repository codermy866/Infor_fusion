#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2 Enhanced (整合5.0所有优势)
融合3.1、4.0和5.0的优势：
1. 保留3.1的所有优点（显式对齐、自适应模态融合、增强Visual Notes）
2. 引入4.0的优势（Frozen VLM + Trainable Adapter、动态知识生成）
3. 🔥 整合5.0的优势：
   - 分层多尺度特征提取（HierarchicalViT）
   - 噪声感知流形超连接（NA-mHC）
   - 动态临床查询演化（ClinicalEvolver）
   - 激进正则化策略（Dropout 0.4, DropPath 0.2）
   - 正交损失（解耦方式）
   - Text Adapter（VLM集成）
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import hashlib
import numpy as np
from typing import Dict, Tuple, Optional, List

# 导入基础组件（从exp_bio3.2本地的src目录导入）
import sys
from pathlib import Path
# 添加exp_bio3.2的src目录到sys.path
local_src_path = Path(__file__).resolve().parent.parent / 'src'
if local_src_path.exists():
    sys.path.insert(0, str(local_src_path.parent))
    from src.models.bida.bio_cot_v2 import DualHeadImageEncoder
    from src.models.bida.memory_bank import NoiseMemoryBank, CenterDiscriminator
    from src.models.bida.losses import SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss
else:
    raise ImportError(f"无法找到src模块，请检查路径: {local_src_path}")

# 导入增强后的 Visual Notes（3.1的优势）
from .visual_notes import VisualNotesModule

# 🔥 导入5.0的模块（5.0的优势）
from .backbones import HierarchicalViT
from .mhc_fusion import NoiseAwareMHC
from .clinical_evolver import ClinicalEvolver

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


class VariationalReliabilityInference(nn.Module):
    """
    Variational modality reliability inference.

    Each modality emits a posterior mean and log-variance. Fusion is performed
    with precision weights, so higher uncertainty lowers that modality's
    contribution. At training time, reparameterized samples propagate reliability
    uncertainty into the downstream diagnosis state.
    """

    def __init__(
        self,
        dim: int = 768,
        num_centers: int = 5,
        modalities: Tuple[str, ...] = ("oct", "colpo", "clinical_prior"),
        center_aware: bool = True,
        dropout: float = 0.1,
        min_logvar: float = -6.0,
        max_logvar: float = 3.0,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.center_aware = center_aware
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.center_embedding = nn.Embedding(num_centers, dim) if center_aware else None
        in_dim = dim * 2 if center_aware else dim
        self.posterior_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim * 2),
                )
                for name in self.modalities
            }
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        center_labels: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, object]:
        mus: Dict[str, torch.Tensor] = {}
        logvars: Dict[str, torch.Tensor] = {}
        samples: Dict[str, torch.Tensor] = {}
        precisions = []

        first_feat = next(iter(features.values()))
        if self.center_embedding is not None and center_labels is not None:
            center_context = self.center_embedding(center_labels)
        elif self.center_embedding is not None:
            center_context = torch.zeros_like(first_feat)
        else:
            center_context = None

        for name in self.modalities:
            feat = features[name]
            head_in = torch.cat([feat, center_context], dim=-1) if center_context is not None else feat
            params = self.posterior_heads[name](head_in)
            mu, raw_logvar = params.chunk(2, dim=-1)
            logvar = raw_logvar.clamp(self.min_logvar, self.max_logvar)
            if sample and self.training:
                eps = torch.randn_like(mu)
                z = mu + torch.exp(0.5 * logvar) * eps
            else:
                z = mu
            precision = torch.exp(-logvar).mean(dim=-1, keepdim=True)
            mus[name] = mu
            logvars[name] = logvar
            samples[name] = z
            precisions.append(precision)

        precision_tensor = torch.stack(precisions, dim=1).clamp_min(1e-6)
        weights = precision_tensor / precision_tensor.sum(dim=1, keepdim=True)
        fused = 0.0
        weight_dict: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.modalities):
            w = weights[:, idx, :]
            weight_dict[name] = w
            fused = fused + w * samples[name]

        kl_terms = []
        for name in self.modalities:
            mu = mus[name]
            logvar = logvars[name]
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            kl_terms.append(kl.mean())

        return {
            "fused": fused,
            "mu": mus,
            "logvar": logvars,
            "samples": samples,
            "weights": weight_dict,
            "kl": torch.stack(kl_terms).mean(),
        }


class CenterEmpiricalPriorMemory(nn.Module):
    """
    Center-specific empirical prior memory for posterior refinement.

    The slots are buffers updated by EMA during training. They are not used as a
    trainable shortcut; instead they provide a lightweight center-conditioned
    empirical context for the sequential belief update.
    """

    def __init__(self, num_centers: int, dim: int, momentum: float = 0.05):
        super().__init__()
        self.num_centers = max(int(num_centers), 1)
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("slots", torch.zeros(self.num_centers, dim))
        self.register_buffer("counts", torch.zeros(self.num_centers))

    def retrieve(self, center_labels: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if center_labels is None:
            return torch.zeros(batch_size, self.dim, device=device)
        labels = center_labels.clamp(min=0, max=self.num_centers - 1)
        return self.slots.to(device)[labels]

    @torch.no_grad()
    def update(self, states: torch.Tensor, center_labels: Optional[torch.Tensor]) -> None:
        if center_labels is None or states.numel() == 0:
            return
        labels = center_labels.detach().clamp(min=0, max=self.num_centers - 1)
        detached = states.detach()
        for center_idx in labels.unique():
            mask = labels == center_idx
            if not torch.any(mask):
                continue
            mean_state = detached[mask].mean(dim=0)
            idx = int(center_idx.item())
            if self.counts[idx] <= 0:
                self.slots[idx].copy_(mean_state)
            else:
                self.slots[idx].mul_(1.0 - self.momentum).add_(mean_state, alpha=self.momentum)
            self.counts[idx] += mask.sum().to(self.counts.dtype)


class SequentialPosteriorRefinement(nn.Module):
    """
    Cached-feature posterior trajectory:

    z0 -> z1(clinical prior) -> z2(colposcopy evidence) -> z3(OCT evidence).
    This makes posterior refinement explicit even when the raw-image
    hierarchical branch is bypassed by cached patch features.
    """

    def __init__(self, dim: int = 768, num_centers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.center_memory = CenterEmpiricalPriorMemory(num_centers, dim)
        self.evidence_proj = nn.ModuleDict(
            {
                "clinical_prior": nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)),
                "colpo": nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)),
                "oct": nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)),
                "center": nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)),
            }
        )
        self.initial_state = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.update_cell = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.state_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _update(self, state: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        updated = self.update_cell(evidence, state)
        return self.state_norm(state + self.dropout(updated))

    def forward(
        self,
        clinical_prior: torch.Tensor,
        colpo: torch.Tensor,
        oct_feat: torch.Tensor,
        center_labels: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> Dict[str, object]:
        batch_size = clinical_prior.shape[0]
        device = clinical_prior.device
        center_prior = self.center_memory.retrieve(center_labels, batch_size, device)

        e_cli = self.evidence_proj["clinical_prior"](clinical_prior)
        e_center = self.evidence_proj["center"](center_prior)
        e_colpo = self.evidence_proj["colpo"](colpo) + e_center
        e_oct = self.evidence_proj["oct"](oct_feat)

        z0 = self.initial_state(clinical_prior)
        z1 = self._update(z0, e_cli)
        z2 = self._update(z1, e_colpo)
        z3 = self._update(z2, e_oct)

        if update_memory:
            self.center_memory.update(z3, center_labels)

        smooth = (F.mse_loss(z1, z0.detach()) + F.mse_loss(z2, z1.detach()) + F.mse_loss(z3, z2.detach())) / 3.0
        return {
            "trajectory": {"z0": z0, "z1": z1, "z2": z2, "z3": z3},
            "evidence": {"clinical_prior": e_cli, "center": e_center, "colpo": e_colpo, "oct": e_oct},
            "z_final": z3,
            "smooth_loss": smooth,
        }


class ASCCPPrototypePrior(nn.Module):
    """
    Discrete clinical-prior prototype manifold.

    Prototypes are initialized from structured guideline-style text descriptions
    and then refined by a trainable residual. This keeps p(z|K) tied to the
    clinical prior text while still allowing dataset-specific adaptation.
    """

    DEFAULT_PROTOTYPES = (
        "normal_or_hpv_negative_low_risk",
        "hpv_positive_nilm_surveillance",
        "ascus_or_lsil_low_grade_abnormality",
        "asch_or_agc_high_risk_triage",
        "hsil_or_cin2plus_high_grade_lesion",
        "cancer_suspicious_invasive_disease",
    )

    def __init__(
        self,
        dim: int = 768,
        temperature: float = 0.15,
        prototype_names: Optional[Tuple[str, ...]] = None,
        prototype_path: Optional[str] = None,
        text_model_name: Optional[str] = None,
        use_text_derived_init: bool = True,
        local_files_only: bool = True,
    ):
        super().__init__()
        prototype_items = self._load_prototype_items(prototype_path, prototype_names)
        self.prototype_names = tuple(item["name"] for item in prototype_items)
        self.prototype_texts = tuple(item["text"] for item in prototype_items)
        self.temperature = temperature
        init_embeddings, init_source = self._build_text_embeddings(
            self.prototype_texts,
            dim=dim,
            text_model_name=text_model_name,
            use_text_derived_init=use_text_derived_init,
            local_files_only=local_files_only,
        )
        self.register_buffer("prototype_text_embeddings", init_embeddings)
        self.prototype_residual = nn.Parameter(torch.zeros_like(init_embeddings))
        self.prototype_residual_scale = nn.Parameter(torch.tensor(0.10))
        self.prototype_norm = nn.LayerNorm(dim)
        self.text_init_source = init_source

    @classmethod
    def _load_prototype_items(
        cls,
        prototype_path: Optional[str],
        prototype_names: Optional[Tuple[str, ...]],
    ) -> List[Dict[str, str]]:
        if prototype_path:
            path = Path(prototype_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    raw_items = json.load(handle)
                items = []
                for item in raw_items:
                    name = str(item.get("name", "")).strip()
                    if not name:
                        continue
                    description = str(item.get("description", "")).strip()
                    prototype_text = str(item.get("prototype_text", "")).strip()
                    category = str(item.get("asccp_category", "")).strip()
                    risk = str(item.get("risk_level", "")).strip()
                    text_parts = [part for part in [name, category, risk, description, prototype_text] if part]
                    items.append({"name": name, "text": ". ".join(text_parts)})
                if items:
                    return items

        names = tuple(prototype_names or cls.DEFAULT_PROTOTYPES)
        return [{"name": name, "text": name.replace("_", " ")} for name in names]

    @staticmethod
    def _hash_text_embedding(text: str, dim: int) -> torch.Tensor:
        values = torch.empty(dim, dtype=torch.float32)
        seed = text.encode("utf-8")
        offset = 0
        counter = 0
        while offset < dim:
            digest = hashlib.sha256(seed + str(counter).encode("utf-8")).digest()
            chunk = torch.tensor(list(digest), dtype=torch.float32) / 127.5 - 1.0
            take = min(chunk.numel(), dim - offset)
            values[offset : offset + take] = chunk[:take]
            offset += take
            counter += 1
        return F.normalize(values, dim=0)

    @classmethod
    def _hash_text_embeddings(cls, texts: Tuple[str, ...], dim: int) -> torch.Tensor:
        return torch.stack([cls._hash_text_embedding(text, dim) for text in texts], dim=0) * 0.02

    @staticmethod
    def _transformer_text_embeddings(
        texts: Tuple[str, ...],
        dim: int,
        text_model_name: str,
        local_files_only: bool,
    ) -> Optional[torch.Tensor]:
        try:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=local_files_only)
            encoder = AutoModel.from_pretrained(text_model_name, local_files_only=local_files_only)
            encoder.eval()
            with torch.no_grad():
                inputs = tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                outputs = encoder(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    emb = outputs.last_hidden_state[:, 0, :]
            if emb.shape[-1] > dim:
                emb = emb[:, :dim]
            elif emb.shape[-1] < dim:
                emb = F.pad(emb, (0, dim - emb.shape[-1]))
            return F.normalize(emb.float(), dim=-1) * 0.02
        except Exception:
            return None

    @classmethod
    def _build_text_embeddings(
        cls,
        texts: Tuple[str, ...],
        dim: int,
        text_model_name: Optional[str],
        use_text_derived_init: bool,
        local_files_only: bool,
    ) -> Tuple[torch.Tensor, str]:
        if use_text_derived_init and text_model_name:
            transformer_embeddings = cls._transformer_text_embeddings(texts, dim, text_model_name, local_files_only)
            if transformer_embeddings is not None:
                return transformer_embeddings, "frozen_transformer_text"
        if use_text_derived_init:
            return cls._hash_text_embeddings(texts, dim), "deterministic_text_hash"
        return torch.randn(len(texts), dim) * 0.02, "random_trainable"

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_norm = F.normalize(z, dim=-1)
        residual_scale = torch.clamp(self.prototype_residual_scale, min=0.0, max=1.0)
        prototypes = self.prototype_norm(self.prototype_text_embeddings + residual_scale * self.prototype_residual)
        proto_norm = F.normalize(prototypes, dim=-1)
        cost = 1.0 - torch.matmul(z_norm, proto_norm.t())
        assignment = F.softmax(-cost / max(self.temperature, 1e-6), dim=-1)
        context = torch.matmul(assignment, prototypes)
        expected_cost = torch.sum(assignment * cost, dim=-1).mean()
        entropy = -torch.sum(assignment * torch.log(assignment + 1e-8), dim=-1).mean()
        return {
            "prototypes": prototypes,
            "assignment": assignment,
            "context": context,
            "cost": cost,
            "ot_loss": expected_cost,
            "entropy": entropy,
        }


class ModalityLikelihoodDecoder(nn.Module):
    """Lightweight feature-space approximation to p(M | z, c)."""

    def __init__(self, dim: int = 768, num_centers: int = 5, center_aware: bool = True, dropout: float = 0.1):
        super().__init__()
        self.center_aware = center_aware
        self.center_embedding = nn.Embedding(num_centers, dim) if center_aware else None
        in_dim = dim * 2 if center_aware else dim
        self.decoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                )
                for name in ("oct", "colpo", "clinical_prior")
            }
        )

    def forward(
        self,
        z: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        center_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if self.center_embedding is not None:
            if center_labels is None:
                center_context = torch.zeros_like(z)
            else:
                labels = center_labels.clamp(min=0, max=self.center_embedding.num_embeddings - 1)
                center_context = self.center_embedding(labels)
            decoder_input = torch.cat([z, center_context], dim=-1)
        else:
            decoder_input = z
        recon = {name: decoder(decoder_input) for name, decoder in self.decoders.items()}
        losses = [F.mse_loss(recon[name], targets[name].detach()) for name in recon.keys() if name in targets]
        loss = torch.stack(losses).mean() if losses else z.new_tensor(0.0)
        return {"recon": recon, "loss": loss}


class TrajectoryCoEReadout(nn.Module):
    """
    Trajectory-conditioned evidence readout.

    This is not a free-text generator. It provides step-wise evidence template
    logits that can be inspected and used by automatic evidence consistency
    tests without claiming clinician replacement.
    """

    TEMPLATE_NAMES = (
        "clinical_prior_supports_low_risk",
        "clinical_prior_supports_high_risk",
        "colposcopy_supports_low_grade",
        "colposcopy_supports_high_grade",
        "oct_confirms_low_risk",
        "oct_confirms_high_risk",
    )

    def __init__(self, dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.template_names = self.TEMPLATE_NAMES
        self.step_decoders = nn.ModuleDict(
            {
                step: nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, len(self.template_names)),
                )
                for step in ("z1", "z2", "z3")
            }
        )

    @staticmethod
    def _normalize_clinical_info(clinical_info: Optional[List[str]], batch_size: int) -> List[str]:
        if clinical_info is None:
            return [""] * batch_size
        if isinstance(clinical_info, str):
            return [clinical_info] * batch_size
        if isinstance(clinical_info, tuple):
            clinical_info = list(clinical_info)
        if isinstance(clinical_info, list):
            values = [str(item) for item in clinical_info]
            if len(values) < batch_size:
                values.extend([""] * (batch_size - len(values)))
            return values[:batch_size]
        return [""] * batch_size

    @classmethod
    def _weak_targets(
        cls,
        labels: torch.Tensor,
        clinical_info: Optional[List[str]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        labels = labels.detach().long().view(-1)
        batch_size = int(labels.shape[0])
        clinical_text = cls._normalize_clinical_info(clinical_info, batch_size)

        clinical_targets = []
        for idx, text in enumerate(clinical_text):
            text_lower = text.lower()
            high_risk_prior = (
                "positive" in text_lower
                or "hpv: 1" in text_lower
                or "16" in text_lower
                or "18" in text_lower
                or "asc-h" in text_lower
                or "agc" in text_lower
                or "hsil" in text_lower
                or "scc" in text_lower
                or "癌" in text_lower
                or "阳性" in text_lower
                or "高危" in text_lower
            )
            if text.strip() == "":
                high_risk_prior = bool(labels[idx].item())
            clinical_targets.append(1 if high_risk_prior else 0)

        high_grade = labels.eq(1)
        return {
            "z1": torch.tensor(clinical_targets, dtype=torch.long, device=device),
            "z2": torch.where(high_grade, torch.full_like(labels, 3), torch.full_like(labels, 2)),
            "z3": torch.where(high_grade, torch.full_like(labels, 5), torch.full_like(labels, 4)),
        }

    def forward(
        self,
        trajectory: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        clinical_info: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        step_inputs = {
            "z1": evidence["clinical_prior"],
            "z2": evidence["colpo"],
            "z3": evidence["oct"],
        }
        logits = {}
        probs = {}
        for step, ev in step_inputs.items():
            step_logits = self.step_decoders[step](torch.cat([trajectory[step], ev], dim=-1))
            logits[step] = step_logits
            probs[step] = F.softmax(step_logits, dim=-1)

        result: Dict[str, object] = {"logits": logits, "probs": probs, "template_names": self.template_names}
        if labels is not None:
            targets = self._weak_targets(labels, clinical_info, trajectory["z1"].device)
            losses = [F.cross_entropy(logits[step], targets[step]) for step in ("z1", "z2", "z3")]
            result["targets"] = targets
            result["supervision_loss"] = torch.stack(losses).mean()
        return result


class BioCOT_v3_2(nn.Module):
    """
    Bio-COT 3.2 Enhanced Version (整合5.0所有优势)
    融合3.1、4.0和5.0的优势：
    - 3.1: 显式对齐、自适应模态融合、增强Visual Notes
    - 4.0: Frozen VLM + Trainable Adapter、动态知识生成
    - 5.0: 分层多尺度特征、噪声感知融合、动态临床演化、激进正则化、正交损失
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_centers: int = 5,
        input_dim: int = 768,
        vlm_json_path: str = None,
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        use_visual_notes: bool = True,
        use_ot: bool = True,
        use_dual: bool = True,
        use_cross_attn: bool = True,
        use_adaptive_gating: bool = True,
        warmup_epochs: int = 10,
        hidden_dim: int = 768,
        # 🔥 5.0新增参数
        use_hierarchical: bool = True,  # 是否使用分层多尺度特征
        extract_layers: Tuple[int, ...] = (2, 5, 8, 11),  # 提取的层索引
        vit_pretrained: bool = True,
        drop_path_rate: float = 0.2,  # ViT的DropPath率
        dropout_rate: float = 0.4,  # 激进正则化
        use_noise_aware: bool = True,  # 是否使用噪声感知融合
        use_clinical_evolver: bool = True,  # 是否使用临床查询演化
        mhc_latent_dim: int = 256,  # NA-mHC的潜在维度
        sinkhorn_iters: int = 3,  # Sinkhorn迭代次数
        mhc_epsilon: float = 0.05,  # Sinkhorn epsilon
        use_text_adapter: bool = True,  # 是否使用Text Adapter（5.0的VLM集成）
        use_variational_reliability: bool = True,
        use_center_aware_reliability: bool = True,
        fusion_strategy: str = "gated",
        direct_fusion_only: bool = False,
        use_posterior_refinement: bool = True,
        use_asccp_prior: bool = True,
        use_modality_likelihood: bool = True,
        use_coe_readout: bool = True,
        use_coe_supervision: bool = True,
        asccp_prototype_path: Optional[str] = None,
        use_text_derived_asccp: bool = True,
        asccp_text_model_name: Optional[str] = None,
        asccp_text_local_files_only: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_centers = num_centers
        self.use_visual_notes = use_visual_notes
        self.use_ot = use_ot
        self.use_dual = use_dual
        self.use_adaptive_gating = use_adaptive_gating
        self.use_hierarchical = use_hierarchical
        self.use_noise_aware = use_noise_aware
        self.use_clinical_evolver = use_clinical_evolver
        self.use_variational_reliability = use_variational_reliability
        self.use_center_aware_reliability = use_center_aware_reliability
        self.fusion_strategy = "variational" if use_variational_reliability else fusion_strategy
        self.direct_fusion_only = direct_fusion_only
        self.use_posterior_refinement = use_posterior_refinement
        self.use_asccp_prior = use_asccp_prior
        self.use_modality_likelihood = use_modality_likelihood
        self.use_coe_readout = use_coe_readout
        self.use_coe_supervision = use_coe_supervision
        self.current_epoch = 0
        
        # ============================================================
        # 🔥 5.0优势1：分层多尺度特征提取（HierarchicalViT）
        # ============================================================
        if use_hierarchical:
            self.visual_encoder = HierarchicalViT(
                model_name="vit_base_patch16_224",
                pretrained=vit_pretrained,
                out_indices=extract_layers,
                drop_path_rate=drop_path_rate
            )
            self.num_stages = len(extract_layers)
        else:
            self.visual_encoder = None
            self.num_stages = 1
        
        # ============================================================
        # 🔥 关键改动1：替换为VLMAugmentedRetriever（4.0的优势）
        # ============================================================
        # 支持消融实验：可以禁用VLM Retriever
        # 注意：use_vlm_retriever 将在 create_bio_cot_v3_2 中设置
        # 这里先初始化为 True，后续会被覆盖
        self.use_vlm_retriever = True  # 默认启用，可通过config覆盖
        
        # 创建知识检索器（如果vlm_json_path存在）
        if vlm_json_path is not None:
            self.knowledge_retriever = VLMAugmentedRetriever(
                vlm_json_path=vlm_json_path,
                visual_dim=embed_dim,
                text_model_name=text_model_name
            )
        else:
            self.knowledge_retriever = None
        
        # 创建语义投影器（仿照exp_bio3.0_improved的方案）
        # 无论是否使用VLM Retriever，都创建note_projector
        # 当禁用VLM Retriever时，使用可学习的静态嵌入作为输入
        self.note_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # 创建可学习的知识嵌入基向量（仿照exp_bio3.0_improved的方案）
        # 当禁用VLM Retriever时，使用这个可学习参数作为note_embeds的替代
        # 初始化为小的随机值，而不是全零
        self.learnable_knowledge_base = nn.Parameter(
            torch.randn(1, embed_dim) * 0.02  # 小的随机初始化
        )
        
        # 🔥 5.0优势6：Text Adapter（VLM集成增强）
        if use_text_adapter:
            self.text_adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(dropout_rate * 0.5)
            )
        else:
            self.text_adapter = None
        
        # ============================================================
        # 🔥 5.0优势2：噪声感知流形超连接（NA-mHC）
        # ============================================================
        if use_noise_aware and use_hierarchical:
            self.mhc_layers = nn.ModuleList([
                NoiseAwareMHC(
                    img_dim=embed_dim,
                    clinical_dim=hidden_dim,
                    num_centers=num_centers,
                    latent_dim=mhc_latent_dim,
                    sinkhorn_iters=sinkhorn_iters,
                    epsilon=mhc_epsilon,
                    dropout=dropout_rate
                )
                for _ in range(self.num_stages)
            ])
        else:
            self.mhc_layers = None
        
        # ============================================================
        # 🔥 5.0优势3：动态临床查询演化（ClinicalEvolver）
        # ============================================================
        if use_clinical_evolver and use_hierarchical:
            self.evolvers = nn.ModuleList([
                ClinicalEvolver(
                    visual_dim=embed_dim,
                    clinical_dim=hidden_dim,
                    dropout=dropout_rate * 0.5
                )
                for _ in range(max(self.num_stages - 1, 0))
            ])
        else:
            self.evolvers = None
        
        # 临床状态初始化
        self.clinical_init = nn.Sequential(
            nn.Linear(7, hidden_dim),  # 假设临床特征维度为7
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.GELU(),
        )
        self.clinical_feature_projector = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.clinical_prior_mix = nn.Parameter(torch.tensor(0.5))
        self.clinical_prior_norm = nn.LayerNorm(embed_dim)

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

        self.variational_reliability = (
            VariationalReliabilityInference(
                dim=embed_dim,
                num_centers=num_centers,
                center_aware=use_center_aware_reliability,
                dropout=dropout_rate * 0.5,
            )
            if use_variational_reliability
            else None
        )
        self.concat_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.modality_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.late_heads = nn.ModuleDict(
            {
                "oct": nn.Linear(embed_dim, num_classes),
                "colpo": nn.Linear(embed_dim, num_classes),
                "clinical_prior": nn.Linear(embed_dim, num_classes),
            }
        )

        self.posterior_refiner = (
            SequentialPosteriorRefinement(dim=embed_dim, num_centers=num_centers, dropout=dropout_rate * 0.5)
            if use_posterior_refinement and not direct_fusion_only
            else None
        )
        self.posterior_mix_norm = nn.LayerNorm(embed_dim)
        self.asccp_prior = (
            ASCCPPrototypePrior(
                dim=embed_dim,
                prototype_path=asccp_prototype_path,
                text_model_name=asccp_text_model_name or text_model_name,
                use_text_derived_init=use_text_derived_asccp,
                local_files_only=asccp_text_local_files_only,
            )
            if use_asccp_prior and not direct_fusion_only
            else None
        )
        self.asccp_context_norm = nn.LayerNorm(embed_dim)
        self.asccp_context_scale = nn.Parameter(torch.tensor(0.1))
        self.modality_likelihood_decoder = (
            ModalityLikelihoodDecoder(
                dim=embed_dim,
                num_centers=num_centers,
                center_aware=True,
                dropout=dropout_rate * 0.5,
            )
            if use_modality_likelihood and not direct_fusion_only
            else None
        )
        self.coe_readout = (
            TrajectoryCoEReadout(dim=embed_dim, dropout=dropout_rate * 0.5)
            if use_coe_readout and use_posterior_refinement and not direct_fusion_only
            else None
        )
        
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
        
        # 6. 分类器（🔥 5.0优势4：激进正则化）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # 🔥 分类前必须加Dropout
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
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
        f_oct: torch.Tensor,     # [B, N, D] 或 [B, C, H, W]（如果使用分层）
        f_colpo: torch.Tensor,   # [B, N, D] 或 [B, C, H, W]（如果使用分层）
        image_names: List[str],  # 图像文件名列表（用于VLM检索）
        clinical_info: Optional[List[str]] = None,  # 临床信息（可选）
        center_labels: Optional[torch.Tensor] = None,  # 中心标签
        labels: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,  # 🔥 5.0新增：临床特征向量 [B, 7]
        return_loss_components: bool = False,
        current_beta: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Logic Loop (Enhanced with 5.0 advantages)
        
        Args:
            f_oct: OCT图像特征 [B, N, D] 或原始图像 [B, C, H, W]
            f_colpo: Colposcopy图像特征 [B, N, D] 或原始图像 [B, C, H, W]
            image_names: 图像文件名列表（必需）
            clinical_info: 临床信息列表（可选）
            center_labels: 中心标签（用于对抗损失和噪声感知）
            clinical_features: 临床特征向量 [B, 7]（5.0新增，用于ClinicalEvolver）
            return_loss_components: 是否返回损失组件
            current_beta: 当前beta值（用于Visual Notes）
        """
        output = {}
        B = f_oct.shape[0]
        device = f_oct.device
        
        # 验证必需参数
        if image_names is None:
            raise ValueError("image_names不能为None！")
        
        if not isinstance(image_names, list):
            image_names = [image_names] if isinstance(image_names, str) else list(image_names)
        
        if len(image_names) != B:
            raise ValueError(f"image_names长度({len(image_names)})与batch大小({B})不匹配！")
        
        # 🔥 5.0优势：分层多尺度特征提取
        if self.use_hierarchical and self.visual_encoder is not None and len(f_oct.shape) == 4:
            # 使用原始图像，通过HierarchicalViT提取分层特征
            # 合并OCT和Colposcopy（简单平均或拼接）
            # 这里假设f_oct是主要模态，f_colpo作为辅助
            images = f_oct  # [B, C, H, W]
            vis_feats_list = self.visual_encoder(images)  # List of [B, N, D]
            
            # 初始化临床状态
            if clinical_features is not None:
                clin_state = self.clinical_init(clinical_features)  # [B, hidden_dim]
            else:
                # 如果没有提供clinical_features，使用零向量
                clin_state = torch.zeros(B, self.embed_dim, device=device)
            
            final_feat = None
            all_noise_probs = []
            
            # 🔥 5.0优势：分层循环推理（NA-mHC + ClinicalEvolver）
            for i in range(self.num_stages):
                feat = vis_feats_list[i]  # [B, N, D]
                
                # (1) NA-mHC融合（噪声感知）
                if self.use_noise_aware and self.mhc_layers is not None:
                    feat_fused, noise_prob = self.mhc_layers[i](feat, clin_state, center_labels)
                    all_noise_probs.append(noise_prob)
                else:
                    feat_fused = feat
                    noise_prob = None
                
                # 记录最后一层特征
                if i == self.num_stages - 1:
                    final_feat = feat_fused
                
                # (2) Clinical Query Evolution（除非是最后一层）
                if i < self.num_stages - 1 and self.use_clinical_evolver and self.evolvers is not None:
                    clin_state = self.evolvers[i](feat_fused, clin_state)
            
            assert final_feat is not None
            f_oct_processed = final_feat  # [B, N, D]
            f_colpo_processed = final_feat  # 简化：使用相同特征（实际可以分别处理）
            
        else:
            # 向后兼容：使用已提取的特征
            f_oct_processed = f_oct
            f_colpo_processed = f_colpo
            all_noise_probs = []
        
        # --- Step 1: 语义锚点生成（使用VLMAugmentedRetriever或静态嵌入）---
        # 仿照exp_bio3.0_improved的方案：使用note_projector处理嵌入
        if self.use_vlm_retriever and self.knowledge_retriever is not None:
            # 使用VLM Retriever获取动态知识嵌入
            note_embeds = self.knowledge_retriever(
                image_names=image_names,
                clinical_info=clinical_info,
                device=str(device)
            )  # [B, embed_dim]
        else:
            # 禁用VLM Retriever：使用可学习的静态嵌入（消融实验）
            # 仿照exp_bio3.0_improved：使用可学习参数作为note_embeds
            note_embeds = self.learnable_knowledge_base.expand(B, -1)  # [B, embed_dim]
        
        # 统一通过note_projector处理（仿照exp_bio3.0_improved）
        # 处理note_embeds维度（兼容[B, D]和[B, 1, D]）
        if note_embeds.dim() == 3:
            note_embeds = note_embeds.squeeze(1)  # [B, D]
        z_sem = self.note_projector(note_embeds)  # [B, embed_dim]
        
        # 🔥 5.0优势6：Text Adapter
        if self.text_adapter is not None:
            z_sem = self.text_adapter(z_sem)

        if clinical_features is not None:
            clinical_tensor = clinical_features.float()
            if clinical_tensor.dim() == 1:
                clinical_tensor = clinical_tensor.unsqueeze(0)
            structured_prior = self.clinical_feature_projector(clinical_tensor)
            clinical_scale = torch.clamp(self.clinical_prior_mix, min=0.0, max=1.0)
            z_sem = self.clinical_prior_norm(z_sem + clinical_scale * structured_prior)
            output["clinical_structured_prior"] = structured_prior
        
        output['z_sem'] = z_sem
        
        # --- Step 2: 视觉笔记引导的特征提取（3.1的优势）---
        f_oct_pooled, attn_oct = self.extract_features(f_oct_processed, z_sem, current_beta)
        f_colpo_pooled, attn_colpo = self.extract_features(f_colpo_processed, z_sem, current_beta)
        
        if attn_oct is not None:
            output['attn_maps'] = [attn_oct, attn_colpo]
            
        # --- Step 3: modality reliability inference / direct fusion ---
        if self.variational_reliability is not None:
            reliability = self.variational_reliability(
                {
                    "oct": f_oct_pooled,
                    "colpo": f_colpo_pooled,
                    "clinical_prior": z_sem,
                },
                center_labels=center_labels,
                sample=self.training,
            )
            f_fused = reliability["fused"]
            output["reliability"] = reliability
            output['fusion_weights'] = reliability["weights"]
        elif self.fusion_strategy == "concat":
            f_fused = self.concat_fusion(torch.cat([f_oct_pooled, f_colpo_pooled, z_sem], dim=-1))
            output['fusion_weights'] = {}
        elif self.fusion_strategy == "late":
            logits_oct = self.late_heads["oct"](f_oct_pooled)
            logits_colpo = self.late_heads["colpo"](f_colpo_pooled)
            logits_prior = self.late_heads["clinical_prior"](z_sem)
            pred = (logits_oct + logits_colpo + logits_prior) / 3.0
            output["pred"] = pred
            output["logits"] = pred
            output["late_logits"] = {
                "oct": logits_oct,
                "colpo": logits_colpo,
                "clinical_prior": logits_prior,
            }
            if return_loss_components:
                output["loss_components"] = {}
            return output
        elif self.fusion_strategy == "cross_attention":
            modality_tokens = torch.stack([f_oct_pooled, f_colpo_pooled, z_sem], dim=1)
            query = modality_tokens.mean(dim=1, keepdim=True)
            f_fused, attn_weights = self.modality_cross_attn(query, modality_tokens, modality_tokens)
            f_fused = f_fused.squeeze(1)
            output["modality_attention"] = attn_weights
            output['fusion_weights'] = {}
        elif self.fusion_strategy == "equal":
            f_fused = (f_oct_pooled + f_colpo_pooled + z_sem) / 3.0
            output['fusion_weights'] = {
                'oct': torch.ones(B, 1, device=device) / 3.0,
                'colpo': torch.ones(B, 1, device=device) / 3.0,
                'clinical_prior': torch.ones(B, 1, device=device) / 3.0,
            }
        elif self.adaptive_fusion is not None:
            f_fused, (w_oct, w_colpo) = self.adaptive_fusion(f_oct_pooled, f_colpo_pooled)
            output['fusion_weights'] = {'oct': w_oct, 'colpo': w_colpo}
        else:
            f_fused = (f_oct_pooled + f_colpo_pooled) / 2.0
            output['fusion_weights'] = {'oct': torch.ones(B, 1, device=device) * 0.5, 
                                        'colpo': torch.ones(B, 1, device=device) * 0.5}

        if self.direct_fusion_only:
            pred = self.classifier(f_fused)
            output['pred'] = pred
            output['logits'] = pred
            if return_loss_components:
                loss_dict = {}
                if self.variational_reliability is not None and "reliability" in output:
                    loss_dict["L_reliability_kl"] = output["reliability"]["kl"]
                output["loss_components"] = loss_dict
            return output

        # --- Step 4: explicit cached-feature posterior refinement ---
        modality_targets = {
            "oct": f_oct_pooled,
            "colpo": f_colpo_pooled,
            "clinical_prior": z_sem,
        }
        posterior = None
        if self.posterior_refiner is not None:
            refine_features = modality_targets
            if self.variational_reliability is not None and "reliability" in output:
                samples = output["reliability"].get("samples", {})
                refine_features = {
                    "oct": samples.get("oct", f_oct_pooled),
                    "colpo": samples.get("colpo", f_colpo_pooled),
                    "clinical_prior": samples.get("clinical_prior", z_sem),
                }
            posterior = self.posterior_refiner(
                clinical_prior=refine_features["clinical_prior"],
                colpo=refine_features["colpo"],
                oct_feat=refine_features["oct"],
                center_labels=center_labels,
                update_memory=self.training,
            )
            output["posterior_trajectory"] = posterior["trajectory"]
            output["posterior_evidence"] = posterior["evidence"]
            f_fused = self.posterior_mix_norm(f_fused + posterior["z_final"])
            if self.coe_readout is not None:
                output["coe_readout"] = self.coe_readout(
                    posterior["trajectory"],
                    posterior["evidence"],
                    labels=labels if self.use_coe_supervision else None,
                    clinical_info=clinical_info,
                )
        
        # --- Step 5: 双头因果解耦 ---
        if self.use_dual:
            z_causal, z_noise = self.dual_head(f_fused)
        else:
            z_causal = self.dual_head(f_fused)
            z_noise = None

        asccp_prior = None
        if self.asccp_prior is not None:
            asccp_prior = self.asccp_prior(z_causal)
            scale = torch.clamp(self.asccp_context_scale, min=0.0, max=1.0)
            z_causal = self.asccp_context_norm(z_causal + scale * asccp_prior["context"])
            output["asccp_prior"] = asccp_prior

        output['z_causal'] = z_causal
        output['z_noise'] = z_noise
        
        # --- Step 6: 最终诊断（语义-因果特征融合）---
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
        
        # 🔥 5.0优势：返回噪声概率图（用于Noise Regularization Loss）
        if all_noise_probs:
            output['noise_probs'] = all_noise_probs
        
        # --- Step 6: Loss Calculation (Logic Loop)（3.1的优势：显式对齐）---
        if return_loss_components:
            loss_dict = {}
            
            # 6.1 OT Loss
            if self.use_ot:
                loss_dict['L_ot'] = self.ot_loss(z_causal, z_sem)

            if posterior is not None:
                loss_dict["L_posterior_smooth"] = posterior["smooth_loss"]

            if asccp_prior is not None:
                loss_dict["L_asccp_ot"] = asccp_prior["ot_loss"]
                loss_dict["asccp_entropy"] = asccp_prior["entropy"].detach()

            if self.modality_likelihood_decoder is not None:
                likelihood_state = posterior["z_final"] if posterior is not None else z_causal
                modality_likelihood = self.modality_likelihood_decoder(
                    likelihood_state,
                    modality_targets,
                    center_labels=center_labels,
                )
                output["modality_likelihood"] = modality_likelihood
                loss_dict["L_modality_likelihood"] = modality_likelihood["loss"]

            if self.variational_reliability is not None and "reliability" in output:
                loss_dict["L_reliability_kl"] = output["reliability"]["kl"]

            coe_readout = output.get("coe_readout")
            if isinstance(coe_readout, dict) and "supervision_loss" in coe_readout:
                loss_dict["L_coe"] = coe_readout["supervision_loss"]
            
            # 🔥 5.0优势5：正交损失（解耦方式）
            if self.use_dual and z_noise is not None:
                zc = F.normalize(z_causal, dim=1)
                zn = F.normalize(z_noise, dim=1)
                loss_dict['L_ortho'] = torch.mean(torch.abs(torch.sum(zc * zn, dim=1)))
            
            # 🔥 5.0优势：Noise Regularization Loss
            if 'noise_probs' in output and output['noise_probs']:
                # 鼓励噪声概率图稀疏（大部分patch应该是干净的）
                noise_reg_loss = sum([torch.mean(noise_prob) for noise_prob in output['noise_probs']])
                loss_dict['L_noise'] = noise_reg_loss / len(output['noise_probs'])
            
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
    """Factory function to create BioCOT_v3_2 model (整合5.0优势)"""
    # 检查是否禁用VLM Retriever
    use_vlm_retriever = getattr(config, 'use_vlm_retriever', True)
    
    # 如果禁用VLM Retriever，vlm_json_path可以为None
    vlm_json_path = getattr(config, 'vlm_json_path', None) if use_vlm_retriever else None
    
    model = BioCOT_v3_2(
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_centers=config.num_centers,
        input_dim=config.input_dim,
        vlm_json_path=vlm_json_path,
        text_model_name=getattr(config, 'text_model_name', "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        use_visual_notes=config.use_visual_notes,
        use_ot=config.use_ot,
        use_dual=config.use_dual,
        use_cross_attn=config.use_cross_attn,
        use_adaptive_gating=getattr(config, 'use_adaptive_gating', True),
        warmup_epochs=config.warmup_epochs,
        hidden_dim=getattr(config, 'hidden_dim', 768),
        # 🔥 5.0新增参数
        use_hierarchical=getattr(config, 'use_hierarchical', True),
        extract_layers=getattr(config, 'extract_layers', (2, 5, 8, 11)),
        vit_pretrained=getattr(config, 'vit_pretrained', True),
        drop_path_rate=getattr(config, 'drop_path_rate', 0.2),
        dropout_rate=getattr(config, 'dropout_rate', 0.4),
        use_noise_aware=getattr(config, 'use_noise_aware', True),
        use_clinical_evolver=getattr(config, 'use_clinical_evolver', True),
        mhc_latent_dim=getattr(config, 'mhc_latent_dim', 256),
        sinkhorn_iters=getattr(config, 'sinkhorn_iters', 3),
        mhc_epsilon=getattr(config, 'mhc_epsilon', 0.05),
        use_text_adapter=getattr(config, 'use_text_adapter', True),
        use_variational_reliability=getattr(config, 'use_variational_reliability', True),
        use_center_aware_reliability=getattr(config, 'use_center_aware_reliability', True),
        fusion_strategy=getattr(config, 'fusion_strategy', 'gated'),
        direct_fusion_only=getattr(config, 'direct_fusion_only', False),
        use_posterior_refinement=getattr(config, 'use_posterior_refinement', True),
        use_asccp_prior=getattr(config, 'use_asccp_prior', True),
        use_modality_likelihood=getattr(config, 'use_modality_likelihood', True),
        use_coe_readout=getattr(config, 'use_coe_readout', True),
        use_coe_supervision=getattr(config, 'use_coe_supervision', True),
        asccp_prototype_path=getattr(config, 'asccp_prototype_path', None),
        use_text_derived_asccp=getattr(config, 'use_text_derived_asccp', True),
        asccp_text_model_name=getattr(config, 'asccp_text_model_name', None),
        asccp_text_local_files_only=getattr(config, 'asccp_text_local_files_only', True),
    )
    # 设置是否使用VLM Retriever（用于消融实验）
    model.use_vlm_retriever = use_vlm_retriever
    return model
