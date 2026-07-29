"""Active Information Acquisition v2.

The model separates risk estimation from source acquisition. Frozen semantic
queries add a gated residual to a capacity-matched patch pooling path. A
budget-conditioned value-of-information policy uses only the currently
acquired state and an optional source-training-only prototype memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_STOP = 0
ACTION_COLPOSCOPY = 1
ACTION_OCT = 2
ACTION_NAMES = ("stop", "add_colposcopy", "add_oct")


@dataclass(frozen=True)
class ActiveFusionV2Config:
    patch_dim: int = 768
    vlm_dim: int = 1536
    case_vlm_dim: int = 4096
    clinical_dim: int = 14
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.15
    query_mode: str = "qwen"  # qwen, no_query, random
    fusion_mode: str = "concat"  # concat, late, gated, cross_attention, dmome
    query_seed: int = 20260729
    use_memory: bool = True
    use_bicer: bool = True
    memory_prototypes: int = 32
    memory_temperature: float = 0.15
    memory_gate_init: float = -2.0
    intervention_fraction: float = 0.20
    bicer_margin: float = 0.03
    pos_weight: float = 1.0


class SemanticResidualQueryEncoder(nn.Module):
    """Capacity-matched base pooling with an optional semantic residual."""

    def __init__(self, config: ActiveFusionV2Config, concept_tokens: torch.Tensor):
        super().__init__()
        self.config = config
        if concept_tokens.ndim != 2 or concept_tokens.shape[1] != config.vlm_dim:
            raise ValueError("concept_tokens have an invalid shape")
        if config.query_mode == "random":
            generator = torch.Generator().manual_seed(config.query_seed)
            randomized = torch.randn(
                concept_tokens.shape,
                generator=generator,
                dtype=torch.float32,
            )
            randomized = (
                randomized * concept_tokens.float().std().clamp_min(1e-6)
                + concept_tokens.float().mean()
            )
            concept_tokens = randomized
        self.register_buffer("concept_tokens", concept_tokens.float())
        self.patch_norm = nn.LayerNorm(config.patch_dim)
        self.patch_proj = nn.Linear(config.patch_dim, config.hidden_dim)
        self.base_score = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.query_proj = nn.Linear(config.vlm_dim, config.hidden_dim, bias=False)
        self.case_vlm_norm = nn.LayerNorm(config.case_vlm_dim)
        self.case_vlm_proj = nn.Linear(config.case_vlm_dim, config.hidden_dim, bias=False)
        self.semantic_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.semantic_residual = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -1.5)
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(
        self,
        patches: torch.Tensor,
        case_semantic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if patches.ndim != 3 or patches.shape[-1] != self.config.patch_dim:
            raise ValueError("patches must have shape [B,N,patch_dim]")
        keys = self.patch_proj(self.patch_norm(patches.float()))
        base_logits = self.base_score(keys).squeeze(-1)
        base_attention = torch.softmax(base_logits, dim=-1)
        base = torch.sum(keys * base_attention.unsqueeze(-1), dim=1)
        if self.config.query_mode == "no_query":
            semantic = torch.zeros_like(base)
            semantic_attention = base_attention
            gate = torch.zeros(base.shape[0], device=base.device, dtype=base.dtype)
        elif self.config.query_mode in {"case_vlm", "generic_vlm", "shuffled_vlm"}:
            if case_semantic is None:
                semantic = torch.zeros_like(base)
                semantic_attention = base_attention
                gate = torch.zeros(base.shape[0], device=base.device, dtype=base.dtype)
            else:
                if case_semantic.shape != (patches.shape[0], self.config.case_vlm_dim):
                    raise ValueError("case_semantic has an invalid shape")
                semantic = self.semantic_residual(
                    self.case_vlm_proj(self.case_vlm_norm(case_semantic.float()))
                )
                semantic_attention = base_attention
                gate = torch.sigmoid(
                    self.gate(torch.cat([base, semantic], dim=-1))
                ).squeeze(-1)
        else:
            queries = self.query_proj(self.concept_tokens).unsqueeze(0).expand(
                patches.shape[0], -1, -1
            )
            attended, weights = self.semantic_attention(
                queries,
                keys,
                keys,
                need_weights=True,
                average_attn_weights=True,
            )
            semantic = self.semantic_residual(attended.mean(dim=1))
            semantic_attention = weights.mean(dim=1)
            gate = torch.sigmoid(self.gate(torch.cat([base, semantic], dim=-1))).squeeze(-1)
        gated_semantic = gate.unsqueeze(-1) * semantic
        evidence = self.output(torch.cat([base, gated_semantic], dim=-1))
        return evidence, {
            "attention": semantic_attention,
            "base_attention": base_attention,
            "semantic_gate": gate,
            "semantic_residual": semantic,
        }


class BoundedUtilityMemory(nn.Module):
    """Deterministic source-only utility prototypes."""

    def __init__(self, key_dim: int, config: ActiveFusionV2Config):
        super().__init__()
        self.key_dim = int(key_dim)
        self.num_prototypes = int(config.memory_prototypes)
        self.temperature = float(config.memory_temperature)
        self.register_buffer("keys", torch.zeros(self.num_prototypes, key_dim))
        self.register_buffer("values", torch.zeros(self.num_prototypes, 2))
        self.register_buffer("counts", torch.zeros(self.num_prototypes))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, keys: torch.Tensor, values: torch.Tensor, iterations: int = 20) -> None:
        if keys.ndim != 2 or keys.shape[1] != self.key_dim:
            raise ValueError("invalid memory keys")
        if values.shape != (keys.shape[0], 2):
            raise ValueError("invalid memory values")
        if len(keys) < self.num_prototypes:
            raise ValueError("not enough source rows for memory")
        x = F.normalize(keys.detach().float(), dim=-1)
        v = values.detach().float()
        selected = [0]
        minimum = 1.0 - x @ x[0]
        for _ in range(1, self.num_prototypes):
            index = int(torch.argmax(minimum).item())
            selected.append(index)
            minimum = torch.minimum(minimum, 1.0 - x @ x[index])
        centroids = x[selected].clone()
        assignments = torch.zeros(len(x), dtype=torch.long, device=x.device)
        for _ in range(iterations):
            assignments = torch.argmax(x @ centroids.T, dim=1)
            updated = []
            for cluster in range(self.num_prototypes):
                mask = assignments == cluster
                updated.append(x[mask].mean(0) if mask.any() else centroids[cluster])
            new = F.normalize(torch.stack(updated), dim=-1)
            if torch.max(torch.abs(new - centroids)).item() < 1e-5:
                centroids = new
                break
            centroids = new
        global_value = v.mean(0)
        prototype_values = []
        counts = []
        for cluster in range(self.num_prototypes):
            mask = assignments == cluster
            counts.append(float(mask.sum().item()))
            prototype_values.append(v[mask].mean(0) if mask.any() else global_value)
        self.keys.copy_(centroids.to(self.keys))
        self.values.copy_(torch.stack(prototype_values).to(self.values))
        self.counts.copy_(torch.tensor(counts, device=self.counts.device))
        self.fitted.fill_(True)

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.fitted.item()):
            return (
                torch.zeros(len(query), 2, device=query.device, dtype=query.dtype),
                torch.zeros(
                    len(query),
                    self.num_prototypes,
                    device=query.device,
                    dtype=query.dtype,
                ),
            )
        weights = torch.softmax(
            F.normalize(query.float(), dim=-1) @ self.keys.T / self.temperature,
            dim=-1,
        )
        return (weights @ self.values).to(query.dtype), weights.to(query.dtype)


class ActiveFusionV2(nn.Module):
    """Budget-conditioned value-of-information fusion model."""

    def __init__(
        self,
        config: ActiveFusionV2Config,
        concept_tokens: Mapping[str, torch.Tensor],
    ):
        super().__init__()
        self.config = config
        if set(concept_tokens) != {"colposcopy", "oct"}:
            raise ValueError("concept tokens must contain colposcopy and oct")
        self.col_encoder = SemanticResidualQueryEncoder(
            config,
            concept_tokens["colposcopy"],
        )
        self.oct_encoder = SemanticResidualQueryEncoder(
            config,
            concept_tokens["oct"],
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        state_input = config.hidden_dim * 3 + 3
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.modality_risk_heads = nn.ModuleList(
            [nn.Linear(config.hidden_dim, 1) for _ in range(3)]
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.fusion_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.modality_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.LayerNorm(config.hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.pooled_risk_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.case_semantic_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )
        self.utility_memory = BoundedUtilityMemory(config.hidden_dim + 3, config)
        self.memory_gate = nn.Parameter(torch.tensor(float(config.memory_gate_init)))

    def encode_modalities(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        col_case_semantic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        clinical_state = self.clinical_encoder(clinical.float())
        col, col_diag = self.col_encoder(col_patches, col_case_semantic)
        oct_evidence, oct_diag = self.oct_encoder(oct_patches)
        return {
            "clinical_evidence": clinical_state,
            "colposcopy_evidence": col,
            "oct_evidence": oct_evidence,
            "attention_colposcopy": col_diag["attention"],
            "attention_oct": oct_diag["attention"],
            "base_attention_colposcopy": col_diag["base_attention"],
            "base_attention_oct": oct_diag["base_attention"],
            "semantic_gate_colposcopy": col_diag["semantic_gate"],
            "semantic_gate_oct": oct_diag["semantic_gate"],
            "case_semantic_logit": self.case_semantic_head(
                col_diag["semantic_residual"]
            ).squeeze(-1),
        }

    def state_from_mask(
        self,
        encoded: Mapping[str, torch.Tensor],
        acquired_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = acquired_mask.float()
        clinical = encoded["clinical_evidence"]
        col = encoded["colposcopy_evidence"] * mask[:, 1:2]
        oct_evidence = encoded["oct_evidence"] * mask[:, 2:3]
        return self.state_encoder(torch.cat([clinical, col, oct_evidence, mask], dim=-1))

    def state_outputs(
        self,
        encoded: Mapping[str, torch.Tensor],
        acquired_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state = self.state_from_mask(encoded, acquired_mask)
        mode = self.config.fusion_mode
        evidence = torch.stack(
            [
                encoded["clinical_evidence"],
                encoded["colposcopy_evidence"],
                encoded["oct_evidence"],
            ],
            dim=1,
        )
        available = acquired_mask.bool()
        if mode == "concat":
            logit = self.risk_head(state).squeeze(-1)
        elif mode == "late":
            logits = torch.stack(
                [
                    head(evidence[:, index]).squeeze(-1)
                    for index, head in enumerate(self.modality_risk_heads)
                ],
                dim=-1,
            )
            logit = (
                logits.masked_fill(~available, 0.0).sum(dim=-1)
                / available.sum(dim=-1).clamp_min(1)
            )
        elif mode == "gated":
            gate = self.fusion_gate(evidence).squeeze(-1)
            weight = torch.softmax(gate.masked_fill(~available, -1e9), dim=-1)
            pooled = torch.sum(evidence * weight.unsqueeze(-1), dim=1)
            logit = self.pooled_risk_head(pooled).squeeze(-1)
        elif mode == "cross_attention":
            query = encoded["clinical_evidence"].unsqueeze(1)
            attended, _ = self.fusion_attention(
                query,
                evidence,
                evidence,
                key_padding_mask=~available,
                need_weights=False,
            )
            logit = self.pooled_risk_head(
                attended.squeeze(1) + encoded["clinical_evidence"]
            ).squeeze(-1)
        elif mode == "dmome":
            expert = torch.stack(
                [
                    module(evidence[:, index])
                    for index, module in enumerate(self.modality_experts)
                ],
                dim=1,
            )
            gate = self.fusion_gate(evidence).squeeze(-1)
            weight = torch.softmax(gate.masked_fill(~available, -1e9), dim=-1)
            pooled = torch.sum(expert * weight.unsqueeze(-1), dim=1)
            logit = self.pooled_risk_head(pooled).squeeze(-1)
        else:
            raise ValueError(f"unknown fusion_mode {mode}")
        key = torch.cat([state, acquired_mask.float()], dim=-1)
        predicted = self.utility_head(key)
        if self.config.use_memory:
            memory, weights = self.utility_memory.retrieve(key)
            utility = predicted + torch.sigmoid(self.memory_gate) * memory
        else:
            memory = torch.zeros_like(predicted)
            weights = torch.zeros(
                len(state),
                self.config.memory_prototypes,
                device=state.device,
                dtype=state.dtype,
            )
            utility = predicted
        invalid = acquired_mask[:, 1:].bool()
        utility = utility.masked_fill(invalid, -1e9)
        return {
            "state": state,
            "logit": logit,
            "utility": utility,
            "predicted_utility": predicted,
            "memory_utility": memory,
            "memory_weights": weights,
            "memory_key": key,
        }

    def all_subset_outputs(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        col_case_semantic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(
            clinical,
            col_patches,
            oct_patches,
            col_case_semantic,
        )
        batch = len(clinical)
        masks = {
            "clinical": torch.tensor([1, 0, 0], device=clinical.device).expand(batch, -1),
            "clinical_colposcopy": torch.tensor([1, 1, 0], device=clinical.device).expand(batch, -1),
            "clinical_oct": torch.tensor([1, 0, 1], device=clinical.device).expand(batch, -1),
            "all": torch.tensor([1, 1, 1], device=clinical.device).expand(batch, -1),
        }
        result: Dict[str, torch.Tensor] = {**encoded}
        for name, mask in masks.items():
            output = self.state_outputs(encoded, mask)
            result[f"mask_{name}"] = mask
            for key, value in output.items():
                result[f"{key}_{name}"] = value
        return result

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor(
            self.config.pos_weight,
            device=logits.device,
            dtype=logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            pos_weight=pos_weight,
        )

    @staticmethod
    def _sample_utility(
        current_logit: torch.Tensor,
        next_logit: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        current = F.binary_cross_entropy_with_logits(
            current_logit,
            labels.float(),
            reduction="none",
        )
        following = F.binary_cross_entropy_with_logits(
            next_logit,
            labels.float(),
            reduction="none",
        )
        return (current - following).clamp(-2.0, 2.0).detach()

    def utility_targets(
        self,
        outputs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        initial = torch.stack(
            [
                self._sample_utility(
                    outputs["logit_clinical"],
                    outputs["logit_clinical_colposcopy"],
                    labels,
                ),
                self._sample_utility(
                    outputs["logit_clinical"],
                    outputs["logit_clinical_oct"],
                    labels,
                ),
            ],
            dim=-1,
        )
        after_col = torch.stack(
            [
                torch.full_like(labels.float(), -2.0),
                self._sample_utility(
                    outputs["logit_clinical_colposcopy"],
                    outputs["logit_all"],
                    labels,
                ),
            ],
            dim=-1,
        )
        after_oct = torch.stack(
            [
                self._sample_utility(
                    outputs["logit_clinical_oct"],
                    outputs["logit_all"],
                    labels,
                ),
                torch.full_like(labels.float(), -2.0),
            ],
            dim=-1,
        )
        return {
            "clinical": initial,
            "clinical_colposcopy": after_col,
            "clinical_oct": after_oct,
        }

    @staticmethod
    def _replace_patches(patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        result = patches.clone()
        replacement = patches.mean(dim=1, keepdim=True)
        result.scatter_(
            1,
            indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1]),
            replacement.expand(-1, indices.shape[1], -1),
        )
        return result

    def bicer_loss(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        labels: torch.Tensor,
        factual: Mapping[str, torch.Tensor],
        col_case_semantic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, patch_count, _ = col_patches.shape
        count = max(1, int(round(self.config.intervention_fraction * patch_count)))
        direction = labels.float() * 2.0 - 1.0
        factual_logit = factual["logit_all"]
        modality_terms = []
        gaps = []
        for modality, patches, other in (
            ("colposcopy", col_patches, oct_patches),
            ("oct", oct_patches, col_patches),
        ):
            attention = factual[f"attention_{modality}"].detach()
            high = torch.topk(attention, count, dim=-1).indices
            low = torch.topk(-attention, count, dim=-1).indices
            base = torch.arange(count, device=patches.device).unsqueeze(0)
            offsets = torch.arange(batch, device=patches.device).unsqueeze(1)
            random_one = (base + offsets * max(1, count)) % patch_count
            random_two = (base + (offsets + batch) * max(1, count + 1)) % patch_count
            variants = [
                self._replace_patches(patches, high),
                self._replace_patches(patches, low),
                self._replace_patches(patches, random_one),
                self._replace_patches(patches, random_two),
            ]
            logits = []
            for variant in variants:
                if modality == "colposcopy":
                    altered = self.all_subset_outputs(
                        clinical, variant, other, col_case_semantic
                    )["logit_all"]
                else:
                    altered = self.all_subset_outputs(
                        clinical, other, variant, col_case_semantic
                    )["logit_all"]
                logits.append(altered)
            effects = [direction * (factual_logit - logit) for logit in logits]
            targeted = effects[0]
            matched_control = torch.stack(effects[1:], dim=0).mean(0)
            gap = targeted - matched_control
            modality_terms.append(F.relu(self.config.bicer_margin - gap).mean())
            gaps.append(gap.mean())
        return torch.stack(modality_terms).mean(), torch.stack(gaps).mean()

    def training_losses(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        labels: torch.Tensor,
        col_case_semantic: Optional[torch.Tensor] = None,
        *,
        lambda_utility: float = 0.5,
        lambda_brier: float = 0.2,
        lambda_bicer: float = 0.0,
        lambda_case_semantic: float = 0.15,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.all_subset_outputs(
            clinical,
            col_patches,
            oct_patches,
            col_case_semantic,
        )
        subset_names = ("clinical", "clinical_colposcopy", "clinical_oct", "all")
        classification = torch.stack(
            [self._classification_loss(outputs[f"logit_{name}"], labels) for name in subset_names]
        ).mean()
        brier = torch.stack(
            [
                (torch.sigmoid(outputs[f"logit_{name}"]) - labels.float()).pow(2).mean()
                for name in subset_names
            ]
        ).mean()
        targets = self.utility_targets(outputs, labels)
        utility = torch.stack(
            [
                F.smooth_l1_loss(
                    outputs[f"predicted_utility_{name}"],
                    targets[name],
                )
                for name in ("clinical", "clinical_colposcopy", "clinical_oct")
            ]
        ).mean()
        case_semantic = torch.zeros((), device=labels.device)
        if self.config.query_mode in {"case_vlm", "generic_vlm", "shuffled_vlm"}:
            case_semantic = self._classification_loss(
                outputs["case_semantic_logit"],
                labels,
            )
        monotonic = torch.zeros((), device=labels.device)
        if self.config.fusion_mode == "dmome":
            direction = labels.float() * 2.0 - 1.0
            correct = {
                name: direction * outputs[f"logit_{name}"]
                for name in subset_names
            }
            monotonic = torch.stack(
                [
                    F.relu(0.02 + correct["clinical"] - correct["clinical_colposcopy"]),
                    F.relu(0.02 + correct["clinical"] - correct["clinical_oct"]),
                    F.relu(0.02 + correct["clinical_colposcopy"] - correct["all"]),
                    F.relu(0.02 + correct["clinical_oct"] - correct["all"]),
                ],
                dim=0,
            ).mean()
        bicer = torch.zeros((), device=labels.device)
        bicer_gap = torch.zeros((), device=labels.device)
        if self.config.use_bicer and lambda_bicer > 0:
            bicer, bicer_gap = self.bicer_loss(
                clinical,
                col_patches,
                oct_patches,
                labels,
                outputs,
                col_case_semantic,
            )
        total = (
            classification
            + lambda_brier * brier
            + lambda_utility * utility
            + lambda_bicer * bicer
            + lambda_case_semantic * case_semantic
            + 0.15 * monotonic
        )
        return total, {
            "loss": total,
            "classification": classification,
            "brier": brier,
            "utility": utility,
            "case_semantic": case_semantic,
            "monotonic": monotonic,
            "bicer": bicer,
            "bicer_gap": bicer_gap,
            "outputs": outputs,
        }

    @torch.no_grad()
    def fit_utility_memory(
        self,
        batches: Iterable[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Optional[torch.Tensor],
            ]
        ],
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()
        keys = []
        values = []
        for batch in batches:
            if len(batch) == 4:
                clinical, col, oct_patches, labels = batch
                col_case_semantic = None
            else:
                clinical, col, oct_patches, labels, col_case_semantic = batch
            outputs = self.all_subset_outputs(
                clinical,
                col,
                oct_patches,
                col_case_semantic,
            )
            targets = self.utility_targets(outputs, labels)
            for name in ("clinical", "clinical_colposcopy", "clinical_oct"):
                keys.append(outputs[f"memory_key_{name}"])
                values.append(targets[name])
        all_keys = torch.cat(keys)
        all_values = torch.cat(values)
        self.utility_memory.fit(all_keys, all_values)
        self.train(was_training)
        return {
            "rows": float(len(all_keys)),
            "prototypes": float(self.utility_memory.num_prototypes),
            "nonempty_prototypes": float((self.utility_memory.counts > 0).sum().item()),
        }

    @torch.no_grad()
    def run_policy(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        col_case_semantic: Optional[torch.Tensor] = None,
        *,
        policy: str = "learned",
        cost_weight: float = 0.2,
        colposcopy_cost: float = 1.0,
        oct_cost: float = 1.0,
        uncertainty_threshold: float = 0.55,
        random_seed: int = 0,
        random_acquisition_probability: float = 0.5,
        labels_for_oracle: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(
            clinical,
            col_patches,
            oct_patches,
            col_case_semantic,
        )
        batch = len(clinical)
        device = clinical.device
        acquired = torch.tensor([1, 0, 0], device=device).expand(batch, -1).clone()
        stopped = torch.zeros(batch, dtype=torch.bool, device=device)
        actions = torch.zeros(batch, 2, dtype=torch.long, device=device)
        scenario_cost = torch.zeros(batch, device=device)
        generator = torch.Generator(device=device).manual_seed(int(random_seed))
        oracle_targets = None
        if labels_for_oracle is not None:
            all_outputs = self.all_subset_outputs(
                clinical,
                col_patches,
                oct_patches,
                col_case_semantic,
            )
            oracle_targets = self.utility_targets(all_outputs, labels_for_oracle)
        for step in range(2):
            current = self.state_outputs(encoded, acquired)
            utility = current["utility"].clone()
            available_cost = torch.tensor(
                [colposcopy_cost, oct_cost],
                device=device,
                dtype=utility.dtype,
            )
            if policy == "learned":
                scores = utility - float(cost_weight) * available_cost
                best_value, best_index = scores.max(dim=-1)
                action = best_index + 1
                action = torch.where(
                    best_value > 0,
                    action,
                    torch.full_like(action, ACTION_STOP),
                )
            elif policy == "uncertainty":
                probability = torch.sigmoid(current["logit"])
                uncertainty = 1.0 - torch.abs(probability - 0.5) * 2.0
                candidate = torch.where(
                    acquired[:, 1].bool(),
                    torch.full((batch,), ACTION_OCT, device=device),
                    torch.full((batch,), ACTION_COLPOSCOPY, device=device),
                )
                action = torch.where(
                    uncertainty > uncertainty_threshold,
                    candidate,
                    torch.full_like(candidate, ACTION_STOP),
                )
            elif policy == "cheapest_first":
                candidate = (
                    ACTION_COLPOSCOPY
                    if colposcopy_cost <= oct_cost
                    else ACTION_OCT
                )
                candidate_tensor = torch.full((batch,), candidate, device=device)
                invalid = acquired.gather(1, candidate_tensor.unsqueeze(1)).squeeze(1).bool()
                fallback = torch.full(
                    (batch,),
                    ACTION_OCT if candidate == ACTION_COLPOSCOPY else ACTION_COLPOSCOPY,
                    device=device,
                )
                candidate_tensor = torch.where(invalid, fallback, candidate_tensor)
                probability = torch.sigmoid(current["logit"])
                uncertainty = 1.0 - torch.abs(probability - 0.5) * 2.0
                action = torch.where(
                    uncertainty > uncertainty_threshold,
                    candidate_tensor,
                    torch.full_like(candidate_tensor, ACTION_STOP),
                )
            elif policy == "fixed_order" or policy == "static_all":
                action = torch.where(
                    acquired[:, 1].bool(),
                    torch.full((batch,), ACTION_OCT, device=device),
                    torch.full((batch,), ACTION_COLPOSCOPY, device=device),
                )
            elif policy == "clinical_only":
                action = torch.full((batch,), ACTION_STOP, device=device)
            elif policy == "random":
                acquire = (
                    torch.rand(batch, generator=generator, device=device)
                    < random_acquisition_probability
                )
                random_index = torch.randint(0, 2, (batch,), generator=generator, device=device)
                candidate = random_index + 1
                invalid = acquired.gather(1, candidate.unsqueeze(1)).squeeze(1).bool()
                candidate = torch.where(
                    invalid,
                    torch.where(
                        acquired[:, 1].bool(),
                        torch.full_like(candidate, ACTION_OCT),
                        torch.full_like(candidate, ACTION_COLPOSCOPY),
                    ),
                    candidate,
                )
                action = torch.where(acquire, candidate, torch.full_like(candidate, ACTION_STOP))
            elif policy == "oracle":
                if oracle_targets is None:
                    raise ValueError("oracle requires labels_for_oracle")
                if step == 0:
                    target = oracle_targets["clinical"]
                else:
                    target = torch.where(
                        acquired[:, 1:2].bool(),
                        oracle_targets["clinical_colposcopy"],
                        oracle_targets["clinical_oct"],
                    )
                score = target - float(cost_weight) * available_cost
                best_value, best_index = score.max(dim=-1)
                action = torch.where(
                    best_value > 0,
                    best_index + 1,
                    torch.full_like(best_index, ACTION_STOP),
                )
            else:
                raise ValueError(f"unknown policy {policy}")
            invalid = acquired.gather(1, action.unsqueeze(1)).squeeze(1).bool()
            action = torch.where(invalid | stopped, torch.full_like(action, ACTION_STOP), action)
            actions[:, step] = action
            stopped = stopped | (action == ACTION_STOP)
            col_mask = action == ACTION_COLPOSCOPY
            oct_mask = action == ACTION_OCT
            acquired[col_mask, 1] = 1
            acquired[oct_mask, 2] = 1
            scenario_cost[col_mask] += float(colposcopy_cost)
            scenario_cost[oct_mask] += float(oct_cost)
        final = self.state_outputs(encoded, acquired)
        return {
            "logit": final["logit"],
            "probability": torch.sigmoid(final["logit"]),
            "actions": actions,
            "acquired_mask": acquired,
            "acquisition_count": acquired[:, 1:].sum(dim=-1).float(),
            "cost": scenario_cost,
            "final_state": final["state"],
        }


__all__ = [
    "ACTION_STOP",
    "ACTION_COLPOSCOPY",
    "ACTION_OCT",
    "ACTION_NAMES",
    "ActiveFusionV2",
    "ActiveFusionV2Config",
    "BoundedUtilityMemory",
    "SemanticResidualQueryEncoder",
]
