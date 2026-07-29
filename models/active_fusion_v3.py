"""Shift-aware latent active fusion with a frozen VLM semantic anchor.

The VLM embedding is not concatenated into the prediction head. It anchors the
colposcopy shared latent and modulates evidence precision through visual-
semantic agreement. Acquisition uses a lower confidence bound on
counterfactual utility. A separate CIN3+ head can force acquisition when the
source-calibrated safety state is indeterminate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.active_fusion_v2 import (
    ACTION_COLPOSCOPY,
    ACTION_NAMES,
    ACTION_OCT,
    ACTION_STOP,
    ActiveFusionV2,
    ActiveFusionV2Config,
)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float) -> torch.Tensor:
    return _GradientReversal.apply(value, coefficient)


@dataclass(frozen=True)
class ActiveFusionV3Config(ActiveFusionV2Config):
    query_mode: str = "no_query"
    fusion_mode: str = "latent_precision"
    use_memory: bool = False
    semantic_anchor_mode: str = "medical"  # medical, generic, shuffled, none
    use_semantic_anchor: bool = True
    use_domain_adversary: bool = True
    use_reliability: bool = True
    use_precision_fusion: bool = True
    use_utility_lcb: bool = True
    use_safety_gate: bool = True
    num_centres: int = 5
    private_dim: int = 64
    private_evidence_weight: float = 0.25
    adversary_coefficient: float = 0.25
    utility_lcb_beta: float = 0.5
    safety_error_beta: float = 1.0
    safety_utility_weight: float = 0.50
    anchor_temperature: float = 0.15
    use_concept_distillation: bool = False
    use_concept_token_fusion: bool = False
    use_concept_risk_expert: bool = False
    concept_expert_gate_mode: str = "learned"  # learned, constant, quality
    use_vlm_quality_reliability: bool = False


class ActiveFusionV3(ActiveFusionV2):
    """Shared-private latent fusion and reliable active acquisition."""

    def __init__(
        self,
        config: ActiveFusionV3Config,
        concept_tokens: Mapping[str, torch.Tensor],
    ):
        base_config = ActiveFusionV2Config(
            patch_dim=config.patch_dim,
            vlm_dim=config.vlm_dim,
            case_vlm_dim=config.case_vlm_dim,
            clinical_dim=config.clinical_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            query_mode="no_query",
            fusion_mode="concat",
            query_seed=config.query_seed,
            use_memory=False,
            use_bicer=config.use_bicer,
            intervention_fraction=config.intervention_fraction,
            bicer_margin=config.bicer_margin,
            pos_weight=config.pos_weight,
        )
        super().__init__(base_config, concept_tokens)
        self.config = config
        h = config.hidden_dim
        p = config.private_dim

        def shared_projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, h),
                nn.GELU(),
                nn.LayerNorm(h),
            )

        def private_projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, p),
                nn.GELU(),
                nn.LayerNorm(p),
                nn.Linear(p, h),
            )

        self.shared_projectors = nn.ModuleList(
            [shared_projector() for _ in range(3)]
        )
        self.private_projectors = nn.ModuleList(
            [private_projector() for _ in range(3)]
        )
        self.anchor_projector = nn.Sequential(
            nn.LayerNorm(config.case_vlm_dim),
            nn.Linear(config.case_vlm_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.concept_decoder = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, config.case_vlm_dim),
            nn.Sigmoid(),
        )
        self.concept_gate = nn.Sequential(
            nn.Linear(h * 2, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
        )
        nn.init.constant_(self.concept_gate[-1].bias, -1.0)
        # Deliberately low-rank: the seven bounded concepts passed a linear
        # source probe, whereas a high-dimensional nonlinear expert did not.
        self.concept_cin2_head = nn.Linear(config.case_vlm_dim, 1)
        self.concept_cin3_head = nn.Linear(config.case_vlm_dim, 1)
        self.concept_expert_scale_logit = nn.Parameter(torch.tensor(-1.10))
        self.reliability_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h * 2, h // 2),
                    nn.GELU(),
                    nn.Linear(h // 2, 1),
                )
                for _ in range(3)
            ]
        )
        self.latent_state_encoder = nn.Sequential(
            nn.Linear(h * 3 + 3, h * 2),
            nn.LayerNorm(h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
            nn.LayerNorm(h),
        )
        self.cin3_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, 1),
        )
        self.safety_error_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
            nn.Softplus(),
        )
        self.utility_logvar_head = nn.Sequential(
            nn.Linear(h + 3, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, 2),
        )
        self.shared_centre_discriminator = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, config.num_centres),
        )
        self.private_centre_classifier = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, config.num_centres),
        )

    def encode_modalities(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        col_case_semantic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base = super().encode_modalities(
            clinical,
            col_patches,
            oct_patches,
            None,
        )
        raw = [
            base["clinical_evidence"],
            base["colposcopy_evidence"],
            base["oct_evidence"],
        ]
        shared = [
            F.normalize(module(value), dim=-1)
            for module, value in zip(self.shared_projectors, raw)
        ]
        private = [
            module(value)
            for module, value in zip(self.private_projectors, raw)
        ]
        if col_case_semantic is None:
            anchor = torch.zeros_like(shared[1])
            raw_semantic = torch.zeros(
                len(clinical),
                self.config.case_vlm_dim,
                device=clinical.device,
                dtype=clinical.dtype,
            )
            anchor_agreement = torch.zeros(
                len(clinical), device=clinical.device, dtype=clinical.dtype
            )
        else:
            raw_semantic = col_case_semantic.float()
            anchor = F.normalize(
                self.anchor_projector(raw_semantic),
                dim=-1,
            )
            anchor_agreement = (shared[1] * anchor).sum(dim=-1)

        precision_logits = []
        for index in range(3):
            context = shared[0] if index else shared[index]
            precision_logits.append(
                self.reliability_heads[index](
                    torch.cat([shared[index], context], dim=-1)
                ).squeeze(-1)
            )
        if self.config.use_semantic_anchor:
            precision_logits[1] = precision_logits[1] + anchor_agreement
        if (
            self.config.use_vlm_quality_reliability
            and col_case_semantic is not None
            and col_case_semantic.shape[1] >= 7
        ):
            # Both structured prompts reserve positions 5 and 6 for image
            # quality and assessment uncertainty on [0,1].
            quality_margin = raw_semantic[:, 5] - raw_semantic[:, 6]
            precision_logits[1] = precision_logits[1] + 2.0 * quality_margin
        if self.config.use_reliability and self.config.use_precision_fusion:
            precision = [
                F.softplus(value).clamp(0.05, 10.0)
                for value in precision_logits
            ]
        else:
            precision = [torch.ones_like(value) for value in precision_logits]

        evidence = [
            shared[index]
            + float(self.config.private_evidence_weight) * private[index]
            for index in range(3)
        ]
        concept_gate = torch.zeros(
            len(clinical), device=clinical.device, dtype=clinical.dtype
        )
        if (
            self.config.use_semantic_anchor
            and self.config.use_concept_token_fusion
            and col_case_semantic is not None
        ):
            concept_gate = torch.sigmoid(
                self.concept_gate(
                    torch.cat([shared[1], anchor], dim=-1)
                )
            ).squeeze(-1)
            evidence[1] = evidence[1] + concept_gate.unsqueeze(-1) * anchor
        elif (
            self.config.use_concept_risk_expert
            and self.config.concept_expert_gate_mode == "constant"
        ):
            concept_gate = torch.sigmoid(
                self.concept_expert_scale_logit
            ).expand(len(clinical))
        elif (
            self.config.use_concept_risk_expert
            and self.config.concept_expert_gate_mode == "quality"
            and col_case_semantic is not None
            and col_case_semantic.shape[1] >= 7
        ):
            concept_gate = torch.sigmoid(
                4.0 * (raw_semantic[:, 5] - raw_semantic[:, 6])
            )
        shared_consensus = torch.stack(shared, dim=1).mean(dim=1)
        if (
            self.config.use_concept_risk_expert
            and self.config.use_concept_token_fusion
        ):
            shared_consensus = F.normalize(
                shared_consensus + concept_gate.unsqueeze(-1) * anchor,
                dim=-1,
            )
        private_consensus = torch.stack(private, dim=1).mean(dim=1)
        return {
            **base,
            "clinical_evidence": evidence[0],
            "colposcopy_evidence": evidence[1],
            "oct_evidence": evidence[2],
            "shared_clinical": shared[0],
            "shared_colposcopy": shared[1],
            "shared_oct": shared[2],
            "private_clinical": private[0],
            "private_colposcopy": private[1],
            "private_oct": private[2],
            "shared_consensus": shared_consensus,
            "private_consensus": private_consensus,
            "semantic_anchor": anchor,
            "raw_semantic": raw_semantic,
            "anchor_agreement": anchor_agreement,
            "concept_gate": concept_gate,
            "concept_cin2_logit": self.concept_cin2_head(raw_semantic).squeeze(-1),
            "concept_cin3_logit": self.concept_cin3_head(raw_semantic).squeeze(-1),
            "precision_clinical": precision[0],
            "precision_colposcopy": precision[1],
            "precision_oct": precision[2],
        }

    def _latent_state(
        self,
        encoded: Mapping[str, torch.Tensor],
        acquired_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = acquired_mask.float()
        values = torch.stack(
            [
                encoded["clinical_evidence"],
                encoded["colposcopy_evidence"],
                encoded["oct_evidence"],
            ],
            dim=1,
        )
        precision = torch.stack(
            [
                encoded["precision_clinical"],
                encoded["precision_colposcopy"],
                encoded["precision_oct"],
            ],
            dim=1,
        )
        available_precision = precision * mask
        normalized = (
            available_precision
            / available_precision.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        weighted = values * normalized.unsqueeze(-1)
        state = self.latent_state_encoder(
            torch.cat(
                [
                    weighted[:, 0],
                    weighted[:, 1],
                    weighted[:, 2],
                    mask,
                ],
                dim=-1,
            )
        )
        return state, normalized

    def state_from_mask(
        self,
        encoded: Mapping[str, torch.Tensor],
        acquired_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._latent_state(encoded, acquired_mask)[0]

    def state_outputs(
        self,
        encoded: Mapping[str, torch.Tensor],
        acquired_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state, precision_weight = self._latent_state(encoded, acquired_mask)
        logit = self.risk_head(state).squeeze(-1)
        cin3_logit = self.cin3_head(state).squeeze(-1)
        if self.config.use_concept_risk_expert:
            concept_weight = (
                acquired_mask[:, 1].float() * encoded["concept_gate"]
            )
            logit = (
                logit
                + concept_weight * encoded["concept_cin2_logit"]
            )
            cin3_logit = (
                cin3_logit
                + concept_weight * encoded["concept_cin3_logit"]
            )
        safety_error = self.safety_error_head(state).squeeze(-1).clamp_max(1.0)
        key = torch.cat([state, acquired_mask.float()], dim=-1)
        utility_mean = self.utility_head(key)
        utility_logvar = self.utility_logvar_head(key).clamp(-5.0, 3.0)
        utility_std = torch.exp(0.5 * utility_logvar)
        if self.config.use_reliability and self.config.use_utility_lcb:
            utility = (
                utility_mean
                - float(self.config.utility_lcb_beta) * utility_std
            )
        else:
            utility = utility_mean
        invalid = acquired_mask[:, 1:].bool()
        utility = utility.masked_fill(invalid, -1e9)
        return {
            "state": state,
            "logit": logit,
            "cin3_logit": cin3_logit,
            "safety_error": safety_error,
            "precision_weight": precision_weight,
            "utility": utility,
            "predicted_utility": utility_mean,
            "utility_logvar": utility_logvar,
            "utility_std": utility_std,
            "memory_utility": torch.zeros_like(utility_mean),
            "memory_weights": torch.zeros(
                len(state), 1, device=state.device, dtype=state.dtype
            ),
            "memory_key": key,
        }

    @staticmethod
    def _orthogonality(
        shared: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        shared = F.normalize(shared, dim=-1)
        private = F.normalize(private, dim=-1)
        return (shared * private).sum(dim=-1).pow(2).mean()

    def training_losses(
        self,
        clinical: torch.Tensor,
        col_patches: torch.Tensor,
        oct_patches: torch.Tensor,
        labels: torch.Tensor,
        col_case_semantic: Optional[torch.Tensor] = None,
        *,
        cin3_labels: Optional[torch.Tensor] = None,
        centre_labels: Optional[torch.Tensor] = None,
        lambda_utility: float = 0.5,
        lambda_brier: float = 0.2,
        lambda_bicer: float = 0.0,
        lambda_case_semantic: float = 0.0,
        lambda_anchor: float = 0.10,
        lambda_domain: float = 0.05,
        lambda_private_domain: float = 0.025,
        lambda_orthogonal: float = 0.025,
        lambda_cin3: float = 0.35,
        lambda_safety_error: float = 0.10,
        lambda_concept: float = 0.15,
        lambda_concept_risk: float = 0.20,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_total, pieces = super().training_losses(
            clinical,
            col_patches,
            oct_patches,
            labels,
            col_case_semantic,
            lambda_utility=0.0,
            lambda_brier=lambda_brier,
            lambda_bicer=lambda_bicer,
            lambda_case_semantic=0.0,
        )
        outputs = pieces["outputs"]
        device = labels.device
        subset_names = ("clinical", "clinical_colposcopy", "clinical_oct", "all")

        targets = self.utility_targets(outputs, labels)
        safety_targets = None
        if cin3_labels is not None:
            safety_targets = {
                "clinical": torch.stack(
                    [
                        self._sample_utility(
                            outputs["cin3_logit_clinical"],
                            outputs["cin3_logit_clinical_colposcopy"],
                            cin3_labels,
                        ),
                        self._sample_utility(
                            outputs["cin3_logit_clinical"],
                            outputs["cin3_logit_clinical_oct"],
                            cin3_labels,
                        ),
                    ],
                    dim=-1,
                ),
                "clinical_colposcopy": torch.stack(
                    [
                        torch.full_like(cin3_labels.float(), -2.0),
                        self._sample_utility(
                            outputs["cin3_logit_clinical_colposcopy"],
                            outputs["cin3_logit_all"],
                            cin3_labels,
                        ),
                    ],
                    dim=-1,
                ),
                "clinical_oct": torch.stack(
                    [
                        self._sample_utility(
                            outputs["cin3_logit_clinical_oct"],
                            outputs["cin3_logit_all"],
                            cin3_labels,
                        ),
                        torch.full_like(cin3_labels.float(), -2.0),
                    ],
                    dim=-1,
                ),
            }
            targets = {
                name: value
                + float(self.config.safety_utility_weight)
                * safety_targets[name]
                for name, value in targets.items()
            }
        utility_regression = torch.stack(
            [
                F.smooth_l1_loss(
                    outputs[f"predicted_utility_{name}"],
                    targets[name],
                )
                for name in ("clinical", "clinical_colposcopy", "clinical_oct")
            ]
        ).mean()
        heteroscedastic = []
        for name in ("clinical", "clinical_colposcopy", "clinical_oct"):
            residual = targets[name] - outputs[f"predicted_utility_{name}"]
            logvar = outputs[f"utility_logvar_{name}"]
            heteroscedastic.append(
                0.5 * (torch.exp(-logvar) * residual.pow(2) + logvar)
            )
        reliability = torch.stack(
            [value.mean() for value in heteroscedastic]
        ).mean()

        cin3 = torch.zeros((), device=device)
        safety_error_loss = torch.zeros((), device=device)
        if cin3_labels is not None:
            cin3 = torch.stack(
                [
                    F.binary_cross_entropy_with_logits(
                        outputs[f"cin3_logit_{name}"],
                        cin3_labels.float(),
                    )
                    for name in subset_names
                ]
            ).mean()
            safety_error_loss = torch.stack(
                [
                    F.smooth_l1_loss(
                        outputs[f"safety_error_{name}"],
                        (
                            torch.sigmoid(outputs[f"cin3_logit_{name}"])
                            - cin3_labels.float()
                        ).abs().detach(),
                    )
                    for name in subset_names
                ]
            ).mean()

        anchor = torch.zeros((), device=device)
        concept = torch.zeros((), device=device)
        concept_risk = torch.zeros((), device=device)
        if self.config.use_semantic_anchor and col_case_semantic is not None:
            anchor = (
                1.0
                - F.cosine_similarity(
                    outputs["shared_colposcopy"],
                    outputs["semantic_anchor"],
                    dim=-1,
                )
            ).mean()
            if self.config.use_concept_distillation:
                concept = F.smooth_l1_loss(
                    self.concept_decoder(outputs["shared_colposcopy"]),
                    col_case_semantic.float(),
                )
        if (
            self.config.use_concept_risk_expert
            and col_case_semantic is not None
        ):
            concept_risk = self._classification_loss(
                outputs["concept_cin2_logit"],
                labels,
            )
            if cin3_labels is not None:
                concept_risk = concept_risk + F.binary_cross_entropy_with_logits(
                    outputs["concept_cin3_logit"],
                    cin3_labels.float(),
                )

        domain = torch.zeros((), device=device)
        private_domain = torch.zeros((), device=device)
        if centre_labels is not None:
            if self.config.use_domain_adversary:
                domain_logits = self.shared_centre_discriminator(
                    gradient_reverse(
                        outputs["shared_consensus"],
                        self.config.adversary_coefficient,
                    )
                )
                domain = F.cross_entropy(domain_logits, centre_labels)
            private_logits = self.private_centre_classifier(
                outputs["private_consensus"]
            )
            private_domain = F.cross_entropy(private_logits, centre_labels)

        orthogonal = torch.stack(
            [
                self._orthogonality(
                    outputs[f"shared_{name}"],
                    outputs[f"private_{name}"],
                )
                for name in ("clinical", "colposcopy", "oct")
            ]
        ).mean()

        total = (
            base_total
            + lambda_utility * utility_regression
            + 0.25 * reliability
            + lambda_anchor * anchor
            + lambda_concept * concept
            + lambda_concept_risk * concept_risk
            + lambda_domain * domain
            + lambda_private_domain * private_domain
            + lambda_orthogonal * orthogonal
            + lambda_cin3 * cin3
            + lambda_safety_error * safety_error_loss
        )
        pieces.update(
            {
                "loss": total,
                "utility": utility_regression,
                "reliability": reliability,
                "anchor": anchor,
                "concept": concept,
                "concept_risk": concept_risk,
                "domain": domain,
                "private_domain": private_domain,
                "orthogonal": orthogonal,
                "cin3": cin3,
                "safety_error": safety_error_loss,
            }
        )
        return total, pieces

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
        safety_threshold: float = 0.10,
        apply_safety_gate: Optional[bool] = None,
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
        use_safety = (
            self.config.use_safety_gate
            if apply_safety_gate is None
            else bool(apply_safety_gate)
        )
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
                action = torch.where(
                    best_value > 0,
                    best_index + 1,
                    torch.full_like(best_index, ACTION_STOP),
                )
                if use_safety:
                    p3 = torch.sigmoid(current["cin3_logit"])
                    upper = (
                        p3
                        + float(self.config.safety_error_beta)
                        * current["safety_error"]
                    ).clamp_max(1.0)
                    indeterminate = (p3 < float(safety_threshold)) & (
                        upper >= float(safety_threshold)
                    )
                    available = ~acquired[:, 1:].bool()
                    forced_index = utility.masked_fill(~available, -1e9).argmax(dim=-1)
                    can_force = available.any(dim=-1)
                    force = (
                        indeterminate
                        & can_force
                        & (action == ACTION_STOP)
                        & (~stopped)
                    )
                    action = torch.where(force, forced_index + 1, action)
            elif policy in {
                "uncertainty",
                "cheapest_first",
                "fixed_order",
                "static_all",
                "clinical_only",
                "random",
            }:
                # Reuse the established policy definitions one batch at a time
                # is not possible without re-encoding; implement the same rules.
                probability = torch.sigmoid(current["logit"])
                uncertainty = 1.0 - torch.abs(probability - 0.5) * 2.0
                if policy == "clinical_only":
                    action = torch.full((batch,), ACTION_STOP, device=device)
                elif policy in {"fixed_order", "static_all"}:
                    action = torch.where(
                        acquired[:, 1].bool(),
                        torch.full((batch,), ACTION_OCT, device=device),
                        torch.full((batch,), ACTION_COLPOSCOPY, device=device),
                    )
                elif policy == "uncertainty":
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
                    first = (
                        ACTION_COLPOSCOPY
                        if colposcopy_cost <= oct_cost
                        else ACTION_OCT
                    )
                    candidate = torch.full((batch,), first, device=device)
                    invalid_first = acquired.gather(
                        1, candidate.unsqueeze(1)
                    ).squeeze(1).bool()
                    fallback = torch.full(
                        (batch,),
                        ACTION_OCT if first == ACTION_COLPOSCOPY else ACTION_COLPOSCOPY,
                        device=device,
                    )
                    candidate = torch.where(invalid_first, fallback, candidate)
                    action = torch.where(
                        uncertainty > uncertainty_threshold,
                        candidate,
                        torch.full_like(candidate, ACTION_STOP),
                    )
                else:
                    acquire = (
                        torch.rand(batch, generator=generator, device=device)
                        < random_acquisition_probability
                    )
                    candidate = (
                        torch.randint(
                            0, 2, (batch,), generator=generator, device=device
                        )
                        + 1
                    )
                    invalid_random = acquired.gather(
                        1, candidate.unsqueeze(1)
                    ).squeeze(1).bool()
                    candidate = torch.where(
                        invalid_random,
                        torch.where(
                            acquired[:, 1].bool(),
                            torch.full_like(candidate, ACTION_OCT),
                            torch.full_like(candidate, ACTION_COLPOSCOPY),
                        ),
                        candidate,
                    )
                    action = torch.where(
                        acquire,
                        candidate,
                        torch.full_like(candidate, ACTION_STOP),
                    )
            else:
                raise ValueError(f"unsupported v3 policy {policy}")
            invalid = acquired.gather(1, action.unsqueeze(1)).squeeze(1).bool()
            action = torch.where(
                invalid | stopped,
                torch.full_like(action, ACTION_STOP),
                action,
            )
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
            "cin3_logit": final["cin3_logit"],
            "cin3_probability": torch.sigmoid(final["cin3_logit"]),
            "safety_error": final["safety_error"],
            "actions": actions,
            "acquired_mask": acquired,
            "acquisition_count": acquired[:, 1:].sum(dim=-1).float(),
            "cost": scenario_cost,
            "final_state": final["state"],
            "final_precision_weight": final["precision_weight"],
        }


__all__ = [
    "ACTION_STOP",
    "ACTION_COLPOSCOPY",
    "ACTION_OCT",
    "ACTION_NAMES",
    "ActiveFusionV3",
    "ActiveFusionV3Config",
    "gradient_reverse",
]
