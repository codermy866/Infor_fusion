#!/usr/bin/env python3
"""Source-only development and locked outer evaluation for latent active fusion v3."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from models.active_fusion_v2 import ACTION_NAMES  # noqa: E402
from models.active_fusion_v3 import ActiveFusionV3, ActiveFusionV3Config  # noqa: E402
from scripts.active_fusion_v2.run_v2 import (  # noqa: E402
    COST_SCENARIOS,
    COST_WEIGHT_GRID,
    DEFAULT_CACHE,
    DEFAULT_CASE_VLM,
    DEFAULT_SPLIT_ROOT,
    DEFAULT_TOKENS,
    EXPECTED_FOLDS,
    FeatureRows,
    choose_safety_threshold,
    choose_youden_threshold,
    collect_intervention_responses,
    derive_source_train_validation,
    discover_folds,
    ece_score,
    load_frame,
    make_loader,
    move_batch,
    seed_everything,
    sha256_file,
    summarize_interventions,
    utc_now,
    validation_checkpoint_score,
    write_json,
)


SCHEMA = "shift_safe_latent_active_fusion_v3"
DEFAULT_OUTPUT = EXP_ROOT / "outputs/shift_safe_vlm_v3_20260729"
CENTRE_INDEX = {
    "十堰市人民医院": 0,
    "恩施州中心医院": 1,
    "武大人民医院": 2,
    "荆州市第一人民医院": 3,
    "襄阳市中心医院": 4,
}
ARM_CONFIGS: Dict[str, Dict[str, object]] = {
    "v3_full_medical": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_no_vlm": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_generic_vlm": {
        "semantic_mode": "generic_vlm",
        "anchor_mode": "generic",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_shuffled_vlm": {
        "semantic_mode": "shuffled_vlm",
        "anchor_mode": "shuffled",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_no_domain": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": False,
        "use_reliability": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_mean_utility": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_no_safety": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_safety": False,
        "use_bicer": True,
        "use_safety_utility": False,
    },
    "v3_no_bicer": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_safety": True,
        "use_bicer": False,
    },
    "v3_precision_mean": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_uniform_lcb": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical",
        "use_anchor": True,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": False,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_medical": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_no_vlm": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_generic": {
        "semantic_mode": "generic_vlm",
        "anchor_mode": "generic_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_shuffled": {
        "semantic_mode": "shuffled_vlm",
        "anchor_mode": "shuffled_medical_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_fused_medical": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_concept_fusion": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_fused_no_vlm": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_fused_generic": {
        "semantic_mode": "generic_vlm",
        "anchor_mode": "generic_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_concept_fusion": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_fused_shuffled": {
        "semantic_mode": "shuffled_vlm",
        "anchor_mode": "shuffled_medical_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_concept_fusion": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_expert_medical": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_concept_fusion": True,
        "use_concept_expert": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_expert_no_vlm": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": False,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_expert_generic": {
        "semantic_mode": "generic_vlm",
        "anchor_mode": "generic_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_concept_fusion": True,
        "use_concept_expert": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_concept_expert_shuffled": {
        "semantic_mode": "shuffled_vlm",
        "anchor_mode": "shuffled_medical_concepts",
        "feature_source": "concepts",
        "use_anchor": True,
        "use_concept": True,
        "use_concept_fusion": True,
        "use_concept_expert": True,
        "use_domain": True,
        "use_reliability": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_reliability_medical": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "medical_quality",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": False,
        "use_vlm_reliability": True,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_reliability_no_vlm": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": False,
        "use_vlm_reliability": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_reliability_generic": {
        "semantic_mode": "generic_vlm",
        "anchor_mode": "generic_quality",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": False,
        "use_vlm_reliability": True,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_reliability_shuffled": {
        "semantic_mode": "shuffled_vlm",
        "anchor_mode": "shuffled_quality",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": False,
        "use_vlm_reliability": True,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_latent_full": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_latent_no_domain": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": False,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_latent_mean_utility": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": False,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_latent_uniform_fusion": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": False,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_latent_no_safety": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": False,
        "use_safety_utility": False,
        "use_bicer": True,
    },
    "v3_latent_no_bicer": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": False,
    },
    "v3_latent_shared_only": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "use_anchor": False,
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
        "private_evidence_weight": 0.0,
    },
    "v3_vlm_poe_medical": {
        "semantic_mode": "case_vlm",
        "anchor_mode": "paired_medical_concepts",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": True,
        "concept_expert_gate_mode": "constant",
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_poe_no_vlm": {
        "semantic_mode": "none",
        "anchor_mode": "none",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": False,
        "concept_expert_gate_mode": "constant",
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_poe_generic": {
        "semantic_mode": "generic_vlm",
        "anchor_mode": "paired_generic_concepts",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": True,
        "concept_expert_gate_mode": "constant",
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
    "v3_vlm_poe_shuffled": {
        "semantic_mode": "shuffled_vlm",
        "anchor_mode": "shuffled_paired_medical_concepts",
        "feature_source": "concepts",
        "use_anchor": False,
        "use_concept": False,
        "use_concept_fusion": False,
        "use_concept_expert": True,
        "concept_expert_gate_mode": "constant",
        "use_domain": True,
        "use_reliability": True,
        "use_precision": True,
        "use_lcb": True,
        "use_safety": True,
        "use_bicer": True,
    },
}


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(average_precision_score(y, p)) if np.any(y == 1) else None


def select_by_retention_and_safety(
    candidates: Sequence[dict[str, object]],
    *,
    static_auprc: float,
    retention: float,
    cin3_sensitivity_floor: float,
) -> tuple[dict[str, object], bool]:
    """Select acquisition cost only after both source constraints are met."""
    viable = [
        candidate
        for candidate in candidates
        if (
            float(candidate["metrics"]["cin2_auprc"])
            >= retention * static_auprc
            and float(candidate["metrics"]["cin3_sensitivity"])
            >= cin3_sensitivity_floor
        )
    ]
    if viable:
        return (
            min(
                viable,
                key=lambda item: (
                    float(item["metrics"]["mean_acquisition_count"]),
                    float(item["metrics"]["cin2_brier"]),
                    -float(item["metrics"]["cin2_auprc"]),
                ),
            ),
            True,
        )
    # A failed source constraint remains explicit. The least-violating setting
    # is retained for diagnosis but is not eligible for a safety claim.
    return (
        min(
            candidates,
            key=lambda item: (
                max(
                    0.0,
                    retention * static_auprc
                    - float(item["metrics"]["cin2_auprc"]),
                )
                + max(
                    0.0,
                    cin3_sensitivity_floor
                    - float(item["metrics"]["cin3_sensitivity"]),
                ),
                float(item["metrics"]["mean_acquisition_count"]),
                float(item["metrics"]["cin2_brier"]),
            ),
        ),
        False,
    )


def summarize(frame: pd.DataFrame, cin2_threshold: float = 0.5, cin3_threshold: float = 0.5):
    y2 = frame.y2.to_numpy(dtype=int)
    y3 = frame.y3.to_numpy(dtype=int)
    p2 = frame.probability.to_numpy(dtype=float)
    p3 = frame.cin3_probability.to_numpy(dtype=float)
    pred2 = p2 >= cin2_threshold
    pred3 = p3 >= cin3_threshold
    positive2, negative2, positive3 = y2 == 1, y2 == 0, y3 == 1
    true_negative = int((~pred2 & negative2).sum())
    false_negative = int((~pred2 & positive2).sum())
    return {
        "n": int(len(frame)),
        "cin2_auroc": safe_auc(y2, p2),
        "cin2_auprc": safe_auprc(y2, p2),
        "cin2_brier": float(brier_score_loss(y2, p2)),
        "cin2_ece": float(ece_score(y2, p2)),
        "cin2_sensitivity": float(pred2[positive2].mean()) if positive2.any() else None,
        "cin2_specificity": float((~pred2[negative2]).mean()) if negative2.any() else None,
        "cin2_npv": float(true_negative / max(1, true_negative + false_negative)),
        "cin3_auroc": safe_auc(y3, p3),
        "cin3_auprc": safe_auprc(y3, p3),
        "cin3_brier": float(brier_score_loss(y3, p3)),
        "cin3_sensitivity": float(pred3[positive3].mean()) if positive3.any() else None,
        "cin3_false_negatives": int((~pred3 & positive3).sum()),
        "safety_referral_rate": float(pred3.mean()),
        "mean_acquisition_count": float(frame.acquisition_count.mean()),
        "mean_scenario_cost": float(frame.cost.mean()),
        "colposcopy_acquisition_rate": float(frame.acquired_colposcopy.mean()),
        "oct_acquisition_rate": float(frame.acquired_oct.mean()),
        "mean_colposcopy_images_triggered": float(
            (frame.acquired_colposcopy * frame.col_count).mean()
        ),
        "mean_oct_bscans_triggered": float(
            (frame.acquired_oct * frame.oct_count).mean()
        ),
    }


@torch.no_grad()
def collect_subsets(model: ActiveFusionV3, loader: DataLoader, device: torch.device):
    model.eval()
    names = ("clinical", "clinical_colposcopy", "clinical_oct", "all")
    rows = {name: [] for name in names}
    for batch in loader:
        clinical, col, oct_patches, _, semantic = move_batch(batch, device)
        outputs = model.all_subset_outputs(clinical, col, oct_patches, semantic)
        for name in names:
            p2 = torch.sigmoid(outputs[f"logit_{name}"]).cpu().numpy()
            p3 = torch.sigmoid(outputs[f"cin3_logit_{name}"]).cpu().numpy()
            mask = outputs[f"mask_{name}"][0].cpu().numpy()
            for index in range(len(p2)):
                rows[name].append(
                    {
                        "case_hash": batch["case_hash"][index],
                        "center_name": batch["center_name"][index],
                        "y2": int(batch["y2"][index]),
                        "y3": int(batch["y3"][index]),
                        "probability": float(p2[index]),
                        "cin3_probability": float(p3[index]),
                        "acquired_colposcopy": int(mask[1]),
                        "acquired_oct": int(mask[2]),
                        "acquisition_count": int(mask[1] + mask[2]),
                        "cost": float(mask[1] + mask[2]),
                        "col_count": float(batch["col_count"][index]),
                        "oct_count": float(batch["oct_count"][index]),
                    }
                )
    return {name: pd.DataFrame(value) for name, value in rows.items()}


@torch.no_grad()
def collect_latent_features(
    model: ActiveFusionV3,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    shared, private, centres, orthogonal = [], [], [], []
    for batch in loader:
        clinical, col, oct_patches, _, semantic = move_batch(batch, device)
        outputs = model.all_subset_outputs(
            clinical, col, oct_patches, semantic
        )
        shared_value = outputs["shared_consensus"]
        private_value = outputs["private_consensus"]
        shared.append(shared_value.cpu().numpy())
        private.append(private_value.cpu().numpy())
        centres.extend(CENTRE_INDEX[name] for name in batch["center_name"])
        orthogonal.append(
            (
                F.cosine_similarity(shared_value, private_value, dim=-1)
                .pow(2)
                .cpu()
                .numpy()
            )
        )
    return {
        "shared": np.concatenate(shared),
        "private": np.concatenate(private),
        "centre": np.asarray(centres, dtype=int),
        "orthogonality": np.concatenate(orthogonal),
    }


def source_centre_probe(
    train_latent: Mapping[str, np.ndarray],
    validation_latent: Mapping[str, np.ndarray],
) -> dict[str, float]:
    result = {}
    for name in ("shared", "private"):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=20260729,
            ),
        )
        probe.fit(train_latent[name], train_latent["centre"])
        prediction = probe.predict(validation_latent[name])
        result[f"{name}_centre_balanced_accuracy"] = float(
            balanced_accuracy_score(
                validation_latent["centre"],
                prediction,
            )
        )
    result["validation_shared_private_cosine_squared"] = float(
        validation_latent["orthogonality"].mean()
    )
    result["interpretation"] = (
        "source-only post-hoc linear probes; lower shared and higher private "
        "centre predictability support, but do not prove, latent decoupling"
    )
    return result


@torch.no_grad()
def collect_policy(
    model: ActiveFusionV3,
    loader: DataLoader,
    device: torch.device,
    *,
    policy: str,
    seed: int,
    cost_weight: float = 0.2,
    uncertainty_threshold: float = 0.55,
    random_probability: float = 0.5,
    colposcopy_cost: float = 1.0,
    oct_cost: float = 1.0,
    safety_threshold: float = 0.10,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch_index, batch in enumerate(loader):
        clinical, col, oct_patches, _, semantic = move_batch(batch, device)
        output = model.run_policy(
            clinical,
            col,
            oct_patches,
            semantic,
            policy=policy,
            cost_weight=cost_weight,
            uncertainty_threshold=uncertainty_threshold,
            random_acquisition_probability=random_probability,
            random_seed=seed * 100003 + batch_index,
            colposcopy_cost=colposcopy_cost,
            oct_cost=oct_cost,
            safety_threshold=safety_threshold,
        )
        p2 = output["probability"].cpu().numpy()
        p3 = output["cin3_probability"].cpu().numpy()
        safety_error = output["safety_error"].cpu().numpy()
        acquired = output["acquired_mask"].cpu().numpy()
        actions = output["actions"].cpu().numpy()
        cost = output["cost"].cpu().numpy()
        for index in range(len(p2)):
            rows.append(
                {
                    "case_hash": batch["case_hash"][index],
                    "center_name": batch["center_name"][index],
                    "y2": int(batch["y2"][index]),
                    "y3": int(batch["y3"][index]),
                    "probability": float(p2[index]),
                    "cin3_probability": float(p3[index]),
                    "safety_error": float(safety_error[index]),
                    "action_1": ACTION_NAMES[int(actions[index, 0])],
                    "action_2": ACTION_NAMES[int(actions[index, 1])],
                    "acquired_colposcopy": int(acquired[index, 1]),
                    "acquired_oct": int(acquired[index, 2]),
                    "acquisition_count": int(output["acquisition_count"][index]),
                    "cost": float(cost[index]),
                    "col_count": float(batch["col_count"][index]),
                    "oct_count": float(batch["oct_count"][index]),
                }
            )
    return pd.DataFrame(rows)


def train_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    case_vlm_features: Mapping[str, Mapping[str, torch.Tensor]],
    tokens: Mapping[str, torch.Tensor],
    arm: str,
    args: argparse.Namespace,
    device: torch.device,
    job_dir: Path,
):
    seed_everything(args.seed)
    arm_config = ARM_CONFIGS[arm]
    prevalence = float(train_frame.pathology_cin2plus.mean())
    pos_weight = float(np.clip((1.0 - prevalence) / max(prevalence, 1e-6), 1.0, 8.0))
    config = ActiveFusionV3Config(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        semantic_anchor_mode=str(arm_config["anchor_mode"]),
        use_semantic_anchor=bool(arm_config["use_anchor"]),
        use_domain_adversary=bool(arm_config["use_domain"]),
        use_reliability=bool(arm_config["use_reliability"]),
        use_precision_fusion=bool(
            arm_config.get("use_precision", arm_config["use_reliability"])
        ),
        use_utility_lcb=bool(
            arm_config.get("use_lcb", arm_config["use_reliability"])
        ),
        use_safety_gate=bool(arm_config["use_safety"]),
        use_bicer=bool(arm_config["use_bicer"]),
        pos_weight=pos_weight,
        utility_lcb_beta=args.utility_lcb_beta,
        adversary_coefficient=args.adversary_coefficient,
        private_evidence_weight=float(
            arm_config.get("private_evidence_weight", 0.25)
        ),
        safety_utility_weight=(
            args.safety_utility_weight
            if bool(arm_config.get("use_safety_utility", True))
            else 0.0
        ),
        case_vlm_dim=int(
            next(iter(case_vlm_features.values()))["medical"].numel()
        ),
        use_concept_distillation=bool(arm_config.get("use_concept", False)),
        use_concept_token_fusion=bool(
            arm_config.get("use_concept_fusion", False)
        ),
        use_concept_risk_expert=bool(
            arm_config.get("use_concept_expert", False)
        ),
        concept_expert_gate_mode=str(
            arm_config.get("concept_expert_gate_mode", "learned")
        ),
        use_vlm_quality_reliability=bool(
            arm_config.get("use_vlm_reliability", False)
        ),
    )
    model = ActiveFusionV3(config, tokens).to(device)
    semantic_mode = str(arm_config["semantic_mode"])
    train_loader = make_loader(
        train_frame,
        cache,
        case_vlm_features,
        semantic_mode,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_frame,
        cache,
        case_vlm_features,
        semantic_mode,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    tracked = (
        "loss",
        "classification",
        "brier",
        "utility",
        "reliability",
        "anchor",
        "concept",
        "concept_risk",
        "domain",
        "private_domain",
        "orthogonal",
        "cin3",
        "safety_error",
        "bicer",
        "bicer_gap",
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            clinical, col, oct_patches, labels, semantic = move_batch(batch, device)
            cin3 = batch["y3"].to(device).long()
            centre = torch.tensor(
                [CENTRE_INDEX[name] for name in batch["center_name"]],
                device=device,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = model.training_losses(
                clinical,
                col,
                oct_patches,
                labels,
                semantic,
                cin3_labels=cin3,
                centre_labels=centre,
                lambda_utility=args.lambda_utility,
                lambda_brier=args.lambda_brier,
                lambda_bicer=args.lambda_bicer if bool(arm_config["use_bicer"]) else 0.0,
                lambda_anchor=args.lambda_anchor if bool(arm_config["use_anchor"]) else 0.0,
                lambda_concept=args.lambda_concept if bool(arm_config.get("use_concept", False)) else 0.0,
                lambda_domain=args.lambda_domain if bool(arm_config["use_domain"]) else 0.0,
                lambda_cin3=args.lambda_cin3,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batches += 1
            for key in tracked:
                totals[key] = totals.get(key, 0.0) + float(pieces[key].detach())
        subsets = collect_subsets(model, val_loader, device)
        score = validation_checkpoint_score(subsets)
        # Give the multitask head a small, fixed role in early stopping.
        cin3_brier = brier_score_loss(
            subsets["all"].y3,
            subsets["all"].cin3_probability,
        )
        score = float(score + 0.15 * cin3_brier)
        record = {"epoch": float(epoch), "validation_score": score}
        record.update({key: value / batches for key, value in totals.items()})
        history.append(record)
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        print(
            f"[{arm}] epoch={epoch} val={score:.6f} "
            f"train={record['loss']:.6f} best={best_epoch}",
            flush=True,
        )
        if stale >= args.early_stop:
            break
    if best_state is None:
        raise RuntimeError("v3 training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "schema": SCHEMA,
        "arm": arm,
        "seed": args.seed,
        "best_source_validation_epoch": best_epoch,
        "best_source_validation_score": best_score,
        "config": asdict(config),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, job_dir / "model_best_source_validation.pt")
    return model, val_loader, history


def tune_source(
    model: ActiveFusionV3,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
):
    subsets = collect_subsets(model, val_loader, device)
    static = subsets["all"]
    safety_threshold = choose_safety_threshold(
        static.y3.to_numpy(dtype=int),
        static.cin3_probability.to_numpy(dtype=float),
        floor=args.safety_sensitivity_floor,
    )
    static_metrics = summarize(
        static,
        cin3_threshold=safety_threshold,
    )
    candidates = []
    for weight in COST_WEIGHT_GRID:
        frame = collect_policy(
            model,
            val_loader,
            device,
            policy="learned",
            seed=args.seed,
            cost_weight=weight,
            safety_threshold=safety_threshold,
        )
        candidates.append(
            {
                "policy": "learned",
                "cost_weight": weight,
                "metrics": summarize(
                    frame,
                    cin3_threshold=safety_threshold,
                ),
            }
        )
    selected, constraint_satisfied = select_by_retention_and_safety(
        candidates,
        static_auprc=float(static_metrics["cin2_auprc"]),
        retention=args.performance_retention,
        cin3_sensitivity_floor=args.safety_sensitivity_floor,
    )
    return {
        "selection_data": "source_validation_only",
        "performance_retention": args.performance_retention,
        "safety_sensitivity_floor": args.safety_sensitivity_floor,
        "joint_source_constraint_satisfied": constraint_satisfied,
        "cin3_safety_threshold": safety_threshold,
        "static_all": static_metrics,
        "learned_candidates": candidates,
        "selected_learned": selected,
    }


def evaluate(
    model: ActiveFusionV3,
    loader: DataLoader,
    selection: Mapping[str, object],
    device: torch.device,
    args: argparse.Namespace,
):
    chosen = selection["selected_learned"]
    safety_threshold = float(selection["cin3_safety_threshold"])
    primary = collect_policy(
        model,
        loader,
        device,
        policy="learned",
        seed=args.seed,
        cost_weight=float(chosen["cost_weight"]),
        safety_threshold=safety_threshold,
    )
    curves = []
    for scenario, (col_cost, oct_cost) in COST_SCENARIOS.items():
        for weight in COST_WEIGHT_GRID:
            frame = collect_policy(
                model,
                loader,
                device,
                policy="learned",
                seed=args.seed,
                cost_weight=weight,
                colposcopy_cost=col_cost,
                oct_cost=oct_cost,
                safety_threshold=safety_threshold,
            )
            curves.append(
                {
                    "scenario": scenario,
                    "colposcopy_relative_cost": col_cost,
                    "oct_relative_cost": oct_cost,
                    "cost_weight": weight,
                    **summarize(frame, cin3_threshold=safety_threshold),
                }
            )
    return primary, curves


def run_one(
    fold_path: Path,
    arm: str,
    args: argparse.Namespace,
    cache,
    case_vlm_features,
    semantic_cache_path: Path,
    tokens,
    device: torch.device,
):
    job_dir = args.output / args.mode / fold_path.name / f"seed_{args.seed}" / arm
    job_dir.mkdir(parents=True, exist_ok=True)
    completion_path = job_dir / "completion.json"
    expected_status = (
        "formal_target_evaluation_complete"
        if args.mode == "formal"
        else "source_development_complete_target_outcomes_not_loaded_by_runner"
    )
    semantic_cache_sha256 = sha256_file(semantic_cache_path)
    if completion_path.exists():
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completed.get("schema") == SCHEMA
            and completed.get("status") == expected_status
            and completed.get("arm") == arm
            and int(completed.get("seed")) == args.seed
            and completed.get("input_hashes", {}).get(
                "semantic_feature_cache"
            )
            == semantic_cache_sha256
        ):
            print(f"[resume] {fold_path.name} {args.seed} {arm}", flush=True)
            return completed
    train_path = fold_path / "train_labels.csv"
    val_path = fold_path / "val_labels.csv"
    test_path = fold_path / "external_test_labels.csv"
    train_frame, val_frame, split_audit = derive_source_train_validation(
        load_frame(train_path),
        load_frame(val_path),
        seed=args.seed,
    )
    held_center = pd.read_csv(
        test_path,
        usecols=["center_name"],
        encoding="utf-8-sig",
    ).center_name.iloc[0]
    model, val_loader, history = train_model(
        train_frame,
        val_frame,
        cache,
        case_vlm_features,
        tokens,
        arm,
        args,
        device,
        job_dir,
    )
    train_audit_loader = make_loader(
        train_frame,
        cache,
        case_vlm_features,
        str(ARM_CONFIGS[arm]["semantic_mode"]),
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    latent_centre_probe = source_centre_probe(
        collect_latent_features(model, train_audit_loader, device),
        collect_latent_features(model, val_loader, device),
    )
    selection = tune_source(model, val_loader, device, args)
    selected = selection["selected_learned"]
    validation_primary = collect_policy(
        model,
        val_loader,
        device,
        policy="learned",
        seed=args.seed,
        cost_weight=float(selected["cost_weight"]),
        safety_threshold=float(selection["cin3_safety_threshold"]),
    )
    selection["cin2_operating_threshold"] = choose_youden_threshold(
        validation_primary.y2.to_numpy(dtype=int),
        validation_primary.probability.to_numpy(dtype=float),
    )
    source_interventions = pd.DataFrame()
    if bool(ARM_CONFIGS[arm]["use_bicer"]):
        source_interventions = collect_intervention_responses(
            model,
            val_loader,
            device,
        )
        source_interventions.to_csv(
            job_dir / "source_validation_interventions.csv",
            index=False,
        )
    frozen = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "mode": args.mode,
        "fold": fold_path.name,
        "held_out_center_identity_only": str(held_center),
        "arm": arm,
        "seed": args.seed,
        "source_train_n": len(train_frame),
        "source_validation_n": len(val_frame),
        "source_split_audit": split_audit,
        "target_outcomes_used_for_training_or_selection": False,
        "selection": selection,
        "source_validation_metrics": summarize(
            validation_primary,
            cin2_threshold=float(selection["cin2_operating_threshold"]),
            cin3_threshold=float(selection["cin3_safety_threshold"]),
        ),
        "source_latent_centre_probe": latent_centre_probe,
        "source_validation_interventions": (
            summarize_interventions(source_interventions)
            if len(source_interventions)
            else []
        ),
        "training_history": history,
        "input_hashes": {
            "train": sha256_file(train_path),
            "validation": sha256_file(val_path),
            "concept_tokens": sha256_file(args.concept_tokens),
            "semantic_feature_cache": semantic_cache_sha256,
        },
        "claim_boundary": (
            "CIN3+ safety is calibrated on source validation and is not a "
            "guarantee under target-centre shift."
        ),
    }
    write_json(job_dir / "frozen_source_development.json", frozen)
    if args.mode == "development":
        completion = {**frozen, "status": expected_status}
        write_json(completion_path, completion)
        return completion

    test_frame = load_frame(test_path)
    test_loader = make_loader(
        test_frame,
        cache,
        case_vlm_features,
        str(ARM_CONFIGS[arm]["semantic_mode"]),
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    predictions, curves = evaluate(
        model,
        test_loader,
        selection,
        device,
        args,
    )
    predictions.to_csv(
        job_dir / "target_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(curves).to_csv(
        job_dir / "performance_cost_curve.csv",
        index=False,
    )
    target_interventions = pd.DataFrame()
    if bool(ARM_CONFIGS[arm]["use_bicer"]):
        target_interventions = collect_intervention_responses(
            model,
            test_loader,
            device,
        )
        target_interventions.to_csv(
            job_dir / "target_interventions.csv",
            index=False,
        )
    completion = {
        **frozen,
        "status": expected_status,
        "target_n": len(test_frame),
        "primary_metrics": summarize(
            predictions,
            cin2_threshold=float(selection["cin2_operating_threshold"]),
            cin3_threshold=float(selection["cin3_safety_threshold"]),
        ),
        "target_interventions": (
            summarize_interventions(target_interventions)
            if len(target_interventions)
            else []
        ),
        "target_predictions_sha256": sha256_file(
            job_dir / "target_predictions.csv"
        ),
    }
    write_json(completion_path, completion)
    return completion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), default="development")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--concept-tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--case-vlm", type=Path, default=DEFAULT_CASE_VLM)
    parser.add_argument(
        "--concept-cache",
        type=Path,
        default=DEFAULT_OUTPUT
        / "shared/qwen3vl_structured_colposcopy_concepts.pt",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", action="append", default=[])
    parser.add_argument("--arm", action="append", choices=tuple(ARM_CONFIGS), default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--formal-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lambda-utility", type=float, default=0.5)
    parser.add_argument("--lambda-brier", type=float, default=0.2)
    parser.add_argument("--lambda-bicer", type=float, default=0.15)
    parser.add_argument("--lambda-anchor", type=float, default=0.10)
    parser.add_argument("--lambda-concept", type=float, default=0.15)
    parser.add_argument("--lambda-domain", type=float, default=0.05)
    parser.add_argument("--lambda-cin3", type=float, default=0.35)
    parser.add_argument("--utility-lcb-beta", type=float, default=0.5)
    parser.add_argument("--adversary-coefficient", type=float, default=0.25)
    parser.add_argument("--performance-retention", type=float, default=0.97)
    parser.add_argument("--safety-sensitivity-floor", type=float, default=0.95)
    parser.add_argument("--safety-utility-weight", type=float, default=0.50)
    parser.add_argument(
        "--development-all-folds",
        action="store_true",
        help="Run source-only development for every outer fold without opening target outcomes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folds = discover_folds(args.split_root)
    if args.fold:
        folds = [fold for fold in folds if fold.name in set(args.fold)]
    arms = args.arm or list(ARM_CONFIGS)
    seeds = args.formal_seeds or [args.seed]
    if args.mode == "formal":
        if {fold.name for fold in folds} != EXPECTED_FOLDS:
            raise ValueError("formal v3 mode requires exactly five locked folds")
        if len(set(seeds)) < 3:
            raise ValueError("formal v3 mode requires at least three seeds")
    elif not args.development_all_folds:
        folds = folds[:1]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache = cache_payload.get("features", cache_payload)
    token_payload = torch.load(args.concept_tokens, map_location="cpu", weights_only=False)
    tokens = token_payload["tokens"]
    vlm_payload = torch.load(args.case_vlm, map_location="cpu", weights_only=False)
    embedding_features = vlm_payload["features"]
    need_concepts = any(
        ARM_CONFIGS[arm].get("feature_source") == "concepts"
        for arm in arms
    )
    concept_features = None
    if need_concepts:
        concept_payload = torch.load(
            args.concept_cache,
            map_location="cpu",
            weights_only=False,
        )
        concept_features = concept_payload["features"]
        if len(concept_features) != 1897:
            raise ValueError("structured concept cache must contain 1,897 cases")
    results = []
    for seed in seeds:
        args.seed = int(seed)
        for fold in folds:
            for arm in arms:
                selected_features = (
                    concept_features
                    if ARM_CONFIGS[arm].get("feature_source") == "concepts"
                    else embedding_features
                )
                selected_feature_path = (
                    args.concept_cache
                    if ARM_CONFIGS[arm].get("feature_source") == "concepts"
                    else args.case_vlm
                )
                results.append(
                    run_one(
                        fold,
                        arm,
                        args,
                        cache,
                        selected_features,
                        selected_feature_path,
                        tokens,
                        device,
                    )
                )
    write_json(
        args.output / f"{args.mode}_campaign.json",
        {
            "schema": SCHEMA,
            "mode": args.mode,
            "environment": sys.executable,
            "cuda_visible_devices": str(
                __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")
            ),
            "arms": arms,
            "folds": [fold.name for fold in folds],
            "seeds": seeds,
            "jobs": len(results),
            "target_label_policy": (
                "not loaded by development runner"
                if args.mode == "development"
                else "opened after source checkpoint and policy freeze"
            ),
        },
    )


if __name__ == "__main__":
    main()
