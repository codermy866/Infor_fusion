#!/usr/bin/env python3
"""Source-developed, frozen-policy evaluation for Active Fusion v2.

Development mode deliberately exits before the held-out outcome table is read.
Formal mode is guarded to require all five LOCO folds and three seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from models.active_fusion_v2 import ACTION_NAMES, ActiveFusionV2, ActiveFusionV2Config  # noqa: E402
from paper_revision.scripts.clinical_variable_mapping import clinical_features_from_row  # noqa: E402


SCHEMA = "active_information_acquisition_v2"
DEFAULT_SPLIT_ROOT = EXP_ROOT / "outputs/bicer_trace_coe_20260716/splits/loco"
DEFAULT_CACHE = EXP_ROOT / "paper_revision/cache/patch_features_final_1897.pt"
DEFAULT_TOKENS = (
    EXP_ROOT / "outputs/active_trace_coe_20260729/shared/vlm_concept_tokens_qwen2vl2b.pt"
)
DEFAULT_CASE_VLM = (
    EXP_ROOT / "outputs/active_fusion_v2_20260729/shared/qwen3vl_colposcopy_embeddings.pt"
)
DEFAULT_OUTPUT = EXP_ROOT / "outputs/active_fusion_v2_20260729"
EXPECTED_FOLDS = {
    "loco_十堰市人民医院",
    "loco_恩施州中心医院",
    "loco_武大人民医院",
    "loco_荆州市第一人民医院",
    "loco_襄阳市中心医院",
}
ARM_CONFIGS: Dict[str, Dict[str, object]] = {
    "static_no_query": dict(query_mode="no_query", fusion_mode="concat", use_memory=False, use_bicer=False, active=False),
    "static_random_query": dict(query_mode="random", fusion_mode="concat", use_memory=False, use_bicer=False, active=False),
    "static_srqf": dict(query_mode="qwen", fusion_mode="concat", use_memory=False, use_bicer=False, active=False),
    "static_generic_vlm": dict(query_mode="generic_vlm", fusion_mode="concat", use_memory=False, use_bicer=False, active=False),
    "static_shuffled_vlm": dict(query_mode="shuffled_vlm", fusion_mode="concat", use_memory=False, use_bicer=False, active=False),
    "static_pcvlf": dict(query_mode="case_vlm", fusion_mode="concat", use_memory=False, use_bicer=False, active=False),
    "late_fusion": dict(query_mode="no_query", fusion_mode="late", use_memory=False, use_bicer=False, active=False),
    "gated_fusion": dict(query_mode="no_query", fusion_mode="gated", use_memory=False, use_bicer=False, active=False),
    "cross_attention_fusion": dict(query_mode="no_query", fusion_mode="cross_attention", use_memory=False, use_bicer=False, active=False),
    "simmlm_dmome": dict(query_mode="no_query", fusion_mode="dmome", use_memory=False, use_bicer=False, active=False),
    "bvoi_no_memory": dict(query_mode="no_query", fusion_mode="concat", use_memory=False, use_bicer=False, active=True),
    "bvoi_bicer": dict(query_mode="no_query", fusion_mode="concat", use_memory=False, use_bicer=True, active=True),
    "bvoi_memory": dict(query_mode="no_query", fusion_mode="concat", use_memory=True, use_bicer=False, active=True),
    "bvoi_memory_bicer": dict(query_mode="no_query", fusion_mode="concat", use_memory=True, use_bicer=True, active=True),
}
COST_WEIGHT_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20)
UNCERTAINTY_GRID = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
RANDOM_PROBABILITY_GRID = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
COST_SCENARIOS = {
    "acquisition_count": (1.0, 1.0),
    "moderate_oct": (1.0, 2.0),
    "high_oct": (1.0, 4.0),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def case_key(row: pd.Series) -> str:
    return f"{row.get('patient_id', row.get('ID', ''))}||{row.get('oct_id', row.get('OCT', ''))}"


class FeatureRows(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cache: Mapping[str, Mapping[str, torch.Tensor]],
        case_vlm_features: Mapping[str, Mapping[str, torch.Tensor]],
        semantic_mode: str,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.cache = feature_cache
        self.keys = [case_key(row) for _, row in self.frame.iterrows()]
        missing = [key for key in self.keys if key not in feature_cache]
        if missing:
            raise KeyError(f"{len(missing)} cases missing from cache: {missing[:3]}")
        self.case_vlm_features = case_vlm_features
        self.semantic_mode = semantic_mode
        missing_vlm = [key for key in self.keys if key not in case_vlm_features]
        if missing_vlm:
            raise KeyError(f"{len(missing_vlm)} cases missing Qwen3-VL features: {missing_vlm[:3]}")
        first_feature = next(iter(case_vlm_features.values()))
        self.semantic_dim = int(first_feature["medical"].numel())
        ordered = sorted(self.keys, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        self.shuffled_key = {
            key: ordered[(index + 1) % len(ordered)]
            for index, key in enumerate(ordered)
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        key = self.keys[index]
        if self.semantic_mode == "generic_vlm":
            semantic = self.case_vlm_features[key]["generic"]
        elif self.semantic_mode == "shuffled_vlm":
            semantic = self.case_vlm_features[self.shuffled_key[key]]["medical"]
        elif self.semantic_mode == "case_vlm":
            semantic = self.case_vlm_features[key]["medical"]
        else:
            semantic = torch.zeros(self.semantic_dim, dtype=torch.float16)
        col_count = row.get("col_count", row.get("colposcopy_num_images", 0))
        oct_count = row.get("oct_count", row.get("oct_num_bscans", 0))
        return {
            "clinical": torch.tensor(clinical_features_from_row(row), dtype=torch.float32),
            "colposcopy": self.cache[key]["colpo"],
            "oct": self.cache[key]["oct"],
            "case_semantic": semantic,
            "y2": torch.tensor(int(row["pathology_cin2plus"]), dtype=torch.long),
            "y3": torch.tensor(int(row["pathology_cin3plus"]), dtype=torch.long),
            "case_hash": hashlib.sha256(key.encode("utf-8")).hexdigest()[:24],
            "center_name": str(row["center_name"]),
            "col_count": float(col_count) if pd.notna(col_count) else 0.0,
            "oct_count": float(oct_count) if pd.notna(oct_count) else 0.0,
        }


def collate_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "clinical": torch.stack([row["clinical"] for row in rows]),
        "colposcopy": torch.stack([row["colposcopy"] for row in rows]),
        "oct": torch.stack([row["oct"] for row in rows]),
        "case_semantic": torch.stack([row["case_semantic"] for row in rows]),
        "y2": torch.stack([row["y2"] for row in rows]),
        "y3": torch.stack([row["y3"] for row in rows]),
        "case_hash": [str(row["case_hash"]) for row in rows],
        "center_name": [str(row["center_name"]) for row in rows],
        "col_count": torch.tensor([float(row["col_count"]) for row in rows]),
        "oct_count": torch.tensor([float(row["oct_count"]) for row in rows]),
    }


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "patient_id", "oct_id", "pathology_cin2plus",
        "pathology_cin3plus", "center_name",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def move_batch(batch: Mapping[str, object], device: torch.device):
    return (
        batch["clinical"].to(device, non_blocking=True).float(),
        batch["colposcopy"].to(device, non_blocking=True).float(),
        batch["oct"].to(device, non_blocking=True).float(),
        batch["y2"].to(device, non_blocking=True).long(),
        batch["case_semantic"].to(device, non_blocking=True).float(),
    )


def ece_score(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        mask = (probability >= edges[index]) & upper
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return result


def derive_source_train_validation(
    original_train: pd.DataFrame,
    original_validation: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create a patient-disjoint, stratified split from all four source centres."""
    source = pd.concat([original_train, original_validation], ignore_index=True)
    source = source.drop_duplicates(subset=["patient_id", "oct_id"], keep="first").reset_index(drop=True)
    strata = (
        source["center_name"].astype(str)
        + "::"
        + source["pathology_cin2plus"].astype(int).astype(str)
    )
    groups = source["patient_id"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, validation_index = next(
        splitter.split(np.zeros(len(source)), strata, groups)
    )
    train = source.iloc[train_index].reset_index(drop=True)
    validation = source.iloc[validation_index].reset_index(drop=True)
    overlap = set(train["patient_id"].astype(str)) & set(validation["patient_id"].astype(str))
    if overlap:
        raise RuntimeError("derived source split is not patient-disjoint")
    if train["pathology_cin2plus"].nunique() != 2 or validation["pathology_cin2plus"].nunique() != 2:
        raise RuntimeError("derived source split must contain both CIN2+ classes")
    audit = {
        "method": "StratifiedGroupKFold_first_split",
        "n_splits": 5,
        "seed": seed,
        "strata": "source_center_by_CIN2plus",
        "group": "patient_id",
        "source_centers": sorted(source["center_name"].astype(str).unique().tolist()),
        "train_n": len(train),
        "validation_n": len(validation),
        "train_cin2plus": train["pathology_cin2plus"].value_counts().sort_index().to_dict(),
        "validation_cin2plus": validation["pathology_cin2plus"].value_counts().sort_index().to_dict(),
        "patient_overlap_n": 0,
    }
    return train, validation, audit


def make_loader(
    frame: pd.DataFrame,
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    case_vlm_features: Mapping[str, Mapping[str, torch.Tensor]],
    semantic_mode: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FeatureRows(frame, cache, case_vlm_features, semantic_mode),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_rows,
        generator=generator,
        drop_last=False,
    )


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(average_precision_score(y, p)) if np.any(y == 1) else None


def choose_youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best = (-float("inf"), 0.5)
    for threshold in candidates:
        prediction = p >= threshold
        sensitivity = prediction[y == 1].mean()
        specificity = (~prediction[y == 0]).mean()
        score = float(sensitivity + specificity - 1.0)
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def choose_safety_threshold(y3: np.ndarray, p: np.ndarray, floor: float = 0.95) -> float:
    positives = p[y3 == 1]
    if len(positives) == 0:
        return 0.0
    return float(np.quantile(positives, 1.0 - floor, method="lower"))


def summarize_predictions(
    frame: pd.DataFrame,
    *,
    cin2_threshold: float = 0.5,
    cin3_threshold: float = 0.5,
) -> Dict[str, float | int | None]:
    y = frame["y2"].to_numpy(dtype=int)
    y3 = frame["y3"].to_numpy(dtype=int)
    p = frame["probability"].to_numpy(dtype=float)
    prediction = p >= cin2_threshold
    safety_prediction = p >= cin3_threshold
    positive = y == 1
    negative = y == 0
    positive3 = y3 == 1
    true_negative = int((~prediction & negative).sum())
    false_negative = int((~prediction & positive).sum())
    return {
        "n": int(len(frame)),
        "cin2_auroc": safe_auc(y, p),
        "cin2_auprc": safe_auprc(y, p),
        "cin2_brier": float(brier_score_loss(y, p)),
        "cin2_ece": float(ece_score(y, p)),
        "cin2_sensitivity": (
            float(prediction[positive].mean()) if positive.any() else None
        ),
        "cin2_specificity": (
            float((~prediction[negative]).mean()) if negative.any() else None
        ),
        "cin2_npv": float(true_negative / max(1, true_negative + false_negative)),
        "cin3_sensitivity": (
            float(safety_prediction[positive3].mean()) if positive3.any() else None
        ),
        "cin3_false_negatives": int((~safety_prediction & positive3).sum()),
        "safety_referral_rate": float(safety_prediction.mean()),
        "mean_acquisition_count": float(frame["acquisition_count"].mean()),
        "mean_scenario_cost": float(frame["cost"].mean()),
        "colposcopy_acquisition_rate": float(frame["acquired_colposcopy"].mean()),
        "oct_acquisition_rate": float(frame["acquired_oct"].mean()),
        "mean_colposcopy_images_triggered": float(
            (frame["acquired_colposcopy"] * frame["col_count"]).mean()
        ),
        "mean_oct_bscans_triggered": float(
            (frame["acquired_oct"] * frame["oct_count"]).mean()
        ),
    }


@torch.no_grad()
def collect_subset_predictions(
    model: ActiveFusionV2,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, pd.DataFrame]:
    model.eval()
    names = ("clinical", "clinical_colposcopy", "clinical_oct", "all")
    rows: Dict[str, list[dict[str, object]]] = {name: [] for name in names}
    for batch in loader:
        clinical, col, oct_patches, _, case_semantic = move_batch(batch, device)
        outputs = model.all_subset_outputs(
            clinical, col, oct_patches, case_semantic
        )
        for name in names:
            probability = torch.sigmoid(outputs[f"logit_{name}"]).cpu().numpy()
            mask = outputs[f"mask_{name}"][0].cpu().numpy()
            for index, value in enumerate(probability):
                rows[name].append(
                    {
                        "case_hash": batch["case_hash"][index],
                        "center_name": batch["center_name"][index],
                        "y2": int(batch["y2"][index]),
                        "y3": int(batch["y3"][index]),
                        "probability": float(value),
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
def collect_policy_predictions(
    model: ActiveFusionV2,
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
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader):
        clinical, col, oct_patches, _, case_semantic = move_batch(batch, device)
        output = model.run_policy(
            clinical,
            col,
            oct_patches,
            case_semantic,
            policy=policy,
            cost_weight=cost_weight,
            uncertainty_threshold=uncertainty_threshold,
            random_acquisition_probability=random_probability,
            random_seed=seed * 100003 + batch_index,
            colposcopy_cost=colposcopy_cost,
            oct_cost=oct_cost,
        )
        probability = output["probability"].cpu().numpy()
        acquired = output["acquired_mask"].cpu().numpy()
        actions = output["actions"].cpu().numpy()
        cost = output["cost"].cpu().numpy()
        for index, value in enumerate(probability):
            rows.append(
                {
                    "case_hash": batch["case_hash"][index],
                    "center_name": batch["center_name"][index],
                    "y2": int(batch["y2"][index]),
                    "y3": int(batch["y3"][index]),
                    "probability": float(value),
                    "action_1": ACTION_NAMES[int(actions[index, 0])],
                    "action_2": ACTION_NAMES[int(actions[index, 1])],
                    "acquired_colposcopy": int(acquired[index, 1]),
                    "acquired_oct": int(acquired[index, 2]),
                    "acquisition_count": int(output["acquisition_count"][index].item()),
                    "cost": float(cost[index]),
                    "col_count": float(batch["col_count"][index]),
                    "oct_count": float(batch["oct_count"][index]),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_intervention_responses(
    model: ActiveFusionV2,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate high-attention against low and two deterministic random controls."""
    model.eval()
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader):
        clinical, col, oct_patches, labels, case_semantic = move_batch(batch, device)
        factual = model.all_subset_outputs(
            clinical, col, oct_patches, case_semantic
        )
        factual_logit = factual["logit_all"]
        direction = labels.float() * 2.0 - 1.0
        for modality, patches, other in (
            ("colposcopy", col, oct_patches),
            ("oct", oct_patches, col),
        ):
            patch_count = patches.shape[1]
            count = max(1, int(round(model.config.intervention_fraction * patch_count)))
            attention = factual[f"attention_{modality}"]
            high = torch.topk(attention, count, dim=-1).indices
            low = torch.topk(-attention, count, dim=-1).indices
            base = torch.arange(count, device=device).unsqueeze(0)
            offsets = torch.arange(len(labels), device=device).unsqueeze(1)
            random_one = (base + offsets * max(1, count) + batch_index) % patch_count
            random_two = (
                base + (offsets + len(labels)) * max(1, count + 1) + batch_index
            ) % patch_count
            effects = []
            for indices in (high, low, random_one, random_two):
                variant = model._replace_patches(patches, indices)
                if modality == "colposcopy":
                    altered = model.all_subset_outputs(
                        clinical, variant, other, case_semantic
                    )["logit_all"]
                else:
                    altered = model.all_subset_outputs(
                        clinical, other, variant, case_semantic
                    )["logit_all"]
                effects.append(direction * (factual_logit - altered))
            random_effect = torch.stack(effects[2:], dim=0).mean(0)
            for index in range(len(labels)):
                targeted = float(effects[0][index].item())
                low_effect = float(effects[1][index].item())
                random_value = float(random_effect[index].item())
                rows.append(
                    {
                        "case_hash": batch["case_hash"][index],
                        "center_name": batch["center_name"][index],
                        "modality": modality,
                        "y2": int(batch["y2"][index]),
                        "targeted_effect": targeted,
                        "low_attention_effect": low_effect,
                        "random_effect": random_value,
                        "targeted_minus_random": targeted - random_value,
                        "targeted_minus_low": targeted - low_effect,
                    }
                )
    return pd.DataFrame(rows)


def summarize_interventions(frame: pd.DataFrame) -> list[dict[str, object]]:
    summaries = []
    for modality, group in frame.groupby("modality", sort=True):
        summaries.append(
            {
                "modality": str(modality),
                "n": int(len(group)),
                "targeted_effect_mean": float(group["targeted_effect"].mean()),
                "random_effect_mean": float(group["random_effect"].mean()),
                "low_attention_effect_mean": float(group["low_attention_effect"].mean()),
                "targeted_minus_random_mean": float(group["targeted_minus_random"].mean()),
                "targeted_minus_low_mean": float(group["targeted_minus_low"].mean()),
            }
        )
    return summaries


def validation_checkpoint_score(frames: Mapping[str, pd.DataFrame]) -> float:
    subset_brier = np.mean(
        [brier_score_loss(frame.y2, frame.probability) for frame in frames.values()]
    )
    complete = frames["all"]
    return float(
        subset_brier
        + brier_score_loss(complete.y2, complete.probability)
        - 0.10 * average_precision_score(complete.y2, complete.probability)
    )


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
) -> tuple[ActiveFusionV2, DataLoader, DataLoader, list[dict[str, float]], dict[str, float]]:
    seed_everything(args.seed)
    arm_config = ARM_CONFIGS[arm]
    semantic_mode = str(arm_config["query_mode"])
    prevalence = float(train_frame["pathology_cin2plus"].mean())
    pos_weight = float(np.clip((1.0 - prevalence) / max(prevalence, 1e-6), 1.0, 8.0))
    config = ActiveFusionV2Config(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        query_mode=str(arm_config["query_mode"]),
        fusion_mode=str(arm_config["fusion_mode"]),
        query_seed=args.seed + 20260729,
        use_memory=bool(arm_config["use_memory"]),
        use_bicer=bool(arm_config["use_bicer"]),
        memory_prototypes=args.memory_prototypes,
        pos_weight=pos_weight,
    )
    model = ActiveFusionV2(config, tokens).to(device)
    train_loader = make_loader(
        train_frame, cache, case_vlm_features, semantic_mode,
        batch_size=args.batch_size, shuffle=True,
        seed=args.seed, num_workers=args.num_workers,
    )
    train_eval_loader = make_loader(
        train_frame, cache, case_vlm_features, semantic_mode,
        batch_size=args.batch_size, shuffle=False,
        seed=args.seed, num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_frame, cache, case_vlm_features, semantic_mode,
        batch_size=args.batch_size, shuffle=False,
        seed=args.seed, num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            clinical, col, oct_patches, labels, case_semantic = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = model.training_losses(
                clinical, col, oct_patches, labels, case_semantic,
                lambda_utility=args.lambda_utility if bool(arm_config["active"]) else 0.0,
                lambda_brier=args.lambda_brier,
                lambda_bicer=args.lambda_bicer if bool(arm_config["use_bicer"]) else 0.0,
                lambda_case_semantic=args.lambda_case_semantic,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batches += 1
            for key in (
                "loss", "classification", "brier", "utility",
                "case_semantic", "monotonic", "bicer", "bicer_gap",
            ):
                totals[key] = totals.get(key, 0.0) + float(pieces[key].detach().item())
        val_subsets = collect_subset_predictions(model, val_loader, device)
        score = validation_checkpoint_score(val_subsets)
        record = {"epoch": float(epoch), "validation_score": score}
        record.update({key: value / batches for key, value in totals.items()})
        history.append(record)
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
        raise RuntimeError("training produced no source-validation checkpoint")
    model.load_state_dict(best_state)
    memory_summary: dict[str, float] = {}
    if bool(arm_config["use_memory"]):
        batches = []
        for batch in train_eval_loader:
            clinical, col, oct_patches, labels, case_semantic = move_batch(batch, device)
            batches.append((clinical, col, oct_patches, labels, case_semantic))
        memory_summary = model.fit_utility_memory(batches)
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
    return model, train_eval_loader, val_loader, history, memory_summary


def select_by_retention(
    candidates: Sequence[dict[str, object]],
    *,
    static_auprc: float,
    retention: float,
) -> dict[str, object]:
    viable = [
        candidate for candidate in candidates
        if float(candidate["metrics"]["cin2_auprc"]) >= retention * static_auprc
    ]
    if viable:
        return min(
            viable,
            key=lambda item: (
                float(item["metrics"]["mean_acquisition_count"]),
                float(item["metrics"]["cin2_brier"]),
                -float(item["metrics"]["cin2_auprc"]),
            ),
        )
    return max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["cin2_auprc"])
            - 0.02 * float(item["metrics"]["mean_acquisition_count"]),
            -float(item["metrics"]["cin2_brier"]),
        ),
    )


def tune_source_validation(
    model: ActiveFusionV2,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    static_frame = collect_policy_predictions(
        model, val_loader, device, policy="static_all", seed=args.seed,
    )
    static_metrics = summarize_predictions(static_frame)
    candidates: list[dict[str, object]] = []
    for weight in COST_WEIGHT_GRID:
        frame = collect_policy_predictions(
            model, val_loader, device, policy="learned", seed=args.seed, cost_weight=weight,
        )
        candidates.append(
            {"policy": "learned", "cost_weight": weight, "metrics": summarize_predictions(frame)}
        )
    selected = select_by_retention(
        candidates,
        static_auprc=float(static_metrics["cin2_auprc"]),
        retention=args.performance_retention,
    )
    audits: list[dict[str, object]] = []
    for threshold in UNCERTAINTY_GRID:
        for policy in ("uncertainty", "cheapest_first"):
            frame = collect_policy_predictions(
                model, val_loader, device, policy=policy, seed=args.seed,
                uncertainty_threshold=threshold,
            )
            audits.append(
                {"policy": policy, "uncertainty_threshold": threshold, "metrics": summarize_predictions(frame)}
            )
    for probability in RANDOM_PROBABILITY_GRID:
        frame = collect_policy_predictions(
            model, val_loader, device, policy="random", seed=args.seed,
            random_probability=probability,
        )
        audits.append(
            {"policy": "random", "random_probability": probability, "metrics": summarize_predictions(frame)}
        )
    audits.append({"policy": "static_all", "metrics": static_metrics})
    audits.append(
        {
            "policy": "clinical_only",
            "metrics": summarize_predictions(
                collect_policy_predictions(
                    model, val_loader, device, policy="clinical_only", seed=args.seed,
                )
            ),
        }
    )
    return {
        "selection_data": "source_validation_only",
        "performance_retention": args.performance_retention,
        "static_all": static_metrics,
        "learned_candidates": candidates,
        "selected_learned": selected,
        "policy_audits": audits,
    }


def discover_folds(split_root: Path) -> list[Path]:
    return sorted(
        path for path in split_root.glob("loco_*")
        if path.is_dir() and (path / "train_labels.csv").exists()
    )


def validate_formal_guard(mode: str, fold_paths: Sequence[Path], seeds: Sequence[int]) -> None:
    if mode != "formal":
        return
    names = {path.name for path in fold_paths}
    if names != EXPECTED_FOLDS:
        raise ValueError(f"formal mode requires exactly five locked folds; got {sorted(names)}")
    if len(set(seeds)) < 3:
        raise ValueError("formal mode requires at least three distinct seeds")


def evaluate_frozen_policy(
    model: ActiveFusionV2,
    test_loader: DataLoader,
    device: torch.device,
    selection: Mapping[str, object],
    args: argparse.Namespace,
    arm: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    is_active = bool(ARM_CONFIGS[arm]["active"])
    if is_active:
        chosen = selection["selected_learned"]
        primary = collect_policy_predictions(
            model, test_loader, device, policy="learned", seed=args.seed,
            cost_weight=float(chosen["cost_weight"]),
        )
    else:
        primary = collect_policy_predictions(
            model, test_loader, device, policy="static_all", seed=args.seed,
        )
    curves: list[dict[str, object]] = []
    if is_active:
        for scenario, (col_cost, oct_cost) in COST_SCENARIOS.items():
            for weight in COST_WEIGHT_GRID:
                frame = collect_policy_predictions(
                    model, test_loader, device, policy="learned", seed=args.seed,
                    cost_weight=weight, colposcopy_cost=col_cost, oct_cost=oct_cost,
                )
                curves.append(
                    {
                        "scenario": scenario,
                        "colposcopy_relative_cost": col_cost,
                        "oct_relative_cost": oct_cost,
                        "cost_weight": weight,
                        **summarize_predictions(frame),
                    }
                )
    return primary, curves


def run_one(
    fold_path: Path,
    arm: str,
    args: argparse.Namespace,
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    case_vlm_features: Mapping[str, Mapping[str, torch.Tensor]],
    tokens: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    job_dir = args.output / args.mode / fold_path.name / f"seed_{args.seed}" / arm
    job_dir.mkdir(parents=True, exist_ok=True)
    completion_path = job_dir / "completion.json"
    if completion_path.exists():
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        expected_status = (
            "formal_target_evaluation_complete"
            if args.mode == "formal"
            else "source_development_complete_target_outcomes_not_loaded_by_runner"
        )
        if (
            completed.get("schema") == SCHEMA
            and completed.get("status") == expected_status
            and completed.get("fold") == fold_path.name
            and completed.get("arm") == arm
            and int(completed.get("seed")) == int(args.seed)
        ):
            print(
                f"[resume] fold={fold_path.name} seed={args.seed} arm={arm}",
                flush=True,
            )
            return completed
    train_path = fold_path / "train_labels.csv"
    val_path = fold_path / "val_labels.csv"
    test_path = fold_path / "external_test_labels.csv"
    original_train = load_frame(train_path)
    original_validation = load_frame(val_path)
    train_frame, val_frame, source_split_audit = derive_source_train_validation(
        original_train,
        original_validation,
        seed=args.seed,
    )
    held_center = pd.read_csv(test_path, usecols=["center_name"], encoding="utf-8-sig")["center_name"].iloc[0]
    model, _, val_loader, history, memory_summary = train_model(
        train_frame, val_frame, cache, case_vlm_features, tokens, arm, args, device, job_dir,
    )
    if bool(ARM_CONFIGS[arm]["active"]):
        selection = tune_source_validation(model, val_loader, device, args)
    else:
        static_validation = summarize_predictions(
            collect_policy_predictions(
                model, val_loader, device, policy="static_all", seed=args.seed,
            )
        )
        selection = {
            "selection_data": "source_validation_only",
            "static_all": static_validation,
            "selected_learned": None,
            "policy_audits": [],
        }
    if bool(ARM_CONFIGS[arm]["active"]):
        selected = selection["selected_learned"]
        validation_primary = collect_policy_predictions(
            model,
            val_loader,
            device,
            policy="learned",
            seed=args.seed,
            cost_weight=float(selected["cost_weight"]),
        )
    else:
        validation_primary = collect_policy_predictions(
            model,
            val_loader,
            device,
            policy="static_all",
            seed=args.seed,
        )
    selection["operating_thresholds"] = {
        "cin2_youden": choose_youden_threshold(
            validation_primary["y2"].to_numpy(dtype=int),
            validation_primary["probability"].to_numpy(dtype=float),
        ),
        "cin3_safety_95pct_source_validation": choose_safety_threshold(
            validation_primary["y3"].to_numpy(dtype=int),
            validation_primary["probability"].to_numpy(dtype=float),
            floor=0.95,
        ),
    }
    source_intervention_summary: list[dict[str, object]] = []
    if arm in {"bvoi_bicer", "bvoi_memory", "bvoi_memory_bicer"}:
        source_interventions = collect_intervention_responses(
            model, val_loader, device,
        )
        source_interventions.to_csv(
            job_dir / "source_validation_interventions.csv",
            index=False,
        )
        source_intervention_summary = summarize_interventions(
            source_interventions
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
        "source_split_audit": source_split_audit,
        "target_outcomes_used_for_training_or_selection": False,
        "claim_boundary": (
            "The development runner does not load target outcomes. The cohort has "
            "historical analyses, so this is leakage-safe model development rather "
            "than a prospectively blinded or pristine confirmatory study."
        ),
        "selection": selection,
        "memory": memory_summary,
        "source_validation_interventions": source_intervention_summary,
        "training_history": history,
        "input_hashes": {
            "train": sha256_file(train_path),
            "validation": sha256_file(val_path),
            "concept_tokens": sha256_file(args.concept_tokens),
            "case_vlm": sha256_file(args.case_vlm),
        },
    }
    write_json(job_dir / "frozen_source_development.json", frozen)
    if args.mode == "development":
        completion = {**frozen, "status": "source_development_complete_target_outcomes_not_loaded_by_runner"}
        write_json(completion_path, completion)
        return completion

    # Formal outcome opening occurs only after model, hyperparameters and policy are frozen.
    test_frame = load_frame(test_path)
    test_loader = make_loader(
        test_frame, cache, case_vlm_features, str(ARM_CONFIGS[arm]["query_mode"]),
        batch_size=args.batch_size, shuffle=False,
        seed=args.seed, num_workers=args.num_workers,
    )
    predictions, curves = evaluate_frozen_policy(
        model, test_loader, device, selection, args, arm,
    )
    predictions.to_csv(job_dir / "target_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(job_dir / "performance_cost_curve.csv", index=False)
    target_intervention_summary: list[dict[str, object]] = []
    if arm in {"bvoi_bicer", "bvoi_memory", "bvoi_memory_bicer"}:
        target_interventions = collect_intervention_responses(
            model, test_loader, device,
        )
        target_interventions.to_csv(
            job_dir / "target_interventions.csv",
            index=False,
        )
        target_intervention_summary = summarize_interventions(
            target_interventions
        )
    result = {
        **frozen,
        "status": "formal_target_evaluation_complete",
        "target_n": len(test_frame),
        "primary_metrics": summarize_predictions(
            predictions,
            cin2_threshold=float(selection["operating_thresholds"]["cin2_youden"]),
            cin3_threshold=float(
                selection["operating_thresholds"]["cin3_safety_95pct_source_validation"]
            ),
        ),
        "target_interventions": target_intervention_summary,
        "target_predictions_sha256": sha256_file(job_dir / "target_predictions.csv"),
    }
    write_json(completion_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), default="development")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--concept-tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--case-vlm", type=Path, default=DEFAULT_CASE_VLM)
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
    parser.add_argument("--memory-prototypes", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lambda-utility", type=float, default=0.5)
    parser.add_argument("--lambda-brier", type=float, default=0.2)
    parser.add_argument("--lambda-bicer", type=float, default=0.15)
    parser.add_argument("--lambda-case-semantic", type=float, default=0.15)
    parser.add_argument("--performance-retention", type=float, default=0.97)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_folds = discover_folds(args.split_root)
    selected_folds = (
        [path for path in all_folds if path.name in set(args.fold)]
        if args.fold else all_folds
    )
    selected_arms = args.arm or list(ARM_CONFIGS)
    formal_seeds = args.formal_seeds or [args.seed]
    validate_formal_guard(args.mode, selected_folds, formal_seeds)
    if args.mode == "development" and len(selected_folds) != 1:
        selected_folds = selected_folds[:1]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache = cache_payload.get("features", cache_payload)
    token_payload = torch.load(args.concept_tokens, map_location="cpu", weights_only=False)
    tokens = token_payload["tokens"]
    case_vlm_payload = torch.load(args.case_vlm, map_location="cpu", weights_only=False)
    case_vlm_features = case_vlm_payload["features"]
    results = []
    for seed in formal_seeds:
        args.seed = int(seed)
        for fold in selected_folds:
            for arm in selected_arms:
                results.append(
                    run_one(
                        fold, arm, args, cache, case_vlm_features, tokens, device,
                    )
                )
    manifest = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "mode": args.mode,
        "device": str(device),
        "folds": [path.name for path in selected_folds],
        "seeds": formal_seeds,
        "arms": selected_arms,
        "target_label_policy": (
            "not loaded by development runner" if args.mode == "development"
            else "opened once after each source-developed model and policy were frozen"
        ),
        "jobs": len(results),
    }
    write_json(args.output / args.mode / "campaign_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
