#!/usr/bin/env python3
"""Fast feature-space baselines, requirement ablations, and faithfulness tests.

This script uses the same pretrained ViT patch extractor as the full model, then
trains compact multimodal fusion heads on cached patient-level features. It is
intended to produce auditable reviewer-facing baselines and ablations quickly,
while the full HyDRA model remains the primary method implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
RESULT_DIR = PAPER_DIR / "results"
PRED_DIR = RESULT_DIR / "predictions"
FEATURE_DIR = RESULT_DIR / "feature_cache"
TABLE_DIR = PAPER_DIR / "tables"
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from config import BioCOT_v3_2_Config
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from metrics_utils import binary_metrics, read_prediction_files, select_threshold
from training.extract_vit_patches import extract_patch_features_with_vit


CENTER_NAMES = {
    "M22105": "Enshi",
    "M20105": "Wuda",
    "M20203": "Wuda",
    "M22102": "Xiangyang",
    "M0008": "Jingzhou",
    "M22101": "Jingzhou",
    "M22104": "Shiyan",
}


def center_from_oct(oct_id: str) -> str:
    return CENTER_NAMES.get(str(oct_id).split("_")[0], str(oct_id).split("_")[0])


def split_csv(config: BioCOT_v3_2_Config, split: str) -> Path:
    root = Path(config.data_root)
    return {
        "train": root / "train_labels.csv",
        "internal_validation": root / "val_labels.csv",
        "external_test": root / "external_test_labels.csv",
    }[split]


def clinical_features_from_batch(batch: Dict[str, object], batch_size: int, device: torch.device) -> torch.Tensor:
    clinical_data = batch.get("clinical_data", {})
    if not isinstance(clinical_data, dict):
        return torch.zeros(batch_size, 7, device=device)
    hpv = clinical_data.get("hpv", torch.zeros(batch_size, device=device))
    age = clinical_data.get("age", torch.zeros(batch_size, device=device))
    tct = clinical_data.get("tct", torch.zeros(batch_size, device=device))
    hpv = hpv.to(device).float() if isinstance(hpv, torch.Tensor) else torch.as_tensor(hpv, device=device).float()
    age = age.to(device).float() if isinstance(age, torch.Tensor) else torch.as_tensor(age, device=device).float()
    if isinstance(tct, torch.Tensor):
        tct = tct.to(device).long()
    else:
        mapping = {
            "NILM": 0,
            "ASC-US": 1,
            "LSIL": 2,
            "ASC-H": 3,
            "HSIL": 4,
            "AGC": 4,
            "SCC": 4,
        }
        values = []
        for item in list(tct) if isinstance(tct, (list, tuple)) else [tct] * batch_size:
            text = str(item).strip().upper()
            values.append(mapping.get(text, 0))
        tct = torch.as_tensor(values[:batch_size], device=device).long()
    tct = torch.clamp(tct, 0, 4)
    return torch.cat([hpv.unsqueeze(1), age.unsqueeze(1) / 100.0, F.one_hot(tct, num_classes=5).float()], dim=1)


def pooled_patch_features(images: torch.Tensor, device: torch.device, vit_batch_size: int) -> torch.Tensor:
    if images.dim() == 5:
        b, n = images.shape[:2]
        flat = images.view(b * n, *images.shape[2:])
        feats = extract_patch_features_with_vit(flat, device, batch_size=vit_batch_size, pretrained=True)
        return feats.mean(dim=1).view(b, n, -1).mean(dim=1)
    return extract_patch_features_with_vit(images, device, batch_size=vit_batch_size, pretrained=True).mean(dim=1)


def build_feature_cache(config: BioCOT_v3_2_Config, split: str, device: torch.device, force: bool = False) -> Path:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_DIR / f"{split}_vit_patient_features.npz"
    if out_path.exists() and not force:
        return out_path

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=str(split_csv(config, split)),
        transform=transform,
        oct_num_frames=config.oct_frames,
        max_col_images=config.colposcopy_images,
        balance_negative_frames=False,
        data_root=str(config.data_root),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    oct_feats, col_feats, cli_feats, labels, centers, case_ids, center_names = [], [], [], [], [], [], []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["label"].shape[0]
            oct_img = batch["oct_images"].to(device, non_blocking=True)
            col_img = batch["colposcopy_images"].to(device, non_blocking=True)
            oct_feats.append(pooled_patch_features(oct_img, device, config.vit_batch_size).cpu().numpy())
            col_feats.append(pooled_patch_features(col_img, device, config.vit_batch_size).cpu().numpy())
            cli_feats.append(clinical_features_from_batch(batch, batch_size, device).cpu().numpy())
            labels.append(batch["label"].numpy())
            centers.append(batch["center_idx"].numpy())
            df_slice = dataset.df.iloc[cursor : cursor + batch_size]
            cursor += batch_size
            for _, row in df_slice.iterrows():
                case_ids.append(str(row.get("ID", row.get("patient_id", ""))))
                center_names.append(center_from_oct(str(row.get("OCT", row.get("oct_id", "")))))

    np.savez_compressed(
        out_path,
        oct=np.concatenate(oct_feats),
        col=np.concatenate(col_feats),
        clinical=np.concatenate(cli_feats),
        y=np.concatenate(labels).astype(np.int64),
        center_idx=np.concatenate(centers).astype(np.int64),
        case_id=np.asarray(case_ids, dtype=object),
        center=np.asarray(center_names, dtype=object),
    )
    print(f"Wrote feature cache: {out_path}")
    return out_path


def load_features(split: str) -> Dict[str, np.ndarray]:
    data = np.load(FEATURE_DIR / f"{split}_vit_patient_features.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


@dataclass
class MethodSpec:
    name: str
    fusion: str
    direct: bool = True
    variational: bool = False
    center_aware: bool = True
    refinement_steps: int = 0
    memory: bool = False
    prior: str = "none"  # none/ot/contrastive/both


class FeatureFusionNet(nn.Module):
    def __init__(self, spec: MethodSpec, hidden: int = 256, num_centers: int = 5):
        super().__init__()
        self.spec = spec
        self.hidden = hidden
        self.oct_proj = nn.Sequential(nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15))
        self.col_proj = nn.Sequential(nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15))
        self.cli_proj = nn.Sequential(nn.Linear(7, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1))
        self.concat_proj = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden, 2))
        self.late_heads = nn.ModuleList([nn.Linear(hidden, 2), nn.Linear(hidden, 2), nn.Linear(hidden, 2)])
        self.center_embedding = nn.Embedding(num_centers, hidden)
        self.rel_heads = nn.ModuleList([nn.Linear(hidden * (2 if spec.center_aware else 1), hidden * 2) for _ in range(3)])
        self.gru = nn.GRUCell(hidden, hidden)
        self.memory_slots = nn.Parameter(torch.randn(num_centers, 4, hidden) * 0.02)
        self.class_prototypes = nn.Parameter(torch.randn(2, hidden) * 0.02)

    def encode(self, oct_x, col_x, cli_x):
        return self.oct_proj(oct_x), self.col_proj(col_x), self.cli_proj(cli_x)

    def variational_fuse(self, feats: List[torch.Tensor], center_idx: torch.Tensor):
        weights, samples, kl_terms, mus, logvars = [], [], [], [], []
        center_context = self.center_embedding(center_idx) if self.spec.center_aware else None
        for idx, feat in enumerate(feats):
            head_in = torch.cat([feat, center_context], dim=-1) if center_context is not None else feat
            mu, raw_logvar = self.rel_heads[idx](head_in).chunk(2, dim=-1)
            logvar = raw_logvar.clamp(-6.0, 3.0)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            precision = torch.exp(-logvar).mean(dim=-1, keepdim=True)
            weights.append(precision)
            samples.append(z)
            mus.append(mu)
            logvars.append(logvar)
            kl_terms.append((-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean())
        w = torch.stack(weights, dim=1).clamp_min(1e-6)
        w = w / w.sum(dim=1, keepdim=True)
        fused = sum(w[:, i, :] * samples[i] for i in range(3))
        return fused, w.squeeze(-1), torch.stack(kl_terms).mean(), mus, logvars

    def forward(self, oct_x, col_x, cli_x, center_idx, y=None):
        f_oct, f_col, f_cli = self.encode(oct_x, col_x, cli_x)
        aux = {}
        weights = None

        if self.spec.fusion == "late":
            logits = sum(head(feat) for head, feat in zip(self.late_heads, [f_oct, f_col, f_cli])) / 3.0
            return logits, {"weights": None, "aux_loss": torch.tensor(0.0, device=oct_x.device)}
        if self.spec.variational:
            z, weights, kl, _, _ = self.variational_fuse([f_oct, f_col, f_cli], center_idx)
            aux["kl"] = kl
        elif self.spec.fusion == "concat":
            z = self.concat_proj(torch.cat([f_oct, f_col, f_cli], dim=-1))
        elif self.spec.fusion == "gated":
            weights = F.softmax(self.gate(torch.cat([f_oct, f_col, f_cli], dim=-1)), dim=-1)
            z = weights[:, 0:1] * f_oct + weights[:, 1:2] * f_col + weights[:, 2:3] * f_cli
        elif self.spec.fusion == "cross_attention":
            tokens = torch.stack([f_oct, f_col, f_cli], dim=1)
            z, attn = self.attn(tokens.mean(dim=1, keepdim=True), tokens, tokens)
            z = z.squeeze(1)
            weights = attn.squeeze(1)
        else:
            weights = torch.ones(oct_x.shape[0], 3, device=oct_x.device) / 3.0
            z = (f_oct + f_col + f_cli) / 3.0

        if self.spec.memory:
            slots = self.memory_slots[center_idx.clamp_min(0).clamp_max(self.memory_slots.shape[0] - 1)]
            attn = F.softmax(torch.einsum("bd,bkd->bk", z, slots) / (self.hidden ** 0.5), dim=-1)
            evidence = torch.einsum("bk,bkd->bd", attn, slots)
            aux["memory_attention"] = attn
        else:
            evidence = torch.zeros_like(z)

        for _ in range(self.spec.refinement_steps):
            z = self.gru(evidence, z)

        aux_loss = torch.tensor(0.0, device=oct_x.device)
        if self.spec.variational:
            aux_loss = aux_loss + 0.005 * aux["kl"]
        if y is not None and self.spec.prior in {"ot", "both"}:
            aux_loss = aux_loss + 0.05 * F.mse_loss(z, self.class_prototypes[y])
        if y is not None and self.spec.prior in {"contrastive", "both"}:
            proto_logits = z @ F.normalize(self.class_prototypes, dim=-1).t()
            aux_loss = aux_loss + 0.1 * F.cross_entropy(proto_logits, y)

        logits = self.classifier(z)
        return logits, {"weights": weights, "aux_loss": aux_loss}


def make_tensors(features: Dict[str, np.ndarray], stats: Optional[Dict[str, np.ndarray]] = None):
    if stats is None:
        stats = {}
        for key in ["oct", "col"]:
            stats[f"{key}_mean"] = features[key].mean(axis=0, keepdims=True)
            stats[f"{key}_std"] = features[key].std(axis=0, keepdims=True) + 1e-6
    arrays = {
        "oct": (features["oct"] - stats["oct_mean"]) / stats["oct_std"],
        "col": (features["col"] - stats["col_mean"]) / stats["col_std"],
        "clinical": features["clinical"],
        "y": features["y"],
        "center_idx": features["center_idx"],
    }
    return arrays, stats


def batch_iter(data: Dict[str, np.ndarray], batch_size: int, shuffle: bool, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data["y"]))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        yield idx


def predict(model: FeatureFusionNet, data: Dict[str, np.ndarray], device: torch.device, missing: str = "none") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    probs, weights_all = [], []
    with torch.no_grad():
        for idx in batch_iter(data, 256, False, 0):
            oct_x = torch.as_tensor(data["oct"][idx], dtype=torch.float32, device=device)
            col_x = torch.as_tensor(data["col"][idx], dtype=torch.float32, device=device)
            cli_x = torch.as_tensor(data["clinical"][idx], dtype=torch.float32, device=device)
            center = torch.as_tensor(data["center_idx"][idx], dtype=torch.long, device=device)
            if missing in {"remove_oct", "random_one_oct"}:
                oct_x.zero_()
            if missing in {"remove_colposcopy", "random_one_col"}:
                col_x.zero_()
            if missing in {"remove_clinical_prior", "random_one_cli"}:
                cli_x.zero_()
            logits, extra = model(oct_x, col_x, cli_x, center)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            w = extra.get("weights")
            if w is not None:
                weights_all.append(w.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(weights_all) if weights_all else None


def train_one(spec: MethodSpec, train, val, external, meta_external, device, seed: int, epochs: int, run_id: str):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = FeatureFusionNet(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train["y"], minlength=2)
    class_weight = torch.as_tensor((len(train["y"]) / np.maximum(counts, 1)) / 2.0, dtype=torch.float32, device=device)
    best_auc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for idx in batch_iter(train, 64, True, seed + epoch):
            oct_x = torch.as_tensor(train["oct"][idx], dtype=torch.float32, device=device)
            col_x = torch.as_tensor(train["col"][idx], dtype=torch.float32, device=device)
            cli_x = torch.as_tensor(train["clinical"][idx], dtype=torch.float32, device=device)
            center = torch.as_tensor(train["center_idx"][idx], dtype=torch.long, device=device)
            y = torch.as_tensor(train["y"][idx], dtype=torch.long, device=device)
            logits, extra = model(oct_x, col_x, cli_x, center, y)
            loss = F.cross_entropy(logits, y, weight=class_weight) + extra["aux_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_prob, _ = predict(model, val, device)
        val_auc = binary_metrics(val["y"], val_prob).auc
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    RESULT_DIR.joinpath("feature_models").mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "spec": spec.__dict__, "best_val_auc": best_auc}, RESULT_DIR / "feature_models" / f"{spec.name}_seed{seed}.pt")

    write_predictions(spec.name, run_id, seed, "internal_validation", val, meta=None, probs=predict(model, val, device)[0], weights=predict(model, val, device)[1])
    ext_probs, ext_weights = predict(model, external, device)
    write_predictions(spec.name, run_id, seed, "external_test", external, meta=meta_external, probs=ext_probs, weights=ext_weights)
    if spec.name == "HyDRA_FeatureVariational":
        for setting in ["remove_oct", "remove_colposcopy", "remove_clinical_prior"]:
            probs, weights = predict(model, external, device, missing=setting)
            write_predictions(f"{spec.name}_{setting}", run_id, seed, "external_test", external, meta_external, probs, weights, modality_setting=setting)
        run_faithfulness(model, spec.name, external, meta_external, device, seed)
    return best_auc


def write_predictions(method, run_id, seed, split, data, meta, probs, weights, modality_setting: str = "none") -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"{method}_run{run_id}_seed{seed}_{split}_{'full' if modality_setting == 'none' else modality_setting}.csv"
    if meta is None:
        case_ids = [f"{split}_{i}" for i in range(len(data["y"]))]
        centers = ["internal"] * len(data["y"])
    else:
        case_ids = [str(x) for x in meta["case_id"]]
        centers = [str(x) for x in meta["center"]]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "run_id", "seed", "split", "case_id", "center", "y_true", "y_prob", "modality_setting", "reliability_oct", "reliability_colposcopy", "reliability_clinical_prior"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, prob in enumerate(probs):
            row = {
                "method": method,
                "run_id": run_id,
                "seed": seed,
                "split": split,
                "case_id": case_ids[i],
                "center": centers[i],
                "y_true": int(data["y"][i]),
                "y_prob": float(prob),
                "modality_setting": modality_setting,
                "reliability_oct": "",
                "reliability_colposcopy": "",
                "reliability_clinical_prior": "",
            }
            if weights is not None:
                row["reliability_oct"] = float(weights[i, 0])
                row["reliability_colposcopy"] = float(weights[i, 1])
                row["reliability_clinical_prior"] = float(weights[i, 2])
            writer.writerow(row)


def run_faithfulness(model, method, external, meta, device, seed: int):
    base_prob, weights = predict(model, external, device)
    if weights is None:
        return
    top_mod = weights.argmax(axis=1)
    setting_map = {0: "remove_oct", 1: "remove_colposcopy", 2: "remove_clinical_prior"}
    drops, random_drops = [], []
    rng = np.random.default_rng(seed)
    for i, mod_idx in enumerate(top_mod):
        removed_prob, _ = predict_single_removed(model, external, i, setting_map[int(mod_idx)], device)
        random_idx = int(rng.integers(0, 3))
        random_prob, _ = predict_single_removed(model, external, i, setting_map[random_idx], device)
        drops.append(base_prob[i] - removed_prob)
        random_drops.append(base_prob[i] - random_prob)

    swap_changes = []
    y = external["y"]
    for i, mod_idx in enumerate(top_mod):
        candidates = np.where(y != y[i])[0]
        if len(candidates) == 0:
            continue
        j = int(rng.choice(candidates))
        swapped_prob = predict_single_swapped(model, external, i, j, int(mod_idx), device)
        swap_changes.append(abs(base_prob[i] - swapped_prob))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "method": method,
                "test": "cited_modality_removal",
                "mean_probability_drop": float(np.mean(drops)),
                "random_removal_drop": float(np.mean(random_drops)),
                "faithfulness_margin": float(np.mean(drops) - np.mean(random_drops)),
                "n": len(drops),
            },
            {
                "method": method,
                "test": "counterfactual_evidence_swap",
                "mean_absolute_probability_change": float(np.mean(swap_changes)) if swap_changes else float("nan"),
                "random_removal_drop": "",
                "faithfulness_margin": "",
                "n": len(swap_changes),
            },
        ]
    )
    summary.to_csv(TABLE_DIR / "coe_faithfulness_automatic_metrics.csv", index=False)

    threshold = select_threshold(external["y"], base_prob)
    pred = (base_prob >= threshold).astype(int)
    cases = pd.DataFrame(
        {
            "case_id": meta["case_id"],
            "center": meta["center"],
            "y_true": external["y"],
            "y_prob": base_prob,
            "y_pred": pred,
            "top_evidence_modality": [setting_map[int(i)].replace("remove_", "") for i in top_mod],
            "reliability_oct": weights[:, 0],
            "reliability_colposcopy": weights[:, 1],
            "reliability_clinical_prior": weights[:, 2],
        }
    )
    cases["error_type"] = np.where(
        (cases.y_true == cases.y_pred) & (cases[["reliability_oct", "reliability_colposcopy", "reliability_clinical_prior"]].max(axis=1) < 0.45),
        "correct_low_evidence_confidence",
        np.where(cases.y_true != cases.y_pred, "wrong_prediction", "correct_prediction"),
    )
    cases.sort_values(["error_type", "y_prob"]).to_csv(TABLE_DIR / "failure_cases_automatic.csv", index=False)


def predict_single_removed(model, data, i, setting, device):
    one = {k: v[i : i + 1].copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    return predict(model, one, device, missing=setting)


def predict_single_swapped(model, data, i, j, mod_idx, device):
    one = {k: v[i : i + 1].copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    if mod_idx == 0:
        one["oct"][0] = data["oct"][j]
    elif mod_idx == 1:
        one["col"][0] = data["col"][j]
    else:
        one["clinical"][0] = data["clinical"][j]
    return predict(model, one, device)[0][0]


def summarize_feature_tables() -> None:
    df = read_prediction_files(PRED_DIR)
    if df.empty:
        return
    rows = []
    for (method, run_id, seed), group in df.groupby(["method", "run_id", "seed"], dropna=False):
        source = group[group["split"] == "internal_validation"]
        threshold = select_threshold(source["y_true"], source["y_prob"]) if not source.empty else select_threshold(group["y_true"], group["y_prob"])
        for split, split_df in group.groupby("split"):
            metric = binary_metrics(split_df["y_true"], split_df["y_prob"], threshold=threshold)
            rows.append({"method": method, "split": split, "run_id": run_id, "seed": seed, **metric.__dict__})
    metrics = pd.DataFrame(rows)

    direct_names = ["Concat_Fusion", "Late_Fusion", "Gated_Fusion", "CrossAttention_Fusion"]
    ablation_prefix = "Ablation_"
    for name, mask in [
        ("direct_fusion_baseline_metrics", metrics["method"].isin(direct_names)),
        ("requirement_ablation_metrics", metrics["method"].astype(str).str.startswith(ablation_prefix) | (metrics["method"] == "HyDRA_FeatureVariational")),
    ]:
        table = metrics[mask & (metrics["split"] == "external_test")].copy()
        if table.empty:
            continue
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        display = table[["method", "n", "auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier"]].copy()
        for col in ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier"]:
            display[col] = display[col].map(lambda x: f"{x:.3f}")
        display.to_csv(TABLE_DIR / f"{name}_formatted.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    config = BioCOT_v3_2_Config()
    config.vit_pretrained = True
    config.use_vlm_retriever = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ["train", "internal_validation", "external_test"]:
        build_feature_cache(config, split, device, force=args.force_features)

    train_raw, val_raw, ext_raw = load_features("train"), load_features("internal_validation"), load_features("external_test")
    train, stats = make_tensors(train_raw)
    val, _ = make_tensors(val_raw, stats)
    external, _ = make_tensors(ext_raw, stats)
    meta_external = {"case_id": ext_raw["case_id"], "center": ext_raw["center"]}

    specs = [
        MethodSpec("Concat_Fusion", "concat"),
        MethodSpec("Late_Fusion", "late"),
        MethodSpec("Gated_Fusion", "gated"),
        MethodSpec("CrossAttention_Fusion", "cross_attention"),
        MethodSpec("HyDRA_FeatureVariational", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_DeterministicGate", "gated", variational=False, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_EqualWeightFusion", "equal", variational=False, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_NoCenterAwareReliability", "variational", variational=True, center_aware=False, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_OneShotNoRefinement", "variational", variational=True, center_aware=True, refinement_steps=0, memory=True, prior="both"),
        MethodSpec("Ablation_NoMemoryRetrieval", "variational", variational=True, center_aware=True, refinement_steps=2, memory=False, prior="both"),
        MethodSpec("Ablation_NoClinicalPriorMatching", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="none"),
        MethodSpec("Ablation_OTOnly", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="ot"),
        MethodSpec("Ablation_ContrastiveOnly", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="contrastive"),
    ]
    summary = []
    for spec in specs:
        best_val_auc = train_one(spec, train, val, external, meta_external, device, args.seed, args.epochs, run_id="feature")
        summary.append({"method": spec.name, "best_internal_auc": best_val_auc})
        print(f"{spec.name}: best internal AUC={best_val_auc:.4f}")

    pd.DataFrame(summary).to_csv(TABLE_DIR / "feature_space_training_summary.csv", index=False)
    summarize_feature_tables()
    print(f"Wrote feature-space experiment outputs to {PRED_DIR} and {TABLE_DIR}")


if __name__ == "__main__":
    main()
