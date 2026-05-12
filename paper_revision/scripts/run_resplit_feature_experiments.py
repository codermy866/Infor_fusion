#!/usr/bin/env python3
"""Feature-space experiments for the full-multimodal resplit protocol.

This script reuses the cached patient-level ViT features extracted from the
complete 985-case multimodal cohort, then evaluates the revised validation
plan:

1. Main external validation: Enshi held out.
2. Historical continuity: Jingzhou + Shiyan held out.
3. Supplementary leave-one-center-out validation across the five full centers.

The output is intended as a fast, auditable companion to the full HyDRA model
training. It produces direct fusion baselines, requirement-level ablations,
missing-modality robustness, center-wise calibration, and automatic CoE
faithfulness proxies under the new resplit policy.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
SPLIT_ROOT = PAPER_DIR / "splits" / "full_multimodal_resplit"
FEATURE_DIR = PAPER_DIR / "results" / "feature_cache"
RESULT_DIR = PAPER_DIR / "results"
PRED_DIR = RESULT_DIR / "resplit_feature_predictions"
MODEL_DIR = RESULT_DIR / "resplit_feature_models"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"

sys.path.insert(0, str(SCRIPT_DIR))

from metrics_utils import binary_metrics, decision_curve_points, select_threshold
from run_feature_space_experiments import FeatureFusionNet, MethodSpec, batch_iter, make_tensors


METRIC_COLS = ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier"]
DIRECT_METHODS = ["Concat_Fusion", "Late_Fusion", "Gated_Fusion", "CrossAttention_Fusion"]
PRIMARY_METHOD = "HyDRA_FeatureVariational"
ABLATION_METHODS = [
    "Ablation_DeterministicGate",
    "Ablation_EqualWeightFusion",
    "Ablation_NoCenterAwareReliability",
    "Ablation_OneShotNoRefinement",
    "Ablation_NoMemoryRetrieval",
    "Ablation_NoClinicalPriorMatching",
    "Ablation_OTOnly",
    "Ablation_ContrastiveOnly",
]


def method_specs() -> List[MethodSpec]:
    return [
        MethodSpec("Concat_Fusion", "concat"),
        MethodSpec("Late_Fusion", "late"),
        MethodSpec("Gated_Fusion", "gated"),
        MethodSpec("CrossAttention_Fusion", "cross_attention"),
        MethodSpec(PRIMARY_METHOD, "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_DeterministicGate", "gated", variational=False, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_EqualWeightFusion", "equal", variational=False, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_NoCenterAwareReliability", "variational", variational=True, center_aware=False, refinement_steps=2, memory=True, prior="both"),
        MethodSpec("Ablation_OneShotNoRefinement", "variational", variational=True, center_aware=True, refinement_steps=0, memory=True, prior="both"),
        MethodSpec("Ablation_NoMemoryRetrieval", "variational", variational=True, center_aware=True, refinement_steps=2, memory=False, prior="both"),
        MethodSpec("Ablation_NoClinicalPriorMatching", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="none"),
        MethodSpec("Ablation_OTOnly", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="ot"),
        MethodSpec("Ablation_ContrastiveOnly", "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="contrastive"),
    ]


def fast_specs() -> List[MethodSpec]:
    return [
        MethodSpec("Late_Fusion", "late"),
        MethodSpec("Gated_Fusion", "gated"),
        MethodSpec(PRIMARY_METHOD, "variational", variational=True, center_aware=True, refinement_steps=2, memory=True, prior="both"),
    ]


def read_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "ID" not in df.columns:
        raise ValueError(f"{path} is missing ID column.")
    if "center_group_id" not in df.columns:
        raise ValueError(f"{path} is missing center_group_id column.")
    return df


def load_master_features() -> Dict[str, Dict[str, object]]:
    master: Dict[str, Dict[str, object]] = {}
    for split in ["train", "internal_validation", "external_test"]:
        path = FEATURE_DIR / f"{split}_vit_patient_features.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached feature file: {path}")
        data = np.load(path, allow_pickle=True)
        for idx, case_id in enumerate(data["case_id"]):
            key = str(case_id)
            if key in master:
                raise ValueError(f"Duplicate case_id in cached features: {key}")
            master[key] = {
                "oct": data["oct"][idx].astype(np.float32),
                "col": data["col"][idx].astype(np.float32),
                "clinical": data["clinical"][idx].astype(np.float32),
                "label": int(data["y"][idx]),
            }
    return master


def features_from_csv(path: Path, master: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = read_label_csv(path)
    missing = [str(case_id) for case_id in df["ID"].astype(str) if str(case_id) not in master]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} cases in {path} are missing from cached features, e.g. {preview}")

    oct_arr, col_arr, cli_arr, labels, centers = [], [], [], [], []
    case_ids, center_names, oct_ids = [], [], []
    for _, row in df.iterrows():
        case_id = str(row["ID"])
        item = master[case_id]
        oct_arr.append(item["oct"])
        col_arr.append(item["col"])
        cli_arr.append(item["clinical"])
        labels.append(int(row["label"]))
        centers.append(int(row["center_group_id"]))
        case_ids.append(case_id)
        center_names.append(str(row.get("center_name", "")))
        oct_ids.append(str(row.get("OCT", row.get("oct_id", ""))))

    features = {
        "oct": np.stack(oct_arr).astype(np.float32),
        "col": np.stack(col_arr).astype(np.float32),
        "clinical": np.stack(cli_arr).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "center_idx": np.asarray(centers, dtype=np.int64),
    }
    meta = {
        "case_id": np.asarray(case_ids, dtype=object),
        "center": np.asarray(center_names, dtype=object),
        "oct_id": np.asarray(oct_ids, dtype=object),
    }
    return features, meta


def load_split(split_dir: Path, master: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_raw, _ = features_from_csv(split_dir / "train_labels.csv", master)
    val_raw, val_meta = features_from_csv(split_dir / "val_labels.csv", master)
    ext_raw, ext_meta = features_from_csv(split_dir / "external_test_labels.csv", master)
    train, stats = make_tensors(train_raw)
    val, _ = make_tensors(val_raw, stats)
    external, _ = make_tensors(ext_raw, stats)
    return train, val, external, {"internal_validation": val_meta, "external_test": ext_meta}


def mask_for_setting(n: int, setting: str, seed: int) -> np.ndarray:
    mask = np.zeros((n, 3), dtype=bool)
    if setting == "full":
        return mask
    if setting == "remove_oct":
        mask[:, 0] = True
    elif setting == "remove_colposcopy":
        mask[:, 1] = True
    elif setting == "remove_clinical_prior":
        mask[:, 2] = True
    elif setting == "random_one_modality":
        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, 3, size=n)
        mask[np.arange(n), chosen] = True
    elif setting == "random_two_modalities":
        rng = np.random.default_rng(seed)
        for i in range(n):
            chosen = rng.choice(3, size=2, replace=False)
            mask[i, chosen] = True
    else:
        raise ValueError(f"Unknown modality setting: {setting}")
    return mask


def predict_masked(
    model: FeatureFusionNet,
    data: Dict[str, np.ndarray],
    device: torch.device,
    setting: str = "full",
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    mask = mask_for_setting(len(data["y"]), setting, seed)
    probs: List[np.ndarray] = []
    weights_all: List[np.ndarray] = []
    with torch.no_grad():
        for idx in batch_iter(data, 256, False, seed):
            oct_x = torch.as_tensor(data["oct"][idx], dtype=torch.float32, device=device)
            col_x = torch.as_tensor(data["col"][idx], dtype=torch.float32, device=device)
            cli_x = torch.as_tensor(data["clinical"][idx], dtype=torch.float32, device=device)
            center = torch.as_tensor(data["center_idx"][idx], dtype=torch.long, device=device)
            batch_mask = mask[idx]
            if batch_mask[:, 0].any():
                oct_x[torch.as_tensor(batch_mask[:, 0], device=device)] = 0.0
            if batch_mask[:, 1].any():
                col_x[torch.as_tensor(batch_mask[:, 1], device=device)] = 0.0
            if batch_mask[:, 2].any():
                cli_x[torch.as_tensor(batch_mask[:, 2], device=device)] = 0.0
            logits, extra = model(oct_x, col_x, cli_x, center)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            weights = extra.get("weights")
            if weights is not None:
                weights_all.append(weights.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(weights_all) if weights_all else None


def train_one(
    spec: MethodSpec,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int = 64,
) -> Tuple[FeatureFusionNet, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = FeatureFusionNet(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train["y"], minlength=2)
    class_weight = torch.as_tensor((len(train["y"]) / np.maximum(counts, 1)) / 2.0, dtype=torch.float32, device=device)
    best_auc = -math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for idx in batch_iter(train, batch_size, True, seed + epoch):
            oct_x = torch.as_tensor(train["oct"][idx], dtype=torch.float32, device=device)
            col_x = torch.as_tensor(train["col"][idx], dtype=torch.float32, device=device)
            cli_x = torch.as_tensor(train["clinical"][idx], dtype=torch.float32, device=device)
            center = torch.as_tensor(train["center_idx"][idx], dtype=torch.long, device=device)
            y = torch.as_tensor(train["y"][idx], dtype=torch.long, device=device)
            logits, extra = model(oct_x, col_x, cli_x, center, y)
            loss = F.cross_entropy(logits, y, weight=class_weight) + extra["aux_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_prob, _ = predict_masked(model, val, device, "full", seed)
        val_metric = binary_metrics(val["y"], val_prob)
        val_score = val_metric.auc if not math.isnan(val_metric.auc) else val_metric.auprc
        if val_score > best_auc:
            best_auc = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_auc)


def write_prediction_file(
    experiment: str,
    method: str,
    seed: int,
    split: str,
    data: Dict[str, np.ndarray],
    meta: Dict[str, np.ndarray],
    probs: np.ndarray,
    weights: Optional[np.ndarray],
    setting: str,
    threshold: float,
) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{experiment}_{method}_seed{seed}_{split}_{setting}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "experiment",
            "method",
            "run_id",
            "seed",
            "split",
            "case_id",
            "center",
            "oct_id",
            "y_true",
            "y_prob",
            "y_pred",
            "modality_setting",
            "threshold",
            "reliability_oct",
            "reliability_colposcopy",
            "reliability_clinical_prior",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pred = (probs >= threshold).astype(int)
        for i, prob in enumerate(probs):
            row = {
                "experiment": experiment,
                "method": method,
                "run_id": experiment,
                "seed": seed,
                "split": split,
                "case_id": str(meta["case_id"][i]),
                "center": str(meta["center"][i]),
                "oct_id": str(meta["oct_id"][i]),
                "y_true": int(data["y"][i]),
                "y_prob": float(prob),
                "y_pred": int(pred[i]),
                "modality_setting": setting,
                "threshold": float(threshold),
                "reliability_oct": "",
                "reliability_colposcopy": "",
                "reliability_clinical_prior": "",
            }
            if weights is not None:
                row["reliability_oct"] = float(weights[i, 0])
                row["reliability_colposcopy"] = float(weights[i, 1])
                row["reliability_clinical_prior"] = float(weights[i, 2])
            writer.writerow(row)
    return out


def metric_row(
    experiment: str,
    method: str,
    seed: int,
    split: str,
    setting: str,
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float,
    best_val_auc: float,
) -> Dict[str, object]:
    metric = binary_metrics(y_true, y_prob, threshold=threshold)
    return {
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "split": split,
        "modality_setting": setting,
        "best_internal_auc": best_val_auc,
        **asdict(metric),
    }


def train_and_evaluate_split(
    experiment: str,
    split_dir: Path,
    specs: List[MethodSpec],
    master: Dict[str, Dict[str, object]],
    device: torch.device,
    seed: int,
    epochs: int,
    include_robustness: bool,
    save_models: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, FeatureFusionNet]]:
    train, val, external, meta = load_split(split_dir, master)
    rows: List[Dict[str, object]] = []
    center_rows: List[Dict[str, object]] = []
    trained: Dict[str, FeatureFusionNet] = {}

    for spec in specs:
        model, best_val_auc = train_one(spec, train, val, device, seed, epochs)
        trained[spec.name] = model
        val_prob, val_weights = predict_masked(model, val, device, "full", seed)
        threshold = select_threshold(val["y"], val_prob)
        ext_prob, ext_weights = predict_masked(model, external, device, "full", seed)

        rows.append(metric_row(experiment, spec.name, seed, "internal_validation", "full", val["y"], val_prob, threshold, best_val_auc))
        rows.append(metric_row(experiment, spec.name, seed, "external_test", "full", external["y"], ext_prob, threshold, best_val_auc))
        write_prediction_file(experiment, spec.name, seed, "internal_validation", val, meta["internal_validation"], val_prob, val_weights, "full", threshold)
        write_prediction_file(experiment, spec.name, seed, "external_test", external, meta["external_test"], ext_prob, ext_weights, "full", threshold)

        ext_meta = pd.DataFrame({"center": meta["external_test"]["center"], "y_true": external["y"], "y_prob": ext_prob})
        for center_name, center_df in ext_meta.groupby("center"):
            center_metric = binary_metrics(center_df["y_true"], center_df["y_prob"], threshold=threshold)
            center_rows.append(
                {
                    "experiment": experiment,
                    "method": spec.name,
                    "seed": seed,
                    "center": center_name,
                    "modality_setting": "full",
                    **asdict(center_metric),
                }
            )

        if include_robustness and spec.name == PRIMARY_METHOD:
            for setting in ["remove_oct", "remove_colposcopy", "remove_clinical_prior", "random_one_modality", "random_two_modalities"]:
                prob, weights = predict_masked(model, external, device, setting, seed + 1009)
                rows.append(metric_row(experiment, spec.name, seed, "external_test", setting, external["y"], prob, threshold, best_val_auc))
                write_prediction_file(experiment, spec.name, seed, "external_test", external, meta["external_test"], prob, weights, setting, threshold)

        if save_models:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "method": spec.name,
                    "spec": spec.__dict__,
                    "best_internal_auc": best_val_auc,
                    "experiment": experiment,
                    "seed": seed,
                },
                MODEL_DIR / f"{experiment}_{spec.name}_seed{seed}.pt",
            )
        print(f"[{experiment}] {spec.name}: best internal score={best_val_auc:.4f}")

    return pd.DataFrame(rows), pd.DataFrame(center_rows), trained


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in METRIC_COLS + ["threshold", "best_internal_auc"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    return display


def write_table(df: pd.DataFrame, name: str, columns: Optional[List[str]] = None) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLE_DIR / f"{name}.csv"
    table_df = df[columns].copy() if columns else df.copy()
    format_metric_table(table_df).to_csv(out_csv, index=False)


def build_faithfulness(
    experiment: str,
    model: FeatureFusionNet,
    external: Dict[str, np.ndarray],
    meta: Dict[str, np.ndarray],
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    base_prob, weights = predict_masked(model, external, device, "full", seed)
    if weights is None:
        return pd.DataFrame()
    threshold = select_threshold(external["y"], base_prob)
    pred = (base_prob >= threshold).astype(int)
    top_modality = weights.argmax(axis=1)
    setting_map = {0: "remove_oct", 1: "remove_colposcopy", 2: "remove_clinical_prior"}

    cited_drops: List[float] = []
    random_drops: List[float] = []
    rng = np.random.default_rng(seed + 2026)
    for idx, modality_idx in enumerate(top_modality):
        sample = {key: value[idx : idx + 1].copy() for key, value in external.items()}
        base_conf = base_prob[idx] if pred[idx] == 1 else 1.0 - base_prob[idx]
        cited_prob, _ = predict_masked(model, sample, device, setting_map[int(modality_idx)], seed)
        cited_conf = cited_prob[0] if pred[idx] == 1 else 1.0 - cited_prob[0]
        random_setting = setting_map[int(rng.integers(0, 3))]
        random_prob, _ = predict_masked(model, sample, device, random_setting, seed)
        random_conf = random_prob[0] if pred[idx] == 1 else 1.0 - random_prob[0]
        cited_drops.append(float(base_conf - cited_conf))
        random_drops.append(float(base_conf - random_conf))

    swap_changes: List[float] = []
    labels = external["y"]
    for idx, modality_idx in enumerate(top_modality):
        candidates = np.where(labels != labels[idx])[0]
        if len(candidates) == 0:
            continue
        other = int(rng.choice(candidates))
        sample = {key: value[idx : idx + 1].copy() for key, value in external.items()}
        if modality_idx == 0:
            sample["oct"][0] = external["oct"][other]
        elif modality_idx == 1:
            sample["col"][0] = external["col"][other]
        else:
            sample["clinical"][0] = external["clinical"][other]
        swap_prob, _ = predict_masked(model, sample, device, "full", seed)
        swap_changes.append(float(abs(base_prob[idx] - swap_prob[0])))

    summary = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "method": PRIMARY_METHOD,
                "test": "cited_modality_removal",
                "mean_confidence_drop": float(np.mean(cited_drops)),
                "random_removal_drop": float(np.mean(random_drops)),
                "faithfulness_margin": float(np.mean(cited_drops) - np.mean(random_drops)),
                "n": len(cited_drops),
            },
            {
                "experiment": experiment,
                "method": PRIMARY_METHOD,
                "test": "counterfactual_evidence_swap",
                "mean_confidence_drop": float(np.mean(swap_changes)) if swap_changes else float("nan"),
                "random_removal_drop": "",
                "faithfulness_margin": "",
                "n": len(swap_changes),
            },
        ]
    )

    failure = pd.DataFrame(
        {
            "experiment": experiment,
            "case_id": meta["case_id"],
            "center": meta["center"],
            "y_true": external["y"],
            "y_prob": base_prob,
            "y_pred": pred,
            "top_evidence_modality": [setting_map[int(x)].replace("remove_", "") for x in top_modality],
            "reliability_oct": weights[:, 0],
            "reliability_colposcopy": weights[:, 1],
            "reliability_clinical_prior": weights[:, 2],
        }
    )
    failure["error_type"] = np.where(failure["y_true"].eq(failure["y_pred"]), "correct_prediction", "wrong_prediction")
    failure.sort_values(["error_type", "y_prob"]).to_csv(TABLE_DIR / f"resplit_feature_{experiment}_failure_cases_automatic.csv", index=False)
    return summary


def write_decision_curve(experiment: str, prediction_rows: pd.DataFrame) -> None:
    selected = prediction_rows[
        prediction_rows["split"].eq("external_test")
        & prediction_rows["modality_setting"].eq("full")
        & prediction_rows["method"].isin([PRIMARY_METHOD, "Late_Fusion", "Gated_Fusion"])
    ].copy()
    frames = []
    for method, group in selected.groupby("method"):
        curve = decision_curve_points(group["y_true"], group["y_prob"], thresholds=np.linspace(0.05, 0.80, 16))
        curve.insert(0, "experiment", experiment)
        curve.insert(1, "method", method)
        frames.append(curve)
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TABLE_DIR / f"resplit_feature_{experiment}_decision_curve.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=180)
        for method, group in out.groupby("method"):
            ax.plot(group["threshold"], group["net_benefit"], marker="o", linewidth=1.7, label=method)
        ref = out[out["method"].eq(out["method"].iloc[0])]
        ax.plot(ref["threshold"], ref["treat_all"], linestyle="--", color="0.45", label="Treat all")
        ax.plot(ref["threshold"], ref["treat_none"], linestyle=":", color="0.15", label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"Decision curve: {experiment}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"resplit_feature_{experiment}_decision_curve.png")
        plt.close(fig)
    except Exception as exc:
        (FIGURE_DIR / f"resplit_feature_{experiment}_decision_curve_plot_error.txt").write_text(str(exc), encoding="utf-8")


def read_prediction_rows(experiment: str) -> pd.DataFrame:
    frames = []
    for path in sorted(PRED_DIR.glob(f"{experiment}_*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_outputs(metrics: pd.DataFrame, centerwise: pd.DataFrame, lco_metrics: pd.DataFrame) -> None:
    main = metrics[metrics["experiment"].eq("recommended_enshi_external")].copy()
    main_external = main[main["split"].eq("external_test") & main["modality_setting"].eq("full")].copy()

    write_table(
        main_external[main_external["method"].isin(DIRECT_METHODS + [PRIMARY_METHOD])],
        "resplit_feature_enshi_direct_fusion_metrics",
        ["method", "n", "positives", "negatives"] + METRIC_COLS,
    )
    write_table(
        main_external[main_external["method"].isin([PRIMARY_METHOD] + ABLATION_METHODS)],
        "resplit_feature_enshi_requirement_ablation_metrics",
        ["method", "n", "positives", "negatives"] + METRIC_COLS,
    )
    robustness = main[main["method"].eq(PRIMARY_METHOD) & main["split"].eq("external_test")].copy()
    write_table(
        robustness,
        "resplit_feature_enshi_missing_modality_robustness_metrics",
        ["modality_setting", "n", "positives", "negatives"] + METRIC_COLS,
    )
    write_table(
        centerwise[centerwise["experiment"].eq("recommended_enshi_external")],
        "resplit_feature_enshi_centerwise_calibration_metrics",
        ["method", "center", "n", "positives", "negatives", "auc", "ece", "brier", "sensitivity", "specificity", "npv"],
    )

    historical = metrics[
        metrics["experiment"].eq("official_jingzhou_shiyan_external")
        & metrics["split"].eq("external_test")
        & metrics["modality_setting"].eq("full")
    ].copy()
    write_table(
        historical,
        "resplit_feature_historical_jingzhou_shiyan_metrics",
        ["method", "n", "positives", "negatives"] + METRIC_COLS,
    )

    if not lco_metrics.empty:
        write_table(
            lco_metrics,
            "resplit_feature_lco_external_metrics",
            ["experiment", "method", "n", "positives", "negatives"] + METRIC_COLS,
        )
        write_table(
            centerwise[centerwise["experiment"].str.startswith("lco_", na=False)],
            "resplit_feature_lco_centerwise_calibration_metrics",
            ["experiment", "method", "center", "n", "positives", "negatives", "auc", "ece", "brier", "sensitivity", "specificity", "npv"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-main", type=int, default=120)
    parser.add_argument("--epochs-lco", type=int, default=80)
    parser.add_argument("--fast-lco", action="store_true", help="Run only Late/Gated/HyDRA on LCO folds.")
    parser.add_argument("--no-lco", action="store_true")
    args = parser.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = load_master_features()
    print(f"Loaded cached features for {len(master)} full-multimodal cases. Device={device}")

    all_metrics: List[pd.DataFrame] = []
    all_centerwise: List[pd.DataFrame] = []
    faithfulness_frames: List[pd.DataFrame] = []

    main_experiments = {
        "recommended_enshi_external": SPLIT_ROOT / "recommended_enshi_external",
        "official_jingzhou_shiyan_external": SPLIT_ROOT / "official_jingzhou_shiyan_external",
    }
    for experiment, split_dir in main_experiments.items():
        metrics, centerwise, trained = train_and_evaluate_split(
            experiment=experiment,
            split_dir=split_dir,
            specs=method_specs(),
            master=master,
            device=device,
            seed=args.seed,
            epochs=args.epochs_main,
            include_robustness=experiment == "recommended_enshi_external",
            save_models=experiment == "recommended_enshi_external",
        )
        all_metrics.append(metrics)
        all_centerwise.append(centerwise)
        pred_rows = read_prediction_rows(experiment)
        if not pred_rows.empty:
            write_decision_curve(experiment, pred_rows)
        if experiment == "recommended_enshi_external" and PRIMARY_METHOD in trained:
            _, _, external, meta = load_split(split_dir, master)
            faithfulness_frames.append(build_faithfulness(experiment, trained[PRIMARY_METHOD], external, meta["external_test"], device, args.seed))

    lco_metric_frames: List[pd.DataFrame] = []
    if not args.no_lco:
        lco_specs = fast_specs() if args.fast_lco else method_specs()
        for split_dir in sorted((SPLIT_ROOT / "leave_one_center_out").glob("lco_*")):
            experiment = split_dir.name
            metrics, centerwise, _ = train_and_evaluate_split(
                experiment=experiment,
                split_dir=split_dir,
                specs=lco_specs,
                master=master,
                device=device,
                seed=args.seed,
                epochs=args.epochs_lco,
                include_robustness=False,
                save_models=False,
            )
            all_metrics.append(metrics)
            all_centerwise.append(centerwise)
            lco_metric_frames.append(metrics[metrics["split"].eq("external_test") & metrics["modality_setting"].eq("full")].copy())

    metrics = pd.concat(all_metrics, ignore_index=True)
    centerwise = pd.concat(all_centerwise, ignore_index=True)
    lco_metrics = pd.concat(lco_metric_frames, ignore_index=True) if lco_metric_frames else pd.DataFrame()

    metrics.to_csv(TABLE_DIR / "resplit_feature_all_metrics.csv", index=False)
    centerwise.to_csv(TABLE_DIR / "resplit_feature_all_centerwise_metrics.csv", index=False)
    if faithfulness_frames:
        faithfulness = pd.concat(faithfulness_frames, ignore_index=True)
        write_table(
            faithfulness,
            "resplit_feature_enshi_coe_faithfulness_automatic_metrics",
        )

    summarize_outputs(metrics, centerwise, lco_metrics)
    print(f"Wrote resplit feature predictions to {PRED_DIR}")
    print(f"Wrote resplit feature tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
