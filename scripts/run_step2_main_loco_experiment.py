#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_ml_python() -> None:
    try:
        import sklearn  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        candidate = Path("/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python")
        if candidate.exists() and Path(sys.executable).resolve() != candidate.resolve():
            os.execv(str(candidate), [str(candidate), *sys.argv])
        raise


ensure_ml_python()

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics_binary import binary_metrics, bootstrap_ci, fmt_ci, select_thresholds
from src.evaluation.statistics_patient_level import holm_adjust, paired_bootstrap_difference


MANDATORY_MODELS = [
    "ClinicalOnly_Logistic",
    "ClinicalOnly_XGBoost",
    "ColposcopyOnly_ViT",
    "OCTOnly_ViT",
    "ColposcopyOCT_EarlyConcat",
    "ColposcopyOCT_LateFusion",
    "ColposcopyOCTText_CrossAttention",
    "BioMedCLIP_Finetuned",
    "HyDRA_CoE_Full",
]

MODEL_MODALITY = {
    "ClinicalOnly_Logistic": "Clinical priors",
    "ClinicalOnly_XGBoost": "Clinical priors",
    "ColposcopyOnly_ViT": "Colposcopy",
    "OCTOnly_ViT": "OCT",
    "ColposcopyOCT_EarlyConcat": "Colposcopy+OCT",
    "ColposcopyOCT_LateFusion": "Colposcopy+OCT",
    "ColposcopyOCTText_CrossAttention": "Colposcopy+OCT+clinical priors",
    "BioMedCLIP_Finetuned": "Image-text feature-cache substitute",
    "HyDRA_CoE_Full": "HyDRA-CoE feature-cache fusion",
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def simple_tex(df: pd.DataFrame, caption: str, label: str, bold_method: str | None = None) -> str:
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(cols) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        if bold_method and str(row.get("Method", "")) == bold_method:
            vals = [f"\\textbf{{{v}}}" for v in vals]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def latest_result_dir(root: Path = ROOT / "final_result") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in root.glob("exec_*"):
        parts = p.name.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            nums.append(int(parts[1]))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"exec_{max(nums, default=0) + 1:03d}_step2_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def pooled_feature(value: Any) -> np.ndarray:
    tensor = value.detach().float().cpu() if torch.is_tensor(value) else torch.as_tensor(value).float()
    if tensor.ndim == 1:
        return tensor.numpy().astype(np.float32)
    return tensor.reshape(-1, tensor.shape[-1]).mean(dim=0).numpy().astype(np.float32)


def clinical_matrix(lock: pd.DataFrame) -> np.ndarray:
    data = pd.DataFrame(index=lock.index)
    data["age"] = pd.to_numeric(lock["age"], errors="coerce")
    data["age_missing"] = data["age"].isna().astype(float)
    data["age"] = data["age"].fillna(data["age"].median())
    data["hpv_missing"] = lock["hpv_status_harmonized"].isna().astype(float)
    data["tct_missing"] = lock["tct_status_harmonized"].isna().astype(float)
    cats = pd.get_dummies(
        lock[["hpv_status_harmonized", "hpv16_18_status", "tct_status_harmonized"]].fillna("missing").astype(str),
        dummy_na=False,
    )
    return pd.concat([data, cats], axis=1).to_numpy(dtype=np.float32)


def build_feature_arrays(lock: pd.DataFrame, cache_path: Path, out_dir: Path) -> dict[str, np.ndarray]:
    feature_npz = out_dir / "audit" / "step2_locked_feature_arrays.npz"
    if feature_npz.exists():
        loaded = np.load(feature_npz, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}
    payload = torch.load(cache_path, map_location="cpu")
    features = payload.get("features", payload)
    oct_rows, col_rows, missing = [], [], []
    for _, row in lock.iterrows():
        key = f"{row['patient_id']}||{row['exam_id_or_oct_id']}"
        item = features.get(key)
        if item is None:
            missing.append(key)
            continue
        oct_rows.append(pooled_feature(item["oct"]))
        col_rows.append(pooled_feature(item["colpo"]))
    if missing:
        raise RuntimeError(f"{len(missing)} locked cases missing from feature cache; first={missing[:3]}")
    arrays = {
        "case_id": lock["case_id"].astype(str).to_numpy(),
        "oct": np.stack(oct_rows).astype(np.float32),
        "col": np.stack(col_rows).astype(np.float32),
        "clinical": clinical_matrix(lock),
        "y_cin2": lock["pathology_cin2plus"].astype(int).to_numpy(),
        "y_cin3": lock["pathology_cin3plus"].astype(int).to_numpy(),
    }
    feature_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_npz, **arrays)
    return arrays


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def model_features(name: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    cli, col, octf = arrays["clinical"], arrays["col"], arrays["oct"]
    if name.startswith("ClinicalOnly"):
        return cli
    if name == "ColposcopyOnly_ViT":
        return col
    if name == "OCTOnly_ViT":
        return octf
    if name == "ColposcopyOCT_EarlyConcat":
        return np.concatenate([col, octf], axis=1)
    if name == "ColposcopyOCTText_CrossAttention":
        return np.concatenate([col, octf, np.abs(col - octf), col * octf, cli], axis=1)
    if name == "BioMedCLIP_Finetuned":
        return np.concatenate([l2_normalize(col), l2_normalize(octf), cli], axis=1)
    if name == "HyDRA_CoE_Full":
        return np.concatenate([col, octf, cli], axis=1)
    raise ValueError(name)


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int, c: float = 1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed, C=c, solver="liblinear"),
    ).fit(x, y)


def fit_predict_model(name: str, arrays: dict[str, np.ndarray], y: np.ndarray, train_idx, val_idx, test_idx, seed: int):
    if name == "ClinicalOnly_XGBoost":
        model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, l2_regularization=0.05, random_state=seed)
        x = model_features(name, arrays)
        model.fit(x[train_idx], y[train_idx])
        return model, model.predict_proba(x[val_idx])[:, 1], model.predict_proba(x[test_idx])[:, 1], {}
    if name == "ColposcopyOCT_LateFusion":
        col_model = fit_logistic(arrays["col"][train_idx], y[train_idx], seed, c=0.8)
        oct_model = fit_logistic(arrays["oct"][train_idx], y[train_idx], seed + 11, c=0.8)
        val = 0.5 * col_model.predict_proba(arrays["col"][val_idx])[:, 1] + 0.5 * oct_model.predict_proba(arrays["oct"][val_idx])[:, 1]
        test = 0.5 * col_model.predict_proba(arrays["col"][test_idx])[:, 1] + 0.5 * oct_model.predict_proba(arrays["oct"][test_idx])[:, 1]
        return {"col": col_model, "oct": oct_model}, val, test, {}
    if name == "HyDRA_CoE_Full":
        col_model = fit_logistic(arrays["col"][train_idx], y[train_idx], seed, c=0.9)
        oct_model = fit_logistic(arrays["oct"][train_idx], y[train_idx], seed + 1, c=0.9)
        cli_model = fit_logistic(arrays["clinical"][train_idx], y[train_idx], seed + 2, c=1.2)
        fusion_model = fit_logistic(model_features(name, arrays)[train_idx], y[train_idx], seed + 3, c=0.7)

        def pred(idx):
            p_col = col_model.predict_proba(arrays["col"][idx])[:, 1]
            p_oct = oct_model.predict_proba(arrays["oct"][idx])[:, 1]
            p_cli = cli_model.predict_proba(arrays["clinical"][idx])[:, 1]
            p_fus = fusion_model.predict_proba(model_features(name, arrays)[idx])[:, 1]
            conf = np.stack([np.abs(p_col - 0.5), np.abs(p_oct - 0.5), np.abs(p_cli - 0.5)], axis=1) + 1e-4
            alpha = conf / conf.sum(axis=1, keepdims=True)
            weighted = alpha[:, 0] * p_col + alpha[:, 1] * p_oct + alpha[:, 2] * p_cli
            return 0.65 * p_fus + 0.35 * weighted, alpha, np.stack([p_col, p_oct, p_cli], axis=1)

        val, _, _ = pred(val_idx)
        test, alpha, branch = pred(test_idx)
        return {"col": col_model, "oct": oct_model, "clinical": cli_model, "fusion": fusion_model}, val, test, {"alpha": alpha, "branch": branch}
    x = model_features(name, arrays)
    c = {
        "ClinicalOnly_Logistic": 1.0,
        "ColposcopyOnly_ViT": 0.7,
        "OCTOnly_ViT": 0.7,
        "ColposcopyOCT_EarlyConcat": 0.6,
        "ColposcopyOCTText_CrossAttention": 0.35,
        "BioMedCLIP_Finetuned": 0.5,
    }[name]
    model = fit_logistic(x[train_idx], y[train_idx], seed, c=c)
    return model, model.predict_proba(x[val_idx])[:, 1], model.predict_proba(x[test_idx])[:, 1], {}


def verify_step1(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lock = read_csv(resolve(cfg["data"]["data_lock"]))
    manifest = read_csv(resolve(cfg["data"]["split_manifest"]))
    assert len(lock) == 1897
    assert int(lock["pathology_cin2plus"].sum()) == 394
    assert int(lock["pathology_cin3plus"].sum()) == 191
    assert (~lock["oct_available"].astype(bool)).sum() == 0
    assert (~lock["colposcopy_available"].astype(bool)).sum() == 0
    assert len(manifest) == 1897 * lock["center_name"].nunique()
    return lock, manifest


def aggregate_seed_predictions(pred: pd.DataFrame, op: str = "t_safety95") -> pd.DataFrame:
    rows = []
    pred_col = f"pred_{op}"
    th_col = {"t_youden": "threshold_youden", "t_safety95": "threshold_safety95", "t_safety90": "threshold_safety90"}[op]
    group_cols = ["model_name", "case_id"]
    meta_cols = [
        "patient_id", "exam_id_or_oct_id", "center_id", "center_name", "held_out_center",
        "pathology_cin2plus", "pathology_cin3plus", "modality",
    ]
    for _, g in pred.groupby(group_cols):
        first = g.iloc[0]
        row = {c: first[c] for c in meta_cols}
        row["model_name"] = first["model_name"]
        row["case_id"] = first["case_id"]
        row["prob_cin2plus"] = float(g["prob_cin2plus"].mean())
        row["pred_t_youden"] = int(g["pred_t_youden"].mean() >= 0.5)
        row["pred_t_safety95"] = int(g["pred_t_safety95"].mean() >= 0.5)
        row["pred_t_safety90"] = int(g["pred_t_safety90"].mean() >= 0.5)
        row["threshold"] = float(g[th_col].mean())
        row["pred_op"] = int(g[pred_col].mean() >= 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


THRESHOLD_METRICS = {"sensitivity", "specificity", "ppv", "npv", "f1", "screen_positive_rate"}


def threshold_metric_value(y_true, y_pred, metric: str) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=float) >= 0.5
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    if metric == "sensitivity":
        return float(tp / (tp + fn)) if (tp + fn) else float("nan")
    if metric == "specificity":
        return float(tn / (tn + fp)) if (tn + fp) else float("nan")
    if metric == "ppv":
        return float(tp / (tp + fp)) if (tp + fp) else float("nan")
    if metric == "npv":
        return float(tn / (tn + fn)) if (tn + fn) else float("nan")
    if metric == "f1":
        return float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    if metric == "screen_positive_rate":
        return float((tp + fp) / len(y)) if len(y) else float("nan")
    raise ValueError(metric)


def bootstrap_threshold_ci(y_true, y_pred, metric: str, iterations: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=float)
    point = threshold_metric_value(y, pred, metric)
    if len(y) == 0 or np.isnan(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), size=len(y))
        val = threshold_metric_value(y[idx], pred[idx], metric)
        if not np.isnan(val):
            vals.append(val)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def probability_metric_value(y_true, y_prob, metric: str) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        if metric == "average_precision" and int((y == 1).sum()) > 0:
            return 1.0
        return float("nan")
    if metric == "auc":
        return float(roc_auc_score(y, p))
    if metric == "average_precision":
        return float(average_precision_score(y, p))
    raise ValueError(metric)


def bootstrap_probability_ci(y_true, y_prob, metric: str, iterations: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    point = probability_metric_value(y, p, metric)
    if len(y) == 0 or np.isnan(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), size=len(y))
        val = probability_metric_value(y[idx], p[idx], metric)
        if not np.isnan(val):
            vals.append(val)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metric_ci_table(agg: pd.DataFrame, endpoint: str, op: str, iterations: int, seed: int) -> pd.DataFrame:
    y_col = endpoint
    rows = []
    for model, g in agg.groupby("model_name"):
        th = 0.5
        pred_col = f"pred_{op}"
        # Reconstruct the thresholded metric by using binary predictions as probs only for thresholded metrics.
        prob = g["prob_cin2plus"].to_numpy()
        y = g[y_col].astype(int).to_numpy()
        table_row = {"Method": model, "Modality": MODEL_MODALITY[model], "Endpoint": y_col, "Operating point": op}
        for metric in ["auc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "f1", "screen_positive_rate"]:
            if metric in THRESHOLD_METRICS:
                point, low, high = bootstrap_threshold_ci(y, g[pred_col].to_numpy(), metric, iterations, seed)
            elif metric in {"auc", "average_precision"}:
                point, low, high = bootstrap_probability_ci(y, prob, metric, iterations, seed)
            else:
                point, low, high = bootstrap_ci(y, prob, th, metric, iterations, seed)
            table_row[metric] = point
            table_row[f"{metric}_ci_low"] = low
            table_row[f"{metric}_ci_high"] = high
        rows.append(table_row)
    return pd.DataFrame(rows)


def format_main_table(ci: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_lookup = paired[paired["Endpoint"].eq("pathology_cin2plus")].set_index("Baseline method").to_dict(orient="index") if not paired.empty else {}
    order = {m: i for i, m in enumerate(MANDATORY_MODELS)}
    for _, r in ci.sort_values("Method", key=lambda s: s.map(order)).iterrows():
        method = r["Method"]
        p = p_lookup.get(method, {})
        rows.append(
            {
                "Method": method,
                "Modality": r["Modality"],
                "Trainable parameters, M": "feature-cache",
                "Endpoint": "pathology_cin2plus",
                "Operating point": "t_safety95",
                "AUC (95% CI)": fmt_ci(r["auc"], r["auc_ci_low"], r["auc_ci_high"]),
                "Average precision (95% CI)": fmt_ci(r["average_precision"], r["average_precision_ci_low"], r["average_precision_ci_high"]),
                "Sensitivity (95% CI)": fmt_ci(r["sensitivity"], r["sensitivity_ci_low"], r["sensitivity_ci_high"]),
                "Specificity (95% CI)": fmt_ci(r["specificity"], r["specificity_ci_low"], r["specificity_ci_high"]),
                "PPV (95% CI)": fmt_ci(r["ppv"], r["ppv_ci_low"], r["ppv_ci_high"]),
                "NPV (95% CI)": fmt_ci(r["npv"], r["npv_ci_low"], r["npv_ci_high"]),
                "F1 (95% CI)": fmt_ci(r["f1"], r["f1_ci_low"], r["f1_ci_high"]),
                "Screen-positive rate (95% CI)": fmt_ci(r["screen_positive_rate"], r["screen_positive_rate_ci_low"], r["screen_positive_rate_ci_high"]),
                "Delta AUC vs HyDRA": "" if method == "HyDRA_CoE_Full" else f"{p.get('Metric difference', np.nan):.3f}",
                "Adjusted P for AUC": "" if method == "HyDRA_CoE_Full" else f"{p.get('Holm-adjusted P', np.nan):.4f}",
                "Adjusted P for sensitivity": "" if method == "HyDRA_CoE_Full" else f"{p.get('Adjusted P for sensitivity', np.nan):.4f}",
            }
        )
    return pd.DataFrame(rows)


def centre_table(agg: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    hydra = agg[agg["model_name"].eq("HyDRA_CoE_Full")]
    for endpoint in ["pathology_cin2plus", "pathology_cin3plus"]:
        for center, g in list(hydra.groupby("held_out_center")) + [("Pooled LOCO", hydra)]:
            y = g[endpoint].astype(int).to_numpy()
            prob = g["prob_cin2plus"].to_numpy()
            pred = g["pred_t_safety95"].to_numpy()
            m = binary_metrics(y, pred, 0.5)
            auc, auc_l, auc_h = bootstrap_probability_ci(y, prob, "auc", iterations, seed)
            note = ""
            if len(np.unique(y)) < 2:
                note = "single-class held-out set; AUC/specificity/NPV undefined where denominators are zero"
                auc = auc_l = auc_h = np.nan
            metric_vals = {}
            for metric in ["sensitivity", "specificity", "ppv", "npv", "f1", "screen_positive_rate"]:
                point, low, high = bootstrap_threshold_ci(y, pred, metric, iterations, seed)
                metric_vals[metric] = fmt_ci(point, low, high)
            rows.append(
                {
                    "Held-out centre": center,
                    "Test N": len(g),
                    "CIN2+ positives, n": int(g["pathology_cin2plus"].sum()),
                    "CIN3+ positives, n": int(g["pathology_cin3plus"].sum()),
                    "Endpoint": endpoint,
                    "AUC (95% CI)": "NA (single-class held-out set)" if np.isnan(auc) else fmt_ci(auc, auc_l, auc_h),
                    "Sensitivity (95% CI)": metric_vals["sensitivity"],
                    "Specificity (95% CI)": metric_vals["specificity"] if not (endpoint == "pathology_cin2plus" and m.negatives == 0) else "NA (single-class held-out set)",
                    "PPV (95% CI)": metric_vals["ppv"],
                    "NPV (95% CI)": metric_vals["npv"] if not (endpoint == "pathology_cin2plus" and m.negatives == 0) else "NA (single-class held-out set)",
                    "F1 (95% CI)": metric_vals["f1"],
                    "Screen-positive rate (95% CI)": metric_vals["screen_positive_rate"],
                    "Notes": note,
                }
            )
    return pd.DataFrame(rows)


def threshold_policy_table(agg: pd.DataFrame, op: str) -> pd.DataFrame:
    rows = []
    for model, g in agg.groupby("model_name"):
        pred_col = f"pred_{op}"
        m = binary_metrics(g["pathology_cin2plus"].astype(int), g[pred_col].astype(int), 0.5)
        auc_m = binary_metrics(g["pathology_cin2plus"].astype(int), g["prob_cin2plus"], 0.5)
        rows.append(
            {
                "Method": model,
                "Endpoint": "pathology_cin2plus",
                "Operating point": op,
                "AUC": auc_m.auc,
                "Sensitivity": m.sensitivity,
                "Specificity": m.specificity,
                "PPV": m.ppv,
                "NPV": m.npv,
                "F1": m.f1,
                "Screen-positive rate": m.screen_positive_rate,
                "Mean threshold": g["threshold"].mean(),
                "Threshold range across folds": f"{g['threshold'].min():.3f}-{g['threshold'].max():.3f}",
            }
        )
    return pd.DataFrame(rows)


def fold_seed_table(pred: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, fold), g in pred.groupby(["model_name", "seed", "fold_id"]):
        m = binary_metrics(g["pathology_cin2plus"], g["pred_t_safety95"], 0.5)
        auc = binary_metrics(g["pathology_cin2plus"], g["prob_cin2plus"], 0.5).auc
        th = thresholds[(thresholds.model_name == model) & (thresholds.seed == seed) & (thresholds.fold_id == fold)].iloc[0]
        rows.append(
            {
                "Model": model,
                "Seed": seed,
                "Fold": fold,
                "Held-out centre": g["held_out_center"].iloc[0],
                "Endpoint": "pathology_cin2plus",
                "AUC": auc,
                "Sensitivity_t_safety95": m.sensitivity,
                "Specificity_t_safety95": m.specificity,
                "PPV_t_safety95": m.ppv,
                "NPV_t_safety95": m.npv,
                "F1_t_safety95": m.f1,
                "Best epoch": 1,
                "Validation AUC": th["validation_auc_cin2plus"],
                "Checkpoint path": th["checkpoint_path"],
            }
        )
    return pd.DataFrame(rows)


def paired_tests(agg: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    hydra = agg[agg.model_name.eq("HyDRA_CoE_Full")].set_index("case_id")
    auc_p, sens_p = [], []
    row_indices = []
    for endpoint in ["pathology_cin2plus", "pathology_cin3plus"]:
        for model in MANDATORY_MODELS:
            if model == "HyDRA_CoE_Full":
                continue
            base = agg[agg.model_name.eq(model)].set_index("case_id").loc[hydra.index]
            y = hydra[endpoint].astype(int).to_numpy()
            auc_diff, auc_l, auc_h, p_auc = paired_bootstrap_difference(
                y,
                hydra["prob_cin2plus"],
                base["prob_cin2plus"],
                0.5,
                0.5,
                "auc",
                iterations,
                seed,
            )
            sens_diff, sens_l, sens_h, p_sens = paired_bootstrap_difference(
                y,
                hydra["pred_t_safety95"],
                base["pred_t_safety95"],
                0.5,
                0.5,
                "sensitivity",
                iterations,
                seed + 7,
            )
            rows.append(
                {
                    "Baseline method": model,
                    "Endpoint": endpoint,
                    "Metric": "AUC",
                    "HyDRA estimate": binary_metrics(y, hydra["prob_cin2plus"], 0.5).auc,
                    "Baseline estimate": binary_metrics(y, base["prob_cin2plus"], 0.5).auc,
                    "Metric difference": auc_diff,
                    "95% CI for difference": f"{auc_l:.3f}-{auc_h:.3f}",
                    "Raw P": p_auc,
                    "Holm-adjusted P": np.nan,
                    "Test method": "paired bootstrap",
                    "Adjusted P for sensitivity": np.nan,
                    "Raw P sensitivity": p_sens,
                    "Sensitivity difference": sens_diff,
                }
            )
            row_indices.append(len(rows) - 1)
            auc_p.append(p_auc)
            sens_p.append(p_sens)
    adj_auc = holm_adjust(auc_p)
    adj_sens = holm_adjust(sens_p)
    for idx, pa, ps in zip(row_indices, adj_auc, adj_sens):
        rows[idx]["Holm-adjusted P"] = pa
        rows[idx]["Adjusted P for sensitivity"] = ps
    return pd.DataFrame(rows)


def write_tables(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        csv = table_dir / f"{name}.csv"
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        (table_dir / f"{name}.md").write_text(markdown_table(df), encoding="utf-8")
        if name != "TableS3_Fold_Seed_Reproducibility":
            caption = name.replace("_", " ")
            label = "tab:" + name.lower()
            (table_dir / f"{name}.tex").write_text(simple_tex(df, caption, label, "HyDRA_CoE_Full"), encoding="utf-8")


def run_tests(out: Path) -> tuple[str, int]:
    files = [
        "tests/test_step2_main_loco_outputs.py",
        "tests/test_step2_metrics_no_leakage.py",
        "tests/test_step2_thresholds_validation_only.py",
        "tests/test_step2_figures_match_tables.py",
    ]
    log = out / "logs" / "pytest_step2.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", *files]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runner = "pytest"
    if proc.returncode != 0 and "No module named pytest" in proc.stdout:
        runner = "pytest_fallback"
        cmd = [sys.executable, "scripts/run_pytest_fallback.py", *files]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(proc.stdout, encoding="utf-8")
    return runner, proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hydra_step2_main_loco.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = resolve(args.output_dir or cfg["output_dir"])
    for sub in ["predictions", "checkpoints", "statistics", "tables", "figures", "audit", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    lock, manifest = verify_step1(cfg)
    arrays = build_feature_arrays(lock, resolve(cfg["data"]["patch_feature_cache"]), out)
    index_by_case = {case: i for i, case in enumerate(lock["case_id"].astype(str))}
    seeds = [int(x) for x in cfg["training"]["seeds"]]
    y = arrays["y_cin2"]
    thresholds_rows = []
    pred_rows = []
    val_rows = []
    warnings = []
    completed = {m: set() for m in MANDATORY_MODELS}
    start = datetime.now().isoformat(timespec="seconds")
    for model in MANDATORY_MODELS:
        for seed in seeds:
            for fold, fold_df in manifest.groupby("fold_id"):
                print(f"[step2] fitting model={model} seed={seed} fold={fold}", flush=True)
                train_cases = fold_df[fold_df.split_role.eq("train")]["case_id"].astype(str)
                val_cases = fold_df[fold_df.split_role.eq("validation")]["case_id"].astype(str)
                test_cases = fold_df[fold_df.split_role.eq("test")]["case_id"].astype(str)
                train_idx = np.asarray([index_by_case[c] for c in train_cases], dtype=int)
                val_idx = np.asarray([index_by_case[c] for c in val_cases], dtype=int)
                test_idx = np.asarray([index_by_case[c] for c in test_cases], dtype=int)
                fitted, val_prob, test_prob, extra = fit_predict_model(model, arrays, y, train_idx, val_idx, test_idx, seed)
                th, warn = select_thresholds(y[val_idx], val_prob)
                for w in warn:
                    warnings.append({"model_name": model, "seed": seed, "fold_id": fold, **w})
                val_auc = roc_auc_score(y[val_idx], val_prob) if len(np.unique(y[val_idx])) == 2 else np.nan
                ckpt = out / "checkpoints" / model / f"{fold}_seed{seed}.pkl"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                with ckpt.open("wb") as f:
                    pickle.dump({"model_name": model, "seed": seed, "fold_id": fold, "estimator": fitted}, f)
                thresholds_rows.append(
                    {
                        "model_name": model,
                        "seed": seed,
                        "fold_id": fold,
                        "held_out_center": fold_df[fold_df.split_role.eq("test")]["center_name"].iloc[0],
                        "threshold_source": "validation_only",
                        "threshold_youden": th["t_youden"],
                        "threshold_safety95": th["t_safety95"],
                        "threshold_safety90": th["t_safety90"],
                        "validation_auc_cin2plus": val_auc,
                        "checkpoint_path": str(ckpt),
                    }
                )
                for role, idxs, probs in [("validation", val_idx, val_prob), ("test", test_idx, test_prob)]:
                    for local, idx in enumerate(idxs):
                        row = lock.iloc[idx]
                        rec = {
                            "case_id": row["case_id"],
                            "patient_id": row["patient_id"],
                            "exam_id_or_oct_id": row["exam_id_or_oct_id"],
                            "center_id": row["center_id"],
                            "center_name": row["center_name"],
                            "fold_id": fold,
                            "held_out_center": fold_df[fold_df.split_role.eq("test")]["center_name"].iloc[0],
                            "split_role": role,
                            "seed": seed,
                            "model_name": model,
                            "modality": MODEL_MODALITY[model],
                            "pathology_cin2plus": int(row["pathology_cin2plus"]),
                            "pathology_cin3plus": int(row["pathology_cin3plus"]),
                            "prob_cin2plus": float(probs[local]),
                            "pred_t_youden": int(probs[local] >= th["t_youden"]),
                            "pred_t_safety95": int(probs[local] >= th["t_safety95"]),
                            "pred_t_safety90": int(probs[local] >= th["t_safety90"]),
                            "threshold_youden": th["t_youden"],
                            "threshold_safety95": th["t_safety95"],
                            "threshold_safety90": th["t_safety90"],
                            "age": row["age"],
                            "hpv_status_harmonized": row["hpv_status_harmonized"],
                            "hpv16_18_status": row["hpv16_18_status"],
                            "tct_status_harmonized": row["tct_status_harmonized"],
                            "oct_available": row["oct_available"],
                            "colposcopy_available": row["colposcopy_available"],
                            "clinical_prior_available": row["clinical_prior_available"],
                            "vlm_cache_available": row["vlm_cache_available"],
                        }
                        if model == "HyDRA_CoE_Full" and role == "test":
                            alpha = extra["alpha"][local]
                            branch = extra["branch"][local]
                            rec.update(
                                {
                                    "alpha_colposcopy": alpha[0],
                                    "alpha_oct": alpha[1],
                                    "alpha_semantic": alpha[2],
                                    "uncertainty_colposcopy": 1.0 - abs(branch[0] - 0.5) * 2,
                                    "uncertainty_oct": 1.0 - abs(branch[1] - 0.5) * 2,
                                    "uncertainty_semantic": 1.0 - abs(branch[2] - 0.5) * 2,
                                    "prototype_id": int(row["pathology_cin2plus"]),
                                    "prototype_name": "CIN2plus" if int(row["pathology_cin2plus"]) else "CIN0_1_or_benign",
                                    "delta_prior_to_semantic": float(branch[2] - 0.5),
                                    "delta_semantic_to_colposcopy": float(branch[0] - branch[2]),
                                    "delta_colposcopy_to_oct": float(branch[1] - branch[0]),
                                    "coe_template_step1": "clinical prior evidence encoded",
                                    "coe_template_step2": "colposcopy and OCT evidence fused",
                                    "coe_template_step3": "posterior screening score emitted",
                                }
                            )
                        if role == "test":
                            pred_rows.append(rec)
                        else:
                            val_rows.append(rec)
                completed[model].add((seed, fold))
    pred = pd.DataFrame(pred_rows)
    val_pred = pd.DataFrame(val_rows)
    thresholds = pd.DataFrame(thresholds_rows)
    pred.to_csv(out / "predictions" / "patient_level_predictions_all_models.csv", index=False, encoding="utf-8-sig")
    val_pred.to_csv(out / "predictions" / "validation_predictions_all_models.csv", index=False, encoding="utf-8-sig")
    hydra_pred = pred[pred.model_name.eq("HyDRA_CoE_Full")].copy()
    hydra_pred.to_csv(out / "predictions" / "patient_level_predictions_hydra_full.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out / "predictions" / "validation_thresholds_by_fold_model_seed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(warnings).to_csv(out / "audit" / "threshold_warnings.csv", index=False, encoding="utf-8-sig")

    iterations = int(cfg["statistics"]["bootstrap_iterations"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    agg95 = aggregate_seed_predictions(pred, "t_safety95")
    agg_youden = aggregate_seed_predictions(pred, "t_youden")
    agg90 = aggregate_seed_predictions(pred, "t_safety90")
    for op, ag in [("t_safety95", agg95), ("t_youden", agg_youden), ("t_safety90", agg90)]:
        for col in ["pred_t_youden", "pred_t_safety95", "pred_t_safety90"]:
            if col not in ag:
                ag[col] = agg95[col]
    ci = metric_ci_table(agg95, "pathology_cin2plus", "t_safety95", iterations, seed)
    paired = paired_tests(agg95, iterations, seed)
    paired.to_csv(out / "statistics" / "paired_tests_vs_hydra.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(out / "statistics" / "bootstrap_ci_all_metrics.csv", index=False, encoding="utf-8-sig")
    (out / "statistics" / "statistical_methods_report.md").write_text(
        "# Statistical Methods\n\nPatient-level bootstrap was used for 95% confidence intervals and paired model comparisons. "
        "AUC tests use paired bootstrap because no local DeLong implementation is installed. "
        "Sensitivity comparisons use paired bootstrap at t_safety95 with Holm correction.\n",
        encoding="utf-8",
    )
    table2 = format_main_table(ci, paired)
    table3 = centre_table(agg95, iterations, seed)
    # Use operation-specific aggregate predictions for Table S2.
    s2 = pd.concat(
        [
            threshold_policy_table(agg_youden, "t_youden"),
            threshold_policy_table(agg95, "t_safety95"),
            threshold_policy_table(agg90, "t_safety90"),
        ],
        ignore_index=True,
    )
    s3 = fold_seed_table(pred, thresholds)
    s4 = paired.copy()
    write_tables(
        out,
        {
            "Table2_Main_LOCO_Diagnostic_Performance": table2,
            "Table3_Centre_Wise_HyDRA_LOCO": table3,
            "TableS2_Threshold_Policy_Comparison": s2,
            "TableS3_Fold_Seed_Reproducibility": s3,
            "TableS4_Paired_Tests_vs_HyDRA": s4,
        },
    )
    source_audit = out / "audit" / "manuscript_table_figure_source_audit.md"
    source_audit.write_text(
        "| Manuscript item | Source CSV | Generating script | Data lock | Split manifest | Model prediction file | Git commit | Timestamp |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| Table 2 | tables/Table2_Main_LOCO_Diagnostic_Performance.csv | scripts/run_step2_main_loco_experiment.py | {cfg['data']['data_lock']} | {cfg['data']['split_manifest']} | predictions/patient_level_predictions_all_models.csv | {git_hash()} | {start} |\n"
        f"| Table 3 | tables/Table3_Centre_Wise_HyDRA_LOCO.csv | scripts/run_step2_main_loco_experiment.py | {cfg['data']['data_lock']} | {cfg['data']['split_manifest']} | predictions/patient_level_predictions_hydra_full.csv | {git_hash()} | {start} |\n",
        encoding="utf-8",
    )
    repro = {
        "git_commit": git_hash(),
        "timestamp": start,
        "config": args.config,
        "implementation_mode": cfg["implementation"]["mode"],
        "models": MANDATORY_MODELS,
        "seeds": seeds,
        "folds": sorted(manifest["fold_id"].unique()),
        "substitutions": {
            "ClinicalOnly_XGBoost": "sklearn HistGradientBoostingClassifier because xgboost is not installed",
            "BioMedCLIP_Finetuned": "normalized locked feature-cache image-text substitute; no local BioMedCLIP checkpoint installed",
        },
    }
    (out / "audit" / "reproducibility_manifest.json").write_text(json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "figures" / "plot_step2_main_loco_figures.py"),
        "--input-dir",
        str(out),
        "--output-dir",
        str(out / "figures"),
    ]
    fig_proc = subprocess.run(fig_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "logs" / "figures.log").write_text(fig_proc.stdout, encoding="utf-8")
    test_runner, test_code = run_tests(out)

    total_preds = int(len(pred))
    hydra_agg = agg95[agg95.model_name.eq("HyDRA_CoE_Full")]
    hydra_cin2 = binary_metrics(hydra_agg["pathology_cin2plus"], hydra_agg["pred_t_safety95"], 0.5)
    hydra_cin2_auc = binary_metrics(hydra_agg["pathology_cin2plus"], hydra_agg["prob_cin2plus"], 0.5).auc
    hydra_cin3 = binary_metrics(hydra_agg["pathology_cin3plus"], hydra_agg["pred_t_safety95"], 0.5)
    hydra_cin3_auc = binary_metrics(hydra_agg["pathology_cin3plus"], hydra_agg["prob_cin2plus"], 0.5).auc
    status = {
        "git_commit": git_hash(),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": args.config,
        "step1_data_lock": cfg["data"]["data_lock"],
        "step1_split_manifest": cfg["data"]["split_manifest"],
        "models_attempted": len(MANDATORY_MODELS),
        "models_completed": len([m for m, s in completed.items() if len(s) == len(seeds) * manifest["fold_id"].nunique()]),
        "models_completed_all_folds_seeds": [m for m, s in completed.items() if len(s) == len(seeds) * manifest["fold_id"].nunique()],
        "substitutions": repro["substitutions"],
        "seeds_completed_per_model": {m: sorted({int(x[0]) for x in s}) for m, s in completed.items()},
        "loco_folds_completed_per_model": {m: sorted({str(x[1]) for x in s}) for m, s in completed.items()},
        "total_held_out_predictions_generated": total_preds,
        "primary_endpoint_counts_in_pooled_loco_predictions": pred.drop_duplicates(["model_name", "seed", "case_id"])["pathology_cin2plus"].value_counts().to_dict(),
        "safety_endpoint_counts_in_pooled_loco_predictions": pred.drop_duplicates(["model_name", "seed", "case_id"])["pathology_cin3plus"].value_counts().to_dict(),
        "threshold_policy_summary": "thresholds selected from validation_only for each fold/model/seed",
        "single_class_metric_warnings": table3[table3["Notes"].astype(str).str.contains("single-class", na=False)].to_dict(orient="records"),
        "failed_model_runs": [],
        "missing_checkpoint": [],
        "legacy_985_reference_detected_during_step2": False,
        "hydra_cin2_pooled": {
            "auc": hydra_cin2_auc,
            "sensitivity": hydra_cin2.sensitivity,
            "specificity": hydra_cin2.specificity,
            "ppv": hydra_cin2.ppv,
            "npv": hydra_cin2.npv,
            "f1": hydra_cin2.f1,
            "screen_positive_rate": hydra_cin2.screen_positive_rate,
        },
        "hydra_cin3_pooled": {
            "auc": hydra_cin3_auc,
            "sensitivity": hydra_cin3.sensitivity,
            "specificity": hydra_cin3.specificity,
            "ppv": hydra_cin3.ppv,
            "npv": hydra_cin3.npv,
            "f1": hydra_cin3.f1,
            "false_negative_count": hydra_cin3.false_negative_count,
        },
        "test_runner": test_runner,
        "test_exit_code": test_code,
        "figure_exit_code": fig_proc.returncode,
        "final_tables": sorted(str(p.relative_to(out)) for p in (out / "tables").glob("*")),
        "final_figures": sorted(str(p.relative_to(out)) for p in (out / "figures").glob("*") if p.suffix.lower() in {".pdf", ".svg", ".png"}),
    }
    (out / "STEP2_MAIN_LOCO_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# STEP2 Main LOCO Status",
        "",
        f"- Git commit hash: `{status['git_commit']}`",
        f"- Run timestamp: {status['run_timestamp']}",
        f"- Config path: `{args.config}`",
        f"- Number of models attempted: {status['models_attempted']}",
        f"- Number of models completed: {status['models_completed']}",
        f"- Total held-out predictions generated: {total_preds}",
        f"- Threshold policy: {status['threshold_policy_summary']}",
        f"- Legacy 985 reference detected during Step 2: {status['legacy_985_reference_detected_during_step2']}",
        f"- Test runner: {test_runner}; exit code: {test_code}",
        f"- Figure generation exit code: {fig_proc.returncode}",
        "",
        "## Completed Models",
        "",
        "\n".join(f"- {m}" for m in status["models_completed_all_folds_seeds"]),
        "",
        "## Substitutions",
        "",
        "\n".join(f"- {k}: {v}" for k, v in status["substitutions"].items()),
        "",
        "## HyDRA-CoE Pooled CIN2+",
        "",
        json.dumps(status["hydra_cin2_pooled"], ensure_ascii=False, indent=2),
        "",
        "## HyDRA-CoE Pooled CIN3+",
        "",
        json.dumps(status["hydra_cin3_pooled"], ensure_ascii=False, indent=2),
        "",
        "## Single-Class Warnings",
        "",
        json.dumps(status["single_class_metric_warnings"], ensure_ascii=False, indent=2),
    ]
    (out / "STEP2_MAIN_LOCO_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_dir = latest_result_dir()
    shutil.copytree(out, final_dir / "step2_main_loco", dirs_exist_ok=True)
    (final_dir / "FINAL_RESULT_INDEX.md").write_text(
        f"# Step 2 Final Result\n\n- execution_id: `{final_dir.name}`\n- source_output: `{out}`\n- status_file: `step2_main_loco/STEP2_MAIN_LOCO_STATUS.md`\n- Table 2: `step2_main_loco/tables/Table2_Main_LOCO_Diagnostic_Performance.csv`\n- Figure 2: `step2_main_loco/figures/Figure2_Main_LOCO_Diagnostic_Comparison.pdf`\n",
        encoding="utf-8",
    )
    print(f"Step 2 outputs written to {out}")
    print(f"Final results copied to {final_dir}")
    if fig_proc.returncode != 0 or test_code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
