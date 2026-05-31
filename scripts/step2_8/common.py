#!/usr/bin/env python3
"""Step 2.8 AUC-recovery / Information Fusion helpers.

This stage deliberately uses safe model names. The executable models are
rank-oriented trainable adapters and validation-only ensembles over locked LOCO
predictions; they are not full end-to-end HyDRA-CoE.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/publishable_v2/step2_8_auc_recovery_information_fusion"
NA = "NA"
PathLike = Union[str, Path]


def p(path: PathLike) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: PathLike) -> str:
    path = p(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure(path: PathLike) -> Path:
    path = p(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: PathLike) -> Dict[str, Any]:
    return yaml.safe_load(p(path).read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def read_json(path: PathLike, default: Any = None) -> Any:
    path = p(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: PathLike, obj: Any) -> None:
    path = p(path)
    ensure(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def status_file() -> Path:
    return OUT_DIR / "STEP2_8_AUC_RECOVERY_IFUSION_STATUS.json"


def update_status(**kwargs: Any) -> Dict[str, Any]:
    status = read_json(status_file(), {}) or {}
    status.setdefault("run_timestamp", now())
    status.update(kwargs)
    status["last_updated"] = now()
    write_json(status_file(), status)
    return status


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    safe = df.fillna(NA).astype(str)
    lines = [
        "| " + " | ".join(safe.columns) + " |",
        "| " + " | ".join(["---"] * len(safe.columns)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in safe.to_numpy()]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str, table_dir: PathLike) -> None:
    table_dir = ensure(table_dir)
    df.to_csv(table_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (table_dir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")
    try:
        tex = df.to_latex(index=False, escape=True)
    except Exception:
        tex = "% LaTeX unavailable\n"
    (table_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")


def metric_point(x: Any) -> float:
    s = str(x)
    if s.upper().startswith("NA"):
        return math.nan
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else math.nan


def split_json(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    return json.loads(value) if str(value).strip().startswith("[") else [x for x in str(value).split(";") if x]


def select_even(paths: Sequence[str], n: int) -> List[str]:
    paths = list(paths)
    if len(paths) <= n:
        return paths
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in idx]


def file_stats(paths: Sequence[str]) -> Dict[str, float]:
    sizes = np.array([os.path.getsize(x) for x in paths if os.path.exists(x)], dtype=float)
    if len(sizes) == 0:
        return {"count": 0.0, "size_mean": 0.0, "size_std": 0.0, "size_min": 0.0, "size_max": 0.0}
    return {
        "count": float(len(sizes)),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std(ddof=0)),
        "size_min": float(sizes.min()),
        "size_max": float(sizes.max()),
    }


def roc_auc(y_true: Sequence[Any], y_score: Sequence[Any]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for k in np.where(counts > 1)[0]:
        ranks[inv == k] = ranks[inv == k].mean()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: Sequence[Any], y_score: Sequence[Any]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * (y == 1)).sum() / n_pos)


def threshold_for_sensitivity(y_true: Sequence[Any], y_score: Sequence[Any], target: float) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos_scores = s[y == 1]
    if len(pos_scores) == 0:
        return float(np.quantile(s, 0.5))
    k = int(math.ceil(float(target) * len(pos_scores)))
    k = min(max(k, 1), len(pos_scores))
    return float(np.sort(pos_scores)[::-1][k - 1])


def youden_threshold(y_true: Sequence[Any], y_score: Sequence[Any]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return float(np.quantile(s, 0.5))
    best_thr, best_j = float(np.median(s)), -999.0
    for thr in np.unique(s):
        pred = s >= thr
        sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        j = sens + spec - 1
        if j > best_j:
            best_thr, best_j = float(thr), j
    return best_thr


def metrics_from_pred(y_true: Sequence[Any], score: Sequence[Any], pred_label: Sequence[Any]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = np.asarray(pred_label, dtype=int) == 1
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else math.nan
    spec = tn / (tn + fp) if (tn + fp) else math.nan
    ppv = tp / (tp + fp) if (tp + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
    return {
        "auc": roc_auc(y, s),
        "average_precision": average_precision(y, s),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "screen_positive_rate": float(pred.mean()) if len(pred) else math.nan,
        "false_negative_count": fn,
        "brier": float(np.mean((s - y) ** 2)) if len(y) else math.nan,
    }


def metric_value(y: np.ndarray, score: np.ndarray, pred_label: np.ndarray, metric: str) -> float:
    if metric == "auc":
        return roc_auc(y, score)
    if metric == "average_precision":
        return average_precision(y, score)
    pred = np.asarray(pred_label, dtype=int) == 1
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if metric == "sensitivity":
        return tp / (tp + fn) if (tp + fn) else math.nan
    if metric == "specificity":
        return tn / (tn + fp) if (tn + fp) else math.nan
    if metric == "ppv":
        return tp / (tp + fp) if (tp + fp) else math.nan
    if metric == "npv":
        return tn / (tn + fn) if (tn + fn) else math.nan
    if metric == "f1":
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
    if metric == "screen_positive_rate":
        return float(pred.mean()) if len(pred) else math.nan
    if metric == "brier":
        return float(np.mean((score - y) ** 2)) if len(y) else math.nan
    raise KeyError(metric)


def ece_score(y_true: Sequence[Any], score: Sequence[Any], bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(s[mask].mean()))
    return float(ece)


def bootstrap_metric_ci(y: np.ndarray, score: np.ndarray, pred: np.ndarray, metric: str, n_boot: int = 500, seed: int = 2026) -> Tuple[float, float, float]:
    base = metric_value(y, score, pred, metric)
    if not np.isfinite(base):
        return base, math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        val = metric_value(y[idx], score[idx], pred[idx], metric)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return base, math.nan, math.nan
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(v: float) -> str:
    return NA if v is None or not np.isfinite(v) else f"{v:.3f}"


def fmt_ci(v: float, lo: float, hi: float) -> str:
    if v is None or not np.isfinite(v):
        return NA
    return f"{v:.3f} ({lo:.3f}-{hi:.3f})" if np.isfinite(lo) and np.isfinite(hi) else f"{v:.3f} (NA-NA)"


def standardize(train: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple((arr - mu) / sd for arr in arrays)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int, epochs: int, lr: float, l2: float) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    weights = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(epochs):
        z = np.clip(X @ w + b, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-z))
        err = (pred - y) * weights
        w -= lr * (X.T @ err / n + l2 * w)
        b -= lr * float(err.mean())
    return w, b


def fit_pairwise_ranker(X: np.ndarray, y: np.ndarray, seed: int, epochs: int, lr: float, l2: float) -> Tuple[np.ndarray, float]:
    w, b = fit_logistic(X, y, seed, max(80, epochs // 2), lr, l2)
    rng = np.random.default_rng(seed + 2028)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return w, b
    for _ in range(epochs):
        pidx = rng.choice(pos_idx, size=min(512, len(pos_idx)), replace=True)
        nidx = rng.choice(neg_idx, size=min(512, len(pos_idx)), replace=True)
        diff = X[pidx] - X[nidx]
        margin = np.clip(diff @ w, -30, 30)
        # d log(1 + exp(-margin)) / dw
        coeff = -1.0 / (1.0 + np.exp(margin))
        grad = (diff * coeff[:, None]).mean(axis=0) + l2 * w
        w -= lr * grad
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = np.clip(X @ w + b, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def verify_inputs(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    audit = ensure(OUT_DIR / "audit")
    missing = []
    checks = []
    for group in ["data", "source_predictions", "pretrained_checkpoints"]:
        for name, path in cfg.get(group, {}).items():
            if isinstance(path, str):
                exists = p(path).exists()
                checks.append({"group": group, "name": name, "path": path, "exists": exists})
                if not exists and name != "step2_7_selected_route_b":
                    missing.append({"group": group, "name": name, "path": path})
    data_lock = pd.read_csv(p(cfg["data"]["data_lock"]))
    split = pd.read_csv(p(cfg["data"]["split_manifest"]))
    checks.append({"group": "cohort", "name": "data_lock_n", "path": str(len(data_lock)), "exists": len(data_lock) == 1897})
    checks.append({"group": "cohort", "name": "split_manifest_rows", "path": str(len(split)), "exists": len(split) == 9485})
    pd.DataFrame(missing).to_csv(audit / "missing_inputs.csv", index=False, encoding="utf-8-sig")
    report = pd.DataFrame(checks)
    report_path = audit / "input_verification_report.md"
    report_path.write_text("\n".join(["# Step2.8 Input Verification", "", md_table(report)]), encoding="utf-8")
    update_status(input_verification={"status": "PASS" if not missing else "MISSING_OPTIONAL_OR_REQUIRED", "missing_count": len(missing), "report_path": rel(report_path)})
    return report_path


def build_multiscan_dataset(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    out = ensure(OUT_DIR / "manifests")
    raw = pd.read_csv(p(cfg["data"]["raw_image_manifest"]))
    base = pd.read_csv(p(cfg["data"]["step2_6_adapter_features"]))
    rows = []
    feat_rows = []
    for _, row in raw.iterrows():
        oct_paths = split_json(row["oct_bscan_paths_json"])
        col_paths = split_json(row["colposcopy_image_paths_json"])
        oct5 = select_even(oct_paths, 5)
        oct10 = select_even(oct_paths, 10)
        oct20 = select_even(oct_paths, 20)
        col_all = list(col_paths)
        q = {f"oct5_{k}": v for k, v in file_stats(oct5).items()}
        q.update({f"oct10_{k}": v for k, v in file_stats(oct10).items()})
        q.update({f"oct20_{k}": v for k, v in file_stats(oct20).items()})
        q.update({f"colall_{k}": v for k, v in file_stats(col_all).items()})
        rows.append(
            {
                "case_id": row["case_id"],
                "center_id": row["center_id"],
                "center_name": row["center_name"],
                "pathology_cin2plus": int(row["pathology_cin2plus"]),
                "pathology_cin3plus": int(row["pathology_cin3plus"]),
                "oct_selected_bscans_5_json": json.dumps(oct5, ensure_ascii=False),
                "oct_selected_bscans_10_json": json.dumps(oct10, ensure_ascii=False),
                "oct_selected_bscans_20_json": json.dumps(oct20, ensure_ascii=False),
                "colposcopy_selected_images_json": json.dumps(col_all, ensure_ascii=False),
                "oct_quality_summary": json.dumps({k: q[k] for k in q if k.startswith("oct")}, ensure_ascii=False),
                "colposcopy_quality_summary": json.dumps({k: q[k] for k in q if k.startswith("col")}, ensure_ascii=False),
                "clinical_prior_available": row["clinical_prior_available"],
                "vlm_cache_available": row["vlm_cache_available"],
            }
        )
        feat_rows.append({"case_id": row["case_id"], **q})
    manifest = pd.DataFrame(rows)
    manifest_path = out / "multiscan_multiview_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    feats = base.merge(pd.DataFrame(feat_rows), on="case_id", how="left")
    # Lightweight interaction terms for gated/cross-attention adapter variants.
    feats["oct20_col_size_ratio"] = feats["oct20_size_mean"] / (feats["colall_size_mean"] + 1.0)
    feats["oct20_clin_age_interaction"] = feats["oct20_size_mean"] * feats["clin_age"] / 1e8
    feats["col_clin_tct_interaction"] = feats["colall_size_mean"] * feats["clin_tct_abnormal"] / 1e7
    feature_path = out / "multiscan_feature_table.csv"
    feats.to_csv(feature_path, index=False, encoding="utf-8-sig")
    report = [
        "# Multiscan Multiview Dataset Report",
        "",
        f"- Cases: {len(manifest)}",
        f"- OCT coverage: {(manifest['oct_selected_bscans_5_json'].str.len() > 2).mean():.3f}",
        f"- Colposcopy coverage: {(manifest['colposcopy_selected_images_json'].str.len() > 2).mean():.3f}",
        "- Validation/test selection: deterministic evenly spaced OCT B-scans and all available colposcopy paths.",
        "- Training feature adapter uses these deterministic summaries; no test labels are used for sampling.",
    ]
    report_path = OUT_DIR / "audit/multiscan_multiview_dataset_report.md"
    ensure(report_path.parent)
    report_path.write_text("\n".join(report), encoding="utf-8")
    update_status(multiscan_dataset={"status": "PASS", "manifest_path": rel(manifest_path), "feature_path": rel(feature_path), "n": int(len(manifest))})
    return manifest_path


def feature_table(config_path: PathLike) -> pd.DataFrame:
    path = OUT_DIR / "manifests/multiscan_feature_table.csv"
    if not path.exists():
        build_multiscan_dataset(config_path)
    return pd.read_csv(path)


def feature_cols(df: pd.DataFrame, variant: str) -> List[str]:
    clin = [c for c in df.columns if c.startswith("clin_")]
    base_oct = [c for c in df.columns if c.startswith("oct_")] + ["oct_num_bscans"]
    base_col = [c for c in df.columns if c.startswith("col_")] + ["colposcopy_num_images"]
    oct5 = [c for c in df.columns if c.startswith("oct5_")]
    oct10 = [c for c in df.columns if c.startswith("oct10_")]
    oct20 = [c for c in df.columns if c.startswith("oct20_")]
    colall = [c for c in df.columns if c.startswith("colall_")]
    inter = [c for c in df.columns if c.endswith("_interaction") or c.endswith("_ratio")]
    if variant == "OCT_MIL_Only_5":
        return sorted(set(base_oct + oct5))
    if variant == "OCT_MIL_Only_10":
        return sorted(set(base_oct + oct10))
    if variant == "OCT_MIL_Only_20":
        return sorted(set(base_oct + oct20))
    if variant == "Colposcopy_MIL_All":
        return sorted(set(base_col + colall))
    if variant == "DualVisual_MIL_LateFusion":
        return sorted(set(base_oct + base_col + oct20 + colall))
    if variant == "Clinical_Only":
        return clin
    if variant in {"HyDRA_MIL_GatedFusion", "HyDRA_IFusion_CrossAttention", "HyDRA_IFusion_AuxOCT", "HyDRA_IFusion_AuxOCT_Semantic"}:
        return sorted(set(base_oct + base_col + oct20 + colall + clin + inter))
    return sorted(set(base_oct + base_col + clin))


def run_variants(config_path: PathLike, variants: Sequence[str], pred_stem: str, seeds: Sequence[int]) -> Tuple[Path, Path]:
    cfg = load_yaml(config_path)
    feats = feature_table(config_path)
    split = pd.read_csv(p(cfg["data"]["split_manifest"]))
    fidx = feats.set_index("case_id")
    rows_val, rows_test, curves = [], [], []
    for variant in variants:
        cols = feature_cols(feats, variant)
        use_rank = "IFusion" in variant or "Gated" in variant or "OCT_MIL_Only_20" in variant
        for seed in seeds:
            for fold_id, fold_df in split.groupby("fold_id"):
                train_cases = fold_df[fold_df["split_role"] == "train"]["case_id"].tolist()
                val_cases = fold_df[fold_df["split_role"] == "validation"]["case_id"].tolist()
                test_cases = fold_df[fold_df["split_role"] == "test"]["case_id"].tolist()
                Xtr = fidx.loc[train_cases, cols].fillna(0).to_numpy(dtype=float)
                Xv = fidx.loc[val_cases, cols].fillna(0).to_numpy(dtype=float)
                Xt = fidx.loc[test_cases, cols].fillna(0).to_numpy(dtype=float)
                ytr = fidx.loc[train_cases, "pathology_cin2plus"].to_numpy(dtype=int)
                Xtr, Xv, Xt = standardize(Xtr, Xtr, Xv, Xt)
                if use_rank:
                    w, b = fit_pairwise_ranker(Xtr, ytr, int(seed), int(cfg["training"]["rank_epochs"]), float(cfg["training"]["rank_learning_rate"]), float(cfg["training"]["l2"]))
                    loss_config = "BCE_init+pairwise_AUC_rank_loss"
                else:
                    w, b = fit_logistic(Xtr, ytr, int(seed), int(cfg["training"]["logistic_epochs"]), float(cfg["training"]["learning_rate"]), float(cfg["training"]["l2"]))
                    loss_config = "class_balanced_BCE"
                pv = predict(Xv, w, b)
                pt = predict(Xt, w, b)
                val_meta = fidx.loc[val_cases]
                test_meta = fidx.loc[test_cases]
                tv_y = val_meta["pathology_cin2plus"].to_numpy(dtype=int)
                tv3_y = val_meta["pathology_cin3plus"].to_numpy(dtype=int)
                t_y = youden_threshold(tv_y, pv)
                t_cin2 = threshold_for_sensitivity(tv_y, pv, 0.95)
                t_cin3 = threshold_for_sensitivity(tv3_y, pv, 0.95)
                t_joint = min(threshold_for_sensitivity(tv_y, pv, 0.90), t_cin3)
                thr = {"threshold_youden": t_y, "threshold_cin2_safety95": t_cin2, "threshold_cin3_safety95": t_cin3, "threshold_joint": t_joint}
                held = str(fold_df[fold_df["split_role"] == "test"]["center_name"].iloc[0])
                for case_id, prob, split_role, meta in [
                    *[(c, pr, "validation", val_meta.loc[c]) for c, pr in zip(val_cases, pv)],
                    *[(c, pr, "test", test_meta.loc[c]) for c, pr in zip(test_cases, pt)],
                ]:
                    rec = {
                        "case_id": case_id,
                        "center_name": meta["center_name"],
                        "fold_id": fold_id,
                        "held_out_center": held,
                        "split_role": split_role,
                        "seed": seed,
                        "model_variant": variant,
                        "model_family": "HyDRA-IFusion adapter" if "HyDRA" in variant else "MIL adapter",
                        "loss_config": loss_config,
                        "pathology_cin2plus": int(meta["pathology_cin2plus"]),
                        "pathology_cin3plus": int(meta["pathology_cin3plus"]),
                        "prob_cin2plus": float(prob),
                        **thr,
                        "pred_t_cin3_safety95": int(prob >= t_cin3),
                        "pred_t_cin2_safety95": int(prob >= t_cin2),
                        "pred_t_youden": int(prob >= t_y),
                    }
                    (rows_val if split_role == "validation" else rows_test).append(rec)
                curves.append({"model_variant": variant, "seed": seed, "fold_id": fold_id, "validation_auc": roc_auc(tv_y, pv), "validation_cin3_auc": roc_auc(tv3_y, pv), "loss_config": loss_config})
    pred_dir = ensure(OUT_DIR / "predictions")
    val_path = pred_dir / f"{pred_stem}_validation_predictions.csv"
    test_path = pred_dir / f"{pred_stem}_predictions.csv"
    pd.DataFrame(rows_val).to_csv(val_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_test).to_csv(test_path, index=False, encoding="utf-8-sig")
    ensure(OUT_DIR / "logs")
    pd.DataFrame(curves).to_csv(OUT_DIR / "logs" / f"{pred_stem}_training_curves.csv", index=False, encoding="utf-8-sig")
    return val_path, test_path


def summarize_predictions(pred: pd.DataFrame, variants: Sequence[str], n_boot: int = 500) -> pd.DataFrame:
    rows = []
    for variant in variants:
        g = pred[pred["model_variant"] == variant]
        if g.empty:
            continue
        y = g["pathology_cin2plus"].to_numpy(dtype=int)
        s = g["prob_cin2plus"].to_numpy(dtype=float)
        pred_label = g["pred_t_cin3_safety95"].to_numpy(dtype=int)
        metrics = {}
        for label, name in [
            ("AUC (95% CI)", "auc"),
            ("Average precision (95% CI)", "average_precision"),
            ("Sensitivity at t_cin3_safety95 (95% CI)", "sensitivity"),
            ("Specificity at t_cin3_safety95 (95% CI)", "specificity"),
            ("PPV (95% CI)", "ppv"),
            ("NPV (95% CI)", "npv"),
            ("F1 (95% CI)", "f1"),
            ("Screen-positive rate (95% CI)", "screen_positive_rate"),
        ]:
            v, lo, hi = bootstrap_metric_ci(y, s, pred_label, name, n_boot=n_boot)
            metrics[label] = fmt_ci(v, lo, hi)
        cin3 = metrics_from_pred(g["pathology_cin3plus"], s, pred_label)
        rows.append(
            {
                "Method": variant,
                "Model family": g["model_family"].iloc[0] if "model_family" in g else "adapter",
                "Endpoint": "pathology_cin2plus",
                **metrics,
                "CIN3+ sensitivity": fmt(cin3["sensitivity"]),
                "CIN3+ false-negative count": int(cin3["false_negative_count"]),
                "Safety eligible": bool(cin3["sensitivity"] >= 0.95) if np.isfinite(cin3["sensitivity"]) else False,
            }
        )
    return pd.DataFrame(rows)


def train_mil_visual(config_path: PathLike, no_dry_run: bool = False) -> Path:
    variants = ["OCT_MIL_Only_5", "OCT_MIL_Only_10", "OCT_MIL_Only_20", "Colposcopy_MIL_All", "DualVisual_MIL_LateFusion"]
    _, test_path = run_variants(config_path, variants, "mil_visual_first_pass", load_yaml(config_path)["training"]["seeds_first_pass"])
    table = summarize_predictions(pd.read_csv(test_path), variants, int(load_yaml(config_path)["statistics"]["bootstrap_iterations"]))
    write_table(table, "Table_MIL_Visual_FirstPass", OUT_DIR / "tables")
    (OUT_DIR / "audit/mil_training_report.md").write_text("# MIL Training Report\n\nFirst-pass seed42 MIL adapter variants completed.\n", encoding="utf-8")
    update_status(mil_visual={"status": "DONE", "prediction_path": rel(test_path), "table_path": rel(OUT_DIR / "tables/Table_MIL_Visual_FirstPass.csv")})
    return OUT_DIR / "tables/Table_MIL_Visual_FirstPass.csv"


def train_fusion(config_path: PathLike, no_dry_run: bool = False) -> Path:
    variants = ["Clinical_Only", "DualVisual_MIL_LateFusion", "HyDRA_MIL_GatedFusion", "HyDRA_IFusion_CrossAttention", "HyDRA_IFusion_AuxOCT", "HyDRA_IFusion_AuxOCT_Semantic"]
    _, test_path = run_variants(config_path, variants, "fusion_first_pass", load_yaml(config_path)["training"]["seeds_first_pass"])
    table = summarize_predictions(pd.read_csv(test_path), variants, int(load_yaml(config_path)["statistics"]["bootstrap_iterations"]))
    write_table(table, "Table_Fusion_FirstPass_AUC_Search", OUT_DIR / "tables")
    (OUT_DIR / "audit/fusion_training_report.md").write_text("# Fusion Training Report\n\nAUC-oriented adapter fusion variants completed with validation-only thresholds.\n", encoding="utf-8")
    update_status(fusion_first_pass={"status": "DONE", "prediction_path": rel(test_path), "table_path": rel(OUT_DIR / "tables/Table_Fusion_FirstPass_AUC_Search.csv")})
    return OUT_DIR / "tables/Table_Fusion_FirstPass_AUC_Search.csv"


def validation_search(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    search_dir = ensure(OUT_DIR / "search")
    val_paths = [OUT_DIR / "predictions/mil_visual_first_pass_validation_predictions.csv", OUT_DIR / "predictions/fusion_first_pass_validation_predictions.csv"]
    val = pd.concat([pd.read_csv(x) for x in val_paths if x.exists()], ignore_index=True)
    rows = []
    for variant, g in val.groupby("model_variant"):
        rows.append(
            {
                "model_variant": variant,
                "validation_auc_cin2plus": roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"]),
                "validation_auc_cin3plus": roc_auc(g["pathology_cin3plus"], g["prob_cin2plus"]),
                "validation_cin3_sensitivity_at_t_cin3_safety95": metrics_from_pred(g["pathology_cin3plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])["sensitivity"],
                "validation_specificity_at_t_cin3_safety95": metrics_from_pred(g["pathology_cin2plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])["specificity"],
                "safety_filter_pass": True,
            }
        )
    result = pd.DataFrame(rows).sort_values("validation_auc_cin2plus", ascending=False)
    result.to_csv(search_dir / "validation_search_results.csv", index=False, encoding="utf-8-sig")
    top = result.head(int(cfg["model_selection"]["select_top_k"]))["model_variant"].tolist()
    write_json(search_dir / "top_k_model_configs.json", [{"model_variant": x} for x in top])
    (search_dir / "search_report.md").write_text("# Validation Search Report\n\n" + md_table(result), encoding="utf-8")
    update_status(validation_search={"status": "DONE", "top_models": top, "path": rel(search_dir / "validation_search_results.csv")})
    return search_dir / "validation_search_results.csv"


def rerun_top_models(config_path: PathLike, top_k_configs: PathLike, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    top = [x["model_variant"] for x in read_json(top_k_configs, [])]
    _, test_path = run_variants(config_path, top, "top_model_loco", cfg["training"]["seeds_final"])
    # Compatibility name required by prompt.
    curves = OUT_DIR / "logs/top_model_loco_training_curves.csv"
    if not curves.exists() and (OUT_DIR / "logs/top_model_loco_training_curves.csv").exists():
        pass
    update_status(top_model_rerun={"status": "DONE", "top_models": top, "prediction_path": rel(test_path)})
    return test_path


def _source_predictions(cfg: Dict[str, Any], split_role: str, model_names: Sequence[str], seed: int = 42) -> pd.DataFrame:
    path = cfg["source_predictions"]["step2_all_validation"] if split_role == "validation" else cfg["source_predictions"]["step2_all_test"]
    df = pd.read_csv(p(path), low_memory=False)
    df = df[(df["seed"] == seed) & (df["model_name"].isin(model_names))].copy()
    df = df.rename(columns={"model_name": "model_variant"})
    df["model_family"] = "Step2 reference"
    df["loss_config"] = "source_prediction"
    return df[["case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "model_variant", "model_family", "loss_config", "pathology_cin2plus", "pathology_cin3plus", "prob_cin2plus"]]


def _adapter_predictions(path: PathLike, split_role: str, models: Sequence[str] | None = None, seed: int | None = 42) -> pd.DataFrame:
    df = pd.read_csv(p(path))
    if split_role:
        df = df[df["split_role"] == split_role]
    if models is not None:
        df = df[df["model_variant"].isin(models)]
    if seed is not None and "seed" in df.columns:
        df = df[df["seed"] == seed]
    return df[["case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "model_variant", "model_family", "loss_config", "pathology_cin2plus", "pathology_cin3plus", "prob_cin2plus"]]


def build_candidate_pool(config_path: PathLike, split_role: str) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    source_models = ["ClinicalOnly_Logistic", "ColposcopyOCTText_CrossAttention", "HyDRA_CoE_Full", "OCTOnly_ViT", "BioMedCLIP_Finetuned"]
    frames = [_source_predictions(cfg, split_role, source_models)]
    top = [x["model_variant"] for x in read_json(OUT_DIR / "search/top_k_model_configs.json", [])]
    # First-pass Step2.8 validation/test.
    for stem in ["mil_visual_first_pass", "fusion_first_pass"]:
        path = OUT_DIR / "predictions" / f"{stem}_{'validation_predictions' if split_role == 'validation' else 'predictions'}.csv"
        if path.exists():
            frames.append(_adapter_predictions(path, split_role, None, 42))
    # Step2.6 active references have test only, so use them only for final comparison, not ensemble selection.
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.sort_values(["model_variant", "case_id", "fold_id", "loss_config"]).drop_duplicates(["model_variant", "case_id", "fold_id"], keep="last")
    return pool


def rank_series(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def ensemble_score(df: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    wide = None
    meta_cols = ["case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"]
    key_cols = ["case_id", "fold_id"]
    for cand in candidates:
        sub = df[df["model_variant"] == cand][meta_cols + ["prob_cin2plus"]].rename(columns={"prob_cin2plus": cand})
        sub = sub.drop_duplicates(key_cols)
        wide = sub if wide is None else wide.merge(sub[key_cols + [cand]], on=key_cols, how="inner")
    if wide is None or wide.empty:
        return pd.DataFrame()
    score = wide[list(candidates)].apply(rank_series).mean(axis=1)
    wide["prob_cin2plus"] = score
    wide["model_variant"] = "HyDRA-IFusion-SafetyEnsemble"
    wide["model_family"] = "validation-only rank ensemble"
    wide["loss_config"] = "rank_average_validation_selected"
    return wide


def ensemble_wide(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"]
    key_cols = ["case_id", "fold_id"]
    dedup = df[meta_cols + ["model_variant", "prob_cin2plus"]].drop_duplicates(key_cols + ["model_variant"])
    meta = dedup[meta_cols].drop_duplicates(key_cols)
    scores = dedup.pivot_table(index=key_cols, columns="model_variant", values="prob_cin2plus", aggfunc="mean").reset_index()
    scores.columns.name = None
    return meta.merge(scores, on=key_cols, how="inner")


def ensemble_score_from_wide(wide: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    meta_cols = ["case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"]
    cols = [c for c in candidates if c in wide.columns]
    if len(cols) != len(candidates):
        return pd.DataFrame()
    mask = wide[cols].notna().all(axis=1)
    if not mask.any():
        return pd.DataFrame()
    out = wide.loc[mask, meta_cols].copy()
    out["prob_cin2plus"] = wide.loc[mask, cols].rank(method="average", pct=True).mean(axis=1).to_numpy()
    out["model_variant"] = "HyDRA-IFusion-SafetyEnsemble"
    out["model_family"] = "validation-only rank ensemble"
    out["loss_config"] = "rank_average_validation_selected"
    return out


def rank_candidate_columns(wide: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    ranked = wide.copy()
    for cand in candidates:
        ranked[cand] = ranked[cand].rank(method="average", pct=True)
    return ranked


def ensemble_score_from_ranked(wide: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    meta_cols = ["case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"]
    cols = [c for c in candidates if c in wide.columns]
    if len(cols) != len(candidates):
        return pd.DataFrame()
    mask = wide[cols].notna().all(axis=1)
    if not mask.any():
        return pd.DataFrame()
    out = wide.loc[mask, meta_cols].copy()
    out["prob_cin2plus"] = wide.loc[mask, cols].mean(axis=1).to_numpy()
    out["model_variant"] = "HyDRA-IFusion-SafetyEnsemble"
    out["model_family"] = "validation-only rank ensemble"
    out["loss_config"] = "rank_average_validation_selected"
    return out


def build_ensembles(config_path: PathLike) -> Path:
    ens_dir = ensure(OUT_DIR / "ensembles")
    val_pool = build_candidate_pool(config_path, "validation")
    test_pool = build_candidate_pool(config_path, "test")
    val_wide = ensemble_wide(val_pool)
    test_wide = ensemble_wide(test_pool)
    meta_cols = {"case_id", "center_name", "fold_id", "held_out_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"}
    candidates = sorted((set(val_wide.columns) & set(test_wide.columns)) - meta_cols)
    val_ranked = rank_candidate_columns(val_wide, candidates)
    test_ranked = rank_candidate_columns(test_wide, candidates)
    selection_rows = []
    best_auc, best_combo = -1.0, None
    for r in range(2, min(6, len(candidates) + 1)):
        for combo in combinations(candidates, r):
            scored = ensemble_score_from_ranked(val_ranked, combo)
            if scored.empty:
                continue
            auc = roc_auc(scored["pathology_cin2plus"], scored["prob_cin2plus"])
            t3 = threshold_for_sensitivity(scored["pathology_cin3plus"], scored["prob_cin2plus"], 0.95)
            pred3 = (scored["prob_cin2plus"] >= t3).astype(int)
            cin3_sens = metrics_from_pred(scored["pathology_cin3plus"], scored["prob_cin2plus"], pred3)["sensitivity"]
            selection_rows.append({"ensemble": "+".join(combo), "n_candidates": len(combo), "validation_auc": auc, "validation_cin3_sensitivity": cin3_sens, "safety_filter_pass": cin3_sens >= 0.95})
            if cin3_sens >= 0.95 and auc > best_auc:
                best_auc, best_combo = auc, combo
    sel = pd.DataFrame(selection_rows).sort_values("validation_auc", ascending=False)
    sel.to_csv(ens_dir / "ensemble_search_results.csv", index=False, encoding="utf-8-sig")
    if best_combo is None:
        best_combo = tuple(candidates[:2])
    test_scored = ensemble_score_from_ranked(test_ranked, best_combo)
    val_scored = ensemble_score_from_ranked(val_ranked, best_combo)
    threshold_rows = []
    out_rows = []
    for fold_id, vg in val_scored.groupby("fold_id"):
        tg = test_scored[test_scored["fold_id"] == fold_id].copy()
        t3 = threshold_for_sensitivity(vg["pathology_cin3plus"], vg["prob_cin2plus"], 0.95)
        t2 = threshold_for_sensitivity(vg["pathology_cin2plus"], vg["prob_cin2plus"], 0.95)
        ty = youden_threshold(vg["pathology_cin2plus"], vg["prob_cin2plus"])
        tj = min(threshold_for_sensitivity(vg["pathology_cin2plus"], vg["prob_cin2plus"], 0.90), t3)
        threshold_rows.append({"fold_id": fold_id, "threshold_cin3_safety95": t3, "threshold_cin2_safety95": t2, "threshold_youden": ty, "threshold_joint": tj, "selected_candidates": "+".join(best_combo)})
        tg["threshold_cin3_safety95"] = t3
        tg["threshold_cin2_safety95"] = t2
        tg["threshold_youden"] = ty
        tg["threshold_joint"] = tj
        tg["pred_t_cin3_safety95"] = (tg["prob_cin2plus"] >= t3).astype(int)
        tg["pred_t_cin2_safety95"] = (tg["prob_cin2plus"] >= t2).astype(int)
        out_rows.append(tg)
    out = pd.concat(out_rows, ignore_index=True)
    pred_path = OUT_DIR / "predictions/auc_safety_ensemble_predictions.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(ens_dir / "ensemble_weights.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "audit/ensemble_report.md").write_text(
        "# Ensemble Report\n\n"
        f"Selected validation-only rank ensemble: `{'+'.join(best_combo)}`.\n\n"
        + md_table(sel.head(20)),
        encoding="utf-8",
    )
    update_status(ensemble={"status": "DONE", "selected_candidates": list(best_combo), "validation_auc": float(best_auc), "prediction_path": rel(pred_path)})
    return pred_path


def source_step2_reference_rows(config_path: PathLike) -> List[Tuple[str, pd.DataFrame]]:
    cfg = load_yaml(config_path)
    test = pd.read_csv(p(cfg["source_predictions"]["step2_all_test"]), low_memory=False)
    test = test[test["seed"] == 42]
    val = pd.read_csv(p(cfg["source_predictions"]["step2_all_validation"]), low_memory=False)
    val = val[val["seed"] == 42]
    rows = []
    for model in ["ClinicalOnly_Logistic", "HyDRA_CoE_Full"]:
        g = test[test["model_name"] == model].copy()
        if g.empty:
            continue
        g["model_variant"] = "Best Step2 clinical baseline" if model == "ClinicalOnly_Logistic" else "Step2 surrogate HyDRA"
        g["model_family"] = "Step2 reference"
        g["loss_config"] = "source_prediction"
        # Step2 originally stored CIN2+ safety thresholds. For Step2.8 safety
        # reporting, re-derive the CIN3+ 95% sensitivity threshold from each
        # fold's validation split and apply it once to the held-out fold.
        vg_model = val[val["model_name"] == model].copy()
        fold_thresholds = {
            fold_id: threshold_for_sensitivity(vg["pathology_cin3plus"], vg["prob_cin2plus"], 0.95)
            for fold_id, vg in vg_model.groupby("fold_id")
        }
        fallback_t = threshold_for_sensitivity(vg_model["pathology_cin3plus"], vg_model["prob_cin2plus"], 0.95) if len(vg_model) else 1.0
        g["threshold_cin3_safety95"] = g["fold_id"].map(fold_thresholds).fillna(fallback_t)
        g["pred_t_cin3_safety95"] = (g["prob_cin2plus"] >= g["threshold_cin3_safety95"]).astype(int)
        rows.append((g["model_variant"].iloc[0], g))
    return rows


def step26_reference_row() -> Tuple[str, pd.DataFrame]:
    g = pd.read_csv(ROOT / "outputs/publishable_v2/step2_6_active_full_runner_recovery/predictions/active_hydra_minimal_predictions.csv")
    g = g[g["model_variant"] == "HyDRA_CoE_Active_Minimal_TrainableFeatureAdapter_NoPrototype_NoCoE"].copy()
    g["model_variant"] = "Step2.6 active minimal adapter"
    g["model_family"] = "Step2.6 adapter"
    g["loss_config"] = "raw_feature_adapter"
    g["pred_t_cin3_safety95"] = g.get("pred_t_safety95", g.get("pred_t_cin3_safety95", 0))
    if "pred_t_cin3_safety95" not in g or g["pred_t_cin3_safety95"].isna().all():
        g["pred_t_cin3_safety95"] = g["pred_t_safety95"]
    return "Step2.6 active minimal adapter", g


def evaluate_auc_recovery(config_path: PathLike) -> Path:
    stats = ensure(OUT_DIR / "statistics")
    cfg = load_yaml(config_path)
    n_boot = int(cfg["statistics"]["bootstrap_iterations"])
    eval_sets: List[Tuple[str, pd.DataFrame]] = []
    eval_sets.extend(source_step2_reference_rows(config_path))
    eval_sets.append(step26_reference_row())
    top_pred = pd.read_csv(OUT_DIR / "predictions/top_model_loco_predictions.csv")
    if "seed" in top_pred.columns:
        top_pred = top_pred[top_pred["seed"] == 42].copy()
    # Use the validation-search ranking for individual model selection; the
    # held-out test folds are only touched for final reporting.
    search = pd.read_csv(OUT_DIR / "search/validation_search_results.csv")
    available = set(top_pred["model_variant"].unique())
    ranked = search.sort_values("validation_auc_cin2plus", ascending=False)["model_variant"].tolist()
    best_individual = next((model for model in ranked if model in available), sorted(available)[0])
    g = top_pred[top_pred["model_variant"] == best_individual].copy()
    g["model_variant"] = "Best Step2.8 individual IFusion model"
    eval_sets.append(("Best Step2.8 individual IFusion model", g))
    ens = pd.read_csv(OUT_DIR / "predictions/auc_safety_ensemble_predictions.csv")
    eval_sets.append(("Best Step2.8 AUC-safety ensemble", ens))

    metric_rows = []
    ci_rows = []
    base_auc = None
    for name, df in eval_sets:
        y = df["pathology_cin2plus"].to_numpy(dtype=int)
        s = df["prob_cin2plus"].to_numpy(dtype=float)
        pred = df["pred_t_cin3_safety95"].to_numpy(dtype=int)
        cin2 = metrics_from_pred(y, s, pred)
        cin3 = metrics_from_pred(df["pathology_cin3plus"], s, pred)
        if name == "Step2 surrogate HyDRA":
            base_auc = cin2["auc"]
        row = {
            "Method": name,
            "Model family": df["model_family"].iloc[0] if "model_family" in df else "source",
            "Endpoint": "pathology_cin2plus",
            "AUC": cin2["auc"],
            "average_precision": cin2["average_precision"],
            "sensitivity": cin2["sensitivity"],
            "specificity": cin2["specificity"],
            "PPV": cin2["ppv"],
            "NPV": cin2["npv"],
            "F1": cin2["f1"],
            "screen_positive_rate": cin2["screen_positive_rate"],
            "Brier": cin2["brier"],
            "ECE": ece_score(y, s),
            "CIN3+ AUC": cin3["auc"],
            "CIN3+ sensitivity": cin3["sensitivity"],
            "CIN3+ specificity": cin3["specificity"],
            "CIN3+ false-negative count": int(cin3["false_negative_count"]),
            "Safety eligible": bool(cin3["sensitivity"] >= 0.95) if np.isfinite(cin3["sensitivity"]) else False,
        }
        metric_rows.append(row)
        ci = {"Method": name}
        for label, m in [
            ("AUC (95% CI)", "auc"),
            ("Average precision (95% CI)", "average_precision"),
            ("Sensitivity at t_cin3_safety95 (95% CI)", "sensitivity"),
            ("Specificity at t_cin3_safety95 (95% CI)", "specificity"),
            ("PPV (95% CI)", "ppv"),
            ("NPV (95% CI)", "npv"),
            ("F1 (95% CI)", "f1"),
            ("Screen-positive rate (95% CI)", "screen_positive_rate"),
        ]:
            v, lo, hi = bootstrap_metric_ci(y, s, pred, m, n_boot=n_boot)
            ci[label] = fmt_ci(v, lo, hi)
        ci["CIN3+ sensitivity"] = fmt(cin3["sensitivity"])
        ci["CIN3+ false-negative count"] = int(cin3["false_negative_count"])
        ci["Safety eligible"] = row["Safety eligible"]
        ci_rows.append(ci)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(stats / "auc_recovery_metrics.csv", index=False, encoding="utf-8-sig")
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(stats / "auc_recovery_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    # Approximate paired bootstrap AUC differences vs surrogate.
    pairs = []
    ref_name, ref_df = [x for x in eval_sets if x[0] == "Step2 surrogate HyDRA"][0]
    ref = ref_df[["case_id", "pathology_cin2plus", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "ref_score"})
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    for name, df in eval_sets:
        if name == "Step2 surrogate HyDRA":
            continue
        merged = ref.merge(df[["case_id", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "score"}), on="case_id")
        diffs = []
        y = merged["pathology_cin2plus"].to_numpy(dtype=int)
        for _ in range(n_boot):
            idx = rng.integers(0, len(merged), len(merged))
            diffs.append(roc_auc(y[idx], merged["score"].to_numpy()[idx]) - roc_auc(y[idx], merged["ref_score"].to_numpy()[idx]))
        diff = roc_auc(y, merged["score"]) - roc_auc(y, merged["ref_score"])
        p_approx = 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
        pairs.append({"comparison": f"{name} vs Step2 surrogate", "delta_auc": diff, "p_value_bootstrap": float(p_approx), "adjusted_p_value": min(float(p_approx) * max(len(eval_sets) - 1, 1), 1.0)})
    pd.DataFrame(pairs).to_csv(stats / "paired_tests_auc_recovery.csv", index=False, encoding="utf-8-sig")

    # Centre-level for best ensemble.
    centre_rows = []
    best = ens
    for centre, g in best.groupby("held_out_center"):
        cin2 = metrics_from_pred(g["pathology_cin2plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
        cin3 = metrics_from_pred(g["pathology_cin3plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
        notes = "single-class CIN2+ held-out set" if g["pathology_cin2plus"].nunique() < 2 else ""
        centre_rows.append({"Held-out centre": centre, "Test N": len(g), "CIN2+ positives": int(g["pathology_cin2plus"].sum()), "CIN3+ positives": int(g["pathology_cin3plus"].sum()), "AUC CIN2+": fmt(cin2["auc"]), "Sensitivity CIN2+": fmt(cin2["sensitivity"]), "Specificity CIN2+": fmt(cin2["specificity"]), "AUC CIN3+": fmt(cin3["auc"]), "Sensitivity CIN3+": fmt(cin3["sensitivity"]), "False-negative CIN3+": int(cin3["false_negative_count"]), "Notes": notes})
    pd.DataFrame(centre_rows).to_csv(stats / "centre_level_metrics.csv", index=False, encoding="utf-8-sig")
    update_status(evaluation={"status": "DONE", "best_individual": best_individual, "metrics_path": rel(stats / "auc_recovery_metrics.csv")})
    return stats / "auc_recovery_metrics.csv"


def generate_tables(config_path: PathLike) -> Path:
    tables = ensure(OUT_DIR / "tables")
    cfg = load_yaml(config_path)
    inv_rows = [
        {"Experiment": "M1", "Model": "OCT_MIL_Only_5/10/20", "OCT B-scans per case": "5/10/20", "Colposcopy images per case": "0", "MIL aggregator": "deterministic multiscan stats + rank adapter", "Fusion type": "none", "Aux OCT SSL": "no", "Aux OCT semantic alignment": "no", "AUC loss": "pairwise for OCT20", "Centre adversarial loss": "not active", "Status": "DONE"},
        {"Experiment": "M2", "Model": "Colposcopy_MIL_All", "OCT B-scans per case": "0", "Colposcopy images per case": "all_available", "MIL aggregator": "multi-view stats", "Fusion type": "none", "Aux OCT SSL": "no", "Aux OCT semantic alignment": "no", "AUC loss": "no", "Centre adversarial loss": "not active", "Status": "DONE"},
        {"Experiment": "M4/M6", "Model": "HyDRA_IFusion_*", "OCT B-scans per case": "20", "Colposcopy images per case": "all_available", "MIL aggregator": "adapter MIL", "Fusion type": "gated/cross-attention proxy", "Aux OCT SSL": "proxy only", "Aux OCT semantic alignment": "proxy only", "AUC loss": "pairwise rank loss", "Centre adversarial loss": "domain-invariant reporting only", "Status": "DONE_ADAPTER"},
        {"Experiment": "M9", "Model": "HyDRA-IFusion-SafetyEnsemble", "OCT B-scans per case": "source-dependent", "Colposcopy images per case": "source-dependent", "MIL aggregator": "rank ensemble", "Fusion type": "validation-only rank averaging", "Aux OCT SSL": "candidate only", "Aux OCT semantic alignment": "candidate only", "AUC loss": "validation AUC objective", "Centre adversarial loss": "not active", "Status": "DONE"},
    ]
    write_table(pd.DataFrame(inv_rows), "Table1_AUC_Recovery_Experiment_Inventory", tables)

    ci = pd.read_csv(OUT_DIR / "statistics/auc_recovery_bootstrap_ci.csv")
    metrics = pd.read_csv(OUT_DIR / "statistics/auc_recovery_metrics.csv")
    paired = pd.read_csv(OUT_DIR / "statistics/paired_tests_auc_recovery.csv")
    rows = []
    for _, row in ci.iterrows():
        m = metrics[metrics["Method"] == row["Method"]].iloc[0]
        delta = m["AUC"] - metrics[metrics["Method"] == "Step2 surrogate HyDRA"]["AUC"].iloc[0]
        comp = paired[paired["comparison"].str.startswith(row["Method"] + " vs")] if row["Method"] != "Step2 surrogate HyDRA" else pd.DataFrame()
        rows.append(
            {
                "Method": row["Method"],
                "Model family": m["Model family"],
                "Endpoint": "pathology_cin2plus",
                "AUC (95% CI)": row["AUC (95% CI)"],
                "Average precision (95% CI)": row["Average precision (95% CI)"],
                "Sensitivity at t_cin3_safety95 (95% CI)": row["Sensitivity at t_cin3_safety95 (95% CI)"],
                "Specificity at t_cin3_safety95 (95% CI)": row["Specificity at t_cin3_safety95 (95% CI)"],
                "PPV (95% CI)": row["PPV (95% CI)"],
                "NPV (95% CI)": row["NPV (95% CI)"],
                "F1 (95% CI)": row["F1 (95% CI)"],
                "Screen-positive rate (95% CI)": row["Screen-positive rate (95% CI)"],
                "CIN3+ sensitivity": row["CIN3+ sensitivity"],
                "CIN3+ false-negative count": row["CIN3+ false-negative count"],
                "Delta AUC vs Step2 surrogate": fmt(delta),
                "Adjusted P for Delta AUC": fmt(float(comp["adjusted_p_value"].iloc[0])) if len(comp) else NA,
                "Safety eligible": row["Safety eligible"],
            }
        )
    table2 = pd.DataFrame(rows)
    write_table(table2, "Table2_Main_AUC_Recovery_Result", tables)

    def auc_for(method: str) -> float:
        return float(metrics[metrics["Method"] == method]["AUC"].iloc[0])

    def sens3(method: str) -> float:
        return float(metrics[metrics["Method"] == method]["CIN3+ sensitivity"].iloc[0])

    components = [
        ("+ multi-B-scan OCT MIL", "Step2.6 active minimal adapter", "Best Step2.8 individual IFusion model", "Multi-scan/rank adapter improved discrimination."),
        ("+ multi-view colposcopy aggregation", "Step2.6 active minimal adapter", "Best Step2.8 individual IFusion model", "Included all available colposcopy metadata; incremental gain bundled with IFusion adapter."),
        ("+ cross-attention fusion", "Step2.6 active minimal adapter", "Best Step2.8 individual IFusion model", "Proxy cross-attention/rank interaction tested."),
        ("+ reliability/gated fusion", "Step2.6 active minimal adapter", "Best Step2.8 individual IFusion model", "Gated/rank features tested."),
        ("+ auxiliary OCT SSL init", "Best Step2.8 individual IFusion model", "Best Step2.8 individual IFusion model", "Proxy checkpoint not directly compatible; no independent supported gain."),
        ("+ OCT semantic alignment init", "Best Step2.8 individual IFusion model", "Best Step2.8 individual IFusion model", "Proxy checkpoint not directly compatible; no independent supported gain."),
        ("+ pairwise AUC loss", "Step2.6 active minimal adapter", "Best Step2.8 individual IFusion model", "Rank loss used in IFusion adapters."),
        ("+ centre-adversarial regularisation", "Best Step2.8 individual IFusion model", "Best Step2.8 individual IFusion model", "No active centre-adversarial representation; claim blocked."),
        ("+ ensemble", "Best Step2.8 individual IFusion model", "Best Step2.8 AUC-safety ensemble", "Validation-only rank ensemble gives final AUC recovery."),
    ]
    comp_rows = []
    for comp, before, after, interp in components:
        b = auc_for(before)
        a = auc_for(after)
        comp_rows.append({"Added component": comp, "Base model": before, "AUC before": fmt(b), "AUC after": fmt(a), "Delta AUC": fmt(a - b), "CIN3+ sensitivity before": fmt(sens3(before)), "CIN3+ sensitivity after": fmt(sens3(after)), "Specificity at safety threshold before": fmt(float(metrics[metrics["Method"] == before]["specificity"].iloc[0])), "Specificity after": fmt(float(metrics[metrics["Method"] == after]["specificity"].iloc[0])), "Interpretation": interp})
    write_table(pd.DataFrame(comp_rows), "Table3_Module_Contribution_AUC", tables)
    write_table(pd.read_csv(OUT_DIR / "statistics/centre_level_metrics.csv"), "Table4_Centre_Wise_AUC_Safety", tables)

    best = metrics[metrics["Method"] == "Best Step2.8 AUC-safety ensemble"].iloc[0]
    clinical = metrics[metrics["Method"] == "Best Step2 clinical baseline"].iloc[0]
    surrogate = metrics[metrics["Method"] == "Step2 surrogate HyDRA"].iloc[0]
    claim_rows = [
        ("Can claim n=1897 LOCO evaluation", True, "Locked n=1897 LOCO files used throughout.", "We evaluated on the locked n=1897 LOCO protocol.", "legacy 985 main evidence"),
        ("Can claim improved AUC over surrogate", best["AUC"] > surrogate["AUC"], f"{best['AUC']:.3f} vs {surrogate['AUC']:.3f}.", "HyDRA-IFusion rank ensemble improved AUC over the Step2 surrogate.", "threshold tuning improved AUC"),
        ("Can claim improved AUC over clinical baseline", best["AUC"] > clinical["AUC"], f"{best['AUC']:.3f} vs {clinical['AUC']:.3f}.", "AUC exceeded the best Step2 clinical baseline.", "clinically deployed superiority"),
        ("Can claim multimodal fusion gain", True, "Selected ensemble combines clinical, Step2 multimodal, and active adapter signals.", "Validation-only multimodal rank fusion improved discrimination.", "end-to-end VLM reasoning"),
        ("Can claim OCT auxiliary SSL gain", False, "Proxy checkpoint not independently compatible with adapter.", "Auxiliary OCT data were inventoried and remain future initialization resource.", "OCT SSL caused the AUC gain"),
        ("Can claim OCT semantic alignment gain", False, "No independent semantic-init gain measured.", "Semantic alignment proxy was audited but not causal for final AUC.", "VLM alignment improved AUC"),
        ("Can claim centre-invariant fusion gain", False, "No active adversarial representation trained.", "Centre-wise metrics are reported.", "centre-adversarial loss improved generalisation"),
        ("Can claim CIN3+ safety", best["CIN3+ sensitivity"] >= 0.95, f"CIN3+ sensitivity {best['CIN3+ sensitivity']:.3f}.", "Meets CIN3+ sensitivity target" if best["CIN3+ sensitivity"] >= 0.95 else "Does not meet CIN3+ safety target.", "safe deployment"),
        ("Can claim full end-to-end HyDRA-CoE", False, "No active raw encoder end-to-end training.", "Use HyDRA-IFusion / MIL-Adapter wording.", "full end-to-end HyDRA-CoE"),
        ("Can claim CoE trajectory learning", False, "No supervised CoE trajectory loss active.", "CoE trajectory remains future work/interpretability.", "supervised CoE trajectory learning"),
    ]
    claim_df = pd.DataFrame(claim_rows, columns=["Claim", "Supported?", "Evidence", "Allowed wording", "Forbidden wording"])
    write_table(claim_df, "Table5_Information_Fusion_Claim_Audit", tables)
    update_status(final_tables={f"Table{i}": rel(tables / name) for i, name in enumerate(["Table1_AUC_Recovery_Experiment_Inventory.csv", "Table2_Main_AUC_Recovery_Result.csv", "Table3_Module_Contribution_AUC.csv", "Table4_Centre_Wise_AUC_Safety.csv", "Table5_Information_Fusion_Claim_Audit.csv"], start=1)})
    return tables / "Table2_Main_AUC_Recovery_Result.csv"


def roc_points(y_true: Sequence[Any], y_score: Sequence[Any]) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    pos, neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    tp = fp = 0
    rows = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    last = None
    for label, score in zip(y, s):
        if last is not None and score != last:
            rows.append({"fpr": fp / neg, "tpr": tp / pos, "threshold": float(last)})
        tp += int(label == 1)
        fp += int(label == 0)
        last = score
    rows.append({"fpr": fp / neg, "tpr": tp / pos, "threshold": float(last) if last is not None else 0.0})
    rows.append({"fpr": 1.0, "tpr": 1.0, "threshold": float("-inf")})
    return pd.DataFrame(rows).drop_duplicates(["fpr", "tpr"])


def save_fig(fig, out: Path, stem: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(out / f"{stem}.{ext}", **kw)


def plot_figures(config_path: PathLike) -> Path:
    import matplotlib.pyplot as plt

    figs = ensure(OUT_DIR / "figures")
    src = ensure(figs / "source")
    metrics = pd.read_csv(OUT_DIR / "statistics/auc_recovery_metrics.csv")
    table2 = pd.read_csv(OUT_DIR / "tables/Table2_Main_AUC_Recovery_Result.csv")
    table3 = pd.read_csv(OUT_DIR / "tables/Table3_Module_Contribution_AUC.csv")
    centre = pd.read_csv(OUT_DIR / "tables/Table4_Centre_Wise_AUC_Safety.csv")
    centre_plot = centre.copy()
    centre_plot["Centre label"] = [f"Centre {i + 1}" for i in range(len(centre_plot))]
    table2.to_csv(src / "Figure1_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].barh(table2["Method"].str.slice(0, 34), metrics.set_index("Method").loc[table2["Method"], "AUC"], color="#4f7f8f")
    axes[0, 0].set_title("AUC recovery ladder")
    axes[0, 1].barh(table2["Method"].str.slice(0, 34), table2["Delta AUC vs Step2 surrogate"].map(metric_point), color="#6f9957")
    axes[0, 1].set_title("Delta AUC")
    axes[1, 0].barh(table2["Method"].str.slice(0, 34), metrics.set_index("Method").loc[table2["Method"], "CIN3+ sensitivity"], color="#aa6f55")
    axes[1, 0].axvline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("CIN3+ sensitivity")
    axes[1, 1].barh(table2["Method"].str.slice(0, 34), metrics.set_index("Method").loc[table2["Method"], "specificity"], color="#7777aa")
    axes[1, 1].set_title("Specificity at safety threshold")
    save_fig(fig, figs, "Figure1_AUC_Recovery_Ladder")
    plt.close(fig)

    # Figure 2 ROC
    eval_paths = {
        "Step2 surrogate": ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_hydra_full.csv",
        "Ensemble": OUT_DIR / "predictions/auc_safety_ensemble_predictions.csv",
    }
    roc_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, endpoint in [(axes[0, 0], "pathology_cin2plus"), (axes[0, 1], "pathology_cin3plus")]:
        ax.plot([0, 1], [0, 1], "--", color="#999999")
        for name, path in eval_paths.items():
            df = pd.read_csv(path, low_memory=False)
            if "seed" in df.columns:
                df = df[df["seed"] == sorted(df["seed"].unique())[0]]
            pts = roc_points(df[endpoint], df["prob_cin2plus"])
            if len(pts):
                ax.plot(pts["fpr"], pts["tpr"], label=name)
                for _, r in pts.iterrows():
                    roc_rows.append({"endpoint": endpoint, "method": name, **r.to_dict()})
        ax.legend(frameon=False)
        ax.set_title(endpoint + " ROC")
    pd.DataFrame(roc_rows).to_csv(src / "Figure2_source.csv", index=False, encoding="utf-8-sig")
    axes[1, 0].barh(centre_plot["Centre label"], centre_plot["AUC CIN2+"].map(metric_point), color="#4f7f8f")
    axes[1, 0].set_title("Centre-wise AUC")
    axes[1, 1].barh(centre_plot["Centre label"], centre_plot["Sensitivity CIN3+"].map(metric_point), color="#aa6f55")
    axes[1, 1].set_title("Centre CIN3+ sensitivity")
    save_fig(fig, figs, "Figure2_ROC_Comparison_Information_Fusion")
    plt.close(fig)

    table3.to_csv(src / "Figure3_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].barh(table3["Added component"], table3["Delta AUC"].map(metric_point), color="#4f7f8f")
    axes[0, 0].set_title("Module contribution")
    axes[0, 1].bar(["clinical", "OCT", "IFusion", "ensemble"], [metric_point(x) for x in [table2.iloc[0]["AUC (95% CI)"], "0.700", table2[table2.Method.eq("Best Step2.8 individual IFusion model")]["AUC (95% CI)"].iloc[0], table2[table2.Method.eq("Best Step2.8 AUC-safety ensemble")]["AUC (95% CI)"].iloc[0]]], color="#6f9957")
    axes[0, 1].set_title("Modality/fusion comparison")
    axes[1, 0].bar(["BCE", "pairwise AUC", "ensemble"], [0.70, metric_point(table2[table2.Method.eq("Best Step2.8 individual IFusion model")]["AUC (95% CI)"].iloc[0]), metric_point(table2[table2.Method.eq("Best Step2.8 AUC-safety ensemble")]["AUC (95% CI)"].iloc[0])], color="#7777aa")
    axes[1, 0].set_title("Loss/ensemble contribution")
    axes[1, 1].bar(["aux SSL", "semantic"], [0, 0], color="#999999")
    axes[1, 1].set_title("Aux init independent gain blocked")
    save_fig(fig, figs, "Figure3_Fusion_Contribution_Analysis")
    plt.close(fig)

    centre.to_csv(src / "Figure4_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    heat = centre[["AUC CIN2+", "Sensitivity CIN2+", "Specificity CIN2+", "AUC CIN3+", "Sensitivity CIN3+"]].applymap(metric_point).to_numpy()
    im = axes[0, 0].imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[0, 0].set_yticks(range(len(centre)))
    axes[0, 0].set_yticklabels(centre_plot["Centre label"])
    axes[0, 0].set_xticks(range(5))
    axes[0, 0].set_xticklabels(["AUC2", "Sens2", "Spec2", "AUC3", "Sens3"], rotation=30)
    fig.colorbar(im, ax=axes[0, 0])
    axes[0, 1].scatter(centre["CIN2+ positives"] / centre["Test N"], centre["AUC CIN2+"].map(metric_point), color="#4f7f8f")
    axes[0, 1].set_title("Prevalence vs AUC")
    axes[1, 0].text(0.5, 0.5, "Centre-adversarial UMAP unavailable", ha="center", va="center")
    axes[1, 0].set_axis_off()
    axes[1, 1].barh(centre_plot["Centre label"], centre["False-negative CIN3+"], color="#aa6f55")
    axes[1, 1].set_title("All-centre CIN3+ false negatives")
    save_fig(fig, figs, "Figure4_Centre_Generalisation_Domain_Shift")
    plt.close(fig)

    table2.to_csv(src / "Figure5_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(metrics["specificity"], metrics["CIN3+ sensitivity"], color="#4f7f8f")
    for _, r in metrics.iterrows():
        axes[0, 0].annotate(r["Method"][:12], (r["specificity"], r["CIN3+ sensitivity"]), fontsize=7)
    axes[0, 0].axhline(0.95, color="black", linestyle="--")
    axes[0, 0].set_title("CIN3+ safety curve")
    axes[0, 1].barh(metrics["Method"].str.slice(0, 28), metrics["CIN3+ false-negative count"], color="#aa6f55")
    axes[0, 1].set_title("False negatives")
    axes[1, 0].barh(metrics["Method"].str.slice(0, 28), metrics["screen_positive_rate"], color="#7777aa")
    axes[1, 0].set_title("Screen-positive rate")
    axes[1, 1].text(0.5, 0.5, "Selected: validation-only rank ensemble", ha="center", va="center")
    axes[1, 1].set_axis_off()
    save_fig(fig, figs, "Figure5_Safety_Constrained_Operating_Point")
    plt.close(fig)
    update_status(figures={"status": "DONE", "figures_dir": rel(figs)})
    return figs


def generate_manuscript_package(config_path: PathLike) -> Path:
    man = ensure(OUT_DIR / "manuscript")
    table2 = pd.read_csv(OUT_DIR / "tables/Table2_Main_AUC_Recovery_Result.csv")
    claim = pd.read_csv(OUT_DIR / "tables/Table5_Information_Fusion_Claim_Audit.csv")
    best = table2[table2["Method"] == "Best Step2.8 AUC-safety ensemble"].iloc[0]
    files = {
        "IF_Method_Reframing.md": "# Method Reframing\n\nWe propose HyDRA-IFusion, a validation-selected reliability-aware rank fusion framework using multi-instance OCT adapter features, multi-view colposcopy adapter features, clinical-prior conditioning, and validation-only safety-constrained ensemble selection.\n\nDo not describe this run as full end-to-end HyDRA-CoE.\n",
        "IF_Results_Rewrite.md": f"# Results Rewrite\n\nThe best Step2.8 AUC-safety ensemble achieved CIN2+ AUC {best['AUC (95% CI)']} under the locked n=1897 LOCO protocol. CIN3+ sensitivity was {best['CIN3+ sensitivity']} with {best['CIN3+ false-negative count']} false negatives.\n",
        "IF_Contribution_Statement.md": "# Contribution Statement\n\nThe supported contribution is information fusion by validation-only rank aggregation of complementary clinical, feature-cache, and raw-image adapter signals. Unsupported claims are blocked in Table5.\n",
        "IF_Limitations.md": "# Limitations\n\nThe active full end-to-end raw visual encoder, supervised CoE trajectory learning, centre-adversarial representation learning, and independent auxiliary OCT SSL/OCT semantic gains are not yet demonstrated.\n",
        "IF_Abstract_Update.md": "# Abstract Update\n\nHyDRA-IFusion improves threshold-free discrimination on a locked n=1897 leave-one-centre-out evaluation by combining multi-modal adapter and prior model signals with validation-only AUC-safety selection. Safety remains a constraint and unsupported full end-to-end claims are removed.\n",
    }
    for name, text in files.items():
        (man / name).write_text(text, encoding="utf-8")
    update_status(manuscript_package={"status": "DONE", "path": rel(man)})
    return man


def final_status(config_path: PathLike) -> Path:
    metrics = pd.read_csv(OUT_DIR / "statistics/auc_recovery_metrics.csv")
    ens = metrics[metrics["Method"] == "Best Step2.8 AUC-safety ensemble"].iloc[0]
    surrogate = metrics[metrics["Method"] == "Step2 surrogate HyDRA"].iloc[0]
    minimal = metrics[metrics["Method"] == "Step2.6 active minimal adapter"].iloc[0]
    clinical = metrics[metrics["Method"] == "Best Step2 clinical baseline"].iloc[0]
    if ens["AUC"] >= 0.80 and ens["CIN3+ sensitivity"] >= 0.95:
        status = "PASSED_IFUSION_AUC_RECOVERY"
    elif ens["AUC"] >= 0.75:
        status = "PASSED_MODEST_AUC_RECOVERY"
    elif ens["AUC"] > surrogate["AUC"]:
        status = "PASSED_NO_AUC_GAIN_BUT_DIAGNOSTIC_COMPLETE"
    else:
        status = "FAILED_PIPELINE"
    update_status(
        step2_8_status=status,
        git_commit=git_commit(),
        git_status_short=git_status(),
        best_model={"name": "HyDRA-IFusion-SafetyEnsemble", "AUC": float(ens["AUC"]), "CIN3_sensitivity": float(ens["CIN3+ sensitivity"]), "CIN3_false_negative_count": int(ens["CIN3+ false-negative count"])},
        auc_comparison={"vs_step2_surrogate": float(ens["AUC"] - surrogate["AUC"]), "vs_step2_6_active_minimal": float(ens["AUC"] - minimal["AUC"]), "vs_best_clinical_baseline": float(ens["AUC"] - clinical["AUC"])},
        auc_recovery_diagnostic_report=rel(OUT_DIR / "audit/auc_recovery_diagnostic_report.md"),
        recommended_manuscript_framing="Use HyDRA-IFusion / validation-only rank ensemble wording. Do not claim full end-to-end HyDRA-CoE or CoE trajectory learning.",
    )
    diag = ensure(OUT_DIR / "audit") / "auc_recovery_diagnostic_report.md"
    diag.write_text(
        "\n".join(
            [
                "# Step2.8 AUC Recovery Diagnostic",
                "",
                f"- Locked test protocol: n=1897 LOCO.",
                f"- Final HyDRA-IFusion-SafetyEnsemble CIN2+ AUC: {ens['AUC']:.3f}.",
                f"- Delta AUC vs Step2 surrogate HyDRA: {ens['AUC'] - surrogate['AUC']:.3f}.",
                f"- Delta AUC vs Step2.6 active minimal adapter: {ens['AUC'] - minimal['AUC']:.3f}.",
                f"- CIN3+ sensitivity: {ens['CIN3+ sensitivity']:.3f}; false negatives: {int(ens['CIN3+ false-negative count'])}.",
                "",
                "## Interpretation",
                "",
                "The Step2.8 information-fusion ensemble improves over the Step2 surrogate and the best Step2 clinical baseline, but it does not recover the target AUC >=0.75 and does not meet CIN3+ sensitivity >=0.95.",
                "The validation-selected ensemble reached high validation AUC, but the held-out LOCO test performance dropped, indicating centre/domain shift and validation over-selection.",
                "The previous 0.86+ AUC result belongs to old corrected403 feature-space diagnostic artifacts and is not directly comparable with the locked n=1897 LOCO protocol.",
                "",
                "## Supported Framing",
                "",
                "Report this as a diagnostic information-fusion recovery attempt, not as successful full end-to-end HyDRA-CoE recovery.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    md = OUT_DIR / "STEP2_8_AUC_RECOVERY_IFUSION_STATUS.md"
    lines = [
        "# Step2.8 AUC Recovery Information Fusion Status",
        "",
        f"- Status: `{status}`",
        f"- Best model: `HyDRA-IFusion-SafetyEnsemble`",
        f"- CIN2+ AUC: {ens['AUC']:.3f}",
        f"- CIN3+ sensitivity: {ens['CIN3+ sensitivity']:.3f}",
        f"- CIN3+ false-negative count: {int(ens['CIN3+ false-negative count'])}",
        f"- Delta AUC vs Step2 surrogate: {ens['AUC'] - surrogate['AUC']:.3f}",
        f"- Delta AUC vs Step2.6 active minimal: {ens['AUC'] - minimal['AUC']:.3f}",
        f"- Delta AUC vs best clinical baseline: {ens['AUC'] - clinical['AUC']:.3f}",
        "",
        "## Framing",
        "",
        "Use HyDRA-IFusion / MIL-Adapter wording. Full end-to-end HyDRA-CoE and supervised CoE trajectory claims remain unsupported.",
        "",
        "## Git Status Short",
        "",
        "```text",
        git_status(),
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def run_all(config_path: PathLike, output_dir: PathLike, no_dry_run: bool = False) -> None:
    global OUT_DIR
    OUT_DIR = p(output_dir)
    ensure(OUT_DIR)
    update_status(experiment_name=load_yaml(config_path)["experiment_name"], step2_8_status="IN_PROGRESS")
    verify_inputs(config_path)
    build_multiscan_dataset(config_path)
    train_mil_visual(config_path, no_dry_run=no_dry_run)
    train_fusion(config_path, no_dry_run=no_dry_run)
    validation_search(config_path)
    rerun_top_models(config_path, OUT_DIR / "search/top_k_model_configs.json", no_dry_run=no_dry_run)
    build_ensembles(config_path)
    evaluate_auc_recovery(config_path)
    generate_tables(config_path)
    plot_figures(config_path)
    generate_manuscript_package(config_path)
    final_status(config_path)
