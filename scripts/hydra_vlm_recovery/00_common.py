#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

LOCAL_PKG_OVERLAY = Path(__file__).resolve().parents[2] / ".routeb_python_pkgs"
if LOCAL_PKG_OVERLAY.exists() and str(LOCAL_PKG_OVERLAY) not in sys.path:
    sys.path.insert(0, str(LOCAL_PKG_OVERLAY))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logit
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/publishable_v2/hydra_vlm_recovery"
DATA_LOCK = ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv"
FEATURE_NPZ = ROOT / "outputs/publishable_v2/step2_main_loco/audit/step2_locked_feature_arrays.npz"
RAW_ADAPTER_CSV = ROOT / "outputs/publishable_v2/step2_6_active_full_runner_recovery/manifests/raw_adapter_features_n1897.csv"
ROUTE_B_TTA = ROOT / "outputs/publishable_v2/if_route_b_master/tables/Table_TTA_Comparison_IF.csv"
ROUTE_B_CENTRE = ROOT / "outputs/publishable_v2/if_route_b_master/tables/Table_Centre_Level_Results_IF.csv"

SCI_PALETTE = [
    "#8b98b3",
    "#abb8cc",
    "#dbb98c",
    "#edd6b8",
    "#b57979",
    "#dea3a2",
    "#b3b0b0",
    "#d9d8d8",
]

CENTRE_LABELS = {
    "十堰市人民医院": "Shiyan",
    "恩施州中心医院": "Enshi",
    "武大人民医院": "Wuhan",
    "荆州市第一人民医院": "Jingzhou",
    "襄阳市中心医院": "Xiangyang",
}


def ensure_dirs() -> None:
    for name in [
        "p00_protocol_lock",
        "vlm01_foldwise_lora",
        "vlm02_feature_quality",
        "vlm03_feature_package",
        "loco01_hydra_vlm_loco",
        "abl01_module_ablation",
        "abl04_vlm_backbone_ablation",
        "coe01_decoder_verification",
        "claim_lock_update",
        "manifests",
        "logs",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=SCI_PALETTE,
        font="Noto Sans CJK JP",
        rc={
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "grid.color": "#e8e8e8",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        },
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False, encoding="utf-8-sig")
        return
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, path: Path) -> str:
    """Write a feature table. Prefer parquet; fallback is pickle plus CSV companion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return "PARQUET"
    except Exception:
        df.to_pickle(path)
        df.to_csv(path.with_suffix(path.suffix + ".csv"), index=False, encoding="utf-8-sig")
        return "PANDAS_PICKLE_FALLBACK_WITH_CSV_COMPANION"


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_pickle(path)
        except Exception:
            csv_path = path.with_suffix(path.suffix + ".csv")
            return pd.read_csv(csv_path)


def save_fig(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def file_manifest(base: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rows.append({"path": rel(p), "size_bytes": p.stat().st_size})
    df = pd.DataFrame(rows)
    write_csv(out_path, df)
    return df


def load_data_lock() -> pd.DataFrame:
    df = pd.read_csv(DATA_LOCK)
    df["centre_label"] = df["center_name"].map(CENTRE_LABELS).fillna(df["center_name"].astype(str))
    return df


def load_feature_arrays() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    dl = load_data_lock()
    z = np.load(FEATURE_NPZ, allow_pickle=True)
    ids = [str(x) for x in z["case_id"]]
    idx = {cid: i for i, cid in enumerate(ids)}
    order = [idx[str(cid)] for cid in dl["case_id"]]
    arrays = {
        "oct": np.asarray(z["oct"])[order].astype(np.float32),
        "col": np.asarray(z["col"])[order].astype(np.float32),
        "clinical": np.asarray(z["clinical"])[order].astype(np.float32),
    }
    return dl, arrays


def infer_feature_groups(df: pd.DataFrame, frozen: bool = False) -> dict[str, list[str]]:
    pref = "frozen_" if frozen else ""
    return {
        "clinical": [c for c in df.columns if c.startswith(f"{pref}clinical_feature_")],
        "colpo": [c for c in df.columns if c.startswith(f"{pref}colpo_feature_")],
        "oct": [c for c in df.columns if c.startswith(f"{pref}oct_feature_")],
        "text": [c for c in df.columns if c.startswith(f"{pref}text_feature_")],
    }


def feature_matrix(df: pd.DataFrame, groups: Iterable[str], frozen: bool = False) -> np.ndarray:
    g = infer_feature_groups(df, frozen=frozen)
    cols: list[str] = []
    for name in groups:
        cols.extend(g.get(name, []))
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return x


def make_folds(dl: pd.DataFrame) -> list[dict]:
    folds = []
    centres = list(dl["center_name"].dropna().drop_duplicates())
    for i, centre in enumerate(centres, start=1):
        tgt = dl["center_name"].eq(centre)
        src = ~tgt
        folds.append(
            {
                "fold_id": f"fold_{i:02d}_{CENTRE_LABELS.get(centre, centre)}",
                "held_out_centre": centre,
                "held_out_centre_label": CENTRE_LABELS.get(centre, centre),
                "source_centres": [c for c in centres if c != centre],
                "target_centre": centre,
                "n_source": int(src.sum()),
                "n_target": int(tgt.sum()),
                "n_CIN2_source": int(dl.loc[src, "pathology_cin2plus"].sum()),
                "n_CIN2_target": int(dl.loc[tgt, "pathology_cin2plus"].sum()),
                "n_CIN3_source": int(dl.loc[src, "pathology_cin3plus"].sum()),
                "n_CIN3_target": int(dl.loc[tgt, "pathology_cin3plus"].sum()),
            }
        )
    return folds


def safe_auc(y, s) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def ece_score(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def bootstrap_auc_ci(y, s, n_boot: int = 300, seed: int = 2026) -> str:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    auc = safe_auc(y, s)
    if math.isnan(auc):
        return "NA"
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return f"{auc:.3f} (NA)"
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"{auc:.3f} ({lo:.3f}-{hi:.3f})"


def select_threshold_for_cin3(y_cin3, scores, target_sens: float = 0.95) -> float:
    y = np.asarray(y_cin3, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return 0.5
    thresholds = np.unique(np.r_[0.0, s, 1.0])
    best_t, best_spec = 0.0, -1.0
    for t in thresholds:
        pred = s >= t
        tp = ((pred) & (y == 1)).sum()
        fn = ((~pred) & (y == 1)).sum()
        tn = ((~pred) & (y == 0)).sum()
        fp = ((pred) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= target_sens and spec >= best_spec:
            best_spec = spec
            best_t = float(t)
    if best_spec < 0:
        pos_scores = np.sort(s[y == 1])
        return float(pos_scores[0]) if len(pos_scores) else 0.5
    return best_t


def eval_binary_metrics(y_cin2, y_cin3, score, threshold) -> dict:
    y2 = np.asarray(y_cin2, dtype=int)
    y3 = np.asarray(y_cin3, dtype=int)
    p = np.asarray(score, dtype=float)
    pred = p >= threshold
    tp = int(((pred) & (y3 == 1)).sum())
    fn = int(((~pred) & (y3 == 1)).sum())
    tn = int(((~pred) & (y3 == 0)).sum())
    fp = int(((pred) & (y3 == 0)).sum())
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    spec = tn / max(tn + fp, 1)
    sens = tp / max(tp + fn, 1)
    return {
        "CIN2+ AUC": safe_auc(y2, p),
        "CIN2+ AUC 95% CI": bootstrap_auc_ci(y2, p),
        "CIN3+ sensitivity": sens,
        "CIN3+ FN": fn,
        "specificity_at_threshold_CIN3": spec,
        "PPV_at_threshold_CIN3": ppv,
        "NPV_at_threshold_CIN3": npv,
        "screen_positive_rate": float(pred.mean()),
        "ECE_CIN2": ece_score(y2, p),
    }


def make_inner_split(y: np.ndarray, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    idx = np.arange(len(y))
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return idx, idx
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(idx, y))
    return train_idx, val_idx


def fit_lr_predict(x_source, y_source, x_target, seed: int = 2026) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(np.nan_to_num(x_source))
    xt = scaler.transform(np.nan_to_num(x_target))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)
    model.fit(xs, y_source)
    ps = model.predict_proba(xs)[:, 1]
    pt = model.predict_proba(xt)[:, 1]
    return ps, pt, model


def linear_probe_fold(src: pd.DataFrame, tgt: pd.DataFrame, groups: list[str], frozen: bool = False, seed: int = 2026) -> dict:
    xs_all = feature_matrix(src, groups, frozen=frozen)
    xt = feature_matrix(tgt, groups, frozen=frozen)
    y2 = src["cin2_label"].astype(int).to_numpy()
    y3 = src["cin3_label"].astype(int).to_numpy()
    tr, val = make_inner_split(y2, seed)
    if xs_all.shape[1] == 0 or len(np.unique(y2[tr])) < 2:
        target_score = np.full(len(tgt), float(y2.mean()))
        val_score = np.full(len(val), float(y2.mean()))
    else:
        _, val_score, _ = fit_lr_predict(xs_all[tr], y2[tr], xs_all[val], seed)
        _, target_score, _ = fit_lr_predict(xs_all, y2, xt, seed)
    threshold = select_threshold_for_cin3(y3[val], val_score)
    metrics = eval_binary_metrics(tgt["cin2_label"], tgt["cin3_label"], target_score, threshold)
    metrics["selected_threshold"] = threshold
    return metrics | {"target_score": target_score}


def mmd_rbf(x, y, max_n: int = 450, seed: int = 2026) -> float:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    y = np.nan_to_num(np.asarray(y, dtype=float))
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    z = np.vstack([x, y])
    if len(z) < 3:
        return float("nan")
    d = pdist(z, metric="sqeuclidean")
    sigma2 = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(2.0 * sigma2, 1e-12)
    kxx = np.exp(-gamma * cdist(x, x, "sqeuclidean")).mean()
    kyy = np.exp(-gamma * cdist(y, y, "sqeuclidean")).mean()
    kxy = np.exp(-gamma * cdist(x, y, "sqeuclidean")).mean()
    return float(kxx + kyy - 2 * kxy)


def centre_gap(centre_df: pd.DataFrame, auc_col: str = "CIN2+ AUC") -> float:
    vals = pd.to_numeric(centre_df[auc_col], errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.max() - vals.min())


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")


def fold_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.glob("fold_*") if p.is_dir()])


def short_float(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def status_json(path: Path, status: str, notes: str, **extra) -> None:
    write_json(path, {"status": status, "notes": notes, **extra})


def load_package_arrays(fold_dir: Path) -> dict:
    names = ["clinical", "colpo", "oct", "text", "combined"]
    out = {}
    for split in ["source", "target"]:
        for name in names:
            out[f"X_{split}_{name}"] = np.load(fold_dir / f"X_{split}_{name}.npy")
        out[f"y_{split}_cin2"] = np.load(fold_dir / f"y_{split}_cin2.npy")
        out[f"y_{split}_cin3"] = np.load(fold_dir / f"y_{split}_cin3.npy")
        out[f"ids_{split}"] = pd.read_csv(fold_dir / f"patient_ids_{split}.csv")
    return out


def _fit_prob(x_train, y_train, x_eval, seed: int = 2026) -> np.ndarray:
    if x_train.shape[1] == 0 or len(np.unique(y_train)) < 2:
        return np.full(x_eval.shape[0], float(np.mean(y_train)))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def run_fold_variant(fold_dir: Path, variant: str, mode: str, groups: list[str], seed: int = 2026) -> dict:
    arr = load_package_arrays(fold_dir)
    y2 = arr["y_source_cin2"].astype(int)
    y3 = arr["y_source_cin3"].astype(int)
    tr, val = make_inner_split(y2, seed)
    if mode == "reliability_gated":
        modality_groups = [g for g in ["clinical", "text", "colpo", "oct"] if g in groups]
        val_scores = []
        target_scores = []
        weights = []
        for g in modality_groups:
            xv = _fit_prob(arr[f"X_source_{g}"][tr], y2[tr], arr[f"X_source_{g}"][val], seed)
            xt = _fit_prob(arr[f"X_source_{g}"], y2, arr[f"X_target_{g}"], seed)
            auc = safe_auc(y2[val], xv)
            w = 0.5 if math.isnan(auc) else max(auc, 0.01)
            val_scores.append(xv)
            target_scores.append(xt)
            weights.append(w)
        if "combined" in groups:
            xv = _fit_prob(arr["X_source_combined"][tr], y2[tr], arr["X_source_combined"][val], seed)
            xt = _fit_prob(arr["X_source_combined"], y2, arr["X_target_combined"], seed)
            auc = safe_auc(y2[val], xv)
            val_scores.append(xv)
            target_scores.append(xt)
            weights.append(0.5 if math.isnan(auc) else max(auc, 0.01))
        w = np.asarray(weights, dtype=float)
        w = w / max(w.sum(), 1e-12)
        val_score = np.average(np.vstack(val_scores), axis=0, weights=w)
        target_score = np.average(np.vstack(target_scores), axis=0, weights=w)
        detail = {f"weight_{i}": float(v) for i, v in enumerate(w)}
    else:
        if len(groups) == 1:
            x_source = arr[f"X_source_{groups[0]}"]
            x_target = arr[f"X_target_{groups[0]}"]
        elif groups == ["combined"] or "combined" in groups:
            x_source = arr["X_source_combined"]
            x_target = arr["X_target_combined"]
        else:
            x_source = np.hstack([arr[f"X_source_{g}"] for g in groups if arr[f"X_source_{g}"].shape[1] > 0])
            x_target = np.hstack([arr[f"X_target_{g}"] for g in groups if arr[f"X_target_{g}"].shape[1] > 0])
        val_score = _fit_prob(x_source[tr], y2[tr], x_source[val], seed)
        target_score = _fit_prob(x_source, y2, x_target, seed)
        detail = {}
    threshold = select_threshold_for_cin3(y3[val], val_score)
    metrics = eval_binary_metrics(arr["y_target_cin2"], arr["y_target_cin3"], target_score, threshold)
    tgt_ids = arr["ids_target"].copy()
    pred = tgt_ids[["patient_id", "case_id", "centre", "centre_label"]].copy()
    pred["variant"] = variant
    pred["score_cin2plus"] = target_score
    pred["cin2_label"] = arr["y_target_cin2"].astype(int)
    pred["cin3_label"] = arr["y_target_cin3"].astype(int)
    pred["selected_threshold"] = threshold
    pred["pred_positive"] = target_score >= threshold
    val_pred = pd.DataFrame(
        {
            "fold": fold_dir.name,
            "variant": variant,
            "score_cin2plus": val_score,
            "cin2_label": y2[val],
            "cin3_label": y3[val],
            "selected_threshold": threshold,
        }
    )
    return {
        "metrics": metrics,
        "predictions": pred,
        "source_validation_predictions": val_pred,
        "threshold": threshold,
        "details": detail,
    }


def aggregate_predictions(pred: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    centre_rows = []
    for centre, g in pred.groupby("centre"):
        threshold = float(g["selected_threshold"].iloc[0]) if "selected_threshold" in g else select_threshold_for_cin3(g["cin3_label"], g["score_cin2plus"])
        m = eval_binary_metrics(g["cin2_label"], g["cin3_label"], g["score_cin2plus"], threshold)
        centre_rows.append({"variant": variant, "centre": centre, "centre_label": g["centre_label"].iloc[0], "n": len(g), **m})
    pooled_t = select_threshold_for_cin3(pred["cin3_label"], pred["score_cin2plus"])
    rows.append({"variant": variant, "centre": "Pooled", "centre_label": "Pooled", "n": len(pred), **eval_binary_metrics(pred["cin2_label"], pred["cin3_label"], pred["score_cin2plus"], pooled_t)})
    centre_df = pd.DataFrame(centre_rows)
    aggregate_df = pd.DataFrame(rows)
    aggregate_df["centre_gap"] = centre_gap(centre_df)
    return aggregate_df, centre_df
