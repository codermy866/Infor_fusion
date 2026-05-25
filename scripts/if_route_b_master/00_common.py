#!/usr/bin/env python3
"""Shared utilities for the IF Route B master package.

The package is deliberately conservative: optional inputs may be missing, and
claim wording must remain weaker than the available evidence.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional visual dependency
    sns = None

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PKG_OVERLAY = ROOT / ".routeb_python_pkgs"
if LOCAL_PKG_OVERLAY.exists() and str(LOCAL_PKG_OVERLAY) not in sys.path:
    # Keep the venv packages first; use this only for missing packages such as umap-learn.
    sys.path.append(str(LOCAL_PKG_OVERLAY))
OUT = ROOT / "outputs/publishable_v2/if_route_b_master"
SEED = 20260525
N_BOOTSTRAP = 200
SCI_PALETTE = ["#8b98b3", "#abb8cc", "#dbb98c", "#edd6b8", "#b57979", "#dea3a2", "#b3b0b0", "#d9d8d8"]
CENTRE_LABELS = {
    "十堰市人民医院": "Shiyan",
    "恩施州中心医院": "Enshi",
    "武大人民医院": "Wuda",
    "荆州市第一人民医院": "Jingzhou",
    "襄阳市中心医院": "Xiangyang",
}

SUBDIRS = [
    "audit",
    "tables",
    "figures",
    "statistics",
    "predictions",
    "paper_sections",
    "paper_prompts",
    "logs",
    "manifests",
]

PATHS = {
    "data_lock": "outputs/publishable_v2/data_lock/data_lock_n1897.csv",
    "protocol_csv": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/tables/Table_Inductive_vs_TTA_Protocol.csv",
    "protocol_md": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/audit/protocol_separation_report.md",
    "centre_tta": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/statistics/centre_level_tta_metrics.csv",
    "xiangyang": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/tables/Table3_Xiangyang_Rescue_Analysis.csv",
    "tta_metrics": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/statistics/tta_metrics.csv",
    "tta_ci": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/statistics/tta_bootstrap_ci.csv",
    "tta_tests": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/statistics/tta_paired_tests.csv",
    "source_preds": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/predictions/source_only_reference_predictions.csv",
    "tta_preds": "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/predictions/tta_candidate_predictions.csv",
    "step29_shift": "outputs/publishable_v2/step2_9_domain_generalisation_recovery/audit/centre_shift_metrics.csv",
    "step29_metrics": "outputs/publishable_v2/step2_9_domain_generalisation_recovery/statistics/dg_recovery_metrics.csv",
    "step29_table": "outputs/publishable_v2/step2_9_domain_generalisation_recovery/tables/Table3_Domain_Generalisation_Recovery.csv",
    "step28_table": "outputs/publishable_v2/step2_8_auc_recovery_information_fusion/tables/Table3_IFusion_Recovery.csv",
    "main_all_models": "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_all_models.csv",
    "main_hydra": "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_hydra_full.csv",
    "feature_npz": "outputs/publishable_v2/step2_main_loco/audit/step2_locked_feature_arrays.npz",
    "feature_table": "outputs/publishable_v2/step2_8_auc_recovery_information_fusion/manifests/multiscan_feature_table.csv",
}

FORBIDDEN_PHRASES = [
    "Full HyDRA-CoE",
    "End-to-end HyDRA-CoE",
    "Safe clinical deployment",
    "Clinically validated chain-of-evidence",
    "Stochastic uncertainty modeling",
    "State-of-the-art performance",
    "Outperforms all baselines",
    "Centre-invariant generalization",
    "Xiangyang was successfully rescued",
    "External validation on any centre not locked by protocol",
]


def p(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    try:
        return str(p(path).relative_to(ROOT))
    except Exception:
        return str(path)


def ensure_dirs() -> None:
    for d in SUBDIRS:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    if sns is not None:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette=SCI_PALETTE,
            font="Noto Sans CJK JP",
            rc={
                "axes.spines.right": False,
                "axes.spines.top": False,
                "axes.edgecolor": "#8a8a8a",
                "grid.color": "#e8e8e8",
                "grid.linewidth": 0.8,
                "axes.labelcolor": "#2f2f2f",
                "xtick.color": "#2f2f2f",
                "ytick.color": "#2f2f2f",
                "legend.frameon": False,
                "figure.dpi": 120,
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    for font_path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]:
        if Path(font_path).exists():
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def centre_label(name: str) -> str:
    return CENTRE_LABELS.get(str(name), str(name))


def centre_label_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(CENTRE_LABELS).fillna(s.astype(str))


def palette(n: int | None = None) -> list[str]:
    if n is None:
        return SCI_PALETTE
    reps = int(np.ceil(n / len(SCI_PALETTE)))
    return (SCI_PALETTE * reps)[:n]


def read_csv(path: str | Path) -> Optional[pd.DataFrame]:
    path = p(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def write_text(path: str | Path, text: str) -> None:
    path = p(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._\n"
    safe = df.copy().fillna("NA").astype(str)
    lines = [
        "| " + " | ".join(safe.columns) + " |",
        "| " + " | ".join(["---"] * len(safe.columns)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in safe.to_numpy()]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str) -> Path:
    ensure_dirs()
    base = OUT / "tables" / stem
    df.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    write_text(base.with_suffix(".md"), md_table(df))
    try:
        tex = df.to_latex(index=False, escape=True)
    except Exception as exc:
        tex = f"% LaTeX export failed: {exc}\n"
    write_text(base.with_suffix(".tex"), tex)
    return base.with_suffix(".csv")


def write_stat(df: pd.DataFrame, name: str) -> Path:
    ensure_dirs()
    out = OUT / "statistics" / name
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_pred(df: pd.DataFrame, name: str) -> Path:
    ensure_dirs()
    out = OUT / "predictions" / name
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_fig(fig, stem: str, caption: str = "") -> None:
    ensure_dirs()
    for ax in fig.axes:
        try:
            if sns is not None:
                sns.despine(ax=ax, trim=False)
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if caption:
        write_text(OUT / "figures" / f"{stem}_caption.txt", caption)


def load_data_lock() -> pd.DataFrame:
    df = read_csv(PATHS["data_lock"])
    if df is None:
        raise FileNotFoundError(PATHS["data_lock"])
    return df


def load_source_preds() -> pd.DataFrame:
    df = read_csv(PATHS["source_preds"])
    if df is None:
        raise FileNotFoundError(PATHS["source_preds"])
    return df


def load_tta_preds() -> pd.DataFrame:
    df = read_csv(PATHS["tta_preds"])
    if df is None:
        raise FileNotFoundError(PATHS["tta_preds"])
    return df


def load_tta_metrics() -> pd.DataFrame:
    df = read_csv(PATHS["tta_metrics"])
    if df is None:
        raise FileNotFoundError(PATHS["tta_metrics"])
    return df


def best_tta_method() -> str:
    metrics = load_tta_metrics()
    tta = metrics[metrics["Track"].astype(str).str.contains("transductive", case=False, na=False)].copy()
    if tta.empty:
        return ""
    tta = tta.sort_values(["AUC", "CIN3+ sensitivity"], ascending=False)
    return str(tta.iloc[0]["Method"])


def best_tta_predictions() -> pd.DataFrame:
    name = best_tta_method()
    tta = load_tta_preds()
    return tta[tta["method"].astype(str).eq(name)].copy()


def detect_columns(df: pd.DataFrame, patterns: Sequence[str]) -> List[str]:
    cols = []
    for col in df.columns:
        low = col.lower()
        if any(pat.lower() in low for pat in patterns):
            cols.append(col)
    return cols


def choose_column(df: pd.DataFrame, patterns: Sequence[str], prefer: Sequence[str] = ()) -> Optional[str]:
    candidates = detect_columns(df, patterns)
    if not candidates:
        return None
    for pref in prefer:
        for c in candidates:
            if c.lower() == pref.lower():
                return c
    candidates = sorted(candidates, key=lambda c: (len(c), c))
    return candidates[0]


def fmt(x, nd: int = 3) -> str:
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def auc_score(y_true: Sequence[float], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: Sequence[float], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.sum() == 0:
        return math.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    ranks = np.arange(1, len(y) + 1)
    precision = tp / ranks
    return float((precision * (y == 1)).sum() / max((y == 1).sum(), 1))


def metrics_at_pred(y_true: Sequence[float], score: Sequence[float], pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    mask = np.isfinite(s)
    y, s, pred = y[mask], s[mask], pred[mask]
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "auc": auc_score(y, s),
        "average_precision": average_precision(y, s),
        "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
        "specificity": tn / (tn + fp) if tn + fp else math.nan,
        "ppv": tp / (tp + fp) if tp + fp else math.nan,
        "npv": tn / (tn + fn) if tn + fn else math.nan,
        "fn": fn,
        "screen_positive_rate": float(pred.mean()) if len(pred) else math.nan,
    }


def threshold_for_sensitivity(y_true: Sequence[int], score: Sequence[float], target: float) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if (y == 1).sum() == 0:
        return math.nan
    candidates = np.unique(s[np.isfinite(s)])
    if len(candidates) == 0:
        return math.nan
    best = candidates.min()
    for t in np.sort(candidates)[::-1]:
        sens = ((s >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)
        if sens >= target:
            best = float(t)
            break
    return float(best)


def ece_score(y_true: Sequence[int], score: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.clip(np.asarray(score, dtype=float), 0, 1)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) == 0:
        return math.nan
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(ece)


def bootstrap_auc_ci(y: Sequence[int], score: Sequence[float], n_boot: int = N_BOOTSTRAP) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    point = auc_score(y, s)
    if not np.isfinite(point) or len(y) < 10:
        return point, math.nan, math.nan
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(auc_score(y[idx], s[idx]))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return point, math.nan, math.nan
    return point, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def centre_gap(df: pd.DataFrame, centre_col: str = "held_out_center") -> float:
    vals = []
    for _, g in df.groupby(centre_col):
        auc = auc_score(g["pathology_cin2plus"], g["prob_cin2plus"])
        if np.isfinite(auc):
            vals.append(auc)
    return float(max(vals) - min(vals)) if vals else math.nan


def net_benefit(y_true: Sequence[int], score: Sequence[float], threshold: float) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = s >= threshold
    n = len(y)
    tp = ((pred) & (y == 1)).sum()
    fp = ((pred) & (y == 0)).sum()
    if threshold >= 1:
        return math.nan
    return float(tp / n - fp / n * threshold / (1 - threshold))


def numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def clinical_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df.get("age", df.get("clin_age", np.nan)), errors="coerce")
    age_med = out["age"].median()
    out["age"] = out["age"].fillna(age_med if np.isfinite(age_med) else 45)
    hpv18 = df.get("hpv16_18_status", pd.Series("", index=df.index)).astype(str).str.lower()
    hpv = df.get("hpv_status_harmonized", pd.Series("", index=df.index)).astype(str).str.lower()
    tct = df.get("tct_status_harmonized", pd.Series("", index=df.index)).astype(str).str.lower()
    out["hpv16_18_positive"] = hpv18.str.contains("detect|positive|pos|16|18", regex=True).astype(float)
    out["hpv_missing"] = (hpv18.str.contains("unavailable|nan|none", regex=True) | hpv.eq("nan")).astype(float)
    out["other_hr_hpv"] = (hpv.str.contains("positive|detect", regex=True) & (out["hpv16_18_positive"] == 0)).astype(float)
    out["tct_abnormal"] = (~tct.isin(["nilm", "negative", "-", "nan", "none", ""])).astype(float)
    out["tct_high_grade"] = tct.str.contains("asc-h|hsil|agc|scc|high", regex=True).astype(float)
    return out


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std < 1e-8] = 1.0
    return (np.nan_to_num(x_train, nan=mean) - mean) / std, (np.nan_to_num(x_test, nan=mean) - mean) / std


def fit_logistic_gd(x: np.ndarray, y: np.ndarray, steps: int = 800, lr: float = 0.05, l2: float = 1e-3) -> np.ndarray:
    x = np.c_[np.ones(len(x)), x]
    y = y.astype(float)
    w = np.zeros(x.shape[1], dtype=float)
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    weights = np.where(y == 1, pos_weight, 1.0)
    for _ in range(steps):
        z = np.clip(x @ w, -30, 30)
        p_hat = 1 / (1 + np.exp(-z))
        grad = x.T @ ((p_hat - y) * weights) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logistic(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones(len(x)), x]
    return 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))


def rbf_mmd2(x: np.ndarray, y: np.ndarray, max_n: int = 300) -> float:
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return math.nan
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    xy = np.vstack([x, y])
    xy = np.nan_to_num(xy, nan=np.nanmean(xy, axis=0))
    xy = (xy - xy.mean(axis=0)) / np.where(xy.std(axis=0) < 1e-8, 1.0, xy.std(axis=0))
    x = xy[: len(x)]
    y = xy[len(x) :]
    sample = xy
    d = ((sample[: min(len(sample), 500), None, :] - sample[None, : min(len(sample), 500), :]) ** 2).sum(axis=2)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(2 * med, 1e-8)
    kxx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    kyy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    kxy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    col_mean = np.nanmean(x, axis=0)
    x = np.nan_to_num(x, nan=col_mean)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def load_locked_features() -> Tuple[pd.DataFrame, np.ndarray, str]:
    lock = load_data_lock()[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
    npz = p(PATHS["feature_npz"])
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        case_ids = z["case_id"].astype(str)
        feat = np.hstack([z["oct"], z["col"], z["clinical"]]).astype(float)
        fdf = pd.DataFrame({"case_id": case_ids})
        meta = fdf.merge(lock, on="case_id", how="left")
        return meta, feat, "frozen_step2_feature_arrays_oct_col_clinical"
    table = read_csv(PATHS["feature_table"])
    if table is not None:
        meta = table[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
        cols = [c for c in table.columns if c not in meta.columns and pd.api.types.is_numeric_dtype(table[c])]
        return meta, table[cols].to_numpy(float), "multiscan_numeric_feature_table"
    preds = read_csv(PATHS["main_hydra"])
    cols = [c for c in ["prob_cin2plus", "alpha_colposcopy", "alpha_oct", "alpha_semantic", "uncertainty_colposcopy", "uncertainty_oct", "uncertainty_semantic"] if c in preds.columns]
    meta = preds[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].drop_duplicates("case_id")
    x = preds.drop_duplicates("case_id")[cols].to_numpy(float)
    return meta, x, "score_feature_surrogate"


def existing_outputs_index() -> pd.DataFrame:
    rows = []
    if OUT.exists():
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                rows.append({"path": rel(path), "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def scan_for_forbidden_phrases() -> pd.DataFrame:
    rows = []
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".md", ".txt", ".tex"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for phrase in FORBIDDEN_PHRASES:
                if phrase in text:
                    rows.append({"file": rel(path), "phrase": phrase})
    return pd.DataFrame(rows, columns=["file", "phrase"])


def append_manifest(experiment_id: str, experiment_name: str, status: str, outputs: Iterable[str], notes: str = "") -> None:
    ensure_dirs()
    path = OUT / "manifests" / "final_execution_manifest.csv"
    row = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "status": status,
                "outputs": ";".join(outputs),
                "notes": notes,
            }
        ]
    )
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev["experiment_id"].astype(str) != experiment_id]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(path, index=False, encoding="utf-8-sig")


def run_module(module_file: str) -> int:
    import subprocess

    cmd = [sys.executable, str(Path(__file__).resolve().parent / module_file)]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    ensure_dirs()
    print(OUT)
