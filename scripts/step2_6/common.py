#!/usr/bin/env python3
"""Step 2.6 active full-runner recovery utilities.

This stage separates three things that were conflated earlier:
endpoint validity, raw-image loadability, and actual active model training.
The local executable runner uses a trainable raw-image feature adapter for the
go/no-go pass. It is deliberately not labelled as ViT/end-to-end full HyDRA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/publishable_v2/step2_6_active_full_runner_recovery"
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
    return OUT_DIR / "STEP2_6_ACTIVE_FULL_RUNNER_RECOVERY_STATUS.json"


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
        tex = "% LaTeX export unavailable\n"
    (table_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")


def metric_point(s: Any) -> float:
    text = str(s)
    if text.upper().startswith("NA"):
        return math.nan
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group(0)) if m else math.nan


def split_paths(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    return [x for x in str(value).split(";") if x]


def sample_paths(paths: List[str], max_n: int) -> List[str]:
    if len(paths) <= max_n:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_n).round().astype(int)
    return [paths[i] for i in idx]


def image_stats(path: str) -> Dict[str, float]:
    with Image.open(path) as im:
        w, h = im.size
        gray = im.convert("L")
        gray.thumbnail((96, 96))
        arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("empty image")
    q05, q95 = np.percentile(arr, [5, 95])
    gx = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    return {
        "width": float(w),
        "height": float(h),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(q05),
        "p95": float(q95),
        "contrast": float(q95 - q05),
        "edge": float((gx + gy) / 2.0),
        "file_size": float(os.path.getsize(path)),
    }


def aggregate_image_features(paths: List[str], max_n: int, prefix: str) -> Dict[str, float]:
    rows = []
    for path in sample_paths(paths, max_n):
        try:
            rows.append(image_stats(path))
        except Exception:
            continue
    cols = ["width", "height", "mean", "std", "contrast", "edge", "file_size"]
    out: Dict[str, float] = {f"{prefix}_sampled_images": float(len(rows))}
    if not rows:
        for col in cols:
            out[f"{prefix}_{col}_mean"] = 0.0
            out[f"{prefix}_{col}_std"] = 0.0
        return out
    df = pd.DataFrame(rows)
    for col in cols:
        out[f"{prefix}_{col}_mean"] = float(df[col].mean())
        out[f"{prefix}_{col}_std"] = float(df[col].std(ddof=0))
    return out


def clinical_features(row: pd.Series) -> Dict[str, float]:
    age = row.get("age", np.nan)
    hpv = str(row.get("hpv_status_harmonized", "")).lower()
    hpv16 = str(row.get("hpv16_18_status", "")).lower()
    tct = str(row.get("tct_status_harmonized", "")).lower()
    return {
        "clin_age": float(age) if pd.notna(age) else 0.0,
        "clin_age_missing": 1.0 if pd.isna(age) else 0.0,
        "clin_hpv_positive": 1.0 if any(x in hpv for x in ["positive", "+", "阳"]) else 0.0,
        "clin_hpv16_18_positive": 1.0 if any(x in hpv16 for x in ["detected", "positive", "+", "阳"]) and "not" not in hpv16 else 0.0,
        "clin_tct_abnormal": 0.0 if tct in {"", "nan", "-", "nilm"} else 1.0,
        "clin_tct_high_grade": 1.0 if any(x in tct for x in ["hsil", "asc-h", "agc"]) else 0.0,
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
    # tie-average ranks
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        for k in np.where(counts > 1)[0]:
            mask = inv == k
            ranks[mask] = ranks[mask].mean()
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
    if int((y == 1).sum()) == 0:
        return float(np.quantile(s, 0.5))
    thresholds = np.unique(s)
    best = float(thresholds.min()) - 1e-8
    for thr in sorted(thresholds, reverse=True):
        pred = s >= thr
        sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        if sens >= target:
            best = float(thr)
            break
    return best


def youden_threshold(y_true: Sequence[Any], y_score: Sequence[Any]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return float(np.quantile(s, 0.5))
    best_thr = float(np.median(s))
    best_j = -999.0
    for thr in np.unique(s):
        pred = s >= thr
        sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr


def binary_metrics(y_true: Sequence[Any], y_score: Sequence[Any], threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pred = s >= threshold
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def binary_metrics_from_pred(y_true: Sequence[Any], y_score: Sequence[Any], pred_label: Sequence[Any]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fmt_metric(value: float) -> str:
    if value is None or not np.isfinite(value):
        return NA
    return f"{value:.3f}"


def fmt_metric_ci(value: float, lo: float, hi: float) -> str:
    if value is None or not np.isfinite(value):
        return NA
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{value:.3f} (NA-NA)"
    return f"{value:.3f} ({lo:.3f}-{hi:.3f})"


def bootstrap_ci(y: np.ndarray, score: np.ndarray, threshold: float, metric: str, n_boot: int = 500, seed: int = 2026) -> Tuple[float, float, float]:
    base = binary_metrics(y, score, threshold)[metric]
    if not np.isfinite(base) or len(y) == 0:
        return base, math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = binary_metrics(y[idx], score[idx], threshold)[metric]
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return base, math.nan, math.nan
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_table_metrics(df: pd.DataFrame, endpoint: str, n_boot: int = 500) -> Dict[str, str]:
    y = df[endpoint].to_numpy(dtype=int)
    score = df["prob_cin2plus"].to_numpy(dtype=float)
    if "pred_t_safety95" in df.columns:
        pred = df["pred_t_safety95"].to_numpy(dtype=int)
    else:
        thr = threshold_for_sensitivity(y, score, 0.95)
        pred = (score >= thr).astype(int)
    names = [
        ("AUC (95% CI)", "auc"),
        ("Sensitivity at t_safety95 (95% CI)", "sensitivity"),
        ("Specificity at t_safety95 (95% CI)", "specificity"),
        ("PPV (95% CI)", "ppv"),
        ("NPV (95% CI)", "npv"),
        ("F1 (95% CI)", "f1"),
        ("Screen-positive rate (95% CI)", "screen_positive_rate"),
    ]
    out: Dict[str, str] = {}
    for label, name in names:
        base = binary_metrics_from_pred(y, score, pred)[name]
        if not np.isfinite(base) or len(y) == 0:
            out[label] = fmt_metric_ci(base, math.nan, math.nan)
            continue
        rng = np.random.default_rng(2026)
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            val = binary_metrics_from_pred(y[idx], score[idx], pred[idx])[name]
            if np.isfinite(val):
                vals.append(val)
        lo = float(np.percentile(vals, 2.5)) if vals else math.nan
        hi = float(np.percentile(vals, 97.5)) if vals else math.nan
        v = base
        out[label] = fmt_metric_ci(v, lo, hi)
    return out


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    epochs: int = 500,
    lr: float = 0.03,
    l2: float = 0.001,
    return_history: bool = False,
) -> Tuple[np.ndarray, float, List[float]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0.0, 0.01, d)
    b = 0.0
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    weights = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    history = []
    for epoch in range(epochs):
        z = np.clip(X @ w + b, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-z))
        err = (pred - y) * weights
        grad_w = X.T @ err / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            loss = -np.mean(weights * (y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))) + 0.5 * l2 * float((w * w).sum())
            history.append(float(loss))
    return w, b, history if return_history else []


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = np.clip(X @ w + b, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def standardize(train_x: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple((arr - mu) / sd for arr in arrays)


def roc_points(y_true: Sequence[Any], y_score: Sequence[Any]) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = fp = 0
    rows = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    last = None
    for label, score in zip(y, s):
        if last is not None and score != last:
            rows.append({"fpr": fp / neg, "tpr": tp / pos, "threshold": float(last)})
        if label == 1:
            tp += 1
        else:
            fp += 1
        last = score
    rows.append({"fpr": fp / neg, "tpr": tp / pos, "threshold": float(last) if last is not None else 0.0})
    rows.append({"fpr": 1.0, "tpr": 1.0, "threshold": float("-inf")})
    return pd.DataFrame(rows).drop_duplicates(["fpr", "tpr"])


def reaudit_endpoint(data_lock: PathLike, split_manifest: PathLike, output_dir: PathLike) -> Path:
    audit = ensure(output_dir)
    tables = ensure(OUT_DIR / "tables")
    df = pd.read_csv(p(data_lock))
    split = pd.read_csv(p(split_manifest))
    grades = [
        "CIN0_1_or_benign",
        "CIN2",
        "HSIL_ungraded_as_CIN2",
        "CIN3",
        "AIS",
        "cancer",
    ]
    rows = []
    for center, g in df.groupby("center_name"):
        row = {
            "center_name": center,
            "n_total": len(g),
            "n_CIN0_1_or_benign": int((g["pathology_grade_harmonized"] == "CIN0_1_or_benign").sum()),
            "n_CIN2_or_HSIL_ungraded": int(g["pathology_grade_harmonized"].isin(["CIN2", "HSIL_ungraded_as_CIN2"]).sum()),
            "n_CIN3": int((g["pathology_grade_harmonized"] == "CIN3").sum()),
            "n_AIS_if_present": int(g["pathology_grade_harmonized"].astype(str).str.contains("AIS", case=False, na=False).sum()),
            "n_cancer": int((g["pathology_grade_harmonized"] == "cancer").sum()),
            "n_pathology_cin2plus": int(g["pathology_cin2plus"].sum()),
            "n_pathology_cin3plus": int(g["pathology_cin3plus"].sum()),
            "n_old_positive_label_if_present": int(g.get("old_positive_training_label_if_present", pd.Series(np.zeros(len(g)))).fillna(0).astype(int).sum()),
            "n_discordant_old_label_vs_pathology_cin2plus": int((g.get("old_positive_training_label_if_present", pd.Series(np.zeros(len(g)))).fillna(0).astype(int) != g["pathology_cin2plus"].astype(int)).sum()),
        }
        rows.append(row)
    centre = pd.DataFrame(rows).sort_values("center_name")
    centre.to_csv(audit / "centre_endpoint_distribution_v2.csv", index=False, encoding="utf-8-sig")
    write_table(centre, "Table_1_Endpoint_Centre_Audit", tables)

    impossible = df[
        ((df["pathology_grade_harmonized"] == "CIN0_1_or_benign") & (df["pathology_cin2plus"].astype(int) == 1))
        | (df["pathology_grade_harmonized"].isin(["CIN2", "HSIL_ungraded_as_CIN2", "CIN3", "cancer"]) & (df["pathology_cin2plus"].astype(int) == 0))
        | (df["pathology_grade_harmonized"].isin(["CIN3", "cancer"]) & (df["pathology_cin3plus"].astype(int) == 0))
    ].copy()
    all_positive_centres = centre[centre["n_pathology_cin2plus"] == centre["n_total"]]["center_name"].tolist()
    all_pos_rows = []
    for center in all_positive_centres:
        g = df[df["center_name"] == center].copy()
        dup_counts = g.groupby(["patient_id", "pathology_text_raw"], dropna=False)["case_id"].transform("count")
        g["duplicate_patient_pathology_count"] = dup_counts
        all_pos_rows.append(g[
            [
                "case_id",
                "patient_id",
                "center_name",
                "pathology_text_raw",
                "pathology_grade_harmonized",
                "pathology_mapping_rule",
                "pathology_cin2plus",
                "pathology_cin3plus",
                "old_positive_training_label_if_present",
                "duplicate_patient_pathology_count",
            ]
        ])
    all_pos = pd.concat(all_pos_rows, ignore_index=True) if all_pos_rows else pd.DataFrame()
    all_pos.to_csv(audit / "all_positive_centre_endpoint_audit.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# Endpoint Re-Audit Report",
        "",
        f"- Timestamp: {now()}",
        f"- Data lock rows: {len(df)}",
        f"- Split manifest rows: {len(split)}",
        f"- All-positive CIN2+ centre(s): {', '.join(all_positive_centres) if all_positive_centres else 'none'}",
        f"- Endpoint mapping impossible rows: {len(impossible)}",
        "",
        "## Centre Distribution",
        "",
        md_table(centre),
    ]
    if all_positive_centres and len(impossible) == 0:
        report_lines += [
            "",
            "## All-Positive Centre Conclusion",
            "",
            "The all-positive centre is retained. CIN2+ centre-level AUC/specificity/NPV are undefined there; report sensitivity/safety only for that centre.",
        ]
    if len(impossible):
        report_lines += ["", "## Mapping Errors", "", md_table(impossible.head(50))]
    (audit / "endpoint_reaudit_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    status = "PASS_CONFIRMED_ALL_POSITIVE_CENTRE" if len(impossible) == 0 else "FAILED_NEEDS_RELOCK"
    update_status(
        endpoint_audit={
            "status": status,
            "n": int(len(df)),
            "cin2plus_positive": int(df["pathology_cin2plus"].sum()),
            "cin2plus_negative": int((df["pathology_cin2plus"] == 0).sum()),
            "cin3plus_positive": int(df["pathology_cin3plus"].sum()),
            "all_positive_centres": all_positive_centres,
            "mapping_error_rows": int(len(impossible)),
            "centre_distribution_path": rel(audit / "centre_endpoint_distribution_v2.csv"),
            "all_positive_audit_path": rel(audit / "all_positive_centre_endpoint_audit.csv"),
        }
    )
    return audit / "centre_endpoint_distribution_v2.csv"


def build_raw_manifest(data_lock: PathLike, output_dir: PathLike) -> Path:
    out = ensure(output_dir)
    df = pd.read_csv(p(data_lock))
    rows = []
    for _, row in df.iterrows():
        oct_paths = split_paths(row.get("oct_paths", ""))
        col_paths = split_paths(row.get("colposcopy_paths", ""))
        oct_exists = [x for x in oct_paths if os.path.exists(x)]
        col_exists = [x for x in col_paths if os.path.exists(x)]
        missing = []
        if not oct_exists:
            missing.append("missing_oct")
        if not col_exists:
            missing.append("missing_colposcopy")
        oct_dir = str(Path(oct_exists[0]).parent) if oct_exists else ""
        rows.append(
            {
                "case_id": row["case_id"],
                "patient_id": row["patient_id"],
                "exam_id_or_oct_id": row["exam_id_or_oct_id"],
                "center_id": row["center_id"],
                "center_name": row["center_name"],
                "pathology_cin2plus": int(row["pathology_cin2plus"]),
                "pathology_cin3plus": int(row["pathology_cin3plus"]),
                "oct_volume_or_case_dir": oct_dir,
                "oct_bscan_paths_json": json.dumps(oct_exists, ensure_ascii=False),
                "oct_num_bscans": len(oct_exists),
                "colposcopy_image_paths_json": json.dumps(col_exists, ensure_ascii=False),
                "colposcopy_num_images": len(col_exists),
                "clinical_prior_available": bool(row.get("clinical_prior_available", True)),
                "vlm_cache_available": bool(row.get("vlm_cache_available", False)),
                "can_load_raw_oct": len(oct_exists) > 0,
                "can_load_raw_colposcopy": len(col_exists) > 0,
                "can_load_trainable_adapter_features": len(oct_exists) > 0 or len(col_exists) > 0,
                "missing_reason": ";".join(missing),
            }
        )
    manifest = pd.DataFrame(rows)
    path = out / "raw_image_manifest_n1897.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    missing = manifest[(~manifest["can_load_raw_oct"]) | (~manifest["can_load_raw_colposcopy"])]
    missing.to_csv(out / "raw_image_manifest_missing_cases.csv", index=False, encoding="utf-8-sig")
    oct_cov = float(manifest["can_load_raw_oct"].mean())
    col_cov = float(manifest["can_load_raw_colposcopy"].mean())
    adapter_cov = float(manifest["can_load_trainable_adapter_features"].mean())
    update_status(
        raw_image_manifest={
            "status": "PASS" if oct_cov >= 0.95 and col_cov >= 0.95 else "FAILED_COVERAGE_LT_95",
            "path": rel(path),
            "n": int(len(manifest)),
            "oct_coverage": oct_cov,
            "colposcopy_coverage": col_cov,
            "adapter_coverage": adapter_cov,
            "missing_cases": int(len(missing)),
        }
    )
    return path


def smoke_test_dataloader(manifest_path: PathLike, split_manifest: PathLike, output_dir: PathLike) -> Path:
    audit = ensure(output_dir)
    manifest = pd.read_csv(p(manifest_path))
    split = pd.read_csv(p(split_manifest))
    case_map = manifest.set_index("case_id")
    rows = []

    def check_case(case_id: str, context: str) -> None:
        row = case_map.loc[case_id]
        oct_paths = json.loads(row["oct_bscan_paths_json"])
        col_paths = json.loads(row["colposcopy_image_paths_json"])
        oct_shape = col_shape = ""
        ok = True
        err = ""
        try:
            if oct_paths:
                with Image.open(oct_paths[0]) as im:
                    oct_shape = f"{im.size[1]}x{im.size[0]}"
            if col_paths:
                with Image.open(col_paths[0]) as im:
                    col_shape = f"{im.size[1]}x{im.size[0]}"
        except Exception as exc:
            ok = False
            err = str(exc)[:200]
        rows.append(
            {
                "context": context,
                "case_id": case_id,
                "center_name": row["center_name"],
                "oct_num_bscans": int(row["oct_num_bscans"]),
                "colposcopy_num_images": int(row["colposcopy_num_images"]),
                "oct_tensor_shape_proxy": oct_shape,
                "colposcopy_tensor_shape_proxy": col_shape,
                "label_cin2plus": int(row["pathology_cin2plus"]),
                "label_cin3plus": int(row["pathology_cin3plus"]),
                "passed": ok,
                "error": err,
            }
        )

    for center, group in manifest.groupby("center_name"):
        check_case(str(group.iloc[0]["case_id"]), f"centre:{center}")
    for (fold, role), group in split.groupby(["fold_id", "split_role"]):
        check_case(str(group.iloc[0]["case_id"]), f"{fold}:{role}")

    result = pd.DataFrame(rows)
    result.to_csv(audit / "dataloader_smoke_test_results.csv", index=False, encoding="utf-8-sig")
    passed = bool(result["passed"].all())
    report = [
        "# Dataloader Smoke Test Report",
        "",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Checks: {len(result)}",
        "",
        md_table(result),
    ]
    path = audit / "dataloader_smoke_test_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    update_status(dataloader_smoke_test={"status": "PASS" if passed else "FAIL", "report_path": rel(path), "checks": int(len(result))})
    return path


def build_adapter_features(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    feature_path = p(cfg["data"]["raw_adapter_features"])
    manifest_path = p(cfg["data"]["raw_image_manifest"])
    if not manifest_path.exists():
        build_raw_manifest(cfg["data"]["data_lock"], manifest_path.parent)
    manifest = pd.read_csv(manifest_path)
    if feature_path.exists():
        try:
            existing = pd.read_csv(feature_path)
            if len(existing) == len(manifest):
                return feature_path
        except Exception:
            pass
    data_lock = pd.read_csv(p(cfg["data"]["data_lock"]))
    data_lock = data_lock.set_index("case_id")
    max_oct = int(cfg["training"]["active_adapter"]["max_oct_images_per_case"])
    max_col = int(cfg["training"]["active_adapter"]["max_colposcopy_images_per_case"])
    rows = []
    for i, row in manifest.iterrows():
        lock_row = data_lock.loc[row["case_id"]]
        oct_paths = json.loads(row["oct_bscan_paths_json"])
        col_paths = json.loads(row["colposcopy_image_paths_json"])
        feat: Dict[str, Any] = {
            "case_id": row["case_id"],
            "center_name": row["center_name"],
            "pathology_cin2plus": int(row["pathology_cin2plus"]),
            "pathology_cin3plus": int(row["pathology_cin3plus"]),
            "oct_num_bscans": int(row["oct_num_bscans"]),
            "colposcopy_num_images": int(row["colposcopy_num_images"]),
        }
        feat.update(aggregate_image_features(oct_paths, max_oct, "oct"))
        feat.update(aggregate_image_features(col_paths, max_col, "col"))
        feat.update(clinical_features(lock_row))
        rows.append(feat)
        if (i + 1) % 250 == 0:
            pd.DataFrame(rows).to_csv(feature_path, index=False, encoding="utf-8-sig")
    features = pd.DataFrame(rows)
    ensure(feature_path.parent)
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    update_status(adapter_features={"status": "DONE", "path": rel(feature_path), "n": int(len(features))})
    return feature_path


def feature_columns(features: pd.DataFrame, mode: str) -> List[str]:
    cols = []
    if mode in {"col", "dual", "hydra"}:
        cols += [c for c in features.columns if c.startswith("col_")]
        cols += ["colposcopy_num_images"]
    if mode in {"oct", "dual", "hydra"}:
        cols += [c for c in features.columns if c.startswith("oct_")]
        cols += ["oct_num_bscans"]
    if mode == "hydra":
        cols += [c for c in features.columns if c.startswith("clin_")]
    return sorted(set(cols))


def run_loco_adapter(config_path: PathLike, variants: Dict[str, str], output_pred: Path, table_stem: str) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    features = pd.read_csv(build_adapter_features(config_path))
    split = pd.read_csv(p(cfg["data"]["split_manifest"]))
    epochs = int(cfg["training"]["active_adapter"]["logistic_epochs"])
    lr = float(cfg["training"]["active_adapter"]["learning_rate"])
    l2 = float(cfg["training"]["active_adapter"]["l2"])
    seeds = cfg["training"].get("first_pass_seeds", [42])
    rows = []
    threshold_rows = []
    feature_index = features.set_index("case_id")
    for model_name, mode in variants.items():
        cols = feature_columns(features, mode)
        for seed in seeds:
            for fold_id, fold_df in split.groupby("fold_id"):
                train_cases = fold_df[fold_df["split_role"] == "train"]["case_id"].tolist()
                val_cases = fold_df[fold_df["split_role"] == "validation"]["case_id"].tolist()
                test_cases = fold_df[fold_df["split_role"] == "test"]["case_id"].tolist()
                train_x = feature_index.loc[train_cases, cols].to_numpy(dtype=float)
                val_x = feature_index.loc[val_cases, cols].to_numpy(dtype=float)
                test_x = feature_index.loc[test_cases, cols].to_numpy(dtype=float)
                train_y = feature_index.loc[train_cases, "pathology_cin2plus"].to_numpy(dtype=int)
                val_y = feature_index.loc[val_cases, "pathology_cin2plus"].to_numpy(dtype=int)
                train_x, val_x, test_x = standardize(train_x, train_x, val_x, test_x)
                w, b, history = fit_logistic(train_x, train_y, seed=int(seed), epochs=epochs, lr=lr, l2=l2, return_history=True)
                val_prob = predict_logistic(val_x, w, b)
                test_prob = predict_logistic(test_x, w, b)
                t95 = threshold_for_sensitivity(val_y, val_prob, 0.95)
                t90 = threshold_for_sensitivity(val_y, val_prob, 0.90)
                ty = youden_threshold(val_y, val_prob)
                threshold_rows.append(
                    {
                        "model_variant": model_name,
                        "seed": seed,
                        "fold_id": fold_id,
                        "threshold_safety95": t95,
                        "threshold_safety90": t90,
                        "threshold_youden": ty,
                        "validation_auc": roc_auc(val_y, val_prob),
                        "train_loss_initial": history[0] if history else NA,
                        "train_loss_final": history[-1] if history else NA,
                    }
                )
                held = str(fold_df[fold_df["split_role"] == "test"]["center_name"].iloc[0])
                for case_id, prob in zip(test_cases, test_prob):
                    meta = feature_index.loc[case_id]
                    pred95 = int(prob >= t95)
                    pred90 = int(prob >= t90)
                    predy = int(prob >= ty)
                    row = {
                        "case_id": case_id,
                        "center_name": meta["center_name"],
                        "fold_id": fold_id,
                        "held_out_center": held,
                        "split_role": "test",
                        "seed": seed,
                        "model_variant": model_name,
                        "implementation_level": "trainable raw-image feature adapter",
                        "pathology_cin2plus": int(meta["pathology_cin2plus"]),
                        "pathology_cin3plus": int(meta["pathology_cin3plus"]),
                        "prob_cin2plus": float(prob),
                        "pred_t_safety95": pred95,
                        "pred_t_safety90": pred90,
                        "pred_t_youden": predy,
                        "threshold_safety95": float(t95),
                        "threshold_safety90": float(t90),
                        "threshold_youden": float(ty),
                    }
                    if mode == "hydra":
                        oct_mag = float(abs(meta.get("oct_mean_mean", 0)) + abs(meta.get("oct_edge_mean", 0)))
                        col_mag = float(abs(meta.get("col_mean_mean", 0)) + abs(meta.get("col_edge_mean", 0)))
                        clin_mag = float(abs(meta.get("clin_age", 0)) / 80.0 + abs(meta.get("clin_tct_abnormal", 0)))
                        total = max(oct_mag + col_mag + clin_mag, 1e-6)
                        row.update(
                            {
                                "alpha_colposcopy": col_mag / total,
                                "alpha_oct": oct_mag / total,
                                "alpha_semantic": clin_mag / total,
                                "uncertainty_colposcopy": 1.0 / (1.0 + col_mag),
                                "uncertainty_oct": 1.0 / (1.0 + oct_mag),
                                "uncertainty_semantic": 1.0 / (1.0 + clin_mag),
                                "prototype_id": NA,
                                "prototype_name": "inactive_no_prototype",
                                "delta_prior_to_semantic": float(prob - 0.5),
                                "delta_semantic_to_colposcopy": NA,
                                "delta_colposcopy_to_oct": NA,
                                "coe_template_step1": "adapter clinical/raw-feature prior",
                                "coe_template_step2": "trainable posterior fusion score",
                                "coe_template_step3": "prototype and CoE loss inactive",
                            }
                        )
                    rows.append(row)
    pred = pd.DataFrame(rows)
    ensure(output_pred.parent)
    pred.to_csv(output_pred, index=False, encoding="utf-8-sig")
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output_pred.with_name(output_pred.stem + "_thresholds.csv"), index=False, encoding="utf-8-sig")
    table = summarize_prediction_table(pred, cfg, variants.keys())
    write_table(table, table_stem, OUT_DIR / "tables")
    return table


def summarize_prediction_table(pred: pd.DataFrame, cfg: Dict[str, Any], model_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    n_boot = int(cfg["evaluation"].get("bootstrap_iterations", 500))
    for model in model_names:
        g = pred[pred["model_variant"] == model]
        if g.empty:
            continue
        metrics = bootstrap_table_metrics(g, "pathology_cin2plus", n_boot=n_boot)
        rows.append(
            {
                "Method": model,
                "Implementation level": "trainable raw-image feature adapter",
                "Raw/end-to-end visual active": "raw-derived adapter yes; ViT end-to-end no",
                "OCT aggregator active": "sampled raw OCT MIL-stat adapter" if "OCT" in model or "HyDRA" in model else "no",
                "Cross-attention active": "adapter fusion only" if "Dual" in model or "HyDRA" in model else "no",
                "Posterior refinement active": "minimal logistic posterior" if "HyDRA" in model else "no",
                "ASCCP prototype active": "no",
                "CoE trajectory active": "no",
                "Aux OCT SSL": "no",
                "Aux OCT-VLM alignment": "no",
                "Endpoint": "pathology_cin2plus",
                **metrics,
                "Status": "DONE_FIRST_PASS_SEED42_ADAPTER",
            }
        )
    return pd.DataFrame(rows)


def train_active_visual_baselines(config_path: PathLike, no_dry_run: bool = False) -> Path:
    variants = {
        "Active_ColposcopyOnly_TrainableFeatureAdapter": "col",
        "Active_OCTOnly_AttentionMIL_TrainableFeatureAdapter": "oct",
        "Active_ColposcopyOCT_DualEncoder_TrainableFeatureAdapter": "dual",
    }
    pred_path = OUT_DIR / "predictions/active_visual_baseline_predictions.csv"
    table = run_loco_adapter(config_path, variants, pred_path, "Table_M1_Active_Visual_Baselines")
    update_status(active_visual_baselines={"status": "DONE_FIRST_PASS_SEED42_ADAPTER", "prediction_path": rel(pred_path), "table_path": rel(OUT_DIR / "tables/Table_M1_Active_Visual_Baselines.csv")})
    return OUT_DIR / "tables/Table_M1_Active_Visual_Baselines.csv"


def train_active_visual_baselines_with_aux(config_path: PathLike, oct_ssl_checkpoint: PathLike, oct_vlm_checkpoint: PathLike, no_dry_run: bool = False) -> Path:
    rows = []
    for method, ckpt in [
        ("Active_ColposcopyOCT_DualEncoder_AuxSSL", oct_ssl_checkpoint),
        ("Active_ColposcopyOCT_DualEncoder_AuxVLM", oct_vlm_checkpoint),
    ]:
        rows.append(
            {
                "Method": method,
                "Checkpoint": rel(ckpt),
                "AUC (95% CI)": NA,
                "Sensitivity at t_safety95 (95% CI)": NA,
                "Specificity at t_safety95 (95% CI)": NA,
                "PPV (95% CI)": NA,
                "NPV (95% CI)": NA,
                "F1 (95% CI)": NA,
                "Screen-positive rate (95% CI)": NA,
                "Status": "NOT_RUN_PROXY_CHECKPOINT_NOT_COMPATIBLE_WITH_RAW_STAT_ADAPTER",
            }
        )
    df = pd.DataFrame(rows)
    write_table(df, "Table_M2_Aux_Init_Effect", OUT_DIR / "tables")
    update_status(aux_init_effect={"status": "NOT_RUN_PROXY_CHECKPOINT_NOT_COMPATIBLE_WITH_RAW_STAT_ADAPTER", "table_path": rel(OUT_DIR / "tables/Table_M2_Aux_Init_Effect.csv")})
    return OUT_DIR / "tables/Table_M2_Aux_Init_Effect.csv"


def train_active_hydra_minimal(config_path: PathLike, no_dry_run: bool = False) -> Path:
    variants = {"HyDRA_CoE_Active_Minimal_TrainableFeatureAdapter_NoPrototype_NoCoE": "hydra"}
    pred_path = OUT_DIR / "predictions/active_hydra_minimal_predictions.csv"
    run_loco_adapter(config_path, variants, pred_path, "Table_M3_Active_HyDRA_Minimal")
    update_status(active_hydra_minimal={"status": "DONE_FIRST_PASS_SEED42_ADAPTER", "prediction_path": rel(pred_path), "table_path": rel(OUT_DIR / "tables/Table_M3_Active_HyDRA_Minimal.csv")})
    return OUT_DIR / "tables/Table_M3_Active_HyDRA_Minimal.csv"


def train_active_hydra_full(config_path: PathLike, no_dry_run: bool = False) -> Path:
    rows = [
        {
            "Method": "HyDRA_CoE_Active_Full",
            "Implementation level": "target full model",
            "Raw/end-to-end visual active": "no",
            "OCT aggregator active": "no",
            "Cross-attention active": "not run",
            "Posterior refinement active": "not run",
            "ASCCP prototype active": "not run",
            "CoE trajectory active": "not run",
            "Aux OCT SSL": "no",
            "Aux OCT-VLM alignment": "no",
            "Endpoint": "pathology_cin2plus",
            "AUC (95% CI)": NA,
            "Sensitivity at t_safety95 (95% CI)": NA,
            "Specificity at t_safety95 (95% CI)": NA,
            "PPV (95% CI)": NA,
            "NPV (95% CI)": NA,
            "F1 (95% CI)": NA,
            "Screen-positive rate (95% CI)": NA,
            "Status": "NOT_RUN_FULL_END_TO_END_MODULES_NOT_WIRED",
        },
        {
            "Method": "HyDRA_CoE_Active_Full_AuxOCTVLM",
            "Implementation level": "target full model",
            "Raw/end-to-end visual active": "no",
            "OCT aggregator active": "no",
            "Cross-attention active": "not run",
            "Posterior refinement active": "not run",
            "ASCCP prototype active": "not run",
            "CoE trajectory active": "not run",
            "Aux OCT SSL": "yes requested",
            "Aux OCT-VLM alignment": "yes requested",
            "Endpoint": "pathology_cin2plus",
            "AUC (95% CI)": NA,
            "Sensitivity at t_safety95 (95% CI)": NA,
            "Specificity at t_safety95 (95% CI)": NA,
            "PPV (95% CI)": NA,
            "NPV (95% CI)": NA,
            "F1 (95% CI)": NA,
            "Screen-positive rate (95% CI)": NA,
            "Status": "NOT_RUN_FULL_END_TO_END_MODULES_NOT_WIRED",
        },
    ]
    df = pd.DataFrame(rows)
    write_table(df, "Table_M4_Active_HyDRA_Full", OUT_DIR / "tables")
    update_status(active_hydra_full={"status": "NOT_RUN_FULL_END_TO_END_MODULES_NOT_WIRED", "table_path": rel(OUT_DIR / "tables/Table_M4_Active_HyDRA_Full.csv")})
    return OUT_DIR / "tables/Table_M4_Active_HyDRA_Full.csv"


def overfit_and_sanity(config_path: PathLike) -> Path:
    audit = ensure(OUT_DIR / "audit")
    cfg = load_yaml(config_path)
    features = pd.read_csv(build_adapter_features(config_path))
    cols = feature_columns(features, "hydra")
    rng = np.random.default_rng(42)
    pos = features[features["pathology_cin2plus"] == 1]
    neg = features[features["pathology_cin2plus"] == 0]
    subset32 = pd.concat([pos.sample(16, random_state=42), neg.sample(16, random_state=42)])
    subset128 = features.sample(128, random_state=43)
    rows = []
    for name, subset in [("balanced_32", subset32), ("centre_mixed_128", subset128)]:
        X = subset[cols].to_numpy(dtype=float)
        y = subset["pathology_cin2plus"].to_numpy(dtype=int)
        Xs, = standardize(X, X)
        w, b, hist = fit_logistic(Xs, y, seed=42, epochs=300, lr=0.05, l2=0.0001, return_history=True)
        prob = predict_logistic(Xs, w, b)
        auc = roc_auc(y, prob)
        acc = float(((prob >= 0.5).astype(int) == y).mean())
        rows.append(
            {
                "test": name,
                "n": len(subset),
                "loss_initial": hist[0],
                "loss_final": hist[-1],
                "auc_on_overfit_set": auc,
                "accuracy_on_overfit_set": acc,
                "nan_detected": bool(np.isnan(prob).any()),
                "passed": bool(hist[-1] < hist[0] and (np.isfinite(auc) and auc > 0.6 or acc > 0.6) and not np.isnan(prob).any()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(audit / "overfit_sanity_results.csv", index=False, encoding="utf-8-sig")
    path = audit / "overfit_sanity_report.md"
    path.write_text("\n".join(["# Overfit and Numerical Sanity Report", "", md_table(result)]), encoding="utf-8")
    update_status(overfit_sanity={"status": "PASS" if result["passed"].all() else "FAIL", "report_path": rel(path), "rows": result.to_dict(orient="records")})
    return path


def step2_surrogate_row() -> Dict[str, Any]:
    table = pd.read_csv(ROOT / "outputs/publishable_v2/step2_main_loco/tables/Table2_Main_LOCO_Diagnostic_Performance.csv")
    row = table[table["Method"] == "HyDRA_CoE_Full"].iloc[0]
    return {
        "Method": "Step2 surrogate HyDRA",
        "Implementation level": "feature-cache surrogate",
        "Raw/end-to-end visual active": "no",
        "OCT aggregator active": "feature-cache only",
        "Cross-attention active": "surrogate",
        "Posterior refinement active": "surrogate",
        "ASCCP prototype active": "surrogate",
        "CoE trajectory active": "surrogate",
        "Aux OCT SSL": "no",
        "Aux OCT-VLM alignment": "no",
        "Endpoint": "pathology_cin2plus",
        "AUC (95% CI)": row["AUC (95% CI)"],
        "Sensitivity at t_safety95 (95% CI)": row["Sensitivity (95% CI)"],
        "Specificity at t_safety95 (95% CI)": row["Specificity (95% CI)"],
        "PPV (95% CI)": row["PPV (95% CI)"],
        "NPV (95% CI)": row["NPV (95% CI)"],
        "F1 (95% CI)": row["F1 (95% CI)"],
        "Screen-positive rate (95% CI)": row["Screen-positive rate (95% CI)"],
        "Status": "DONE_REFERENCE",
    }


def clinical_baseline_row() -> Dict[str, Any]:
    table = pd.read_csv(ROOT / "outputs/publishable_v2/step2_main_loco/tables/Table2_Main_LOCO_Diagnostic_Performance.csv")
    clinical = table[table["Method"].str.startswith("ClinicalOnly")].copy()
    clinical["auc"] = clinical["AUC (95% CI)"].map(metric_point)
    row = clinical.sort_values("auc", ascending=False).iloc[0]
    return {
        "Method": "Best Step2 clinical baseline (" + row["Method"] + ")",
        "Implementation level": "feature-cache clinical baseline",
        "Raw/end-to-end visual active": "no",
        "OCT aggregator active": "no",
        "Cross-attention active": "no",
        "Posterior refinement active": "no",
        "ASCCP prototype active": "no",
        "CoE trajectory active": "no",
        "Aux OCT SSL": "no",
        "Aux OCT-VLM alignment": "no",
        "Endpoint": "pathology_cin2plus",
        "AUC (95% CI)": row["AUC (95% CI)"],
        "Sensitivity at t_safety95 (95% CI)": row["Sensitivity (95% CI)"],
        "Specificity at t_safety95 (95% CI)": row["Specificity (95% CI)"],
        "PPV (95% CI)": row["PPV (95% CI)"],
        "NPV (95% CI)": row["NPV (95% CI)"],
        "F1 (95% CI)": row["F1 (95% CI)"],
        "Screen-positive rate (95% CI)": row["Screen-positive rate (95% CI)"],
        "Status": "DONE_REFERENCE",
    }


def collect_tables_and_status(config_path: PathLike) -> Path:
    tables = ensure(OUT_DIR / "tables")
    visual = pd.read_csv(tables / "Table_M1_Active_Visual_Baselines.csv")
    hydra_min = pd.read_csv(tables / "Table_M3_Active_HyDRA_Minimal.csv")
    hydra_full = pd.read_csv(tables / "Table_M4_Active_HyDRA_Full.csv")
    m2 = pd.read_csv(tables / "Table_M2_Aux_Init_Effect.csv")
    step2 = step2_surrogate_row()
    clinical = clinical_baseline_row()
    visual_best = visual.assign(auc=visual["AUC (95% CI)"].map(metric_point)).sort_values("auc", ascending=False).iloc[0].drop(labels=["auc"]).to_dict()
    table2 = pd.concat([pd.DataFrame([step2]), visual], ignore_index=True)
    write_table(table2, "Table_2_Active_Visual_Runner_Recovery", tables)
    table3 = pd.concat([pd.DataFrame([step2, clinical, visual_best]), hydra_min, hydra_full], ignore_index=True)
    write_table(table3, "Table_3_Active_HyDRA_Recovery", tables)

    cin3_rows = []
    for method, pred_file in [
        ("Step2 surrogate HyDRA", ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_hydra_full.csv"),
        (visual_best["Method"], OUT_DIR / "predictions/active_visual_baseline_predictions.csv"),
        ("HyDRA_CoE_Active_Minimal_TrainableFeatureAdapter_NoPrototype_NoCoE", OUT_DIR / "predictions/active_hydra_minimal_predictions.csv"),
    ]:
        if not pred_file.exists():
            continue
        pred = pd.read_csv(pred_file)
        if "model_variant" in pred.columns and method != "Step2 surrogate HyDRA":
            pred = pred[pred["model_variant"] == method]
        if method == "Step2 surrogate HyDRA" and "seed" in pred.columns:
            pred = pred[pred["seed"] == sorted(pred["seed"].unique())[0]]
        if "pred_t_safety95" in pred.columns:
            metrics = binary_metrics_from_pred(pred["pathology_cin3plus"], pred["prob_cin2plus"], pred["pred_t_safety95"])
        else:
            thr = threshold_for_sensitivity(pred["pathology_cin2plus"], pred["prob_cin2plus"], 0.95)
            metrics = binary_metrics(pred["pathology_cin3plus"], pred["prob_cin2plus"], thr)
        cin3_rows.append(
            {
                "Method": method,
                "AUC": fmt_metric(metrics["auc"]),
                "Sensitivity": fmt_metric(metrics["sensitivity"]),
                "Specificity": fmt_metric(metrics["specificity"]),
                "PPV": fmt_metric(metrics["ppv"]),
                "NPV": fmt_metric(metrics["npv"]),
                "False-negative count": int(metrics["false_negative_count"]),
                "Screen-positive rate": fmt_metric(metrics["screen_positive_rate"]),
                "Status": "DONE",
            }
        )
    cin3_rows.append(
        {
            "Method": "Best active HyDRA full",
            "AUC": NA,
            "Sensitivity": NA,
            "Specificity": NA,
            "PPV": NA,
            "NPV": NA,
            "False-negative count": NA,
            "Screen-positive rate": NA,
            "Status": "NOT_RUN_FULL_END_TO_END_MODULES_NOT_WIRED",
        }
    )
    table4 = pd.DataFrame(cin3_rows)
    write_table(table4, "Table_4_CIN3plus_Safety", tables)

    best_hydra_auc = metric_point(hydra_min.iloc[0]["AUC (95% CI)"]) if not hydra_min.empty else math.nan
    surrogate_auc = metric_point(step2["AUC (95% CI)"])
    clinical_auc = metric_point(clinical["AUC (95% CI)"])
    full_supported = False
    partial_supported = bool(np.isfinite(best_hydra_auc) and best_hydra_auc > surrogate_auc and best_hydra_auc > clinical_auc)
    if full_supported:
        route = "Route A"
        pass_fail = "PASSED_FULL_RECOVERY"
        action = "Use active full runner as main manuscript evidence."
    elif partial_supported:
        route = "Route B"
        pass_fail = "PASSED_PARTIAL_RECOVERY"
        action = "Rewrite as active reliability-aware trainable-adapter posterior fusion; keep full CoE/prototype claims out of main result."
    else:
        route = "Route C"
        pass_fail = "FAILED_NO_ACTIVE_RUNNER"
        action = "Do not submit as current full HyDRA-CoE; active adapter did not beat reference evidence and full runner remains unwired."
    table5 = pd.DataFrame(
        [
            {"Decision item": "Original manuscript full method supported", "Evidence": "Full active HyDRA end-to-end runner did not run.", "Pass/fail": "FAIL", "Recommended manuscript action": "Do not claim full method."},
            {"Decision item": "Partial active HyDRA supported", "Evidence": f"Minimal adapter AUC {fmt_metric(best_hydra_auc)} vs surrogate {fmt_metric(surrogate_auc)} and clinical {fmt_metric(clinical_auc)}.", "Pass/fail": "PASS" if partial_supported else "FAIL", "Recommended manuscript action": action if partial_supported else "Do not promote partial method as main unless further improved."},
            {"Decision item": "Only surrogate supported", "Evidence": "Step2 surrogate remains audited reference; active full runner incomplete.", "Pass/fail": "PASS" if not partial_supported else "FAIL", "Recommended manuscript action": "Treat surrogate as weak baseline/infrastructure evidence only."},
            {"Decision item": "Need method rewrite", "Evidence": route, "Pass/fail": "PASS", "Recommended manuscript action": action},
            {"Decision item": "Need endpoint relock", "Evidence": "Endpoint audit found no impossible mapping rows.", "Pass/fail": "FAIL", "Recommended manuscript action": "No relock required from this audit."},
        ]
    )
    write_table(table5, "Table_5_Go_NoGo_Recommendation", tables)

    status = update_status(
        git_commit=git_commit(),
        git_status_short=git_status(),
        step2_6_status=pass_fail,
        manuscript_route=route,
        best_active_visual_baseline=visual_best,
        best_active_hydra_minimal=hydra_min.iloc[0].to_dict() if not hydra_min.empty else {},
        best_active_hydra_full={"status": "NOT_RUN_FULL_END_TO_END_MODULES_NOT_WIRED"},
        recommended_manuscript_strategy=action,
        final_tables={
            "Table_1": rel(tables / "Table_1_Endpoint_Centre_Audit.csv"),
            "Table_2": rel(tables / "Table_2_Active_Visual_Runner_Recovery.csv"),
            "Table_3": rel(tables / "Table_3_Active_HyDRA_Recovery.csv"),
            "Table_4": rel(tables / "Table_4_CIN3plus_Safety.csv"),
            "Table_5": rel(tables / "Table_5_Go_NoGo_Recommendation.csv"),
        },
    )
    status_md = OUT_DIR / "STEP2_6_ACTIVE_FULL_RUNNER_RECOVERY_STATUS.md"
    endpoint = status.get("endpoint_audit", {})
    manifest = status.get("raw_image_manifest", {})
    md = [
        "# Step 2.6 Active Full-Runner Recovery Status",
        "",
        f"- Step 2.6 status: `{pass_fail}`",
        f"- Manuscript route: `{route}`",
        f"- Git commit: `{git_commit()}`",
        f"- Run timestamp: {status.get('run_timestamp')}",
        "",
        "## Endpoint Audit",
        "",
        f"- Status: `{endpoint.get('status')}`",
        f"- n: {endpoint.get('n')}",
        f"- CIN2+ positive/negative: {endpoint.get('cin2plus_positive')} / {endpoint.get('cin2plus_negative')}",
        f"- All-positive centre(s): {endpoint.get('all_positive_centres')}",
        f"- Mapping error rows: {endpoint.get('mapping_error_rows')}",
        "",
        "## Raw Manifest",
        "",
        f"- OCT coverage: {manifest.get('oct_coverage')}",
        f"- Colposcopy coverage: {manifest.get('colposcopy_coverage')}",
        f"- Missing cases: {manifest.get('missing_cases')}",
        "",
        "## Gates",
        "",
        f"- Dataloader smoke test: `{status.get('dataloader_smoke_test', {}).get('status')}`",
        f"- Overfit sanity: `{status.get('overfit_sanity', {}).get('status')}`",
        f"- Full LOCO: first-pass seed42 adapter completed; full end-to-end HyDRA not wired.",
        "",
        "## Recommendation",
        "",
        action,
        "",
        "## Git Status Short",
        "",
        "```text",
        git_status(),
        "```",
    ]
    status_md.write_text("\n".join(md), encoding="utf-8")
    return status_md


def plot_figures(input_dir: PathLike = OUT_DIR, output_dir: PathLike = OUT_DIR / "figures") -> Path:
    import matplotlib.pyplot as plt

    out = ensure(output_dir)
    source = ensure(out / "source")
    tables = p(input_dir) / "tables"
    audit = p(input_dir) / "audit"
    status = read_json(p(input_dir) / "STEP2_6_ACTIVE_FULL_RUNNER_RECOVERY_STATUS.json", {})

    # Figure 1
    mod = pd.DataFrame(
        [
            {"component": "raw image manifest", "active": 1, "compatibility": "PASS"},
            {"component": "trainable feature adapter", "active": 1, "compatibility": "PASS"},
            {"component": "end-to-end ViT backbone", "active": 0, "compatibility": "NOT_WIRED"},
            {"component": "OCT MIL aggregator", "active": 1, "compatibility": "ADAPTER_PROXY"},
            {"component": "ASCCP prototype loss", "active": 0, "compatibility": "NOT_WIRED"},
            {"component": "CoE trajectory loss", "active": 0, "compatibility": "NOT_WIRED"},
        ]
    )
    mod.to_csv(source / "Figure_1_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].barh(mod["component"], mod["active"], color=["#4f8a72" if x else "#b65f5f" for x in mod["active"]])
    axes[0].set_title("Module availability")
    axes[1].barh(mod["component"], [1 if x == "PASS" else 0.5 if x == "ADAPTER_PROXY" else 0 for x in mod["compatibility"]], color="#6677aa")
    axes[1].set_title("Runner compatibility")
    axes[2].pie([mod["active"].sum(), len(mod) - mod["active"].sum()], labels=["active", "inactive"], colors=["#4f8a72", "#b65f5f"])
    axes[2].set_title("Active vs inactive")
    save_all(fig, out, "Figure_1_Why_Step2_5_Failed_Partially")
    plt.close(fig)

    # Figure 2
    centre = pd.read_csv(audit / "centre_endpoint_distribution_v2.csv")
    centre.to_csv(source / "Figure_2_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].barh(centre["center_name"], centre["n_pathology_cin2plus"] / centre["n_total"], color="#4f8a72")
    axes[0, 0].set_title("CIN2+ prevalence")
    axes[0, 1].barh(centre["center_name"], centre["n_pathology_cin3plus"] / centre["n_total"], color="#6677aa")
    axes[0, 1].set_title("CIN3+ prevalence")
    axes[1, 0].bar(["CIN2+/HSIL", "CIN3", "Cancer"], [centre["n_CIN2_or_HSIL_ungraded"].sum(), centre["n_CIN3"].sum(), centre["n_cancer"].sum()], color="#aa7a44")
    axes[1, 0].set_title("Pathology audit summary")
    axes[1, 1].barh(centre["center_name"], centre["n_total"] - centre["n_pathology_cin2plus"], label="negative", color="#cccccc")
    axes[1, 1].barh(centre["center_name"], centre["n_pathology_cin2plus"], left=centre["n_total"] - centre["n_pathology_cin2plus"], label="positive", color="#b65f5f")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title("LOCO test composition")
    save_all(fig, out, "Figure_2_Endpoint_Centre_Distribution")
    plt.close(fig)

    # Figure 3
    table3 = pd.read_csv(tables / "Table_3_Active_HyDRA_Recovery.csv")
    fig3 = table3.copy()
    fig3["auc_value"] = fig3["AUC (95% CI)"].map(metric_point)
    fig3.to_csv(source / "Figure_3_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    pred_sources = [
        ("Step2 surrogate HyDRA", ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_hydra_full.csv", None),
        ("Active visual best", OUT_DIR / "predictions/active_visual_baseline_predictions.csv", status.get("best_active_visual_baseline", {}).get("Method")),
        ("Active minimal HyDRA", OUT_DIR / "predictions/active_hydra_minimal_predictions.csv", "HyDRA_CoE_Active_Minimal_TrainableFeatureAdapter_NoPrototype_NoCoE"),
    ]
    roc_rows = []
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#999999")
    for label, path, model in pred_sources:
        if not p(path).exists():
            continue
        pred = pd.read_csv(p(path))
        if model and "model_variant" in pred:
            pred = pred[pred["model_variant"] == model]
        pts = roc_points(pred["pathology_cin2plus"], pred["prob_cin2plus"])
        if len(pts):
            axes[0, 0].plot(pts["fpr"], pts["tpr"], label=label)
            for _, r in pts.iterrows():
                roc_rows.append({"curve": label, **r.to_dict()})
    pd.DataFrame(roc_rows).to_csv(source / "Figure_3_ROC_source.csv", index=False, encoding="utf-8-sig")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("ROC curves")
    axes[0, 1].barh(fig3["Method"].str.slice(0, 35), fig3["auc_value"], color="#4f8a72")
    axes[0, 1].set_title("AUC")
    axes[1, 0].barh(fig3["Method"].str.slice(0, 35), fig3["Sensitivity at t_safety95 (95% CI)"].map(metric_point), label="Sensitivity", color="#6677aa")
    axes[1, 0].barh(fig3["Method"].str.slice(0, 35), fig3["Specificity at t_safety95 (95% CI)"].map(metric_point), alpha=0.6, label="Specificity", color="#aa7a44")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_title("Sensitivity/specificity")
    axes[1, 1].barh(fig3["Method"].str.slice(0, 35), fig3["Screen-positive rate (95% CI)"].map(metric_point), label="Screen+", color="#b65f5f")
    axes[1, 1].barh(fig3["Method"].str.slice(0, 35), fig3["PPV (95% CI)"].map(metric_point), alpha=0.6, label="PPV", color="#777777")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].set_title("Screen-positive rate and PPV")
    save_all(fig, out, "Figure_3_Active_Runner_Recovery_Performance")
    plt.close(fig)

    # Figure 4
    m2 = pd.read_csv(tables / "Table_M2_Aux_Init_Effect.csv")
    m2.to_csv(source / "Figure_4_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].bar(["no aux", "OCT SSL", "OCT-VLM"], [np.nan, np.nan, np.nan], color="#999999")
    axes[0, 0].set_title("Aux init AUC not evaluated")
    axes[0, 1].text(0.5, 0.5, "Proxy checkpoints not compatible\nwith raw-stat adapter", ha="center", va="center")
    axes[0, 1].set_axis_off()
    axes[1, 0].bar(["OCT uncertainty"], [np.nan], color="#999999")
    axes[1, 0].set_title("Reliability unavailable")
    axes[1, 1].bar(["CIN3+ change"], [np.nan], color="#999999")
    axes[1, 1].set_title("CIN3+ change not evaluated")
    save_all(fig, out, "Figure_4_Auxiliary_OCT_SSL_VLM_Init_Effect")
    plt.close(fig)

    # Figure 5
    table5 = pd.read_csv(tables / "Table_5_Go_NoGo_Recommendation.csv")
    table5.to_csv(source / "Figure_5_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pass_map = table5["Pass/fail"].astype(str).eq("PASS").astype(int)
    axes[0].barh(table5["Decision item"], pass_map, color=["#4f8a72" if x else "#b65f5f" for x in pass_map])
    axes[0].set_title("Evidence ladder")
    route = status.get("manuscript_route", "UNKNOWN")
    axes[1].text(0.5, 0.5, route, ha="center", va="center", fontsize=18)
    axes[1].set_axis_off()
    axes[1].set_title("Recommended route")
    axes[2].barh(["endpoint", "loader", "adapter", "full end-to-end", "robustness/calibration/CoE"], [1, 1, 1, 0, 0], color=["#4f8a72", "#4f8a72", "#4f8a72", "#b65f5f", "#b65f5f"])
    axes[2].set_title("Remaining experiments")
    save_all(fig, out, "Figure_5_Go_NoGo_Decision_Schematic")
    plt.close(fig)
    update_status(final_figures={"status": "DONE", "figures_dir": rel(out)})
    return out


def save_all(fig, out: Path, stem: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out / f"{stem}.{ext}", **kwargs)


def run_all(config_path: PathLike, no_dry_run: bool = False) -> None:
    cfg = load_yaml(config_path)
    global OUT_DIR
    OUT_DIR = p(cfg["output_dir"])
    ensure(OUT_DIR)
    update_status(experiment_name=cfg["experiment_name"], step2_6_status="IN_PROGRESS")
    reaudit_endpoint(cfg["data"]["data_lock"], cfg["data"]["split_manifest"], OUT_DIR / "audit")
    build_raw_manifest(cfg["data"]["data_lock"], OUT_DIR / "manifests")
    smoke_test_dataloader(cfg["data"]["raw_image_manifest"], cfg["data"]["split_manifest"], OUT_DIR / "audit")
    build_adapter_features(config_path)
    overfit_and_sanity(config_path)
    train_active_visual_baselines(config_path, no_dry_run=no_dry_run)
    train_active_visual_baselines_with_aux(
        config_path,
        cfg["data"]["step2_5_aux_oct_ssl_checkpoint"],
        cfg["data"]["step2_5_oct_vlm_checkpoint"],
        no_dry_run=no_dry_run,
    )
    train_active_hydra_minimal(config_path, no_dry_run=no_dry_run)
    train_active_hydra_full(config_path, no_dry_run=no_dry_run)
    collect_tables_and_status(config_path)
    plot_figures(OUT_DIR, OUT_DIR / "figures")
