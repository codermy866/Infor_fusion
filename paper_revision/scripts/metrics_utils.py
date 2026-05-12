#!/usr/bin/env python3
"""Shared metrics for Information Fusion revision experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class MetricBundle:
    auc: float
    auprc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    ece: float
    brier: float
    threshold: float
    n: int
    positives: int
    negatives: int


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[int, int, int, int]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    return tn, fp, fn, tp


def roc_auc_binary(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    n_pos = int((y_true_arr == 1).sum())
    n_neg = int((y_true_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_prob_arr)
    ranks = np.empty_like(order, dtype=float)
    sorted_probs = y_prob_arr[order]
    start = 0
    while start < len(sorted_probs):
        end = start + 1
        while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[y_true_arr == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_binary(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    n_pos = int((y_true_arr == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_prob_arr)
    sorted_true = y_true_arr[order]
    tp_cum = np.cumsum(sorted_true == 1)
    fp_cum = np.cumsum(sorted_true == 0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, EPS)
    return float((precision * (sorted_true == 1)).sum() / n_pos)


def expected_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    if y_true_arr.size == 0:
        return float("nan")

    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_arr >= left) & (y_prob_arr <= right)
        else:
            mask = (y_prob_arr >= left) & (y_prob_arr < right)
        if not np.any(mask):
            continue
        acc = y_true_arr[mask].mean()
        conf = y_prob_arr[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def select_threshold(y_true: Iterable[int], y_prob: Iterable[float], rule: str = "youden") -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    if y_true_arr.size == 0:
        return 0.5
    if rule == "fixed_0.5":
        return 0.5

    thresholds = np.unique(np.clip(y_prob_arr, 0.0, 1.0))
    if thresholds.size == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true_arr, y_pred)
        sensitivity = tp / (tp + fn + EPS)
        specificity = tn / (tn + fp + EPS)
        if rule == "youden":
            score = sensitivity + specificity - 1.0
        elif rule.startswith("target_sensitivity:"):
            target = float(rule.split(":", 1)[1])
            score = -abs(sensitivity - target) + 0.01 * specificity
        else:
            raise ValueError(f"Unknown threshold rule: {rule}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: Optional[float] = None,
    threshold_rule: str = "youden",
    n_bins: int = 10,
) -> MetricBundle:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
    if threshold is None:
        threshold = select_threshold(y_true_arr, y_prob_arr, threshold_rule)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_counts(y_true_arr, y_pred)
    positives = int((y_true_arr == 1).sum())
    negatives = int((y_true_arr == 0).sum())

    auc = roc_auc_binary(y_true_arr, y_prob_arr)
    auprc = average_precision_binary(y_true_arr, y_prob_arr)
    ppv = float(tp / (tp + fp + EPS))
    sensitivity = float(tp / (tp + fn + EPS))
    f1 = float(2.0 * ppv * sensitivity / (ppv + sensitivity + EPS))

    return MetricBundle(
        auc=auc,
        auprc=auprc,
        sensitivity=sensitivity,
        specificity=float(tn / (tn + fp + EPS)),
        ppv=ppv,
        npv=float(tn / (tn + fn + EPS)),
        f1=f1,
        ece=expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins),
        brier=float(np.mean((y_prob_arr - y_true_arr) ** 2)),
        threshold=float(threshold),
        n=int(y_true_arr.size),
        positives=positives,
        negatives=negatives,
    )


def summarize_prediction_dataframe(
    df: pd.DataFrame,
    threshold_rule: str = "youden",
    threshold_from_split: str = "internal_validation",
) -> pd.DataFrame:
    required = {"method", "run_id", "seed", "split", "y_true", "y_prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction dataframe is missing columns: {missing}")

    rows = []
    group_cols = ["method", "run_id", "seed"]
    for (method, run_id, seed), run_df in df.groupby(group_cols, dropna=False):
        threshold_source = run_df[run_df["split"] == threshold_from_split]
        if threshold_source.empty:
            threshold_source = run_df
        threshold = select_threshold(threshold_source["y_true"], threshold_source["y_prob"], threshold_rule)

        for split, split_df in run_df.groupby("split"):
            bundle = binary_metrics(split_df["y_true"], split_df["y_prob"], threshold=threshold)
            row = {
                "method": method,
                "run_id": run_id,
                "seed": seed,
                "split": split,
                **bundle.__dict__,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metric_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier"]
    rows = []
    for (method, split), group in run_metrics.groupby(["method", "split"], dropna=False):
        row: Dict[str, object] = {
            "method": method,
            "split": split,
            "runs": int(group.shape[0]),
            "n": int(group["n"].median()) if "n" in group else "",
        }
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[metric] = format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "TBD"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def decision_curve_points(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    n = len(y_true_arr)
    prevalence = y_true_arr.mean() if n else float("nan")
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true_arr, y_pred)
        net_benefit = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
        treat_all = prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(net_benefit),
                "treat_all": float(treat_all),
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def read_prediction_files(prediction_dir: Path) -> pd.DataFrame:
    files = sorted(prediction_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)
