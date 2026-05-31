#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics_binary import binary_metrics

THRESHOLD_METRICS = {"sensitivity", "specificity", "ppv", "npv", "f1", "screen_positive_rate"}


def _threshold_metric_value(y_true, y_pred, metric: str) -> float:
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


def paired_bootstrap_difference(y, p_hydra, p_base, threshold_hydra, threshold_base, metric: str, iterations: int, seed: int):
    y = np.asarray(y, dtype=int)
    ph = np.asarray(p_hydra, dtype=float)
    pb = np.asarray(p_base, dtype=float)
    if metric == "auc":
        point = float(roc_auc_score(y, ph) - roc_auc_score(y, pb)) if len(np.unique(y)) == 2 else float("nan")
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(iterations):
            idx = rng.integers(0, len(y), size=len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(float(roc_auc_score(y[idx], ph[idx]) - roc_auc_score(y[idx], pb[idx])))
        if not vals:
            return point, float("nan"), float("nan"), float("nan")
        vals = np.asarray(vals)
        p_raw = float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
        return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), min(p_raw, 1.0)
    if metric in THRESHOLD_METRICS:
        pred_h = ph >= threshold_hydra
        pred_b = pb >= threshold_base
        point = _threshold_metric_value(y, pred_h, metric) - _threshold_metric_value(y, pred_b, metric)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(iterations):
            idx = rng.integers(0, len(y), size=len(y))
            mh = _threshold_metric_value(y[idx], pred_h[idx], metric)
            mb = _threshold_metric_value(y[idx], pred_b[idx], metric)
            if not np.isnan(mh) and not np.isnan(mb):
                vals.append(mh - mb)
        if not vals:
            return point, float("nan"), float("nan"), float("nan")
        vals = np.asarray(vals)
        p_raw = float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
        return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), min(p_raw, 1.0)
    point = getattr(binary_metrics(y, ph, threshold_hydra), metric) - getattr(binary_metrics(y, pb, threshold_base), metric)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), size=len(y))
        mh = getattr(binary_metrics(y[idx], ph[idx], threshold_hydra), metric)
        mb = getattr(binary_metrics(y[idx], pb[idx], threshold_base), metric)
        if not np.isnan(mh) and not np.isnan(mb):
            vals.append(mh - mb)
    if not vals:
        return point, float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals)
    p_raw = float(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)))
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), min(p_raw, 1.0)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [(i, p) for i, p in enumerate(p_values)]
    valid = sorted([(i, p) for i, p in indexed if not np.isnan(p)], key=lambda x: x[1])
    adjusted = [float("nan")] * len(p_values)
    prev = 0.0
    m = len(valid)
    for rank, (idx, p) in enumerate(valid):
        val = min(1.0, (m - rank) * p)
        val = max(prev, val)
        adjusted[idx] = val
        prev = val
    return adjusted
