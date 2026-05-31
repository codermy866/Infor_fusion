#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    n: int
    positives: int
    negatives: int
    auc: float
    average_precision: float
    brier: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    mcc: float
    screen_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    false_negative_count: int


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> BinaryMetrics:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(y_prob), dtype=float)
    pred = (p >= threshold).astype(int)
    n = int(len(y))
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if len(np.unique(y)) < 2:
        auc = float("nan")
        ap = float("nan") if positives == 0 else 1.0
    else:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p)) if n else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, n)
    bal = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    f1 = float(f1_score(y, pred, zero_division=0)) if n else float("nan")
    mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else float("nan")
    spr = _safe_div(tp + fp, n)
    return BinaryMetrics(
        n=n,
        positives=positives,
        negatives=negatives,
        auc=auc,
        average_precision=ap,
        brier=brier,
        accuracy=accuracy,
        balanced_accuracy=bal,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        f1=f1,
        mcc=mcc,
        screen_positive_rate=spr,
        false_positive_rate=_safe_div(fp, fp + tn),
        false_negative_rate=_safe_div(fn, fn + tp),
        false_negative_count=int(fn),
    )


def select_thresholds(y_true: Iterable[int], y_prob: Iterable[float]) -> tuple[dict[str, float], list[dict[str, object]]]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(y_prob), dtype=float)
    thresholds = np.unique(np.r_[0.0, p, 1.0])
    candidates = []
    for t in thresholds:
        m = binary_metrics(y, p, float(t))
        candidates.append((float(t), m.sensitivity, m.specificity))
    youden = max(candidates, key=lambda x: ((-1 if np.isnan(x[1]) else x[1]) + (-1 if np.isnan(x[2]) else x[2]), x[0]))
    out = {"t_youden": float(youden[0])}
    warnings: list[dict[str, object]] = []
    for name, target in [("t_safety95", 0.95), ("t_safety90", 0.90)]:
        feasible = [x for x in candidates if not np.isnan(x[1]) and x[1] >= target]
        if feasible:
            # Highest validation specificity subject to the sensitivity target.
            best = max(feasible, key=lambda x: ((-1 if np.isnan(x[2]) else x[2]), x[0]))
        else:
            best = max(candidates, key=lambda x: (-1 if np.isnan(x[1]) else x[1], -1 if np.isnan(x[2]) else x[2]))
            warnings.append(
                {
                    "operating_point": name,
                    "target_sensitivity": target,
                    "max_validation_sensitivity": best[1],
                    "selected_threshold": best[0],
                    "warning": "validation_target_sensitivity_not_reached",
                }
            )
        out[name] = float(best[0])
    return out, warnings


def bootstrap_ci(y_true, y_prob, threshold: float, metric_name: str, iterations: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    point = getattr(binary_metrics(y, p, threshold), metric_name)
    if len(y) == 0 or np.isnan(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        val = getattr(binary_metrics(y[idx], p[idx], threshold), metric_name)
        if not np.isnan(val):
            vals.append(val)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt_ci(point: float, low: float, high: float) -> str:
    if point is None or np.isnan(point):
        return "NA"
    if np.isnan(low) or np.isnan(high):
        return f"{point:.3f}"
    return f"{point:.3f} ({low:.3f}-{high:.3f})"
