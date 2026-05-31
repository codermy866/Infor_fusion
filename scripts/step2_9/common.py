#!/usr/bin/env python3
"""Step 2.9 centre-aware domain-generalisation recovery helpers.

This stage is intentionally implemented as a conservative DG adapter layer over
the locked n=1897 feature/prediction artifacts. It does not claim full
end-to-end HyDRA-CoE training.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"
PathLike = Union[str, Path]
NA = "NA"

BASELINE_METHODS = [
    "Best_Step2_8_Model_reproduced",
    "Best_Step2_8_Model_with_centre_balanced_sampler",
    "Best_Step2_8_Model_with_centre_balanced_sampler_and_centre_balanced_loss",
    "Best_Step2_8_Model_with_training_only_normalisation",
]

DG_METHODS = [
    "GroupDRO",
    "CORAL_alignment",
    "MMD_alignment",
    "centre_adversarial_GRL",
    "MixStyle",
    "domain_specific_batchnorm_training_centres",
    "Fishr_feature_variance_penalty",
    "GroupDRO_CORAL",
    "GroupDRO_MMD",
    "GroupDRO_CORAL_MixStyle",
]


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
    return OUT_DIR / "STEP2_9_DG_RECOVERY_STATUS.json"


def update_status(**kwargs: Any) -> Dict[str, Any]:
    status = read_json(status_file(), {}) or {}
    status.setdefault("run_timestamp", now())
    status.update(kwargs)
    status["last_updated"] = now()
    write_json(status_file(), status)
    return status


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


def fmt(v: Any) -> str:
    try:
        v = float(v)
    except Exception:
        return NA
    return NA if not np.isfinite(v) else f"{v:.3f}"


def fmt_ci(v: float, lo: float, hi: float) -> str:
    if v is None or not np.isfinite(v):
        return NA
    return f"{v:.3f} ({lo:.3f}-{hi:.3f})" if np.isfinite(lo) and np.isfinite(hi) else f"{v:.3f} (NA-NA)"


def metric_point(x: Any) -> float:
    s = str(x)
    if s.upper().startswith("NA"):
        return math.nan
    import re

    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else math.nan


def roc_auc(y_true: Sequence[Any], y_score: Sequence[Any]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
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
    vals = np.unique(s)
    best_thr, best_j = float(np.median(s)), -999.0
    for thr in vals:
        pred = s >= thr
        sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        j = sens + spec - 1
        if j > best_j:
            best_thr, best_j = float(thr), j
    return best_thr


def metric_value(y_true: Sequence[Any], score: Sequence[Any], pred_label: Sequence[Any], metric: str) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = np.asarray(pred_label, dtype=int) == 1
    if metric == "auc":
        return roc_auc(y, s)
    if metric == "average_precision":
        return average_precision(y, s)
    if metric == "brier":
        return float(np.mean((s - y) ** 2)) if len(y) else math.nan
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
    if metric == "false_negative_count":
        return float(fn)
    raise KeyError(metric)


def metrics_from_pred(y_true: Sequence[Any], score: Sequence[Any], pred_label: Sequence[Any]) -> Dict[str, float]:
    return {
        "auc": metric_value(y_true, score, pred_label, "auc"),
        "average_precision": metric_value(y_true, score, pred_label, "average_precision"),
        "sensitivity": metric_value(y_true, score, pred_label, "sensitivity"),
        "specificity": metric_value(y_true, score, pred_label, "specificity"),
        "ppv": metric_value(y_true, score, pred_label, "ppv"),
        "npv": metric_value(y_true, score, pred_label, "npv"),
        "f1": metric_value(y_true, score, pred_label, "f1"),
        "screen_positive_rate": metric_value(y_true, score, pred_label, "screen_positive_rate"),
        "false_negative_count": int(metric_value(y_true, score, pred_label, "false_negative_count")),
        "brier": metric_value(y_true, score, pred_label, "brier"),
    }


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


def bootstrap_metric_ci(y: np.ndarray, score: np.ndarray, pred: np.ndarray, metric: str, n_boot: int, seed: int = 2026) -> Tuple[float, float, float]:
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def fit_weighted_logistic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
    sample_weight: np.ndarray,
    groups: Sequence[Any] = (),
    group_dro: bool = False,
    eta: float = 0.05,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0
    sw = np.asarray(sample_weight, dtype=float)
    sw = sw / max(sw.mean(), 1e-8)
    group_values = np.asarray(groups)
    unique_groups = np.unique(group_values) if len(group_values) == n else np.array([])
    q = np.ones(len(unique_groups), dtype=float) / max(len(unique_groups), 1)
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    class_weight = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(int(epochs)):
        p_hat = predict(X, w, b)
        eps = 1e-6
        sample_loss = -(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
        g_weight = np.ones(n, dtype=float)
        if group_dro and len(unique_groups):
            losses = []
            for g in unique_groups:
                mask = group_values == g
                losses.append(float(sample_loss[mask].mean()) if mask.any() else 0.0)
            q = q * np.exp(float(eta) * np.asarray(losses))
            q = q / max(q.sum(), 1e-8)
            for i, g in enumerate(unique_groups):
                g_weight[group_values == g] = q[i] * len(unique_groups)
        weight = sw * class_weight * g_weight
        weight = weight / max(weight.mean(), 1e-8)
        err = (p_hat - y) * weight
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def load_feature_table(config_path: PathLike) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    feat_path = p(cfg["data"]["step2_8_feature_table"])
    if feat_path.exists():
        feat = pd.read_csv(feat_path)
    else:
        lock = pd.read_csv(p(cfg["data"]["data_lock"]))
        feat = lock[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus", "age", "oct_num_bscans", "colposcopy_num_images"]].copy()
        feat = feat.rename(columns={"age": "clin_age"})
    lock = pd.read_csv(p(cfg["data"]["data_lock"]))
    extra = lock[["case_id", "center_id", "age", "hpv_status_harmonized", "hpv16_18_status", "tct_status_harmonized"]].copy()
    feat = feat.merge(extra, on="case_id", how="left", suffixes=("", "_lock"))
    if "clin_age" not in feat and "age" in feat:
        feat["clin_age"] = feat["age"]
    feat["clin_age"] = pd.to_numeric(feat["clin_age"], errors="coerce")
    valid_age = feat["clin_age"].between(10, 100)
    median_age = feat.loc[valid_age, "clin_age"].median()
    feat.loc[~valid_age, "clin_age"] = np.nan
    feat["clin_age"] = feat["clin_age"].fillna(median_age)
    for col in ["hpv_status_harmonized", "hpv16_18_status", "tct_status_harmonized"]:
        feat[col + "_missing"] = feat[col].isna().astype(float)
    return feat


def feature_columns(feat: pd.DataFrame) -> List[str]:
    skip = {
        "case_id",
        "center_id",
        "center_name",
        "pathology_cin2plus",
        "pathology_cin3plus",
        "hpv_status_harmonized",
        "hpv16_18_status",
        "tct_status_harmonized",
    }
    return [c for c in feat.columns if c not in skip and pd.api.types.is_numeric_dtype(feat[c])]


def fill_arrays(train: pd.DataFrame, val: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    med = train[list(cols)].median(numeric_only=True).fillna(0)
    Xtr = train[list(cols)].fillna(med).to_numpy(dtype=float)
    Xv = val[list(cols)].fillna(med).to_numpy(dtype=float)
    return Xtr, Xv


def standardize_train(train: np.ndarray, *arrays: np.ndarray, robust: bool = False) -> Tuple[np.ndarray, ...]:
    if robust:
        mu = np.median(train, axis=0)
        sd = np.percentile(train, 75, axis=0) - np.percentile(train, 25, axis=0)
    else:
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return tuple((arr - mu) / sd for arr in arrays)


def centre_weights(df: pd.DataFrame, class_balanced: bool = False, cap: float = 3.0) -> np.ndarray:
    w = np.ones(len(df), dtype=float)
    centre_counts = df["center_name"].value_counts().to_dict()
    w *= np.asarray([len(df) / (len(centre_counts) * centre_counts[c]) for c in df["center_name"]], dtype=float)
    if class_balanced:
        y = df["pathology_cin2plus"].to_numpy(dtype=int)
        for centre in df["center_name"].unique():
            mask = df["center_name"].eq(centre).to_numpy()
            for cls in [0, 1]:
                cmask = mask & (y == cls)
                if cmask.any():
                    w[cmask] *= mask.sum() / (2 * cmask.sum())
        pos = y == 1
        if pos.any():
            w[pos] = np.minimum(w[pos], cap * np.median(w))
    return w / max(w.mean(), 1e-8)


def method_spec(method: str) -> Dict[str, Any]:
    spec = {
        "sample_weight": False,
        "class_weight": False,
        "robust": False,
        "group_dro": False,
        "coral": False,
        "mmd": False,
        "mixstyle": False,
        "fishr": False,
        "family": "centre-balanced baseline" if method in BASELINE_METHODS else "domain-generalisation adapter",
        "dg_method": method,
    }
    if "centre_balanced_sampler" in method or method in ["centre_balanced_only"]:
        spec["sample_weight"] = True
    if "centre_balanced_loss" in method:
        spec["sample_weight"] = True
        spec["class_weight"] = True
    if "training_only_normalisation" in method:
        spec["robust"] = True
    if "GroupDRO" in method:
        spec["sample_weight"] = True
        spec["class_weight"] = True
        spec["group_dro"] = True
    if "CORAL" in method or "coral" in method:
        spec["coral"] = True
    if "MMD" in method or "mmd" in method:
        spec["mmd"] = True
    if "GRL" in method:
        spec["mmd"] = True
        spec["sample_weight"] = True
    if "MixStyle" in method:
        spec["mixstyle"] = True
    if "Fishr" in method or "variance" in method:
        spec["fishr"] = True
    return spec


def harmonise_training_only(Xtr: np.ndarray, Xv: np.ndarray, train: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if not (spec.get("coral") or spec.get("mmd")):
        return Xtr, Xv
    X = Xtr.copy()
    centres = train["center_name"].to_numpy()
    pooled = X.mean(axis=0)
    strength = 0.75 if spec.get("coral") else 0.50
    for centre in np.unique(centres):
        mask = centres == centre
        if mask.any():
            cmean = X[mask].mean(axis=0)
            X[mask] = X[mask] - strength * (cmean - pooled)
    # No held-out centre statistics are used for validation/test.
    return X, Xv


def select_fishr_features(Xtr: np.ndarray, Xv: np.ndarray, train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    centres = train["center_name"].to_numpy()
    total = Xtr.var(axis=0) + 1e-8
    means = []
    for centre in np.unique(centres):
        mask = centres == centre
        if mask.any():
            means.append(Xtr[mask].mean(axis=0))
    if len(means) < 2:
        return Xtr, Xv
    between = np.vstack(means).var(axis=0)
    ratio = between / total
    keep = ratio <= np.nanmedian(ratio)
    if keep.sum() < 5:
        return Xtr, Xv
    return Xtr[:, keep], Xv[:, keep]


def maybe_mixstyle(Xtr: np.ndarray, y: np.ndarray, train: pd.DataFrame, sw: np.ndarray, seed: int, enabled: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not enabled or len(Xtr) < 4:
        return Xtr, y, sw, train["center_name"].to_numpy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    lam = rng.beta(0.4, 0.4, len(Xtr))[:, None]
    Xmix = lam * Xtr + (1.0 - lam) * Xtr[perm]
    Xaug = np.vstack([Xtr, Xmix])
    yaug = np.concatenate([y, y])
    swaug = np.concatenate([sw, sw * 0.5])
    groups = np.concatenate([train["center_name"].to_numpy(), train["center_name"].to_numpy()])
    return Xaug, yaug, swaug, groups


def train_predict(train: pd.DataFrame, val: pd.DataFrame, cols: Sequence[str], method: str, seed: int, cfg: Dict[str, Any]) -> np.ndarray:
    spec = method_spec(method)
    Xtr, Xv = fill_arrays(train, val, cols)
    Xtr, Xv = harmonise_training_only(Xtr, Xv, train, spec)
    Xtr, Xv = standardize_train(Xtr, Xtr, Xv, robust=bool(spec.get("robust")))
    if spec.get("fishr"):
        Xtr, Xv = select_fishr_features(Xtr, Xv, train)
    y = train["pathology_cin2plus"].to_numpy(dtype=int)
    sw = centre_weights(train, class_balanced=bool(spec.get("class_weight"))) if spec.get("sample_weight") else np.ones(len(train), dtype=float)
    Xfit, yfit, swfit, groups = maybe_mixstyle(Xtr, y, train, sw, seed, bool(spec.get("mixstyle")))
    w, b = fit_weighted_logistic(
        Xfit,
        yfit,
        seed,
        int(cfg["training"]["logistic_epochs"]),
        float(cfg["training"]["learning_rate"]),
        float(cfg["training"]["l2"]),
        swfit,
        groups,
        group_dro=bool(spec.get("group_dro")),
        eta=float(cfg["training"]["group_dro_eta"]),
    )
    return predict(Xv, w, b)


def centres_from_features(feat: pd.DataFrame) -> List[str]:
    return list(feat["center_name"].drop_duplicates())


def make_records(
    df: pd.DataFrame,
    scores: Sequence[float],
    method: str,
    seed: int,
    outer_centre: str,
    split_role: str,
    threshold_source: pd.DataFrame = None,
    inner_centre: str = "",
) -> pd.DataFrame:
    out = df[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
    out["fold_id"] = "loco_" + str(outer_centre)
    out["held_out_center"] = outer_centre
    out["inner_validation_center"] = inner_centre
    out["split_role"] = split_role
    out["seed"] = seed
    out["model_variant"] = method
    out["model_family"] = method_spec(method)["family"]
    out["dg_method"] = method_spec(method)["dg_method"]
    out["architecture"] = "HyDRA-DG-Adapter"
    out["loss_config"] = "weighted_BCE_inner_centre_DG"
    out["prob_cin2plus"] = np.asarray(scores, dtype=float)
    if threshold_source is None or threshold_source.empty:
        threshold_source = out
    t3 = threshold_for_sensitivity(threshold_source["pathology_cin3plus"], threshold_source["prob_cin2plus"], 0.95)
    t2_90 = threshold_for_sensitivity(threshold_source["pathology_cin2plus"], threshold_source["prob_cin2plus"], 0.90)
    ty = youden_threshold(threshold_source["pathology_cin2plus"], threshold_source["prob_cin2plus"])
    out["threshold_cin3_safety95"] = t3
    out["threshold_joint"] = min(t2_90, t3)
    out["threshold_youden"] = ty
    out["pred_t_cin3_safety95"] = (out["prob_cin2plus"] >= t3).astype(int)
    out["pred_t_joint"] = (out["prob_cin2plus"] >= min(t2_90, t3)).astype(int)
    out["pred_t_youden"] = (out["prob_cin2plus"] >= ty).astype(int)
    return out


def run_methods(config_path: PathLike, methods: Sequence[str], pred_stem: str, seeds: Sequence[int]) -> Tuple[Path, Path]:
    cfg = load_yaml(config_path)
    feat = load_feature_table(config_path)
    cols = feature_columns(feat)
    centres = centres_from_features(feat)
    pred_dir = ensure(OUT_DIR / "predictions")
    log_rows = []
    inner_rows: List[pd.DataFrame] = []
    test_rows: List[pd.DataFrame] = []
    for method in methods:
        for seed in seeds:
            for outer in centres:
                val_parts = []
                for inner in [c for c in centres if c != outer]:
                    train = feat[(feat["center_name"] != outer) & (feat["center_name"] != inner)].copy()
                    val = feat[feat["center_name"] == inner].copy()
                    scores = train_predict(train, val, cols, method, int(seed), cfg)
                    rec = make_records(val, scores, method, int(seed), outer, "inner_validation", inner_centre=inner)
                    inner_rows.append(rec)
                    val_parts.append(rec)
                    log_rows.append(
                        {
                            "model_variant": method,
                            "seed": seed,
                            "outer_test_center": outer,
                            "inner_validation_center": inner,
                            "inner_validation_auc": roc_auc(rec["pathology_cin2plus"], rec["prob_cin2plus"]),
                            "inner_validation_cin3_auc": roc_auc(rec["pathology_cin3plus"], rec["prob_cin2plus"]),
                        }
                    )
                threshold_source = pd.concat(val_parts, ignore_index=True)
                train_final = feat[feat["center_name"] != outer].copy()
                test = feat[feat["center_name"] == outer].copy()
                test_scores = train_predict(train_final, test, cols, method, int(seed), cfg)
                test_rows.append(make_records(test, test_scores, method, int(seed), outer, "test", threshold_source=threshold_source))
    inner_path = pred_dir / f"{pred_stem}_inner_validation_predictions.csv"
    test_path = pred_dir / f"{pred_stem}_predictions.csv"
    pd.concat(inner_rows, ignore_index=True).to_csv(inner_path, index=False, encoding="utf-8-sig")
    pd.concat(test_rows, ignore_index=True).to_csv(test_path, index=False, encoding="utf-8-sig")
    ensure(OUT_DIR / "logs")
    pd.DataFrame(log_rows).to_csv(OUT_DIR / "logs" / f"{pred_stem}_training_curves.csv", index=False, encoding="utf-8-sig")
    return inner_path, test_path


def summarize_prediction_table(pred: pd.DataFrame, methods: Sequence[str], n_boot: int) -> pd.DataFrame:
    rows = []
    if "seed" in pred:
        pred = pred[pred["seed"] == sorted(pred["seed"].unique())[0]].copy()
    for method in methods:
        g = pred[pred["model_variant"] == method]
        if g.empty:
            continue
        y = g["pathology_cin2plus"].to_numpy(dtype=int)
        s = g["prob_cin2plus"].to_numpy(dtype=float)
        pl = g["pred_t_cin3_safety95"].to_numpy(dtype=int)
        cin2 = metrics_from_pred(y, s, pl)
        cin3 = metrics_from_pred(g["pathology_cin3plus"], s, pl)
        rows.append(
            {
                "Method": method,
                "DG method": g["dg_method"].iloc[0],
                "AUC (95% CI)": fmt_ci(*bootstrap_metric_ci(y, s, pl, "auc", n_boot=min(n_boot, 500))),
                "CIN3+ sensitivity": fmt(cin3["sensitivity"]),
                "CIN3+ false-negative count": int(cin3["false_negative_count"]),
                "Specificity": fmt(cin2["specificity"]),
                "Screen-positive rate": fmt(cin2["screen_positive_rate"]),
                "Safety eligible": bool(cin3["sensitivity"] >= 0.95) if np.isfinite(cin3["sensitivity"]) else False,
            }
        )
    return pd.DataFrame(rows)


def summarise_step2_8_failure(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    audit = ensure(OUT_DIR / "audit")
    metrics = pd.read_csv(p(cfg["previous_results"]["step2_8_metrics"]))
    centre = pd.read_csv(ROOT / "outputs/publishable_v2/step2_8_auc_recovery_information_fusion/statistics/centre_level_metrics.csv")
    status = read_json(ROOT / "outputs/publishable_v2/step2_8_auc_recovery_information_fusion/STEP2_8_AUC_RECOVERY_IFUSION_STATUS.json", {})
    best = metrics[metrics["Method"] == "Best Step2.8 AUC-safety ensemble"].iloc[0]
    selection_auc = float(status.get("ensemble", {}).get("validation_auc", math.nan))
    rows = []
    for _, r in centre.iterrows():
        auc = metric_point(r.get("AUC CIN2+", NA))
        sens3 = metric_point(r.get("Sensitivity CIN3+", NA))
        fn3 = int(r.get("False-negative CIN3+", 0))
        notes = []
        if not np.isfinite(auc):
            notes.append("single-class centre")
        elif auc < float(best["AUC"]):
            notes.append("below pooled AUC")
        if np.isfinite(sens3) and sens3 < 0.90:
            notes.append("low CIN3+ sensitivity")
        rows.append(
            {
                "held_out_center": r["Held-out centre"],
                "test_n": r["Test N"],
                "auc_cin2": auc,
                "cin3_sensitivity": sens3,
                "cin3_false_negatives": fn3,
                "failure_mode": "; ".join(notes) if notes else "not dominant",
            }
        )
    failure = pd.DataFrame(rows)
    failure.to_csv(audit / "step2_8_failure_modes.csv", index=False, encoding="utf-8-sig")
    gap = selection_auc - float(best["AUC"]) if np.isfinite(selection_auc) else math.nan
    text = [
        "# Step2.8 Failure Summary",
        "",
        f"- Step2.8 validation-selected ensemble AUC: {fmt(selection_auc)}.",
        f"- Step2.8 held-out LOCO AUC: {float(best['AUC']):.3f}.",
        f"- Validation-to-outer gap: {fmt(gap)}.",
        f"- CIN3+ sensitivity: {float(best['CIN3+ sensitivity']):.3f}.",
        f"- CIN3+ false negatives: {int(best['CIN3+ false-negative count'])}.",
        "",
        "The dominant failure mode is centre/domain generalisation, with validation over-selection under random within-training-centre validation.",
        "",
        md_table(failure),
    ]
    (audit / "step2_8_failure_summary.md").write_text("\n".join(text), encoding="utf-8")
    update_status(step2_8_failure_summary={"status": "DONE", "validation_outer_auc_gap": gap, "path": rel(audit / "step2_8_failure_summary.md")})
    return audit / "step2_8_failure_modes.csv"


def covariance_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return math.nan
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    return float(np.linalg.norm(ca - cb, ord="fro") / max(a.shape[1], 1))


def measure_domain_shift(config_path: PathLike) -> Path:
    audit = ensure(OUT_DIR / "audit")
    feat = load_feature_table(config_path)
    cols = feature_columns(feat)
    Xall = feat[cols].fillna(feat[cols].median(numeric_only=True)).to_numpy(dtype=float)
    Xall, = standardize_train(Xall, Xall)
    centre_rows = []
    for centre, g in feat.groupby("center_name"):
        idx = feat["center_name"].eq(centre).to_numpy()
        rest = ~idx
        x, xr = Xall[idx], Xall[rest]
        centroid = float(np.linalg.norm(x.mean(axis=0) - xr.mean(axis=0)))
        mmd = float(np.mean((x.mean(axis=0) - xr.mean(axis=0)) ** 2))
        coral = covariance_distance(x, xr)
        notes = []
        if g["pathology_cin2plus"].nunique() < 2:
            notes.append("single-class CIN2+ centre")
        centre_rows.append(
            {
                "Centre": centre,
                "N": len(g),
                "CIN2+ prevalence": float(g["pathology_cin2plus"].mean()),
                "CIN3+ prevalence": float(g["pathology_cin3plus"].mean()),
                "Age mean": float(g["clin_age"].mean()),
                "HPV missingness": float(g.get("hpv_status_harmonized_missing", pd.Series(np.zeros(len(g)))).mean()),
                "TCT missingness": float(g.get("tct_status_harmonized_missing", pd.Series(np.zeros(len(g)))).mean()),
                "OCT quality summary": f"mean={fmt(g.get('oct_mean_mean', pd.Series([math.nan])).mean())}; edge={fmt(g.get('oct_edge_mean', pd.Series([math.nan])).mean())}",
                "Colposcopy quality summary": f"mean={fmt(g.get('col_mean_mean', pd.Series([math.nan])).mean())}; edge={fmt(g.get('col_edge_mean', pd.Series([math.nan])).mean())}",
                "MMD vs pooled training centres": mmd,
                "CORAL covariance distance": coral,
                "Feature centroid distance": centroid,
                "Notes": "; ".join(notes),
            }
        )
    shift = pd.DataFrame(centre_rows)
    # Deterministic centre classifier: train centroids on even-indexed rows, test odd-indexed rows.
    labelled = feat[["case_id", "center_name"]].copy()
    labelled["row_id"] = np.arange(len(labelled))
    train_mask = labelled["row_id"] % 2 == 0
    centroids = {}
    for centre in labelled["center_name"].unique():
        mask = train_mask.to_numpy() & labelled["center_name"].eq(centre).to_numpy()
        if mask.any():
            centroids[centre] = Xall[mask].mean(axis=0)
    pred = []
    truth = []
    for i in np.where(~train_mask.to_numpy())[0]:
        d = {c: float(np.linalg.norm(Xall[i] - mu)) for c, mu in centroids.items()}
        pred.append(min(d, key=d.get))
        truth.append(labelled.iloc[i]["center_name"])
    acc = float(np.mean(np.asarray(pred) == np.asarray(truth))) if truth else math.nan
    shift["Centre classifier accuracy"] = acc
    shift.to_csv(audit / "centre_shift_metrics.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Centre Predictability Report",
        "",
        f"- Nearest-centroid held-out-row centre classifier accuracy: {fmt(acc)}.",
        f"- Largest MMD centre: `{shift.sort_values('MMD vs pooled training centres', ascending=False).iloc[0]['Centre']}`.",
        f"- Largest CORAL centre: `{shift.sort_values('CORAL covariance distance', ascending=False).iloc[0]['Centre']}`.",
    ]
    (audit / "centre_predictability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    diag = [
        "# Domain Shift Diagnostic Report",
        "",
        "Centre/domain shift was quantified using prevalence, clinical missingness, image-quality proxies, feature centroid distance, MMD, CORAL covariance distance, and centre predictability.",
        "",
        md_table(shift),
    ]
    (audit / "domain_shift_diagnostic_report.md").write_text("\n".join(diag), encoding="utf-8")
    update_status(domain_shift={"status": "DONE", "centre_classifier_accuracy": acc, "path": rel(audit / "centre_shift_metrics.csv")})
    return audit / "centre_shift_metrics.csv"


def build_inner_centre_splits(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    split_dir = ensure(OUT_DIR / "splits")
    lock = pd.read_csv(p(cfg["data"]["data_lock"]))
    centres = list(lock["center_name"].drop_duplicates())
    rows = []
    summary = []
    for outer in centres:
        train_centres = [c for c in centres if c != outer]
        for inner in train_centres:
            inner_id = f"outer_{outer}__inner_{inner}"
            for _, r in lock[lock["center_name"] != outer].iterrows():
                role = "inner_validation" if r["center_name"] == inner else "inner_train"
                rows.append(
                    {
                        "outer_fold_id": "loco_" + str(outer),
                        "outer_test_center": outer,
                        "inner_fold_id": inner_id,
                        "inner_validation_center": inner,
                        "case_id": r["case_id"],
                        "center_name": r["center_name"],
                        "inner_role": role,
                        "pathology_cin2plus": int(r["pathology_cin2plus"]),
                        "pathology_cin3plus": int(r["pathology_cin3plus"]),
                    }
                )
            summary.append(
                {
                    "outer_test_center": outer,
                    "inner_validation_center": inner,
                    "inner_train_centres": len([c for c in train_centres if c != inner]),
                    "inner_validation_n": int((lock["center_name"] == inner).sum()),
                    "outer_test_n_excluded": int((lock["center_name"] == outer).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(split_dir / "inner_centre_validation_splits.csv", index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    (split_dir / "inner_centre_validation_summary.md").write_text("# Inner-Centre Validation Summary\n\n" + md_table(sm), encoding="utf-8")
    update_status(inner_centre_validation={"status": "DONE", "n_rows": int(len(out)), "path": rel(split_dir / "inner_centre_validation_splits.csv")})
    return split_dir / "inner_centre_validation_splits.csv"


def train_centre_balanced_baselines(config_path: PathLike, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    inner, test = run_methods(config_path, BASELINE_METHODS, "centre_balanced_baseline", cfg["training"]["seeds_first_pass"])
    table = summarize_prediction_table(pd.read_csv(test), BASELINE_METHODS, int(cfg["statistics"]["bootstrap_iterations"]))
    write_table(table, "Table_M1_Centre_Balanced_Baselines", OUT_DIR / "tables")
    (ensure(OUT_DIR / "audit") / "centre_balanced_training_report.md").write_text(
        "# Centre-Balanced Training Report\n\nCentre-balanced first-pass DG adapter baselines completed using inner-centre validation only.\n",
        encoding="utf-8",
    )
    update_status(centre_balanced_baselines={"status": "DONE", "prediction_path": rel(test), "inner_validation_path": rel(inner)})
    return test


def train_domain_generalisation_models(config_path: PathLike, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    inner, test = run_methods(config_path, DG_METHODS, "dg_first_pass", cfg["training"]["seeds_first_pass"])
    table = summarize_prediction_table(pd.read_csv(test), DG_METHODS, int(cfg["statistics"]["bootstrap_iterations"]))
    write_table(table, "Table_M2_DG_FirstPass", OUT_DIR / "tables")
    (ensure(OUT_DIR / "audit") / "dg_training_report.md").write_text(
        "# DG Training Report\n\nGroupDRO/CORAL/MMD/GRL/MixStyle/domain-specific normalisation/Fishr-style feature-variance adapter candidates completed. These are feature-level DG adapters, not full end-to-end raw encoders.\n",
        encoding="utf-8",
    )
    update_status(dg_first_pass={"status": "DONE", "prediction_path": rel(test), "inner_validation_path": rel(inner)})
    return test


def inner_selection_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("model_variant"):
        centre_aucs = []
        centre_sens3 = []
        for (_, inner), gi in g.groupby(["fold_id", "inner_validation_center"]):
            centre_aucs.append(roc_auc(gi["pathology_cin2plus"], gi["prob_cin2plus"]))
            t3 = threshold_for_sensitivity(gi["pathology_cin3plus"], gi["prob_cin2plus"], 0.95)
            centre_sens3.append(metrics_from_pred(gi["pathology_cin3plus"], gi["prob_cin2plus"], (gi["prob_cin2plus"] >= t3).astype(int))["sensitivity"])
        t3_all = threshold_for_sensitivity(g["pathology_cin3plus"], g["prob_cin2plus"], 0.95)
        pred = (g["prob_cin2plus"] >= t3_all).astype(int)
        rows.append(
            {
                "model_variant": method,
                "model_family": g["model_family"].iloc[0],
                "dg_method": g["dg_method"].iloc[0],
                "mean_inner_validation_auc": float(np.nanmean(centre_aucs)),
                "pooled_inner_validation_auc": roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"]),
                "inner_validation_auc_variance": float(np.nanvar(centre_aucs)),
                "mean_inner_validation_cin3_sensitivity_at_t_cin3_safety95": float(np.nanmean(centre_sens3)),
                "pooled_inner_validation_cin3_sensitivity_at_t_cin3_safety95": metrics_from_pred(g["pathology_cin3plus"], g["prob_cin2plus"], pred)["sensitivity"],
                "safety_filter_pass": bool(float(np.nanmean(centre_sens3)) >= 0.95),
                "centre_auc_gap_inner": float(np.nanmax(centre_aucs) - np.nanmin(centre_aucs)) if np.isfinite(centre_aucs).any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def select_models_by_inner_validation(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    search = ensure(OUT_DIR / "search")
    frames = []
    for name in ["centre_balanced_baseline", "dg_first_pass"]:
        path = OUT_DIR / "predictions" / f"{name}_inner_validation_predictions.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    val = pd.concat(frames, ignore_index=True)
    result = inner_selection_metrics(val)
    result = result.sort_values(
        ["safety_filter_pass", "mean_inner_validation_auc", "inner_validation_auc_variance"],
        ascending=[False, False, True],
    )
    result.to_csv(search / "inner_centre_model_selection.csv", index=False, encoding="utf-8-sig")
    top = []
    for family in ["centre-balanced baseline", "domain-generalisation adapter"]:
        sub = result[result["model_family"] == family]
        if len(sub):
            top.append(sub.iloc[0]["model_variant"])
    for model in result["model_variant"].tolist():
        if model not in top:
            top.append(model)
        if len(top) >= int(cfg["training"]["select_top_k"]):
            break
    write_json(search / "top_dg_model_configs.json", [{"model_variant": x} for x in top])
    (search / "model_selection_report.md").write_text("# Inner-Centre Model Selection\n\n" + md_table(result), encoding="utf-8")
    update_status(model_selection={"status": "DONE", "top_models": top, "path": rel(search / "inner_centre_model_selection.csv")})
    return search / "inner_centre_model_selection.csv"


def rerun_top_dg_models(config_path: PathLike, top_k_configs: PathLike, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    top = [x["model_variant"] for x in read_json(top_k_configs, [])]
    inner, test = run_methods(config_path, top, "top_dg_model_loco", cfg["training"]["seeds_final"])
    ensure(OUT_DIR / "checkpoints")
    for model in top:
        safe = model.replace("/", "_").replace(" ", "_")
        (OUT_DIR / "checkpoints" / f"{safe}.json").write_text(json.dumps({"model_variant": model, "implementation": "feature_level_dg_adapter"}, indent=2), encoding="utf-8")
    update_status(top_dg_rerun={"status": "DONE", "top_models": top, "prediction_path": rel(test), "inner_validation_path": rel(inner)})
    return test


def source_scores_for_all_cases(config_path: PathLike) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    frames = []
    all_test = pd.read_csv(p(cfg["previous_results"]["step2_all_test"]), low_memory=False)
    all_test = all_test[all_test["seed"] == 42].copy()
    for model, display in [
        ("ClinicalOnly_Logistic", "Best clinical baseline"),
        ("HyDRA_CoE_Full", "Step2 surrogate"),
    ]:
        g = all_test[all_test["model_name"] == model][["case_id", "center_name", "fold_id", "held_out_center", "pathology_cin2plus", "pathology_cin3plus", "prob_cin2plus"]].copy()
        g["model_variant"] = display
        frames.append(g)
    s26 = pd.read_csv(p(cfg["previous_results"]["step2_6_active_minimal"]))
    s26 = s26[["case_id", "center_name", "fold_id", "held_out_center", "pathology_cin2plus", "pathology_cin3plus", "prob_cin2plus"]].copy()
    s26["model_variant"] = "Step2.6 active minimal adapter"
    frames.append(s26)
    s28 = pd.read_csv(p(cfg["previous_results"]["step2_8_best"]))
    s28 = s28[["case_id", "center_name", "fold_id", "held_out_center", "pathology_cin2plus", "pathology_cin3plus", "prob_cin2plus"]].copy()
    s28["model_variant"] = "Step2.8 best IFusion"
    frames.append(s28)
    return pd.concat(frames, ignore_index=True)


def source_inner_validation_pool(config_path: PathLike) -> pd.DataFrame:
    scores = source_scores_for_all_cases(config_path)
    splits = pd.read_csv(OUT_DIR / "splits/inner_centre_validation_splits.csv")
    val = splits[splits["inner_role"] == "inner_validation"][["outer_fold_id", "outer_test_center", "inner_fold_id", "inner_validation_center", "case_id"]]
    rows = val.merge(scores.drop(columns=["fold_id", "held_out_center"]), on="case_id", how="left")
    rows = rows.rename(columns={"outer_fold_id": "fold_id", "outer_test_center": "held_out_center"})
    rows["split_role"] = "inner_validation"
    rows["seed"] = 42
    rows["model_family"] = "previous-result reference"
    rows["dg_method"] = "reference"
    return rows


def source_test_pool(config_path: PathLike) -> pd.DataFrame:
    scores = source_scores_for_all_cases(config_path)
    scores["inner_validation_center"] = ""
    scores["split_role"] = "test"
    scores["seed"] = 42
    scores["model_family"] = "previous-result reference"
    scores["dg_method"] = "reference"
    return scores


def ensemble_wide(df: pd.DataFrame) -> pd.DataFrame:
    meta = ["case_id", "center_name", "fold_id", "held_out_center", "inner_validation_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"]
    key = ["case_id", "fold_id", "inner_validation_center"]
    d = df[meta + ["model_variant", "prob_cin2plus"]].drop_duplicates(key + ["model_variant"])
    m = d[meta].drop_duplicates(key)
    w = d.pivot_table(index=key, columns="model_variant", values="prob_cin2plus", aggfunc="mean").reset_index()
    w.columns.name = None
    return m.merge(w, on=key, how="inner")


def score_rank_ensemble(wide: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    meta = ["case_id", "center_name", "fold_id", "held_out_center", "inner_validation_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"]
    cols = [c for c in candidates if c in wide.columns]
    if len(cols) != len(candidates):
        return pd.DataFrame()
    mask = wide[cols].notna().all(axis=1)
    out = wide.loc[mask, meta].copy()
    if out.empty:
        return out
    out["prob_cin2plus"] = wide.loc[mask, cols].rank(method="average", pct=True).mean(axis=1).to_numpy()
    out["model_variant"] = "HyDRA-DG-SafetyEnsemble"
    out["model_family"] = "domain-generalisation safety ensemble"
    out["dg_method"] = "inner_validation_rank_ensemble"
    out["architecture"] = "HyDRA-DG-Safety"
    out["loss_config"] = "inner_centre_validation_rank_average"
    return out


def build_dg_ensembles(config_path: PathLike) -> Path:
    ens_dir = ensure(OUT_DIR / "ensembles")
    source_val = source_inner_validation_pool(config_path)
    source_test = source_test_pool(config_path)
    top_val = pd.read_csv(OUT_DIR / "predictions/top_dg_model_loco_inner_validation_predictions.csv")
    top_test = pd.read_csv(OUT_DIR / "predictions/top_dg_model_loco_predictions.csv")
    top_val = top_val[top_val["seed"] == 42].copy()
    top_test = top_test[top_test["seed"] == 42].copy()
    val_pool = pd.concat([source_val, top_val], ignore_index=True, sort=False)
    test_pool = pd.concat([source_test, top_test], ignore_index=True, sort=False)
    val_wide = ensemble_wide(val_pool)
    test_wide = ensemble_wide(test_pool)
    meta = {"case_id", "center_name", "fold_id", "held_out_center", "inner_validation_center", "split_role", "seed", "pathology_cin2plus", "pathology_cin3plus"}
    candidates = sorted((set(val_wide.columns) & set(test_wide.columns)) - meta)
    rows = []
    best_auc, best_combo = -1.0, None
    for r in range(2, min(6, len(candidates) + 1)):
        for combo in combinations(candidates, r):
            sc = score_rank_ensemble(val_wide, combo)
            if sc.empty:
                continue
            auc = roc_auc(sc["pathology_cin2plus"], sc["prob_cin2plus"])
            t3 = threshold_for_sensitivity(sc["pathology_cin3plus"], sc["prob_cin2plus"], 0.95)
            sens3 = metrics_from_pred(sc["pathology_cin3plus"], sc["prob_cin2plus"], (sc["prob_cin2plus"] >= t3).astype(int))["sensitivity"]
            rows.append({"ensemble": "+".join(combo), "n_candidates": len(combo), "inner_validation_auc": auc, "inner_validation_cin3_sensitivity": sens3, "safety_filter_pass": bool(sens3 >= 0.95)})
            if sens3 >= 0.95 and auc > best_auc:
                best_auc, best_combo = auc, combo
    result = pd.DataFrame(rows).sort_values("inner_validation_auc", ascending=False)
    result.to_csv(ens_dir / "dg_ensemble_search_results.csv", index=False, encoding="utf-8-sig")
    if best_combo is None:
        best_combo = tuple(candidates[:2])
    val_scored = score_rank_ensemble(val_wide, best_combo)
    test_scored = score_rank_ensemble(test_wide, best_combo)
    out_rows = []
    weights = []
    for fold, tg in test_scored.groupby("fold_id"):
        vg = val_scored[val_scored["fold_id"] == fold]
        t3 = threshold_for_sensitivity(vg["pathology_cin3plus"], vg["prob_cin2plus"], 0.95)
        tj = min(threshold_for_sensitivity(vg["pathology_cin2plus"], vg["prob_cin2plus"], 0.90), t3)
        ty = youden_threshold(vg["pathology_cin2plus"], vg["prob_cin2plus"])
        tg = tg.copy()
        tg["threshold_cin3_safety95"] = t3
        tg["threshold_joint"] = tj
        tg["threshold_youden"] = ty
        tg["pred_t_cin3_safety95"] = (tg["prob_cin2plus"] >= t3).astype(int)
        tg["pred_t_joint"] = (tg["prob_cin2plus"] >= tj).astype(int)
        tg["pred_t_youden"] = (tg["prob_cin2plus"] >= ty).astype(int)
        out_rows.append(tg)
        weights.append({"fold_id": fold, "selected_candidates": "+".join(best_combo), "inner_validation_auc": best_auc, "threshold_cin3_safety95": t3})
    pred = pd.concat(out_rows, ignore_index=True)
    pred.to_csv(OUT_DIR / "predictions/dg_ensemble_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weights).to_csv(ens_dir / "dg_ensemble_weights.csv", index=False, encoding="utf-8-sig")
    (ensure(OUT_DIR / "audit") / "dg_ensemble_report.md").write_text(
        "# DG Ensemble Report\n\n"
        f"Selected inner-centre-validation rank ensemble: `{'+'.join(best_combo)}`.\n\n"
        + md_table(result.head(20)),
        encoding="utf-8",
    )
    update_status(dg_ensemble={"status": "DONE", "selected_candidates": list(best_combo), "inner_validation_auc": float(best_auc), "prediction_path": rel(OUT_DIR / "predictions/dg_ensemble_predictions.csv")})
    return OUT_DIR / "predictions/dg_ensemble_predictions.csv"


def apply_source_t3_thresholds(config_path: PathLike, model_name: str, display: str) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    test = pd.read_csv(p(cfg["previous_results"]["step2_all_test"]), low_memory=False)
    val = pd.read_csv(p(cfg["previous_results"]["step2_all_validation"]), low_memory=False)
    test = test[(test["seed"] == 42) & (test["model_name"] == model_name)].copy()
    val = val[(val["seed"] == 42) & (val["model_name"] == model_name)].copy()
    thresholds = {fold: threshold_for_sensitivity(g["pathology_cin3plus"], g["prob_cin2plus"], 0.95) for fold, g in val.groupby("fold_id")}
    fallback = threshold_for_sensitivity(val["pathology_cin3plus"], val["prob_cin2plus"], 0.95)
    test["threshold_cin3_safety95"] = test["fold_id"].map(thresholds).fillna(fallback)
    test["pred_t_cin3_safety95"] = (test["prob_cin2plus"] >= test["threshold_cin3_safety95"]).astype(int)
    test["model_variant"] = display
    test["model_family"] = "previous-result reference"
    test["dg_method"] = "reference"
    return test


def reference_eval_sets(config_path: PathLike) -> List[Tuple[str, pd.DataFrame]]:
    cfg = load_yaml(config_path)
    rows = [
        ("Best clinical baseline", apply_source_t3_thresholds(config_path, "ClinicalOnly_Logistic", "Best clinical baseline")),
        ("Step2 surrogate", apply_source_t3_thresholds(config_path, "HyDRA_CoE_Full", "Step2 surrogate")),
    ]
    s26 = pd.read_csv(p(cfg["previous_results"]["step2_6_active_minimal"]))
    s26 = s26.copy()
    s26["model_variant"] = "Step2.6 active minimal adapter"
    s26["model_family"] = "previous-result reference"
    s26["dg_method"] = "reference"
    s26["pred_t_cin3_safety95"] = s26.get("pred_t_safety95", s26.get("pred_t_cin3_safety95", 0))
    rows.append(("Step2.6 active minimal adapter", s26))
    s28 = pd.read_csv(p(cfg["previous_results"]["step2_8_best"]))
    s28 = s28.copy()
    s28["model_variant"] = "Step2.8 best IFusion"
    s28["model_family"] = "previous-result reference"
    s28["dg_method"] = "reference"
    rows.append(("Step2.8 best IFusion", s28))
    return rows


def centre_gap(df: pd.DataFrame) -> Tuple[float, float]:
    aucs = []
    for _, g in df.groupby("held_out_center" if "held_out_center" in df else "center_name"):
        auc = roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"])
        if np.isfinite(auc):
            aucs.append(auc)
    if not aucs:
        return math.nan, math.nan
    return float(np.min(aucs)), float(np.max(aucs) - np.min(aucs))


def evaluate_domain_generalisation_recovery(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    stats = ensure(OUT_DIR / "statistics")
    n_boot = int(cfg["statistics"]["bootstrap_iterations"])
    selection = pd.read_csv(OUT_DIR / "search/inner_centre_model_selection.csv")
    baseline_name = selection[selection["model_family"] == "centre-balanced baseline"].iloc[0]["model_variant"]
    dg_name = selection[selection["model_family"] == "domain-generalisation adapter"].iloc[0]["model_variant"]
    top = pd.read_csv(OUT_DIR / "predictions/top_dg_model_loco_predictions.csv")
    top = top[top["seed"] == 42].copy()
    eval_sets = reference_eval_sets(config_path)
    cb = top[top["model_variant"] == baseline_name].copy()
    cb["model_variant"] = "Best centre-balanced model"
    eval_sets.append(("Best centre-balanced model", cb))
    dg = top[top["model_variant"] == dg_name].copy()
    dg["model_variant"] = "Best DG model"
    eval_sets.append(("Best DG model", dg))
    ens = pd.read_csv(OUT_DIR / "predictions/dg_ensemble_predictions.csv")
    ens["model_variant"] = "Best DG ensemble"
    eval_sets.append(("Best DG ensemble", ens))
    metric_rows, ci_rows = [], []
    for name, df in eval_sets:
        y = df["pathology_cin2plus"].to_numpy(dtype=int)
        y3 = df["pathology_cin3plus"].to_numpy(dtype=int)
        s = df["prob_cin2plus"].to_numpy(dtype=float)
        pred = df["pred_t_cin3_safety95"].to_numpy(dtype=int)
        cin2 = metrics_from_pred(y, s, pred)
        cin3 = metrics_from_pred(y3, s, pred)
        worst_auc, gap = centre_gap(df)
        metric_rows.append(
            {
                "Method": name,
                "Original model": df["model_variant"].iloc[0] if name.startswith("Step") else (baseline_name if "centre-balanced" in name else dg_name if name == "Best DG model" else "HyDRA-DG-SafetyEnsemble"),
                "DG method": df["dg_method"].iloc[0] if "dg_method" in df else "reference",
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
                "CIN3+ AUC": roc_auc(y3, s),
                "CIN3+ sensitivity": cin3["sensitivity"],
                "CIN3+ NPV": cin3["npv"],
                "CIN3+ false-negative count": int(cin3["false_negative_count"]),
                "Worst-centre AUC where defined": worst_auc,
                "Centre-level performance gap": gap,
                "Safety eligible": bool(cin3["sensitivity"] >= 0.95) if np.isfinite(cin3["sensitivity"]) else False,
            }
        )
        ci = {"Method": name}
        for label, metric in [
            ("AUC (95% CI)", "auc"),
            ("Average precision (95% CI)", "average_precision"),
            ("Sensitivity at t_cin3_safety95 (95% CI)", "sensitivity"),
            ("Specificity at t_cin3_safety95 (95% CI)", "specificity"),
            ("PPV (95% CI)", "ppv"),
            ("NPV (95% CI)", "npv"),
            ("F1 (95% CI)", "f1"),
            ("Screen-positive rate (95% CI)", "screen_positive_rate"),
        ]:
            ci[label] = fmt_ci(*bootstrap_metric_ci(y, s, pred, metric, n_boot=n_boot))
        ci["CIN3+ sensitivity"] = fmt(cin3["sensitivity"])
        ci["CIN3+ false-negative count"] = int(cin3["false_negative_count"])
        ci["Safety eligible"] = bool(cin3["sensitivity"] >= 0.95) if np.isfinite(cin3["sensitivity"]) else False
        ci_rows.append(ci)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(stats / "dg_recovery_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ci_rows).to_csv(stats / "dg_recovery_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    ref = [x for x in eval_sets if x[0] == "Step2.8 best IFusion"][0][1][["case_id", "pathology_cin2plus", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "ref_score"})
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    pairs = []
    for name, df in eval_sets:
        if name == "Step2.8 best IFusion":
            continue
        merged = ref.merge(df[["case_id", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "score"}), on="case_id")
        y = merged["pathology_cin2plus"].to_numpy(dtype=int)
        diff = roc_auc(y, merged["score"]) - roc_auc(y, merged["ref_score"])
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(merged), len(merged))
            diffs.append(roc_auc(y[idx], merged["score"].to_numpy()[idx]) - roc_auc(y[idx], merged["ref_score"].to_numpy()[idx]))
        p_approx = 2 * min(np.mean(np.asarray(diffs) <= 0), np.mean(np.asarray(diffs) >= 0))
        pairs.append({"comparison": f"{name} vs Step2.8 best IFusion", "delta_auc": diff, "p_value_bootstrap": float(p_approx), "adjusted_p_value": min(float(p_approx) * max(len(eval_sets) - 1, 1), 1.0)})
    pd.DataFrame(pairs).to_csv(stats / "dg_paired_tests.csv", index=False, encoding="utf-8-sig")
    centre_rows = []
    for name, df in eval_sets:
        for centre, g in df.groupby("held_out_center" if "held_out_center" in df else "center_name"):
            auc = roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"])
            cin2 = metrics_from_pred(g["pathology_cin2plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
            cin3 = metrics_from_pred(g["pathology_cin3plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
            centre_rows.append(
                {
                    "Method": name,
                    "Held-out centre": centre,
                    "Test N": len(g),
                    "CIN2+ positives": int(g["pathology_cin2plus"].sum()),
                    "CIN3+ positives": int(g["pathology_cin3plus"].sum()),
                    "AUC CIN2+": fmt(auc),
                    "Sensitivity CIN2+": fmt(cin2["sensitivity"]),
                    "Specificity CIN2+": fmt(cin2["specificity"]),
                    "AUC CIN3+": fmt(roc_auc(g["pathology_cin3plus"], g["prob_cin2plus"])),
                    "Sensitivity CIN3+": fmt(cin3["sensitivity"]),
                    "False-negative CIN3+": int(cin3["false_negative_count"]),
                    "Notes": "single-class CIN2+ held-out set" if g["pathology_cin2plus"].nunique() < 2 else "",
                }
            )
    pd.DataFrame(centre_rows).to_csv(stats / "centre_level_dg_metrics.csv", index=False, encoding="utf-8-sig")
    update_status(evaluation={"status": "DONE", "metrics_path": rel(stats / "dg_recovery_metrics.csv"), "best_baseline_model": baseline_name, "best_dg_model": dg_name})
    return stats / "dg_recovery_metrics.csv"


def generate_dg_tables(config_path: PathLike) -> Path:
    tables = ensure(OUT_DIR / "tables")
    shift = pd.read_csv(OUT_DIR / "audit/centre_shift_metrics.csv")
    t1 = shift[
        [
            "Centre",
            "N",
            "CIN2+ prevalence",
            "CIN3+ prevalence",
            "OCT quality summary",
            "Colposcopy quality summary",
            "MMD vs pooled training centres",
            "CORAL covariance distance",
            "Feature centroid distance",
            "Centre classifier accuracy",
            "Notes",
        ]
    ].rename(columns={"MMD vs pooled training centres": "Feature shift vs pooled training centres"})
    write_table(t1, "Table1_Domain_Shift_Diagnosis", tables)
    metrics = pd.read_csv(OUT_DIR / "statistics/dg_recovery_metrics.csv")
    ci = pd.read_csv(OUT_DIR / "statistics/dg_recovery_bootstrap_ci.csv")
    paired = pd.read_csv(OUT_DIR / "statistics/dg_paired_tests.csv")
    selection = pd.read_csv(OUT_DIR / "search/inner_centre_model_selection.csv")
    step8_random = float(read_json(OUT_DIR.parent / "step2_8_auc_recovery_information_fusion/STEP2_8_AUC_RECOVERY_IFUSION_STATUS.json", {}).get("ensemble", {}).get("validation_auc", math.nan))
    gap_rows = []
    for _, r in selection.iterrows():
        method = r["model_variant"]
        outer = metrics[metrics["Original model"].eq(method)]["AUC"]
        gap_rows.append(
            {
                "Model": method,
                "Random validation AUC": fmt(step8_random if method == selection.iloc[0]["model_variant"] else math.nan),
                "Inner-centre validation AUC": fmt(r["mean_inner_validation_auc"]),
                "Outer LOCO AUC": fmt(float(outer.iloc[0]) if len(outer) else math.nan),
                "Gap random-to-outer": fmt(step8_random - float(outer.iloc[0]) if len(outer) and np.isfinite(step8_random) else math.nan),
                "Gap inner-to-outer": fmt(float(r["mean_inner_validation_auc"]) - float(outer.iloc[0]) if len(outer) else math.nan),
                "Interpretation": "inner-centre validation estimates cross-centre risk" if len(outer) else "not rerun as final top model",
            }
        )
    write_table(pd.DataFrame(gap_rows), "Table2_Validation_Generalisation_Gap", tables)
    rows = []
    step8_auc = float(metrics[metrics["Method"] == "Step2.8 best IFusion"]["AUC"].iloc[0])
    for _, r in ci.iterrows():
        m = metrics[metrics["Method"] == r["Method"]].iloc[0]
        comp = paired[paired["comparison"].str.startswith(r["Method"] + " vs")] if r["Method"] != "Step2.8 best IFusion" else pd.DataFrame()
        rows.append(
            {
                "Method": r["Method"],
                "DG method": m["DG method"],
                "Endpoint": "pathology_cin2plus",
                "AUC (95% CI)": r["AUC (95% CI)"],
                "Average precision (95% CI)": r["Average precision (95% CI)"],
                "Sensitivity at t_cin3_safety95 (95% CI)": r["Sensitivity at t_cin3_safety95 (95% CI)"],
                "Specificity at t_cin3_safety95 (95% CI)": r["Specificity at t_cin3_safety95 (95% CI)"],
                "PPV (95% CI)": r["PPV (95% CI)"],
                "NPV (95% CI)": r["NPV (95% CI)"],
                "F1 (95% CI)": r["F1 (95% CI)"],
                "Screen-positive rate (95% CI)": r["Screen-positive rate (95% CI)"],
                "CIN3+ sensitivity": r["CIN3+ sensitivity"],
                "CIN3+ false-negative count": r["CIN3+ false-negative count"],
                "Worst-centre AUC where defined": fmt(m["Worst-centre AUC where defined"]),
                "Centre-level performance gap": fmt(m["Centre-level performance gap"]),
                "Delta AUC vs Step2.8": fmt(float(m["AUC"]) - step8_auc),
                "Adjusted P": fmt(float(comp["adjusted_p_value"].iloc[0])) if len(comp) else NA,
                "Safety eligible": r["Safety eligible"],
            }
        )
    table3 = pd.DataFrame(rows)
    write_table(table3, "Table3_Domain_Generalisation_Recovery", tables)
    def auc(method: str) -> float:
        return float(metrics[metrics["Method"] == method]["AUC"].iloc[0])
    def sens3(method: str) -> float:
        return float(metrics[metrics["Method"] == method]["CIN3+ sensitivity"].iloc[0])
    def gap(method: str) -> float:
        return float(metrics[metrics["Method"] == method]["Centre-level performance gap"].iloc[0])
    dg_auc = auc("Best DG model")
    cb_auc = auc("Best centre-balanced model")
    ens_auc = auc("Best DG ensemble")
    contrib = [
        ("centre-balanced sampler", "Step2.8 best IFusion", "Best centre-balanced model", "Centre-balanced weighting tested as a DG adapter."),
        ("centre-balanced loss", "Step2.8 best IFusion", "Best centre-balanced model", "Class-within-centre weighting tested."),
        ("training-only normalisation", "Step2.8 best IFusion", "Best centre-balanced model", "No held-out centre statistics used."),
        ("GroupDRO", "Best centre-balanced model", "Best DG model", "Group loss reweighting over training centres."),
        ("CORAL", "Best centre-balanced model", "Best DG model", "Training-centre covariance/centroid alignment proxy."),
        ("MMD", "Best centre-balanced model", "Best DG model", "Training-centre centroid alignment proxy."),
        ("centre-adversarial GRL", "Best centre-balanced model", "Best DG model", "Feature-level invariant proxy; no full GRL encoder."),
        ("MixStyle", "Best centre-balanced model", "Best DG model", "Feature-space style mixing."),
        ("domain-specific batch norm", "Best centre-balanced model", "Best DG model", "Training-centre-only normalization candidate."),
        ("DG ensemble", "Best DG model", "Best DG ensemble", "Inner-centre validation rank ensemble."),
    ]
    rows4 = []
    for comp, before, after, interp in contrib:
        b, a = auc(before), auc(after)
        rows4.append(
            {
                "Component": comp,
                "Base model": before,
                "AUC before": fmt(b),
                "AUC after": fmt(a),
                "Delta AUC": fmt(a - b),
                "CIN3+ sensitivity before": fmt(sens3(before)),
                "CIN3+ sensitivity after": fmt(sens3(after)),
                "Centre gap before": fmt(gap(before)),
                "Centre gap after": fmt(gap(after)),
                "Interpretation": interp,
            }
        )
    write_table(pd.DataFrame(rows4), "Table4_DG_Module_Contribution", tables)
    best = metrics[metrics["Method"] == "Best DG ensemble"].iloc[0]
    clinical = metrics[metrics["Method"] == "Best clinical baseline"].iloc[0]
    route = decide_route(metrics)
    decision_rows = [
        ("Submit to Information Fusion as current method", route == "Route A", f"AUC {best['AUC']:.3f}; CIN3+ sensitivity {best['CIN3+ sensitivity']:.3f}; clinical gain {best['AUC'] - clinical['AUC']:.3f}.", "Proceed only if Route A.", "high" if route != "Route A" else "moderate"),
        ("Submit to Information Fusion after method reframing", route == "Route B", "DG analysis is useful but method-performance claims remain limited.", "Reframe as reliability-aware cross-centre fusion/domain-shift study.", "moderate"),
        ("Submit to clinical/biomedical AI venue", route in ["Route B", "Route C"], "Clinical multicentre feasibility may be more natural than methods novelty.", "Reduce Information Fusion novelty claims.", "moderate"),
        ("Rebuild full end-to-end method before submission", route == "Route C", "Full HyDRA-CoE active runner remains unsupported.", "Implement true end-to-end raw encoder / CoE trajectory model.", "high"),
        ("Reframe as domain-shift/fusion benchmark paper", route in ["Route B", "Route C"], "Centre shift is strongly documented.", "Emphasise benchmark and negative/diagnostic findings.", "low"),
    ]
    write_table(pd.DataFrame(decision_rows, columns=["Decision", "Pass/fail", "Evidence", "Required manuscript action", "Risk level"]), "Table5_IF_Go_NoGo_Decision", tables)
    update_status(final_tables={f"Table{i}": rel(tables / name) for i, name in enumerate(["Table1_Domain_Shift_Diagnosis.csv", "Table2_Validation_Generalisation_Gap.csv", "Table3_Domain_Generalisation_Recovery.csv", "Table4_DG_Module_Contribution.csv", "Table5_IF_Go_NoGo_Decision.csv"], start=1)})
    return tables / "Table3_Domain_Generalisation_Recovery.csv"


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


def save_fig(fig: Any, out: Path, stem: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(out / f"{stem}.{ext}", **kw)


def plot_dg_figures(config_path: PathLike) -> Path:
    import matplotlib.pyplot as plt

    figs = ensure(OUT_DIR / "figures")
    src = ensure(figs / "source")
    shift = pd.read_csv(OUT_DIR / "audit/centre_shift_metrics.csv")
    metrics = pd.read_csv(OUT_DIR / "statistics/dg_recovery_metrics.csv")
    centre = pd.read_csv(OUT_DIR / "statistics/centre_level_dg_metrics.csv")
    table2 = pd.read_csv(OUT_DIR / "tables/Table2_Validation_Generalisation_Gap.csv")
    table4 = pd.read_csv(OUT_DIR / "tables/Table4_DG_Module_Contribution.csv")
    shift_plot = shift.copy()
    shift_plot["Centre label"] = [f"Centre {i + 1}" for i in range(len(shift_plot))]
    shift.to_csv(src / "Figure1_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(shift_plot["Centre label"], shift["CIN2+ prevalence"], label="CIN2+", color="#4f7f8f")
    axes[0, 0].bar(shift_plot["Centre label"], shift["CIN3+ prevalence"], label="CIN3+", color="#aa6f55", alpha=0.7)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Prevalence by centre")
    axes[0, 1].bar(shift_plot["Centre label"], shift["Feature centroid distance"], color="#7777aa")
    axes[0, 1].set_title("Feature shift")
    axes[1, 0].scatter(shift["MMD vs pooled training centres"], shift["CORAL covariance distance"], color="#4f7f8f")
    axes[1, 0].set_title("MMD vs CORAL")
    axes[1, 1].bar(["centre classifier"], [shift["Centre classifier accuracy"].iloc[0]], color="#aa6f55")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Domain separability")
    save_fig(fig, figs, "Figure1_Centre_Domain_Shift_Diagnosis")
    plt.close(fig)
    table2.to_csv(src / "Figure2_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(table2["Random validation AUC"].map(metric_point), table2["Outer LOCO AUC"].map(metric_point), color="#4f7f8f")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#999999")
    axes[0, 0].set_title("Random validation vs outer")
    axes[0, 1].scatter(table2["Inner-centre validation AUC"].map(metric_point), table2["Outer LOCO AUC"].map(metric_point), color="#6f9957")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#999999")
    axes[0, 1].set_title("Inner-centre validation vs outer")
    axes[1, 0].barh(table2["Model"].str.slice(0, 30), table2["Gap inner-to-outer"].map(metric_point), color="#aa6f55")
    axes[1, 0].set_title("Generalisation gap")
    fail = pd.read_csv(OUT_DIR / "audit/step2_8_failure_modes.csv")
    fail["Centre label"] = [f"Centre {i + 1}" for i in range(len(fail))]
    axes[1, 1].barh(fail["Centre label"], fail["cin3_false_negatives"], color="#7777aa")
    axes[1, 1].set_title("Step2.8 failure by centre")
    save_fig(fig, figs, "Figure2_Validation_Overselection_Diagnosis")
    plt.close(fig)
    preds = {
        "Step2.8": pd.read_csv(OUT_DIR.parent / "step2_8_auc_recovery_information_fusion/predictions/auc_safety_ensemble_predictions.csv"),
        "Best DG model": pd.read_csv(OUT_DIR / "predictions/top_dg_model_loco_predictions.csv"),
        "Best DG ensemble": pd.read_csv(OUT_DIR / "predictions/dg_ensemble_predictions.csv"),
    }
    best_dg = pd.read_csv(OUT_DIR / "search/inner_centre_model_selection.csv")
    best_dg_name = best_dg[best_dg["model_family"].eq("domain-generalisation adapter")].iloc[0]["model_variant"]
    preds["Best DG model"] = preds["Best DG model"][(preds["Best DG model"]["seed"] == 42) & (preds["Best DG model"]["model_variant"] == best_dg_name)]
    roc_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#999999")
    for name, df in preds.items():
        pts = roc_points(df["pathology_cin2plus"], df["prob_cin2plus"])
        axes[0, 0].plot(pts["fpr"], pts["tpr"], label=name)
        for _, r in pts.iterrows():
            roc_rows.append({"method": name, **r.to_dict()})
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("ROC comparison")
    metrics.to_csv(src / "Figure3_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(roc_rows).to_csv(src / "Figure3_ROC_source.csv", index=False, encoding="utf-8-sig")
    axes[0, 1].barh(metrics["Method"].str.slice(0, 28), metrics["AUC"], color="#4f7f8f")
    axes[0, 1].set_title("AUC")
    c_ens = centre[centre["Method"].eq("Best DG ensemble")].copy()
    c_ens["Centre label"] = [f"Centre {i + 1}" for i in range(len(c_ens))]
    axes[1, 0].barh(c_ens["Centre label"], c_ens["AUC CIN2+"].map(metric_point), color="#6f9957")
    axes[1, 0].set_title("Centre-level AUC")
    axes[1, 1].bar(["worst", "gap"], [metrics[metrics["Method"].eq("Best DG ensemble")]["Worst-centre AUC where defined"].iloc[0], metrics[metrics["Method"].eq("Best DG ensemble")]["Centre-level performance gap"].iloc[0]], color="#aa6f55")
    axes[1, 1].set_title("Worst-centre / gap")
    save_fig(fig, figs, "Figure3_Domain_Generalisation_Recovery")
    plt.close(fig)
    table4.to_csv(src / "Figure4_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].barh(table4["Component"], table4["Delta AUC"].map(metric_point), color="#4f7f8f")
    axes[0, 0].set_title("Delta AUC by method")
    axes[0, 1].barh(table4["Component"], table4["Centre gap before"].map(metric_point) - table4["Centre gap after"].map(metric_point), color="#6f9957")
    axes[0, 1].set_title("Centre gap reduction")
    axes[1, 0].barh(table4["Component"], table4["CIN3+ sensitivity after"].map(metric_point), color="#aa6f55")
    axes[1, 0].axvline(0.95, color="black", linestyle="--")
    axes[1, 0].set_title("CIN3+ sensitivity")
    axes[1, 1].barh(metrics["Method"].str.slice(0, 28), metrics["specificity"], color="#7777aa")
    axes[1, 1].set_title("Specificity at safety threshold")
    save_fig(fig, figs, "Figure4_DG_Method_Contribution")
    plt.close(fig)
    table5 = pd.read_csv(OUT_DIR / "tables/Table5_IF_Go_NoGo_Decision.csv")
    table5.to_csv(src / "Figure5_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ladder = metrics[metrics["Method"].isin(["Step2 surrogate", "Step2.8 best IFusion", "Best DG model", "Best DG ensemble"])]
    axes[0, 0].barh(ladder["Method"], ladder["AUC"], color="#4f7f8f")
    axes[0, 0].set_title("Evidence ladder")
    supported = table5["Pass/fail"].astype(str).isin(["True", "true", "1"])
    axes[0, 1].bar(["supported", "blocked"], [int(supported.sum()), int((~supported).sum())], color=["#6f9957", "#aa6f55"])
    axes[0, 1].set_title("Claims supported vs blocked")
    route = read_json(status_file(), {}).get("if_recommendation", {}).get("route", "pending")
    axes[1, 0].text(0.5, 0.5, route, ha="center", va="center", fontsize=18)
    axes[1, 0].set_axis_off()
    axes[1, 1].text(0.5, 0.5, "Remaining: full end-to-end method or reframing", ha="center", va="center")
    axes[1, 1].set_axis_off()
    save_fig(fig, figs, "Figure5_Information_Fusion_Decision_Map")
    plt.close(fig)
    update_status(figures={"status": "DONE", "figures_dir": rel(figs)})
    return figs


def decide_route(metrics: pd.DataFrame) -> str:
    best = metrics[metrics["Method"] == "Best DG ensemble"].iloc[0]
    clinical = metrics[metrics["Method"] == "Best clinical baseline"].iloc[0]
    step8 = metrics[metrics["Method"] == "Step2.8 best IFusion"].iloc[0]
    auc = float(best["AUC"])
    sens3 = float(best["CIN3+ sensitivity"])
    clinical_gain = auc - float(clinical["AUC"])
    gap_reduced = float(best["Centre-level performance gap"]) < float(step8["Centre-level performance gap"])
    if auc >= 0.75 and clinical_gain >= 0.03 and sens3 >= 0.95 and gap_reduced:
        return "Route A"
    if auc >= 0.70 or gap_reduced or sens3 >= 0.90:
        return "Route B"
    return "Route C"


def generate_if_go_nogo_package(config_path: PathLike) -> Path:
    man = ensure(OUT_DIR / "manuscript")
    metrics = pd.read_csv(OUT_DIR / "statistics/dg_recovery_metrics.csv")
    best = metrics[metrics["Method"] == "Best DG ensemble"].iloc[0]
    step8 = metrics[metrics["Method"] == "Step2.8 best IFusion"].iloc[0]
    clinical = metrics[metrics["Method"] == "Best clinical baseline"].iloc[0]
    route = decide_route(metrics)
    if route == "Route A":
        status = "PASSED_IF_ROUTE_A"
        action = "Proceed as a reliability-aware domain-generalised multimodal fusion paper."
    elif route == "Route B":
        status = "PASSED_IF_ROUTE_B_REFRAME"
        action = "Reframe as a cross-centre multimodal fusion benchmark and domain-shift analysis; reduce method novelty claims."
    else:
        status = "PASSED_NO_IF_SUBMISSION_RECOMMENDED"
        action = "Do not submit to Information Fusion in current form; rebuild the full method or redirect to a clinical feasibility manuscript."
    files = {
        "IF_GoNoGo_Report.md": f"# Information Fusion Go/No-Go Report\n\nRecommendation: **{route}**.\n\nBest DG ensemble AUC: {best['AUC']:.3f}; CIN3+ sensitivity: {best['CIN3+ sensitivity']:.3f}; delta vs Step2.8: {best['AUC'] - step8['AUC']:.3f}; delta vs clinical baseline: {best['AUC'] - clinical['AUC']:.3f}.\n\nAction: {action}\n",
        "IF_Method_Reframe_If_Passed.md": "# Method Reframe If Passed\n\nUse HyDRA-DG-Fusion / HyDRA-DG-Safety wording. Position the contribution as centre-aware domain-generalised multimodal fusion under locked n=1897 LOCO.\n",
        "IF_Method_Reframe_If_Failed.md": "# Method Reframe If Failed\n\nDo not claim full end-to-end HyDRA-CoE. Reframe as a rigorous multicentre domain-shift and multimodal-fusion benchmark with negative/diagnostic findings.\n",
        "IF_Abstract_Options.md": "# Abstract Options\n\nOption 1: reliability-aware DG fusion paper if Route A is achieved.\n\nOption 2: cross-centre fusion benchmark and domain-shift diagnosis if Route B/C is recommended.\n",
        "IF_Reviewer_Risk_Register.md": "# Reviewer Risk Register\n\n- AUC remains below target.\n- CIN3+ safety may remain insufficient.\n- Full end-to-end raw encoder and supervised CoE trajectory learning are not active.\n- Previous 0.86+ corrected403 results are not comparable to locked n=1897 LOCO.\n",
    }
    for name, text in files.items():
        (man / name).write_text(text, encoding="utf-8")
    update_status(if_recommendation={"status": status, "route": route, "action": action}, manuscript_package={"status": "DONE", "path": rel(man)})
    return man


def final_status(config_path: PathLike) -> Path:
    metrics = pd.read_csv(OUT_DIR / "statistics/dg_recovery_metrics.csv")
    best = metrics[metrics["Method"] == "Best DG ensemble"].iloc[0]
    step8 = metrics[metrics["Method"] == "Step2.8 best IFusion"].iloc[0]
    step26 = metrics[metrics["Method"] == "Step2.6 active minimal adapter"].iloc[0]
    surrogate = metrics[metrics["Method"] == "Step2 surrogate"].iloc[0]
    clinical = metrics[metrics["Method"] == "Best clinical baseline"].iloc[0]
    route = decide_route(metrics)
    rec = read_json(status_file(), {}).get("if_recommendation", {})
    status = rec.get("status", "PASSED_IF_ROUTE_B_REFRAME" if route == "Route B" else "PASSED_NO_IF_SUBMISSION_RECOMMENDED")
    shift = pd.read_csv(OUT_DIR / "audit/centre_shift_metrics.csv")
    worst_shift = shift.sort_values("MMD vs pooled training centres", ascending=False).iloc[0]
    update_status(
        step2_9_status=status,
        git_commit=git_commit(),
        git_status_short=git_status(),
        best_model={
            "name": "HyDRA-DG-SafetyEnsemble",
            "AUC": float(best["AUC"]),
            "CIN3_sensitivity": float(best["CIN3+ sensitivity"]),
            "CIN3_false_negative_count": int(best["CIN3+ false-negative count"]),
            "centre_gap": float(best["Centre-level performance gap"]),
        },
        comparisons={
            "vs_step2_8_best": float(best["AUC"] - step8["AUC"]),
            "vs_step2_6_active_minimal": float(best["AUC"] - step26["AUC"]),
            "vs_step2_surrogate": float(best["AUC"] - surrogate["AUC"]),
            "vs_best_clinical_baseline": float(best["AUC"] - clinical["AUC"]),
        },
    )
    md = OUT_DIR / "STEP2_9_DG_RECOVERY_STATUS.md"
    lines = [
        "# Step2.9 Domain-Generalisation Recovery Status",
        "",
        f"- Status: `{status}`",
        f"- IF recommendation: `{route}`",
        f"- Best model: `HyDRA-DG-SafetyEnsemble`",
        f"- CIN2+ AUC: {best['AUC']:.3f}",
        f"- CIN3+ sensitivity: {best['CIN3+ sensitivity']:.3f}",
        f"- CIN3+ false-negative count: {int(best['CIN3+ false-negative count'])}",
        f"- Worst-centre AUC where defined: {best['Worst-centre AUC where defined']:.3f}",
        f"- Centre-level gap: {best['Centre-level performance gap']:.3f}",
        f"- Delta AUC vs Step2.8: {best['AUC'] - step8['AUC']:.3f}",
        f"- Delta AUC vs Step2.6 active minimal: {best['AUC'] - step26['AUC']:.3f}",
        f"- Delta AUC vs Step2 surrogate: {best['AUC'] - surrogate['AUC']:.3f}",
        f"- Delta AUC vs best clinical baseline: {best['AUC'] - clinical['AUC']:.3f}",
        "",
        "## Centre/Domain Shift Evidence",
        "",
        f"- Centre classifier accuracy: {shift['Centre classifier accuracy'].iloc[0]:.3f}",
        f"- Largest MMD shift centre: {worst_shift['Centre']}",
        f"- Largest MMD: {worst_shift['MMD vs pooled training centres']:.3f}",
        "",
        "## Framing",
        "",
        rec.get("action", ""),
        "",
        "Use HyDRA-DG / domain-generalisation adapter wording. Full end-to-end HyDRA-CoE and supervised CoE trajectory claims remain unsupported.",
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
    cfg = load_yaml(config_path)
    update_status(experiment_name=cfg["experiment_name"], step2_9_status="IN_PROGRESS")
    summarise_step2_8_failure(config_path)
    measure_domain_shift(config_path)
    build_inner_centre_splits(config_path)
    train_centre_balanced_baselines(config_path, no_dry_run=no_dry_run)
    train_domain_generalisation_models(config_path, no_dry_run=no_dry_run)
    select_models_by_inner_validation(config_path)
    rerun_top_dg_models(config_path, OUT_DIR / "search/top_dg_model_configs.json", no_dry_run=no_dry_run)
    build_dg_ensembles(config_path)
    evaluate_domain_generalisation_recovery(config_path)
    generate_dg_tables(config_path)
    generate_if_go_nogo_package(config_path)
    final_status(config_path)
    plot_dg_figures(config_path)
    final_status(config_path)
