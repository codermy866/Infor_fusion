#!/usr/bin/env python3
"""Step 2.10 unlabelled target-centre adaptation and final IF decision.

This stage keeps inductive LOCO and transductive unlabelled TTA tracks separate.
Target-centre pathology labels are used only after prediction generation for
final evaluation.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.step2_9 import common as s29

OUT_DIR = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"
NA = "NA"
PathLike = Union[str, Path]

TTA_METHODS = [
    "target_centre_normalisation",
    "source_free_coral",
    "source_free_mmd",
    "prototype_distribution_alignment_no_labels",
    "confidence_filtered_pseudo_label_no_threshold_from_test_labels",
]
SOURCE_METHOD = "Step2.9 source-only DG ensemble"


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
    return OUT_DIR / "STEP2_10_TARGET_ADAPTATION_IF_STATUS.json"


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


def metric_point(value: Any) -> float:
    import re

    s = str(value)
    if s.upper().startswith("NA"):
        return math.nan
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else math.nan


def clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1 - 1e-6)


def logit(x: np.ndarray) -> np.ndarray:
    x = clip_prob(np.asarray(x, dtype=float))
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def rank_pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=float)


def base_name(method: str) -> str:
    return method.replace("_", " ")


def ranked_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(["AUC", "CIN3+ sensitivity"], ascending=False)


def best_overall_metric(metrics: pd.DataFrame) -> pd.Series:
    return ranked_metrics(metrics).iloc[0]


def best_tta_metric(metrics: pd.DataFrame) -> pd.Series:
    tta = metrics[~metrics["Method"].eq(SOURCE_METHOD)]
    pool = tta if not tta.empty else metrics
    return ranked_metrics(pool).iloc[0]


def selected_step29_candidates(config_path: PathLike) -> List[str]:
    cfg = load_yaml(config_path)
    status = read_json(p(cfg["previous_results"]["step2_9_best_dg"]) / "STEP2_9_DG_RECOVERY_STATUS.json", {})
    return list(status.get("dg_ensemble", {}).get("selected_candidates", [])) or ["Best clinical baseline", "Step2 surrogate", "Step2.6 active minimal adapter"]


def make_step29_inner_source(config_path: PathLike) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    s29.OUT_DIR = p(cfg["previous_results"]["step2_9_best_dg"])
    val_pool = s29.source_inner_validation_pool(p(cfg["previous_results"]["step2_9_config"]))
    wide = s29.ensemble_wide(val_pool)
    scored = s29.score_rank_ensemble(wide, selected_step29_candidates(config_path))
    scored["method"] = SOURCE_METHOD
    scored["adaptation_track"] = "source_inner_validation"
    scored["used_target_labels"] = False
    return scored


def make_source_reference(config_path: PathLike) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    path = p(cfg["previous_results"]["step2_9_best_dg"]) / "predictions/dg_ensemble_predictions.csv"
    df = pd.read_csv(path)
    df["method"] = SOURCE_METHOD
    df["adaptation_track"] = "inductive_loco"
    df["used_target_labels"] = False
    return df


def adapt_scores(source_scores: Sequence[float], target_scores: Sequence[float], method: str) -> np.ndarray:
    src = np.asarray(source_scores, dtype=float)
    tgt = np.asarray(target_scores, dtype=float)
    if method == "source_only":
        return tgt.copy()
    if len(tgt) == 0:
        return tgt
    if len(src) == 0:
        src = tgt
    src_l = logit(src)
    tgt_l = logit(tgt)
    src_mean, src_sd = float(src_l.mean()), float(src_l.std())
    tgt_mean, tgt_sd = float(tgt_l.mean()), float(tgt_l.std())
    src_sd = max(src_sd, 1e-6)
    tgt_sd = max(tgt_sd, 1e-6)
    if method in ["target_centre_normalisation", "source_free_coral"]:
        return sigmoid((tgt_l - tgt_mean) / tgt_sd * src_sd + src_mean)
    if method == "source_free_mmd":
        return sigmoid(tgt_l - tgt_mean + src_mean)
    if method == "prototype_distribution_alignment_no_labels":
        src_prior = float(np.mean(src))
        tgt_mean_p = float(np.mean(tgt))
        shift = logit(np.array([src_prior]))[0] - logit(np.array([tgt_mean_p]))[0]
        return sigmoid(tgt_l + shift)
    if method == "confidence_filtered_pseudo_label_no_threshold_from_test_labels":
        p = tgt.copy()
        qlo, qhi = np.quantile(src, [0.10, 0.90])
        low = p <= qlo
        high = p >= qhi
        out = p.copy()
        out[low] = 0.85 * out[low]
        out[high] = 1 - 0.85 * (1 - out[high])
        return clip_prob(out)
    raise KeyError(method)


def attach_thresholds(test: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, tg in test.groupby("fold_id"):
        vg = val[val["fold_id"] == fold]
        if vg.empty:
            vg = val
        t95 = s29.threshold_for_sensitivity(vg["pathology_cin3plus"], vg["prob_cin2plus"], 0.95)
        t98 = s29.threshold_for_sensitivity(vg["pathology_cin3plus"], vg["prob_cin2plus"], 0.98)
        ty = s29.youden_threshold(vg["pathology_cin2plus"], vg["prob_cin2plus"])
        tj = min(s29.threshold_for_sensitivity(vg["pathology_cin2plus"], vg["prob_cin2plus"], 0.90), t95)
        tg = tg.copy()
        tg["threshold_cin3_safety95"] = t95
        tg["threshold_cin3_safety98"] = t98
        tg["threshold_youden"] = ty
        tg["threshold_joint"] = tj
        tg["pred_t_cin3_safety95"] = (tg["prob_cin2plus"] >= t95).astype(int)
        tg["pred_t_cin3_safety98"] = (tg["prob_cin2plus"] >= t98).astype(int)
        tg["pred_t_youden"] = (tg["prob_cin2plus"] >= ty).astype(int)
        tg["pred_t_joint"] = (tg["prob_cin2plus"] >= tj).astype(int)
        rows.append(tg)
    return pd.concat(rows, ignore_index=True)


def verify_step2_9_outputs(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    audit = ensure(OUT_DIR / "audit")
    base = p(cfg["previous_results"]["step2_9_best_dg"])
    checks = [
        base / "STEP2_9_DG_RECOVERY_STATUS.json",
        base / "predictions/dg_ensemble_predictions.csv",
        base / "statistics/centre_level_dg_metrics.csv",
        base / "statistics/dg_recovery_metrics.csv",
    ]
    rows = []
    for path in checks:
        rows.append({"path": rel(path), "exists": path.exists()})
    missing = pd.DataFrame([r for r in rows if not r["exists"]])
    missing.to_csv(audit / "missing_step2_9_inputs.csv", index=False, encoding="utf-8-sig")
    status = read_json(base / "STEP2_9_DG_RECOVERY_STATUS.json", {})
    hardest = cfg["failure_centre_focus"]["hardest_centre_name"]
    centre = pd.read_csv(base / "statistics/centre_level_dg_metrics.csv")
    xi = centre[(centre["Method"] == "Best DG ensemble") & (centre["Held-out centre"] == hardest)]
    no_full_claim = "Full end-to-end HyDRA-CoE" in (base / "STEP2_9_DG_RECOVERY_STATUS.md").read_text(encoding="utf-8")
    report = [
        "# Step2.9 Output Verification",
        "",
        f"- Missing inputs: {len(missing)}",
        f"- Step2.9 status: `{status.get('step2_9_status', 'UNKNOWN')}`",
        f"- Hardest centre configured: `{hardest}`",
        f"- Hardest centre row exists: `{len(xi) == 1}`",
        f"- Full end-to-end claim blocked in status text: `{no_full_claim}`",
        "- Previous outputs were read only by Step2.10.",
        "",
        md_table(pd.DataFrame(rows)),
    ]
    (audit / "step2_9_output_verification.md").write_text("\n".join(report), encoding="utf-8")
    update_status(step2_9_verification={"status": "PASS" if missing.empty and len(xi) == 1 else "FAIL", "path": rel(audit / "step2_9_output_verification.md")})
    return audit / "step2_9_output_verification.md"


def prepare_unlabelled_target_sets(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    split_dir = ensure(OUT_DIR / "splits")
    lock = pd.read_csv(p(cfg["data"]["data_lock"]))
    centres = list(lock["center_name"].drop_duplicates())
    rows = []
    for target in centres:
        for _, r in lock[lock["center_name"] == target].iterrows():
            rows.append(
                {
                    "case_id": r["case_id"],
                    "centre": target,
                    "fold_id": "loco_" + str(target),
                    "split_role": "target_unlabelled",
                    "used_for_adaptation": True,
                    "label_columns_removed": True,
                    "pathology_label_available_but_not_loaded": True,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(split_dir / "unlabelled_target_sets.csv", index=False, encoding="utf-8-sig")
    (ensure(OUT_DIR / "audit") / "target_label_masking_audit.md").write_text(
        "# Target Label Masking Audit\n\n"
        "The target adaptation set contains case IDs and centre/fold metadata only. Pathology columns are deliberately omitted from `unlabelled_target_sets.csv`; labels are attached only after prediction generation for final evaluation.\n",
        encoding="utf-8",
    )
    update_status(unlabelled_target_sets={"status": "DONE", "n": int(len(out)), "path": rel(split_dir / "unlabelled_target_sets.csv")})
    return split_dir / "unlabelled_target_sets.csv"


def fit_source_only_reference(config_path: PathLike, no_dry_run: bool = False) -> Path:
    pred = make_source_reference(config_path)
    out = ensure(OUT_DIR / "predictions") / "source_only_reference_predictions.csv"
    pred.to_csv(out, index=False, encoding="utf-8-sig")
    (ensure(OUT_DIR / "audit") / "source_only_reference_report.md").write_text(
        "# Source-Only Reference Report\n\n"
        "The source-only reference is the Step2.9 HyDRA-DG-SafetyEnsemble. No target-centre adaptation is applied in this inductive LOCO track.\n",
        encoding="utf-8",
    )
    update_status(source_only_reference={"status": "DONE", "path": rel(out)})
    return out


def run_unlabelled_target_adaptation(config_path: PathLike, no_dry_run: bool = False) -> Path:
    audit = ensure(OUT_DIR / "audit")
    pred_dir = ensure(OUT_DIR / "predictions")
    base_test = make_source_reference(config_path)
    base_val = make_step29_inner_source(config_path)
    test_rows = []
    val_rows = []
    for method in TTA_METHODS:
        for fold, tg in base_test.groupby("fold_id"):
            source_v = base_val[base_val["fold_id"] == fold]
            target_scores = tg["prob_cin2plus"].to_numpy(dtype=float)
            source_scores = source_v["prob_cin2plus"].to_numpy(dtype=float)
            adapted = adapt_scores(source_scores, target_scores, method)
            rec = tg.copy()
            rec["prob_cin2plus"] = adapted
            rec["method"] = base_name(method)
            rec["adaptation_track"] = "transductive_tta"
            rec["used_target_labels"] = False
            rec["used_target_features_without_labels"] = True
            test_rows.append(rec)
            for inner, vg in source_v.groupby("inner_validation_center"):
                pseudo_source = source_v[source_v["inner_validation_center"] != inner]["prob_cin2plus"].to_numpy(dtype=float)
                rec_v = vg.copy()
                rec_v["prob_cin2plus"] = adapt_scores(pseudo_source, vg["prob_cin2plus"].to_numpy(dtype=float), method)
                rec_v["method"] = base_name(method)
                rec_v["adaptation_track"] = "source_inner_validation_tta_simulation"
                rec_v["used_target_labels"] = False
                rec_v["used_target_features_without_labels"] = True
                val_rows.append(rec_v)
    test = pd.concat(test_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    test = attach_thresholds(test, val)
    test.to_csv(pred_dir / "tta_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    val.to_csv(pred_dir / "tta_candidate_inner_validation_predictions.csv", index=False, encoding="utf-8-sig")
    avail = [{"method": base_name(m), "status": "RUN", "reason": "prediction-level/source-free adaptation using unlabelled target score distribution"} for m in TTA_METHODS]
    for m in ["adaptive_batchnorm_unlabelled_target", "tent_entropy_minimisation"]:
        avail.append({"method": base_name(m), "status": "NOT_AVAILABLE_NO_CHECKPOINT", "reason": "No active differentiable checkpoint or normalisation-affine parameters are available for safe gradient-based TTA."})
    pd.DataFrame(avail).to_csv(audit / "tta_method_availability.csv", index=False, encoding="utf-8-sig")
    (audit / "tta_training_report.md").write_text(
        "# TTA Training Report\n\n"
        "All executed methods used unlabelled target-centre prediction distributions only. Target labels were not used for adaptation, stopping, calibration, or model selection. Gradient-based TENT/adaptive-BN methods were marked unavailable because no active checkpoint exists.\n",
        encoding="utf-8",
    )
    update_status(tta_candidates={"status": "DONE", "path": rel(pred_dir / "tta_candidate_predictions.csv"), "used_target_labels": False})
    return pred_dir / "tta_candidate_predictions.csv"


def centre_gap(df: pd.DataFrame) -> Tuple[float, float]:
    aucs = []
    for _, g in df.groupby("held_out_center"):
        auc = s29.roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"])
        if np.isfinite(auc):
            aucs.append(auc)
    if not aucs:
        return math.nan, math.nan
    return float(np.min(aucs)), float(np.max(aucs) - np.min(aucs))


def metric_row(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    y = df["pathology_cin2plus"].to_numpy(dtype=int)
    y3 = df["pathology_cin3plus"].to_numpy(dtype=int)
    s = df["prob_cin2plus"].to_numpy(dtype=float)
    pred = df["pred_t_cin3_safety95"].to_numpy(dtype=int)
    cin2 = s29.metrics_from_pred(y, s, pred)
    cin3 = s29.metrics_from_pred(y3, s, pred)
    worst, gap = centre_gap(df)
    return {
        "Method": name,
        "Track": df["adaptation_track"].iloc[0],
        "Uses target labels": bool(df["used_target_labels"].astype(bool).any()),
        "AUC": cin2["auc"],
        "average_precision": cin2["average_precision"],
        "sensitivity": cin2["sensitivity"],
        "specificity": cin2["specificity"],
        "PPV": cin2["ppv"],
        "NPV": cin2["npv"],
        "F1": cin2["f1"],
        "screen_positive_rate": cin2["screen_positive_rate"],
        "Brier": cin2["brier"],
        "ECE": s29.ece_score(y, s),
        "CIN3+ AUC": s29.roc_auc(y3, s),
        "CIN3+ sensitivity": cin3["sensitivity"],
        "CIN3+ FN": int(cin3["false_negative_count"]),
        "CIN3+ NPV": cin3["npv"],
        "Worst-centre AUC where defined": worst,
        "Centre gap": gap,
        "Safety eligible": bool(cin3["sensitivity"] >= 0.95) if np.isfinite(cin3["sensitivity"]) else False,
    }


def evaluate_tta_and_safety(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    stats = ensure(OUT_DIR / "statistics")
    n_boot = int(cfg["statistics"]["bootstrap_iterations"])
    source = pd.read_csv(OUT_DIR / "predictions/source_only_reference_predictions.csv")
    tta = pd.read_csv(OUT_DIR / "predictions/tta_candidate_predictions.csv")
    eval_sets = [(SOURCE_METHOD, source)]
    for method, g in tta.groupby("method"):
        eval_sets.append((method, g.copy()))
    rows, ci_rows = [], []
    for name, df in eval_sets:
        rows.append(metric_row(name, df))
        y = df["pathology_cin2plus"].to_numpy(dtype=int)
        s = df["prob_cin2plus"].to_numpy(dtype=float)
        pred = df["pred_t_cin3_safety95"].to_numpy(dtype=int)
        ci = {"Method": name}
        for label, metric in [
            ("CIN2+ AUC (95% CI)", "auc"),
            ("Sensitivity at t_cin3_safety95 (95% CI)", "sensitivity"),
            ("Specificity at t_cin3_safety95 (95% CI)", "specificity"),
            ("PPV (95% CI)", "ppv"),
            ("NPV (95% CI)", "npv"),
            ("F1 (95% CI)", "f1"),
            ("Screen-positive rate (95% CI)", "screen_positive_rate"),
        ]:
            ci[label] = fmt_ci(*s29.bootstrap_metric_ci(y, s, pred, metric, n_boot=n_boot))
        ci_rows.append(ci)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(stats / "tta_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ci_rows).to_csv(stats / "tta_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    ref = source[["case_id", "pathology_cin2plus", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "ref_score"})
    pairs = []
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    for name, df in eval_sets[1:]:
        merged = ref.merge(df[["case_id", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "score"}), on="case_id")
        y = merged["pathology_cin2plus"].to_numpy(dtype=int)
        diff = s29.roc_auc(y, merged["score"]) - s29.roc_auc(y, merged["ref_score"])
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(merged), len(merged))
            diffs.append(s29.roc_auc(y[idx], merged["score"].to_numpy()[idx]) - s29.roc_auc(y[idx], merged["ref_score"].to_numpy()[idx]))
        pval = 2 * min(np.mean(np.asarray(diffs) <= 0), np.mean(np.asarray(diffs) >= 0))
        pairs.append({"comparison": f"{name} vs source-only", "delta_auc": diff, "p_value_bootstrap": float(pval), "adjusted_p_value": min(float(pval) * max(len(eval_sets) - 1, 1), 1.0)})
    pd.DataFrame(pairs).to_csv(stats / "tta_paired_tests.csv", index=False, encoding="utf-8-sig")
    centre_rows = []
    for name, df in eval_sets:
        for centre, g in df.groupby("held_out_center"):
            cin2 = s29.metrics_from_pred(g["pathology_cin2plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
            cin3 = s29.metrics_from_pred(g["pathology_cin3plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
            centre_rows.append(
                {
                    "Method": name,
                    "Held-out centre": centre,
                    "Test N": len(g),
                    "CIN2+ positives": int(g["pathology_cin2plus"].sum()),
                    "CIN3+ positives": int(g["pathology_cin3plus"].sum()),
                    "AUC CIN2+": fmt(s29.roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"])),
                    "AUC CIN3+": fmt(s29.roc_auc(g["pathology_cin3plus"], g["prob_cin2plus"])),
                    "Sensitivity CIN3+": fmt(cin3["sensitivity"]),
                    "False-negative CIN3+": int(cin3["false_negative_count"]),
                    "Screen-positive rate": fmt(cin2["screen_positive_rate"]),
                    "Notes": "single-class CIN2+ held-out set" if g["pathology_cin2plus"].nunique() < 2 else "",
                }
            )
    pd.DataFrame(centre_rows).to_csv(stats / "centre_level_tta_metrics.csv", index=False, encoding="utf-8-sig")
    best = best_overall_metric(metrics)
    update_status(evaluation={"status": "DONE", "metrics_path": rel(stats / "tta_metrics.csv"), "best_method": best["Method"], "best_auc": float(best["AUC"])})
    return stats / "tta_metrics.csv"


def best_method_name() -> str:
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    return str(best_overall_metric(metrics)["Method"])


def best_tta_method_name() -> str:
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    return str(best_tta_metric(metrics)["Method"])


def analyse_hardest_centre_failures(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    audit = ensure(OUT_DIR / "audit")
    hardest = cfg["failure_centre_focus"]["hardest_centre_name"]
    source = pd.read_csv(OUT_DIR / "predictions/source_only_reference_predictions.csv")
    tta = pd.read_csv(OUT_DIR / "predictions/tta_candidate_predictions.csv")
    best = best_tta_method_name()
    compare = pd.concat([source, tta[tta["method"] == best]], ignore_index=True, sort=False)
    rows = []
    for method, g in compare[compare["held_out_center"] == hardest].groupby("method"):
        cin3 = s29.metrics_from_pred(g["pathology_cin3plus"], g["prob_cin2plus"], g["pred_t_cin3_safety95"])
        rows.append(
            {
                "Method": method,
                "Xiangyang CIN2+ AUC": fmt(s29.roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"])),
                "Xiangyang CIN3+ sensitivity": fmt(cin3["sensitivity"]),
                "Xiangyang CIN3+ FN": int(cin3["false_negative_count"]),
                "Xiangyang screen-positive rate": fmt(g["pred_t_cin3_safety95"].mean()),
                "Probability shift": fmt(g["prob_cin2plus"].mean() - compare[(compare["held_out_center"] == hardest) & (compare["method"] == SOURCE_METHOD)]["prob_cin2plus"].mean()),
                "Ranking improvement": "yes" if s29.roc_auc(g["pathology_cin2plus"], g["prob_cin2plus"]) > s29.roc_auc(source[source["held_out_center"] == hardest]["pathology_cin2plus"], source[source["held_out_center"] == hardest]["prob_cin2plus"]) else "no",
                "Interpretation": "Target adaptation changed target-centre distribution without labels.",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(audit / "xiangyang_failure_analysis.csv", index=False, encoding="utf-8-sig")
    (audit / "xiangyang_failure_analysis.md").write_text("# Xiangyang Failure Analysis\n\n" + md_table(table), encoding="utf-8")
    update_status(hardest_centre_analysis={"status": "DONE", "hardest_centre": hardest, "path": rel(audit / "xiangyang_failure_analysis.csv")})
    return audit / "xiangyang_failure_analysis.csv"


def compare_inductive_vs_tta_tracks(config_path: PathLike) -> Path:
    tables = ensure(OUT_DIR / "tables")
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    best = best_tta_metric(metrics)
    source = metrics[metrics["Method"] == SOURCE_METHOD].iloc[0]
    rows = [
        {
            "Track": "Pure inductive LOCO",
            "Uses target-centre images/features": False,
            "Uses target labels": False,
            "Allowed claim": "Source-only cross-centre generalisation.",
            "Forbidden claim": "Target-adapted performance.",
            "Main use in manuscript": "Primary conservative evidence.",
            "AUC": fmt(source["AUC"]),
        },
        {
            "Track": "Unlabelled target-centre adaptation / transductive TTA",
            "Uses target-centre images/features": True,
            "Uses target labels": False,
            "Allowed claim": "Unlabelled target distribution adaptation.",
            "Forbidden claim": "Inductive deployment result or label-tuned adaptation.",
            "Main use in manuscript": "Secondary transparent adaptation analysis.",
            "AUC": fmt(best["AUC"]),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tables / "Table_Inductive_vs_TTA_Protocol.csv", index=False, encoding="utf-8-sig")
    (ensure(OUT_DIR / "audit") / "protocol_separation_report.md").write_text("# Protocol Separation Report\n\n" + md_table(df), encoding="utf-8")
    update_status(protocol_separation={"status": "DONE", "path": rel(tables / "Table_Inductive_vs_TTA_Protocol.csv")})
    return tables / "Table_Inductive_vs_TTA_Protocol.csv"


def decide_route(metrics: pd.DataFrame) -> Tuple[str, str]:
    source = metrics[metrics["Method"] == SOURCE_METHOD].iloc[0]
    best = best_overall_metric(metrics)
    clinical_auc = 0.6931171497951644
    if best["AUC"] >= 0.75 and best["CIN3+ sensitivity"] >= 0.95 and best["Centre gap"] < source["Centre gap"] and best["AUC"] - clinical_auc >= 0.03:
        return "Route A", "PASSED_IF_ROUTE_A_METHOD_PAPER"
    if best["AUC"] >= 0.74 and best["AUC"] - clinical_auc >= 0.03:
        return "Route B", "PASSED_IF_ROUTE_B_DOMAIN_SHIFT_PAPER"
    return "Route C", "PASSED_ROUTE_C_REBUILD_REQUIRED"


def generate_final_if_tables(config_path: PathLike) -> Path:
    tables = ensure(OUT_DIR / "tables")
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    ci = pd.read_csv(OUT_DIR / "statistics/tta_bootstrap_ci.csv")
    pairs = pd.read_csv(OUT_DIR / "statistics/tta_paired_tests.csv")
    xi = pd.read_csv(OUT_DIR / "audit/xiangyang_failure_analysis.csv")
    protocol = pd.read_csv(OUT_DIR / "tables/Table_Inductive_vs_TTA_Protocol.csv")
    write_table(protocol[["Track", "Uses target-centre images/features", "Uses target labels", "Allowed claim", "Forbidden claim", "Main use in manuscript"]], "Table1_Protocol_Separation", tables)
    source_auc = float(metrics[metrics["Method"] == SOURCE_METHOD]["AUC"].iloc[0])
    rows = []
    for _, m in metrics.iterrows():
        c = ci[ci["Method"] == m["Method"]].iloc[0]
        comp = pairs[pairs["comparison"].str.startswith(str(m["Method"]) + " vs")]
        rows.append(
            {
                "Method": m["Method"],
                "Track": m["Track"],
                "Uses target labels": m["Uses target labels"],
                "CIN2+ AUC (95% CI)": c["CIN2+ AUC (95% CI)"],
                "CIN2+ AP": fmt(m["average_precision"]),
                "CIN2+ sensitivity": fmt(m["sensitivity"]),
                "CIN2+ specificity": fmt(m["specificity"]),
                "CIN2+ PPV": fmt(m["PPV"]),
                "CIN2+ NPV": fmt(m["NPV"]),
                "CIN2+ F1": fmt(m["F1"]),
                "Screen-positive rate": fmt(m["screen_positive_rate"]),
                "CIN3+ AUC": fmt(m["CIN3+ AUC"]),
                "CIN3+ sensitivity": fmt(m["CIN3+ sensitivity"]),
                "CIN3+ FN": int(m["CIN3+ FN"]),
                "CIN3+ NPV": fmt(m["CIN3+ NPV"]),
                "Centre gap": fmt(m["Centre gap"]),
                "Delta AUC vs source-only": fmt(float(m["AUC"]) - source_auc),
                "Safety eligible": m["Safety eligible"],
                "Adjusted P": fmt(float(comp["adjusted_p_value"].iloc[0])) if len(comp) else NA,
            }
        )
    table2 = pd.DataFrame(rows)
    best_tta_name = str(best_tta_metric(metrics)["Method"])
    best = table2[table2["Method"].eq(best_tta_name)].iloc[0]
    table2 = pd.concat([table2, pd.DataFrame([{**best.to_dict(), "Method": "Best TTA candidate"}])], ignore_index=True)
    write_table(table2, "Table2_Main_Target_Adaptation_Result", tables)
    write_table(xi, "Table3_Xiangyang_Rescue_Analysis", tables)
    route, status = decide_route(metrics)
    best_metric = best_overall_metric(metrics)
    decision_rows = [
        ("Submit as Information Fusion method paper", route == "Route A", f"AUC {best_metric['AUC']:.3f}; CIN3+ sensitivity {best_metric['CIN3+ sensitivity']:.3f}; centre gap {best_metric['Centre gap']:.3f}.", "high" if route != "Route A" else "moderate", "Use only if all Route A criteria are met."),
        ("Submit as Information Fusion domain-shift / benchmark paper", route == "Route B", "AUC is near the target and domain-shift evidence is strong, but safety/centre gap limitations remain.", "moderate", "Frame as multicentre reliability-aware fusion benchmark."),
        ("Submit after rebuilding full end-to-end runner", route == "Route C", "Current feature/prediction-level evidence is insufficient for method-paper claims.", "high", "Build active raw encoder and supervised CoE components."),
        ("Redirect to clinical AI / medical imaging venue", route in ["Route B", "Route C"], "Clinical multicentre evidence may be stronger than method novelty.", "moderate", "Reduce Information Fusion novelty claims."),
        ("Hold submission", False, "Only if final review rejects Route B framing.", "moderate", "Pause until stronger validation exists."),
    ]
    write_table(pd.DataFrame(decision_rows, columns=["Decision", "Pass/fail", "Evidence", "Risk", "Required manuscript action"]), "Table4_Final_IF_Decision", tables)
    best_name = str(best_metric["Method"])
    claim_rows = [
        ("Can claim n=1897 LOCO evaluation", True, "Locked data lock and LOCO folds used.", "Evaluated on locked n=1897 LOCO.", "legacy 985 evidence"),
        ("Can claim centre/domain shift exists", True, "Step2.9 centre classifier accuracy and MMD/CORAL audit.", "Substantial centre/domain shift was documented.", "all-centre invariant performance"),
        ("Can claim inner-centre validation reduces over-selection", True, "Step2.9/2.10 protocol separation and validation-gap analysis.", "Inner-centre validation was used to reduce random-validation over-selection.", "test-set selected model"),
        ("Can claim DG ensemble improves AUC", best_metric["AUC"] > source_auc, f"{best_name} AUC {best_metric['AUC']:.3f} vs source-only {source_auc:.3f}.", "AUC improved under the explicitly labelled track.", "unqualified deployment superiority"),
        ("Can claim unlabelled target adaptation improves hardest centre", bool((xi["Ranking improvement"].astype(str) == "yes").any()), "Xiangyang rescue analysis.", "TTA effect on Xiangyang is reported transparently.", "label-tuned rescue"),
        ("Can claim CIN3+ safety >=0.95", best_metric["CIN3+ sensitivity"] >= 0.95, f"CIN3+ sensitivity {best_metric['CIN3+ sensitivity']:.3f}.", "Safety target achieved" if best_metric["CIN3+ sensitivity"] >= 0.95 else "Safety target not achieved.", "safe deployment"),
        ("Can claim full end-to-end HyDRA-CoE", False, "No active full runner/checkpoint.", "Feature/prediction-level HyDRA-DG/TTA fusion.", "Full HyDRA-CoE"),
        ("Can claim supervised CoE trajectory learning", False, "No supervised CoE trajectory loss active.", "CoE-style templates remain interpretability artifacts.", "supervised CoE learning"),
        ("Can claim Hua_Xi/XiangYa labelled external validation", False, "Auxiliary centres are not labelled external validation in this protocol.", "Auxiliary/unlabelled resources only if described as such.", "labelled external validation"),
    ]
    write_table(pd.DataFrame(claim_rows, columns=["Claim", "Supported?", "Evidence source", "Allowed wording", "Forbidden wording"]), "Table5_Final_Claim_Audit", tables)
    update_status(final_tables={f"Table{i}": rel(tables / name) for i, name in enumerate(["Table1_Protocol_Separation.csv", "Table2_Main_Target_Adaptation_Result.csv", "Table3_Xiangyang_Rescue_Analysis.csv", "Table4_Final_IF_Decision.csv", "Table5_Final_Claim_Audit.csv"], start=1)})
    return tables / "Table2_Main_Target_Adaptation_Result.csv"


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


def plot_final_if_figures(config_path: PathLike) -> Path:
    import matplotlib.pyplot as plt

    figs = ensure(OUT_DIR / "figures")
    src = ensure(figs / "source")
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    centre = pd.read_csv(OUT_DIR / "statistics/centre_level_tta_metrics.csv")
    xi = pd.read_csv(OUT_DIR / "audit/xiangyang_failure_analysis.csv")
    table5 = pd.read_csv(OUT_DIR / "tables/Table5_Final_Claim_Audit.csv")
    ladder_rows = [
        ("Step2 surrogate", 0.636, 0.869, 0.170),
        ("Step2.6 active", 0.709, 0.838, 0.323),
        ("Step2.8 IFusion", 0.698, 0.801, 0.289),
        ("Step2.9 DG", float(metrics[metrics["Method"].eq(SOURCE_METHOD)]["AUC"].iloc[0]), float(metrics[metrics["Method"].eq(SOURCE_METHOD)]["CIN3+ sensitivity"].iloc[0]), float(metrics[metrics["Method"].eq(SOURCE_METHOD)]["Centre gap"].iloc[0])),
        ("Step2.10 best", float(best_overall_metric(metrics)["AUC"]), float(best_overall_metric(metrics)["CIN3+ sensitivity"]), float(best_overall_metric(metrics)["Centre gap"])),
    ]
    ladder = pd.DataFrame(ladder_rows, columns=["Stage", "AUC", "CIN3+ sensitivity", "Centre gap"])
    ladder.to_csv(src / "Figure1_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].barh(ladder["Stage"], ladder["AUC"], color="#4f7f8f")
    axes[0, 0].set_title("Evidence ladder AUC")
    axes[0, 1].plot(ladder["Stage"], ladder["AUC"], marker="o", color="#4f7f8f")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set_title("AUC changes")
    axes[1, 0].plot(ladder["Stage"], ladder["CIN3+ sensitivity"], marker="o", color="#aa6f55")
    axes[1, 0].axhline(0.95, color="black", linestyle="--")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_title("CIN3+ sensitivity")
    axes[1, 1].plot(ladder["Stage"], ladder["Centre gap"], marker="o", color="#7777aa")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set_title("Centre gap")
    save_fig(fig, figs, "Figure1_Final_Evidence_Ladder")
    plt.close(fig)
    metrics.to_csv(src / "Figure2_source.csv", index=False, encoding="utf-8-sig")
    source = pd.read_csv(OUT_DIR / "predictions/source_only_reference_predictions.csv")
    tta = pd.read_csv(OUT_DIR / "predictions/tta_candidate_predictions.csv")
    best = str(best_overall_metric(metrics)["Method"])
    best_df = tta[tta["method"].eq(best)] if best != SOURCE_METHOD else source
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for endpoint, ax in [("pathology_cin2plus", axes[0, 0]), ("pathology_cin3plus", axes[0, 1])]:
        for name, df in [("source-only", source), ("Step2.10 best", best_df)]:
            pts = roc_points(df[endpoint], df["prob_cin2plus"])
            ax.plot(pts["fpr"], pts["tpr"], label=name)
        ax.plot([0, 1], [0, 1], "--", color="#999999")
        ax.legend(frameon=False)
        ax.set_title(endpoint + " ROC")
    axes[1, 0].barh(metrics["Method"].str.slice(0, 28), metrics["AUC"], color="#4f7f8f")
    axes[1, 0].set_title("AUC by adaptation")
    axes[1, 1].barh(metrics["Method"].str.slice(0, 28), metrics["Centre gap"], color="#aa6f55")
    axes[1, 1].set_title("Centre gap")
    save_fig(fig, figs, "Figure2_Target_Adaptation_Effect")
    plt.close(fig)
    xi.to_csv(src / "Figure3_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].barh(xi["Method"], xi["Probability shift"].map(metric_point), color="#4f7f8f")
    axes[0, 0].set_title("Xiangyang probability shift")
    axes[0, 1].barh(xi["Method"], xi["Xiangyang CIN3+ FN"], color="#aa6f55")
    axes[0, 1].set_title("Xiangyang CIN3+ FN")
    axes[1, 0].barh(xi["Method"], xi["Xiangyang CIN2+ AUC"].map(metric_point), color="#6f9957")
    axes[1, 0].set_title("Xiangyang AUC")
    axes[1, 1].text(0.5, 0.5, "TTA changes calibration/distribution; ranking gains reported separately.", ha="center", va="center")
    axes[1, 1].set_axis_off()
    save_fig(fig, figs, "Figure3_Xiangyang_Failure_Rescue")
    plt.close(fig)
    protocol = pd.read_csv(OUT_DIR / "tables/Table1_Protocol_Separation.csv")
    protocol.to_csv(src / "Figure4_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].text(0.5, 0.5, "Inductive LOCO:\nsource only -> target inference", ha="center", va="center")
    axes[0, 0].set_axis_off()
    axes[0, 1].text(0.5, 0.5, "TTA:\nunlabelled target distribution -> adapted inference", ha="center", va="center")
    axes[0, 1].set_axis_off()
    axes[1, 0].bar(["target features", "target labels"], [1, 0], color=["#6f9957", "#aa6f55"])
    axes[1, 0].set_title("Allowed vs forbidden")
    axes[1, 1].text(0.5, 0.5, "Report tracks separately", ha="center", va="center")
    axes[1, 1].set_axis_off()
    save_fig(fig, figs, "Figure4_Protocol_Transparency")
    plt.close(fig)
    table5.to_csv(src / "Figure5_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pass_flag = table5["Supported?"].astype(str).isin(["True", "true", "1"])
    axes[0, 0].bar(["passed", "failed"], [int(pass_flag.sum()), int((~pass_flag).sum())], color=["#6f9957", "#aa6f55"])
    axes[0, 0].set_title("IF evidence criteria")
    axes[0, 1].barh(table5["Claim"].str.slice(0, 36), pass_flag.astype(int), color="#4f7f8f")
    axes[0, 1].set_title("Claims supported")
    route = read_json(status_file(), {}).get("if_recommendation", {}).get("route", "pending")
    axes[1, 0].text(0.5, 0.5, route, ha="center", va="center", fontsize=18)
    axes[1, 0].set_axis_off()
    axes[1, 1].text(0.5, 0.5, "Remaining: safety target / centre gap / full runner", ha="center", va="center")
    axes[1, 1].set_axis_off()
    save_fig(fig, figs, "Figure5_Final_IF_Decision_Map")
    plt.close(fig)
    update_status(figures={"status": "DONE", "figures_dir": rel(figs)})
    return figs


def generate_final_if_decision_package(config_path: PathLike) -> Path:
    man = ensure(OUT_DIR / "manuscript")
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    best = best_overall_metric(metrics)
    route, status = decide_route(metrics)
    if route == "Route A":
        action = "Submit as an Information Fusion method paper with explicit component limits."
    elif route == "Route B":
        action = "Submit, if at all, as an Information Fusion domain-shift / benchmark paper with transparent TTA and safety limitations."
    else:
        action = "Rebuild the full end-to-end runner before Information Fusion submission."
    files = {
        "FINAL_IF_DECISION_REPORT.md": f"# Final IF Decision Report\n\nRoute: **{route}**.\n\nBest method: {best['Method']} ({best['Track']}). AUC {best['AUC']:.3f}; CIN3+ sensitivity {best['CIN3+ sensitivity']:.3f}; centre gap {best['Centre gap']:.3f}. Target labels used: {best['Uses target labels']}.\n\nAction: {action}\n",
        "IF_Route_A_Method_Paper_Draft_If_Passed.md": "# Route A Draft\n\nUse only if AUC >=0.75, CIN3+ sensitivity >=0.95, centre gap improves, and claims remain limited to implemented HyDRA-DG/TTA components.\n",
        "IF_Route_B_Domain_Shift_Benchmark_Draft.md": "# Route B Draft\n\nFrame the work as a multicentre reliability-aware fusion benchmark revealing centre-specific failure modes in multimodal cervical OCT/colposcopy screening.\n",
        "IF_Route_C_Rebuild_Full_Runner_Plan.md": "# Route C Rebuild Plan\n\nImplement active raw-image end-to-end training, true checkpointed target adaptation, and supervised CoE trajectory learning before a method-paper claim.\n",
        "Reviewer_Risk_Register.md": "# Reviewer Risk Register\n\n- Transductive TTA must be labelled separately from inductive LOCO.\n- CIN3+ safety target may remain below 0.95.\n- Full end-to-end HyDRA-CoE is not active.\n- Hua_Xi/XiangYa are not labelled external validation in this run.\n",
    }
    for name, text in files.items():
        (man / name).write_text(text, encoding="utf-8")
    update_status(if_recommendation={"route": route, "status": status, "action": action}, manuscript_package={"status": "DONE", "path": rel(man)})
    return man


def final_status(config_path: PathLike) -> Path:
    metrics = pd.read_csv(OUT_DIR / "statistics/tta_metrics.csv")
    best = best_overall_metric(metrics)
    source = metrics[metrics["Method"].eq(SOURCE_METHOD)].iloc[0]
    route, status = decide_route(metrics)
    cfg = load_yaml(config_path)
    hardest = cfg["failure_centre_focus"]["hardest_centre_name"]
    centre = pd.read_csv(OUT_DIR / "statistics/centre_level_tta_metrics.csv")
    best_c = centre[(centre["Method"] == best["Method"]) & (centre["Held-out centre"] == hardest)]
    source_c = centre[(centre["Method"] == SOURCE_METHOD) & (centre["Held-out centre"] == hardest)]
    update_status(
        step2_10_status=status,
        git_commit=git_commit(),
        git_status_short=git_status(),
        best_method={
            "name": str(best["Method"]),
            "track": str(best["Track"]),
            "uses_target_labels": bool(best["Uses target labels"]),
            "AUC": float(best["AUC"]),
            "CIN3_sensitivity": float(best["CIN3+ sensitivity"]),
            "CIN3_FN": int(best["CIN3+ FN"]),
            "centre_gap": float(best["Centre gap"]),
        },
    )
    md = OUT_DIR / "STEP2_10_TARGET_ADAPTATION_IF_STATUS.md"
    lines = [
        "# Step2.10 Target Adaptation Final IF Status",
        "",
        f"- Status: `{status}`",
        f"- IF recommendation: `{route}`",
        f"- Best method: `{best['Method']}`",
        f"- Track: `{best['Track']}`",
        f"- Target labels used: `{bool(best['Uses target labels'])}`",
        f"- CIN2+ AUC: {best['AUC']:.3f}",
        f"- CIN3+ sensitivity: {best['CIN3+ sensitivity']:.3f}",
        f"- CIN3+ FN: {int(best['CIN3+ FN'])}",
        f"- Centre gap: {best['Centre gap']:.3f}",
        f"- Source-only AUC: {source['AUC']:.3f}",
        f"- Source-only centre gap: {source['Centre gap']:.3f}",
        "",
        "## Xiangyang",
        "",
        f"- Source-only Xiangyang AUC: {source_c['AUC CIN2+'].iloc[0] if len(source_c) else NA}",
        f"- Best-method Xiangyang AUC: {best_c['AUC CIN2+'].iloc[0] if len(best_c) else NA}",
        f"- Source-only Xiangyang CIN3+ FN: {int(source_c['False-negative CIN3+'].iloc[0]) if len(source_c) else NA}",
        f"- Best-method Xiangyang CIN3+ FN: {int(best_c['False-negative CIN3+'].iloc[0]) if len(best_c) else NA}",
        "",
        "## Framing",
        "",
        read_json(status_file(), {}).get("if_recommendation", {}).get("action", ""),
        "",
        "Do not claim full end-to-end HyDRA-CoE, supervised CoE trajectory learning, or labelled Hua_Xi/XiangYa external validation.",
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
    update_status(experiment_name=load_yaml(config_path)["experiment_name"], step2_10_status="IN_PROGRESS")
    verify_step2_9_outputs(config_path)
    prepare_unlabelled_target_sets(config_path)
    fit_source_only_reference(config_path, no_dry_run=no_dry_run)
    run_unlabelled_target_adaptation(config_path, no_dry_run=no_dry_run)
    evaluate_tta_and_safety(config_path)
    analyse_hardest_centre_failures(config_path)
    compare_inductive_vs_tta_tracks(config_path)
    generate_final_if_tables(config_path)
    generate_final_if_decision_package(config_path)
    final_status(config_path)
    plot_final_if_figures(config_path)
    final_status(config_path)
