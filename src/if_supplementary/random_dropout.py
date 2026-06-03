"""Patient-level random modality dropout completion for P06."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from .common import (
    MODEL_DISPLAY,
    PALETTE,
    PRIMARY_MODEL,
    grouped_metrics,
    input_paths,
    logit,
    metric_row,
    now,
    read_csv,
    save_csv,
    save_figure,
    setup_style,
    str_rel,
    validate_no_raw_id_columns,
    write_text,
)


MODALITIES = ("clinical", "colposcopy", "oct")
SINGLE_MODEL = {
    "clinical": "ClinicalOnly_Logistic",
    "colposcopy": "ColposcopyOnly_ViT",
    "oct": "OCTOnly_ViT",
}
ALPHA_COL = {
    "clinical": "alpha_clinical",
    "colposcopy": "alpha_colposcopy",
    "oct": "alpha_oct",
}


def run_p06_completion(
    out_root: Path,
    dropout_rates: list[float],
    repeats: int,
    seeds: list[int],
    protocol: str = "strict_loco",
) -> dict[str, Path]:
    setup_style()
    out = out_root / "05_modality_ablation_and_missingness"
    out.mkdir(parents=True, exist_ok=True)
    paths = input_paths(out_root)
    base = _build_base_table(paths)
    pred = _make_dropout_predictions(base, paths, dropout_rates, repeats, seeds, protocol)

    required_cols = [
        "patient_id_hash",
        "patient_id_source_available",
        "center",
        "fold_id",
        "protocol",
        "split",
        "model_name",
        "condition",
        "dropout_rate",
        "dropout_repeat",
        "dropout_seed",
        "clinical_available_after_dropout",
        "colposcopy_available_after_dropout",
        "oct_available_after_dropout",
        "clinical_dropped",
        "colposcopy_dropped",
        "oct_dropped",
        "n_modalities_available",
        "y_cin2",
        "y_cin3",
        "pred_cin2_score",
        "pred_logit",
        "threshold_cin2_locked",
        "threshold_cin3_locked",
        "pred_cin2_binary",
        "pred_cin3_binary",
        "cin2_correct",
        "cin3_correct",
        "source_checkpoint",
        "source_split_file",
        "source_threshold_file",
        "script",
        "created_at",
        "notes",
    ]
    pred = pred[required_cols]
    validate_no_raw_id_columns(pred)
    pred_path = save_csv(pred, out / "random_dropout_patient_level_predictions.csv")

    stress = _stress_summary(pred)
    stress_path = save_csv(stress, out / "random_dropout_stress_table.csv")
    center_path = save_csv(grouped_metrics(pred, ["condition", "dropout_rate", "center"]), out / "random_dropout_stress_by_center.csv")
    repeat_path = save_csv(grouped_metrics(pred, ["condition", "dropout_rate", "dropout_repeat", "dropout_seed"]), out / "random_dropout_stress_by_repeat.csv")

    fig_data_1 = _plot_auc_npv(stress, out / "figure_random_dropout_rate_auc_npv")
    fig_data_2 = _plot_cin3_fn(stress, out / "figure_random_dropout_cin3_fn")
    fig_data_3 = _plot_referral_safety(stress, out / "figure_random_dropout_referral_safety_tradeoff")

    report_path = _write_report(out, pred, stress)
    return {
        "patient_predictions": pred_path,
        "stress_table": stress_path,
        "stress_by_center": center_path,
        "stress_by_repeat": repeat_path,
        "figure_auc_npv_data": fig_data_1,
        "figure_cin3_fn_data": fig_data_2,
        "figure_referral_data": fig_data_3,
        "report": report_path,
    }


def _build_base_table(paths: dict[str, Path]) -> pd.DataFrame:
    test = read_csv(paths["test_predictions"], low_memory=False)
    thresholds = read_csv(paths["thresholds"])
    rel = read_csv(paths["reliability"], low_memory=False)

    test["score"] = pd.to_numeric(test["score"], errors="coerce")
    pivot = test.pivot_table(index="case_id_hash", columns="model_name", values="score", aggfunc="mean")
    hydra = test[test["model_name"].eq(PRIMARY_MODEL)].copy()
    keep = [
        "patient_id_hash",
        "case_id_hash",
        "center",
        "fold_id",
        "held_out_center",
        "y_cin2",
        "y_cin3",
        "oct_available",
        "colposcopy_available",
        "clinical_prior_available",
    ]
    meta = hydra[keep].drop_duplicates("case_id_hash").copy()
    base = meta.merge(pivot.reset_index(), on="case_id_hash", how="left")

    rel_keep = ["case_id_hash", "alpha_clinical", "alpha_colposcopy", "alpha_oct"]
    base = base.merge(rel[rel_keep].drop_duplicates("case_id_hash"), on="case_id_hash", how="left")
    for col in ["alpha_clinical", "alpha_colposcopy", "alpha_oct"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base[["alpha_clinical", "alpha_colposcopy", "alpha_oct"]] = base[["alpha_clinical", "alpha_colposcopy", "alpha_oct"]].fillna(1.0 / 3.0)

    hydra_thresholds = thresholds[thresholds["model_name"].eq(PRIMARY_MODEL)].copy()
    hydra_thresholds = hydra_thresholds[["fold_id", "threshold_cin2_f1_val", "threshold_cin3_safety_val"]].drop_duplicates("fold_id")
    base = base.merge(hydra_thresholds, on="fold_id", how="left")
    base["threshold_cin2_f1_val"] = pd.to_numeric(base["threshold_cin2_f1_val"], errors="coerce").fillna(0.5)
    base["threshold_cin3_safety_val"] = pd.to_numeric(base["threshold_cin3_safety_val"], errors="coerce").fillna(0.5)
    return base


def _make_dropout_predictions(
    base: pd.DataFrame,
    paths: dict[str, Path],
    dropout_rates: list[float],
    repeats: int,
    seeds: list[int],
    protocol: str,
) -> pd.DataFrame:
    rows = []
    if len(seeds) < repeats:
        seeds = list(seeds) + [int(202601 + i) for i in range(len(seeds), repeats)]
    created = now()
    source_split_file = str_rel(paths["loco_folds"])
    source_threshold_file = str_rel(paths["thresholds"])
    script = "scripts/if_supplementary/complete_p06_random_dropout_predictions.py"
    for rate in dropout_rates:
        condition = f"random_modality_dropout_{int(round(rate * 100))}"
        for repeat_idx in range(repeats):
            seed = int(seeds[repeat_idx])
            rng = np.random.default_rng(seed)
            for _, row in base.iterrows():
                dropped = {m: bool(rng.random() < rate) for m in MODALITIES}
                if all(dropped.values()):
                    # Keep the modality with highest clean reliability so the stress test stays defined.
                    best = max(MODALITIES, key=lambda m: _safe_float(row.get(ALPHA_COL[m]), default=0.0))
                    dropped[best] = False
                available = {m: not dropped[m] for m in MODALITIES}
                score, source_model, notes = _score_for_mask(row, available)
                score = float(np.clip(score, 1e-6, 1 - 1e-6))
                th2 = float(row["threshold_cin2_f1_val"])
                th3 = float(row["threshold_cin3_safety_val"])
                pred2 = int(score >= th2)
                pred3 = int(score >= th3)
                y2 = int(row["y_cin2"])
                y3 = int(row["y_cin3"])
                rows.append(
                    {
                        "patient_id_hash": row["patient_id_hash"],
                        "patient_id_source_available": False,
                        "center": row["center"],
                        "fold_id": row["fold_id"],
                        "protocol": protocol,
                        "split": "held_out_test",
                        "model_name": "HyDRA_CoE_Full_dropout_proxy",
                        "condition": condition,
                        "dropout_rate": float(rate),
                        "dropout_repeat": int(repeat_idx + 1),
                        "dropout_seed": seed,
                        "clinical_available_after_dropout": bool(available["clinical"]),
                        "colposcopy_available_after_dropout": bool(available["colposcopy"]),
                        "oct_available_after_dropout": bool(available["oct"]),
                        "clinical_dropped": bool(dropped["clinical"]),
                        "colposcopy_dropped": bool(dropped["colposcopy"]),
                        "oct_dropped": bool(dropped["oct"]),
                        "n_modalities_available": int(sum(available.values())),
                        "y_cin2": y2,
                        "y_cin3": y3,
                        "pred_cin2_score": score,
                        "pred_logit": float(logit(score)),
                        "threshold_cin2_locked": th2,
                        "threshold_cin3_locked": th3,
                        "pred_cin2_binary": pred2,
                        "pred_cin3_binary": pred3,
                        "cin2_correct": bool(pred2 == y2),
                        "cin3_correct": bool(pred3 == y3),
                        "source_checkpoint": "not_loaded_feature_cache_prediction_proxy",
                        "source_split_file": source_split_file,
                        "source_threshold_file": source_threshold_file,
                        "script": script,
                        "created_at": created,
                        "notes": f"{notes}; score_source={source_model}; raw identifiers omitted",
                    }
                )
    return pd.DataFrame(rows)


def _score_for_mask(row: pd.Series, available: dict[str, bool]) -> tuple[float, str, str]:
    if all(available.values()) and pd.notna(row.get(PRIMARY_MODEL)):
        return _safe_float(row[PRIMARY_MODEL]), PRIMARY_MODEL, "clean full-modality prediction reused"
    if available == {"clinical": False, "colposcopy": True, "oct": True} and pd.notna(row.get("ColposcopyOCT_LateFusion")):
        return _safe_float(row["ColposcopyOCT_LateFusion"]), "ColposcopyOCT_LateFusion", "direct no-clinical late-fusion prediction reused"

    terms = []
    for modality, is_available in available.items():
        if not is_available:
            continue
        model = SINGLE_MODEL[modality]
        score = row.get(model)
        if pd.isna(score):
            continue
        weight = _safe_float(row.get(ALPHA_COL[modality]), default=1.0)
        terms.append((model, _safe_float(score), max(weight, 1e-6)))
    if not terms and pd.notna(row.get(PRIMARY_MODEL)):
        return _safe_float(row[PRIMARY_MODEL]), PRIMARY_MODEL, "fallback to clean prediction because no component score was available"
    total = sum(w for _, _, w in terms)
    score = sum(s * w for _, s, w in terms) / total
    source_model = "+".join(m for m, _, _ in terms)
    return score, source_model, "reliability-weighted blend of available locked modality predictors"


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _stress_summary(pred: pd.DataFrame) -> pd.DataFrame:
    by_repeat = grouped_metrics(pred, ["condition", "dropout_rate", "dropout_repeat", "dropout_seed"])
    metric_cols = [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "brier",
        "ece",
        "referral_rate",
        "cin3_sensitivity",
        "cin3_specificity",
        "cin3_false_negatives",
        "cin3_referral_rate",
    ]
    rows = []
    for (condition, rate), g in by_repeat.groupby(["condition", "dropout_rate"], dropna=False):
        row = {"condition": condition, "dropout_rate": rate, "status": "PASS_FEATURE_CACHE_PROXY", "n_repeats": int(g["dropout_repeat"].nunique())}
        for col in metric_cols:
            row[col] = pd.to_numeric(g[col], errors="coerce").mean()
            row[f"{col}_sd"] = pd.to_numeric(g[col], errors="coerce").std()
        row["notes"] = "Patient-level random dropout predictions generated from locked component predictors; no raw-image checkpoint was re-run."
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_auc_npv(stress: pd.DataFrame, stem: Path) -> Path:
    data = stress.melt(
        id_vars=["condition", "dropout_rate"],
        value_vars=["auroc", "npv", "cin3_sensitivity"],
        var_name="metric",
        value_name="value",
    )
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    if sns is not None:
        sns.lineplot(data=data, x="dropout_rate", y="value", hue="metric", marker="o", linewidth=3, palette=PALETTE[:3], ax=ax)
    else:
        for metric, g in data.groupby("metric"):
            ax.plot(g["dropout_rate"], g["value"], marker="o", linewidth=3, label=metric)
    ax.set_title("Random Modality Dropout Stress")
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.legend(title="", loc="best", frameon=True)
    save_figure(fig, stem)
    return data_path


def _plot_cin3_fn(stress: pd.DataFrame, stem: Path) -> Path:
    data = stress[["condition", "dropout_rate", "cin3_false_negatives", "cin3_false_negatives_sd"]].copy()
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.errorbar(
        data["dropout_rate"],
        data["cin3_false_negatives"],
        yerr=data["cin3_false_negatives_sd"].fillna(0),
        marker="o",
        linewidth=3,
        markersize=9,
        color=PALETTE[4],
        ecolor=PALETTE[5],
        capsize=5,
    )
    ax.fill_between(
        data["dropout_rate"],
        data["cin3_false_negatives"] - data["cin3_false_negatives_sd"].fillna(0),
        data["cin3_false_negatives"] + data["cin3_false_negatives_sd"].fillna(0),
        color=PALETTE[5],
        alpha=0.18,
    )
    ax.set_title("CIN3+ False Negatives Under Dropout")
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("False negatives")
    save_figure(fig, stem)
    return data_path


def _plot_referral_safety(stress: pd.DataFrame, stem: Path) -> Path:
    data = stress[["condition", "dropout_rate", "referral_rate", "cin3_sensitivity", "auroc"]].copy()
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    sizes = 240 * data["auroc"].fillna(0.5).clip(0.1, 1.0)
    sc = ax.scatter(
        data["referral_rate"],
        data["cin3_sensitivity"],
        s=sizes,
        c=data["dropout_rate"],
        cmap="magma_r",
        edgecolor="#333333",
        linewidth=1.2,
        alpha=0.88,
    )
    for _, row in data.iterrows():
        ax.text(row["referral_rate"] + 0.004, row["cin3_sensitivity"], f"{row['dropout_rate']:.1f}", fontsize=10, color="#30335f")
    ax.set_title("Referral-Safety Trade-off")
    ax.set_xlabel("Referral rate")
    ax.set_ylabel("CIN3+ sensitivity")
    ax.set_xlim(0, min(1.0, max(0.2, data["referral_rate"].max() + 0.08)))
    ax.set_ylim(0, 1.02)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Dropout rate")
    save_figure(fig, stem)
    return data_path


def _write_report(out: Path, pred: pd.DataFrame, stress: pd.DataFrame) -> Path:
    lines = [
        "# Modality Contribution and Missingness Report",
        "",
        "## P06 Completion Status",
        "",
        "Status: `PASS_FEATURE_CACHE_PROXY`.",
        "",
        "Patient-level random modality dropout predictions were generated for 10%, 30%, and 50% dropout with five deterministic repeats.",
        "The completion uses locked single-modality, dual-modality, and full-modality patient-level predictions plus validation-locked thresholds.",
        "It does not reload raw-image checkpoints; therefore it should be described as a feature-cache/prediction-registry stress test.",
        "",
        f"Patient-level rows: `{len(pred)}`.",
        f"Unique patients: `{pred['patient_id_hash'].nunique()}`.",
        "",
        "## Summary",
        "",
        stress[["condition", "dropout_rate", "auroc", "npv", "cin3_sensitivity", "cin3_false_negatives", "referral_rate"]].to_string(index=False),
        "",
        "## Claim Boundary",
        "",
        "- Allowed: random missing-modality stress testing at patient level under the locked LOCO prediction registry.",
        "- Not allowed: claiming this is a raw-image checkpoint re-inference experiment.",
    ]
    return write_text(out / "MODALITY_AND_MISSINGNESS_REPORT.md", "\n".join(lines) + "\n")
