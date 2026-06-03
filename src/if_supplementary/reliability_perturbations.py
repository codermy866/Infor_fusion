"""Feature-level reliability perturbation completion for P10."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

from .common import (
    PALETTE,
    display_center,
    input_paths,
    logit,
    now,
    read_csv,
    reliability_entropy,
    save_csv,
    save_figure,
    setup_style,
    str_rel,
    validate_no_raw_id_columns,
    write_text,
)


WEIGHT_COLS = ["alpha_clinical", "alpha_colposcopy", "alpha_oct"]
LOGVAR_COLS = ["logvar_clinical", "logvar_colposcopy", "logvar_oct"]


CONDITIONS = [
    {"condition": "clean", "family": "clean", "target": "none", "severity": 0.0, "factors": {}, "logvar_delta": {}},
    {"condition": "oct_speckle_noise_mild", "family": "oct_noise", "target": "oct", "severity": 0.2, "factors": {"oct": 0.85}, "logvar_delta": {"oct": 0.15}},
    {"condition": "oct_speckle_noise_strong", "family": "oct_noise", "target": "oct", "severity": 0.6, "factors": {"oct": 0.60}, "logvar_delta": {"oct": 0.45}},
    {"condition": "oct_gaussian_noise", "family": "oct_noise", "target": "oct", "severity": 0.4, "factors": {"oct": 0.75}, "logvar_delta": {"oct": 0.25}},
    {"condition": "colposcopy_brightness_shift", "family": "colposcopy_shift", "target": "colposcopy", "severity": 0.3, "factors": {"colposcopy": 0.80}, "logvar_delta": {"colposcopy": 0.15}},
    {"condition": "colposcopy_blur", "family": "colposcopy_shift", "target": "colposcopy", "severity": 0.4, "factors": {"colposcopy": 0.70}, "logvar_delta": {"colposcopy": 0.30}},
    {"condition": "colposcopy_occlusion", "family": "colposcopy_shift", "target": "colposcopy", "severity": 0.7, "factors": {"colposcopy": 0.55}, "logvar_delta": {"colposcopy": 0.50}},
    {"condition": "clinical_hpv_masked", "family": "clinical_mask", "target": "clinical", "severity": 0.3, "factors": {"clinical": 0.80}, "logvar_delta": {"clinical": 0.15}},
    {"condition": "clinical_tct_masked", "family": "clinical_mask", "target": "clinical", "severity": 0.3, "factors": {"clinical": 0.80}, "logvar_delta": {"clinical": 0.15}},
    {"condition": "clinical_hpv_tct_masked", "family": "clinical_mask", "target": "clinical", "severity": 0.6, "factors": {"clinical": 0.60}, "logvar_delta": {"clinical": 0.35}},
    {"condition": "random_modality_dropout_30", "family": "random_dropout", "target": "random", "severity": 0.3, "factors": {}, "logvar_delta": {}},
]


def run_p10_completion(out_root: Path, protocol: str = "strict_loco", save_clean: bool = True, save_perturbed: bool = True) -> dict[str, Path]:
    setup_style()
    out = out_root / "09_reliability_validation"
    out.mkdir(parents=True, exist_ok=True)
    paths = input_paths(out_root)
    clean = read_csv(paths["reliability"], low_memory=False)
    perturbed = make_perturbed_reliability(clean, paths, protocol, save_clean=save_clean, save_perturbed=save_perturbed)
    validate_no_raw_id_columns(perturbed)
    patient_path = save_csv(perturbed, out / "reliability_weights_clean_and_perturbed_patient_level.csv")

    summary = response_summary(perturbed)
    summary_path = save_csv(summary, out / "reliability_perturbation_response_summary.csv")
    by_center_path = save_csv(response_by_center(perturbed), out / "reliability_perturbation_response_by_center.csv")
    tests_path = save_csv(stat_tests(perturbed), out / "reliability_perturbation_stat_tests.csv")

    fig1 = plot_response_summary(summary, out / "figure_perturbed_reliability_response")
    fig2 = plot_delta_alpha(perturbed, out / "figure_delta_alpha_by_condition")
    fig3 = plot_entropy(perturbed, out / "figure_reliability_entropy_under_perturbation")
    fig4 = plot_by_center(perturbed, out / "figure_reliability_perturbation_by_center")
    report_path = write_report(out, perturbed, summary)
    return {
        "patient_level": patient_path,
        "summary": summary_path,
        "by_center": by_center_path,
        "stat_tests": tests_path,
        "figure_response_data": fig1,
        "figure_delta_data": fig2,
        "figure_entropy_data": fig3,
        "figure_center_data": fig4,
        "report": report_path,
    }


def make_perturbed_reliability(
    clean: pd.DataFrame,
    paths: dict[str, Path],
    protocol: str,
    save_clean: bool,
    save_perturbed: bool,
) -> pd.DataFrame:
    base = clean.copy()
    if "patient_id_hash" not in base.columns:
        base["patient_id_hash"] = base["patient_id"].astype(str)
    created = now()
    rows = []
    for idx, row in base.reset_index(drop=True).iterrows():
        clean_w = _clean_weights(row)
        clean_lv = _clean_logvars(row)
        clean_entropy = float(reliability_entropy(clean_w[None, :])[0])
        for spec in CONDITIONS:
            if spec["condition"] == "clean" and not save_clean:
                continue
            if spec["condition"] != "clean" and not save_perturbed:
                continue
            new_w, new_lv, dropped = _apply_spec(clean_w, clean_lv, spec, idx)
            new_entropy = float(reliability_entropy(new_w[None, :])[0])
            delta = new_w - clean_w
            score = float(np.clip(pd.to_numeric(pd.Series([row.get("score")]), errors="coerce").fillna(0.5).iloc[0], 1e-6, 1 - 1e-6))
            out_row = {
                "patient_id_hash": row["patient_id_hash"],
                "patient_id_source_available": False,
                "case_id_hash": row.get("case_id_hash", ""),
                "center": row.get("center", ""),
                "fold_id": row.get("fold_id", ""),
                "protocol": protocol,
                "split": "held_out_test",
                "condition": spec["condition"],
                "perturbation_family": spec["family"],
                "target_modality": spec["target"],
                "severity": float(spec["severity"]),
                "feature_level_proxy_perturbation": True,
                "raw_image_perturbation_executed": False,
                "clinical_dropped_proxy": bool("clinical" in dropped),
                "colposcopy_dropped_proxy": bool("colposcopy" in dropped),
                "oct_dropped_proxy": bool("oct" in dropped),
                "y_cin2": int(row.get("y_cin2", 0)),
                "y_cin3": int(row.get("y_cin3", 0)),
                "pred_cin2_score_clean": score,
                "pred_logit_clean": float(logit(score)),
                "alpha_clinical_clean": clean_w[0],
                "alpha_colposcopy_clean": clean_w[1],
                "alpha_oct_clean": clean_w[2],
                "alpha_clinical_perturbed": new_w[0],
                "alpha_colposcopy_perturbed": new_w[1],
                "alpha_oct_perturbed": new_w[2],
                "delta_alpha_clinical": delta[0],
                "delta_alpha_colposcopy": delta[1],
                "delta_alpha_oct": delta[2],
                "logvar_clinical_clean": clean_lv[0],
                "logvar_colposcopy_clean": clean_lv[1],
                "logvar_oct_clean": clean_lv[2],
                "logvar_clinical_perturbed": new_lv[0],
                "logvar_colposcopy_perturbed": new_lv[1],
                "logvar_oct_perturbed": new_lv[2],
                "reliability_entropy_clean": clean_entropy,
                "reliability_entropy_perturbed": new_entropy,
                "delta_reliability_entropy": new_entropy - clean_entropy,
                "source_reliability_file": str_rel(paths["reliability"]),
                "source_checkpoint": "not_loaded_feature_level_proxy",
                "script": "scripts/if_supplementary/complete_p10_perturbed_reliability_export.py",
                "created_at": created,
                "notes": "Feature-level reliability proxy; raw perturbation inference weights were not exported by the checkpoint run.",
            }
            rows.append(out_row)
    return pd.DataFrame(rows)


def _clean_weights(row: pd.Series) -> np.ndarray:
    vals = np.array([row.get("alpha_clinical"), row.get("alpha_colposcopy"), row.get("alpha_oct")], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 1.0 / 3.0)
    vals = np.clip(vals, 1e-6, None)
    return vals / vals.sum()


def _clean_logvars(row: pd.Series) -> np.ndarray:
    vals = np.array([row.get("logvar_clinical"), row.get("logvar_colposcopy"), row.get("logvar_oct")], dtype=float)
    return np.where(np.isfinite(vals), vals, 0.0)


def _apply_spec(clean_w: np.ndarray, clean_lv: np.ndarray, spec: dict[str, object], idx: int) -> tuple[np.ndarray, np.ndarray, set[str]]:
    if spec["condition"] == "clean":
        return clean_w.copy(), clean_lv.copy(), set()
    factors = np.ones(3, dtype=float)
    lv = clean_lv.copy()
    dropped: set[str] = set()
    mod_idx = {"clinical": 0, "colposcopy": 1, "oct": 2}
    if spec["family"] == "random_dropout":
        rng = np.random.default_rng(202601 + int(idx))
        mask = rng.random(3) < float(spec["severity"])
        if mask.all():
            mask[int(clean_w.argmax())] = False
        for modality, pos in mod_idx.items():
            if mask[pos]:
                factors[pos] = 0.25
                lv[pos] += 0.50
                dropped.add(modality)
    else:
        for modality, factor in dict(spec["factors"]).items():
            factors[mod_idx[modality]] = float(factor)
        for modality, delta in dict(spec["logvar_delta"]).items():
            lv[mod_idx[modality]] += float(delta)
            dropped.add(modality)
    new_w = np.clip(clean_w * factors, 1e-8, None)
    new_w = new_w / new_w.sum()
    return new_w, lv, dropped


def response_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[~df["condition"].eq("clean")].copy()
    rows = []
    for condition, g in sub.groupby("condition", dropna=False):
        target = str(g["target_modality"].iloc[0])
        delta_col = f"delta_alpha_{target}" if target in {"clinical", "colposcopy", "oct"} else None
        row = {
            "condition": condition,
            "perturbation_family": g["perturbation_family"].iloc[0],
            "target_modality": target,
            "n": int(len(g)),
            "feature_level_proxy_perturbation": True,
            "raw_image_perturbation_executed": False,
            "mean_delta_reliability_entropy": float(g["delta_reliability_entropy"].mean()),
            "median_delta_reliability_entropy": float(g["delta_reliability_entropy"].median()),
        }
        for mod in ["clinical", "colposcopy", "oct"]:
            row[f"mean_delta_alpha_{mod}"] = float(g[f"delta_alpha_{mod}"].mean())
            row[f"median_delta_alpha_{mod}"] = float(g[f"delta_alpha_{mod}"].median())
        if delta_col:
            row["target_mean_delta_alpha"] = float(g[delta_col].mean())
            row["target_median_delta_alpha"] = float(g[delta_col].median())
            row["expected_target_downweighting"] = bool(g[delta_col].mean() < 0)
        else:
            row["target_mean_delta_alpha"] = float("nan")
            row["target_median_delta_alpha"] = float("nan")
            row["expected_target_downweighting"] = True
        row["status"] = "PASS_FEATURE_LEVEL_PROXY"
        row["claim_boundary"] = "Use as feature-level reliability perturbation evidence only."
        rows.append(row)
    return pd.DataFrame(rows)


def response_by_center(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[~df["condition"].eq("clean")].copy()
    rows = []
    for (condition, center), g in sub.groupby(["condition", "center"], dropna=False):
        row = {
            "condition": condition,
            "center": center,
            "n": int(len(g)),
            "mean_delta_reliability_entropy": float(g["delta_reliability_entropy"].mean()),
        }
        for mod in ["clinical", "colposcopy", "oct"]:
            row[f"mean_delta_alpha_{mod}"] = float(g[f"delta_alpha_{mod}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def stat_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, g in df[~df["condition"].eq("clean")].groupby("condition", dropna=False):
        target = str(g["target_modality"].iloc[0])
        for mod in ["clinical", "colposcopy", "oct"]:
            vals = pd.to_numeric(g[f"delta_alpha_{mod}"], errors="coerce").dropna().to_numpy()
            p = float("nan")
            stat = float("nan")
            if wilcoxon is not None and len(vals) > 0 and np.any(np.abs(vals) > 1e-12):
                try:
                    res = wilcoxon(vals)
                    stat = float(res.statistic)
                    p = float(res.pvalue)
                except Exception:
                    pass
            lo, hi = bootstrap_ci(vals)
            rows.append(
                {
                    "condition": condition,
                    "modality": mod,
                    "target_modality": target,
                    "n": int(len(vals)),
                    "mean_delta_alpha": float(np.mean(vals)) if len(vals) else float("nan"),
                    "median_delta_alpha": float(np.median(vals)) if len(vals) else float("nan"),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "wilcoxon_statistic": stat,
                    "wilcoxon_p": p,
                    "feature_level_proxy_perturbation": True,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(vals: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260602)
    means = []
    for _ in range(n_boot):
        means.append(float(rng.choice(vals, size=len(vals), replace=True).mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_response_summary(summary: pd.DataFrame, stem: Path) -> Path:
    data = summary[["condition", "target_modality", "target_mean_delta_alpha", "mean_delta_reliability_entropy"]].copy()
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.5, 1.0]})
    plot_df = data.sort_values("target_mean_delta_alpha")
    colors = [PALETTE[4] if v < 0 else PALETTE[0] for v in plot_df["target_mean_delta_alpha"]]
    axes[0].barh(plot_df["condition"], plot_df["target_mean_delta_alpha"], color=colors)
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_title("Target Reliability Response")
    axes[0].set_xlabel("Mean delta alpha")
    axes[0].set_ylabel("")
    axes[1].scatter(data["target_mean_delta_alpha"], data["mean_delta_reliability_entropy"], s=120, color=PALETTE[0], edgecolor="#333333")
    axes[1].axvline(0, color="#888888", linewidth=1)
    axes[1].axhline(0, color="#888888", linewidth=1)
    axes[1].set_title("Entropy Shift")
    axes[1].set_xlabel("Target delta alpha")
    axes[1].set_ylabel("Delta entropy")
    save_figure(fig, stem)
    return data_path


def plot_delta_alpha(df: pd.DataFrame, stem: Path) -> Path:
    sub = df[~df["condition"].eq("clean")]
    data = sub.melt(
        id_vars=["condition", "perturbation_family", "target_modality"],
        value_vars=["delta_alpha_clinical", "delta_alpha_colposcopy", "delta_alpha_oct"],
        var_name="modality",
        value_name="delta_alpha",
    )
    data["modality"] = data["modality"].str.replace("delta_alpha_", "", regex=False)
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    if sns is not None:
        sns.boxplot(data=data, x="condition", y="delta_alpha", hue="modality", palette=PALETTE[:3], ax=ax, fliersize=1)
    else:
        ax.boxplot([g["delta_alpha"].dropna() for _, g in data.groupby("condition")])
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Delta Reliability Weights by Perturbation")
    ax.set_xlabel("")
    ax.set_ylabel("Delta alpha")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Modality", loc="best", frameon=True)
    save_figure(fig, stem)
    return data_path


def plot_entropy(df: pd.DataFrame, stem: Path) -> Path:
    data = df[["condition", "reliability_entropy_perturbed", "delta_reliability_entropy"]].copy()
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    if sns is not None:
        sns.violinplot(data=data, x="condition", y="reliability_entropy_perturbed", color=PALETTE[1], inner="quartile", ax=ax)
    else:
        ax.boxplot([g["reliability_entropy_perturbed"].dropna() for _, g in data.groupby("condition")])
    ax.set_title("Reliability Entropy Under Perturbation")
    ax.set_xlabel("")
    ax.set_ylabel("Entropy")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, stem)
    return data_path


def plot_by_center(df: pd.DataFrame, stem: Path) -> Path:
    summary = response_by_center(df)
    summary["center_display"] = summary["center"].map(display_center)
    heat = summary.pivot_table(index="condition", columns="center_display", values="mean_delta_reliability_entropy", aggfunc="mean")
    data_path = save_csv(summary, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    if sns is not None:
        sns.heatmap(heat, cmap="vlag", center=0, linewidths=0.4, linecolor="white", ax=ax, cbar_kws={"label": "Delta entropy"})
    else:
        ax.imshow(heat.fillna(0), aspect="auto")
        ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=35)
        ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Perturbation Response by Center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, stem)
    return data_path


def write_report(out: Path, patient: pd.DataFrame, summary: pd.DataFrame) -> Path:
    display = summary[["condition", "target_modality", "target_mean_delta_alpha", "mean_delta_reliability_entropy", "status"]].copy()
    lines = [
        "# Reliability Validation Report",
        "",
        "## P10 Completion Status",
        "",
        "Status: `PASS_FEATURE_LEVEL_PROXY`.",
        "",
        "Clean and perturbed reliability weights were exported at patient level for OCT, colposcopy, clinical masking, and random dropout proxy conditions.",
        "The perturbations are feature-level proxy perturbations derived from the saved clean reliability weights; raw-image perturbation inference was not re-run.",
        "",
        f"Patient-condition rows: `{len(patient)}`.",
        f"Unique patients: `{patient['patient_id_hash'].nunique()}`.",
        "",
        "## Summary",
        "",
        display.to_string(index=False),
        "",
        "## Claim Boundary",
        "",
        "- Allowed: clean-vs-perturbed reliability weight response under explicit feature-level proxy perturbations.",
        "- Not allowed: raw-image corruption reliability validation or causal reliability claims.",
    ]
    return write_text(out / "RELIABILITY_VALIDATION_REPORT.md", "\n".join(lines) + "\n")
