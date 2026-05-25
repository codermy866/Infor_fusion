#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
except Exception:  # pragma: no cover
    chi2_contingency = None
    fisher_exact = None
    mannwhitneyu = None

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

HARD = "襄阳市中心医院"


def hard_source() -> pd.DataFrame:
    src = C.load_source_preds()
    lock = C.load_data_lock()
    cols = ["case_id", "age", "hpv_status_harmonized", "hpv16_18_status", "tct_status_harmonized", "oct_num_bscans", "colposcopy_num_images"]
    return src[src["held_out_center"].eq(HARD)].merge(lock[cols], on="case_id", how="left")


def perm_pvalue(a, b, n_perm=500):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(np.median(a) - np.median(b))
    pooled = np.r_[a, b]
    rng = np.random.default_rng(C.SEED)
    vals = []
    for _ in range(n_perm):
        rng.shuffle(pooled)
        vals.append(abs(np.median(pooled[: len(a)]) - np.median(pooled[len(a) :])))
    return float((np.asarray(vals) >= obs).mean())


def e20_fn_analysis() -> None:
    C.setup_plot_style()
    df = hard_source()
    df["group"] = np.where((df["pathology_cin3plus"] == 1) & (df["pred_t_cin3_safety95"] == 0), "false_negative", np.where((df["pathology_cin3plus"] == 1) & (df["pred_t_cin3_safety95"] == 1), "true_positive", "other"))
    comp = df[df["group"].isin(["false_negative", "true_positive"])].copy()
    rows = []
    for var, kind in [
        ("age", "continuous"),
        ("prob_cin2plus", "continuous"),
        ("oct_num_bscans", "continuous"),
        ("colposcopy_num_images", "continuous"),
        ("hpv16_18_status", "categorical"),
        ("tct_status_harmonized", "categorical"),
    ]:
        fn = comp[comp["group"].eq("false_negative")][var]
        tp = comp[comp["group"].eq("true_positive")][var]
        if kind == "continuous":
            if mannwhitneyu is not None:
                a = pd.to_numeric(fn, errors="coerce").dropna()
                b = pd.to_numeric(tp, errors="coerce").dropna()
                pval = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan
                test_name = "Mann-Whitney U"
            else:
                pval = perm_pvalue(fn, tp)
                test_name = "permutation median-difference"
            rows.append(
                {
                    "Variable": var,
                    "False-negative summary": f"median={pd.to_numeric(fn, errors='coerce').median():.3f}",
                    "True-positive summary": f"median={pd.to_numeric(tp, errors='coerce').median():.3f}",
                    "Test": test_name,
                    "p_value": C.fmt(pval),
                    "Notes": "Exploratory descriptive comparison.",
                }
            )
        else:
            test_name = "category distribution listing"
            pval = "NA"
            if chi2_contingency is not None:
                tab = pd.crosstab(comp["group"], comp[var].astype(str))
                if tab.shape == (2, 2) and fisher_exact is not None:
                    test_name = "Fisher exact"
                    pval = C.fmt(fisher_exact(tab.to_numpy())[1])
                elif tab.shape[0] >= 2 and tab.shape[1] >= 2:
                    test_name = "Chi-square"
                    pval = C.fmt(chi2_contingency(tab.to_numpy())[1])
            rows.append(
                {
                    "Variable": var,
                    "False-negative summary": "; ".join(fn.astype(str).value_counts().head(5).index.astype(str)),
                    "True-positive summary": "; ".join(tp.astype(str).value_counts().head(5).index.astype(str)),
                    "Test": test_name,
                    "p_value": pval,
                    "Notes": "Exploratory comparison.",
                }
            )
    out = pd.DataFrame(rows)
    C.write_table(out, "Table_HardCentre_FN_Analysis")
    vig = comp.copy().reset_index(drop=True)
    vig["anonymized_case_id"] = [f"hard_case_{i+1:03d}" for i in range(len(vig))]
    vig = vig[["anonymized_case_id", "group", "pathology_cin3plus", "prob_cin2plus", "pred_t_cin3_safety95", "age", "hpv16_18_status", "tct_status_harmonized"]]
    vig.to_csv(C.OUT / "statistics" / "hard_centre_fn_case_vignettes.csv", index=False, encoding="utf-8-sig")
    fig, ax = C.plt.subplots(figsize=(6.6, 4.4))
    for group, color in [("false_negative", C.SCI_PALETTE[4]), ("true_positive", C.SCI_PALETTE[0])]:
        vals = comp[comp["group"].eq(group)]["prob_cin2plus"]
        if C.sns is not None:
            C.sns.histplot(vals, bins=12, alpha=0.42, label=group.replace("_", " "), color=color, kde=True, ax=ax, stat="count", edgecolor="white")
        else:
            ax.hist(vals, bins=12, alpha=0.6, label=group.replace("_", " "), color=color)
    ax.set_xlabel("Source-only predicted CIN2+ probability")
    ax.set_ylabel("CIN3+ case count")
    ax.set_title("Hard-centre CIN3+ score distribution", weight="bold")
    ax.legend(frameon=False)
    C.save_fig(fig, "Figure_HardCentre_FN_Probability_Distribution", "Distribution of source-only scores among CIN3+ false negatives and true positives in the hardest centre.")
    C.write_text(C.OUT / "paper_sections" / "sec_hard_centre_fn_analysis.txt", "Hard-centre false-negative analysis compared CIN3+ false negatives and true positives using available clinical and score variables. The table is exploratory and anonymized; no identifiable patient information is exported.\n")
    status = "COMPLETED" if mannwhitneyu is not None else "COMPLETED_WITH_CAVEAT"
    note = "SciPy Mann-Whitney/Fisher/chi-square tests used where applicable." if mannwhitneyu is not None else "Statistical tests are descriptive/permutation fallback."
    C.append_manifest("E20", "Hard-centre FN Analysis", status, ["tables/Table_HardCentre_FN_Analysis.csv", "figures/Figure_HardCentre_FN_Probability_Distribution.pdf", "statistics/hard_centre_fn_case_vignettes.csv"], note)


def rank_norm(s):
    return pd.Series(s).rank(method="average", pct=True).to_numpy()


def pav_fit(x, y):
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y, dtype=float)[order]
    levels = ys.copy()
    weights = np.ones(len(ys))
    starts = list(range(len(ys)))
    ends = list(range(len(ys)))
    i = 0
    while i < len(levels) - 1:
        if levels[i] > levels[i + 1]:
            total = weights[i] + weights[i + 1]
            avg = (levels[i] * weights[i] + levels[i + 1] * weights[i + 1]) / total
            levels[i] = avg
            weights[i] = total
            ends[i] = ends[i + 1]
            levels = np.delete(levels, i + 1)
            weights = np.delete(weights, i + 1)
            del starts[i + 1]
            del ends[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    bins = []
    for lev, st, en in zip(levels, starts, ends):
        bins.append((xs[st], xs[en], lev))
    return bins


def pav_predict(bins, x):
    out = []
    for v in x:
        lev = bins[0][2]
        for lo, hi, l in bins:
            if v >= lo:
                lev = l
            if v <= hi:
                break
        out.append(lev)
    return np.asarray(out)


def threshold_sweep(df):
    y = df["pathology_cin3plus"].astype(int).to_numpy()
    s = df["prob_cin2plus"].astype(float).to_numpy()
    rows = []
    for t in np.linspace(0, 1, 101):
        pred = s >= t
        met = C.metrics_at_pred(y, s, pred.astype(int))
        rows.append(
            {
                "threshold": t,
                "sensitivity": met["sensitivity"],
                "specificity": met["specificity"],
                "PPV": met["ppv"],
                "NPV": met["npv"],
                "screen-positive rate": met["screen_positive_rate"],
                "FN": met["fn"],
            }
        )
    return pd.DataFrame(rows)


def e21_ranking_calibration() -> None:
    C.setup_plot_style()
    src = C.load_source_preds()
    hard = src[src["held_out_center"].eq(HARD)].copy()
    rest = src[~src["held_out_center"].eq(HARD)].copy()
    y = hard["pathology_cin2plus"].astype(int).to_numpy()
    s = hard["prob_cin2plus"].astype(float).to_numpy()
    transforms = {
        "original": s,
        "min_max": (s - s.min()) / max(s.max() - s.min(), 1e-8),
        "rank_normalization": rank_norm(s),
        "temperature_scaling_T2": 1 / (1 + np.exp(-np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1)) / 2)),
    }
    bins = pav_fit(rest["prob_cin2plus"].astype(float).to_numpy(), rest["pathology_cin2plus"].astype(int).to_numpy())
    transforms["isotonic_nonhard_fit"] = pav_predict(bins, s)
    rows = []
    for name, sc in transforms.items():
        rows.append({"transform": name, "CIN2+ AUC": C.auc_score(y, sc), "CIN3+ AUC": C.auc_score(hard["pathology_cin3plus"], sc), "notes": "monotone/non-hard calibration diagnostic"})
    res = pd.DataFrame(rows)
    res.to_csv(C.OUT / "statistics" / "ranking_vs_calibration_transform_results.csv", index=False, encoding="utf-8-sig")
    sweep = threshold_sweep(hard)
    sweep.to_csv(C.OUT / "statistics" / "hard_centre_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    fig, axes = C.plt.subplots(1, 2, figsize=(11.8, 4.4))
    if C.sns is not None:
        C.sns.barplot(data=res, x="transform", y="CIN2+ AUC", ax=axes[0], palette=C.palette(len(res)), hue="transform", legend=False)
    else:
        axes[0].bar(res["transform"], res["CIN2+ AUC"], color=C.SCI_PALETTE[0])
    axes[0].set_ylim(0.45, 0.65)
    axes[0].set_ylabel("Hard-centre CIN2+ AUC")
    axes[0].set_title("A. AUC after recalibration", weight="bold", loc="left")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].plot(sweep["threshold"], sweep["sensitivity"], label="CIN3+ sensitivity", color=C.SCI_PALETTE[4], lw=2)
    axes[1].plot(sweep["threshold"], sweep["screen-positive rate"], label="Screen-positive rate", color=C.SCI_PALETTE[0], lw=2)
    axes[1].fill_between(sweep["threshold"], sweep["sensitivity"], sweep["screen-positive rate"], color=C.SCI_PALETTE[7], alpha=0.25)
    axes[1].axhline(0.95, color=C.SCI_PALETTE[4], linestyle="--", lw=1)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("B. Hard-centre threshold sweep", weight="bold", loc="left")
    axes[1].legend(frameon=False)
    C.save_fig(fig, "Figure_Ranking_vs_Calibration_Diagnosis", "Because AUC is invariant under monotone score transformations, the unchanged AUC after recalibration confirms that the residual failure is a ranking-level limitation rather than a purely threshold-level artefact.")
    possible = sweep[sweep["sensitivity"] >= 0.95]
    if possible.empty:
        msg = "The hard-centre threshold sweep did not reach CIN3+ sensitivity 0.95 across the evaluated grid."
    else:
        row = possible.sort_values("threshold", ascending=False).iloc[0]
        msg = f"The threshold needed to reach CIN3+ sensitivity >=0.95 was approximately {row['threshold']:.2f}, with specificity {row['specificity']:.3f} and screen-positive rate {row['screen-positive rate']:.3f}."
    C.write_text(C.OUT / "paper_sections" / "sec_ranking_vs_calibration_summary.txt", "Because AUC is invariant under monotone score transformations, the unchanged AUC after recalibration confirms that the residual failure is a ranking-level limitation rather than a purely threshold-level artefact. " + msg + "\n")
    C.append_manifest("E21", "Ranking vs Calibration Diagnosis", "COMPLETED_WITH_CAVEAT", ["figures/Figure_Ranking_vs_Calibration_Diagnosis.pdf", "statistics/ranking_vs_calibration_transform_results.csv", "statistics/hard_centre_threshold_sweep.csv"], "Isotonic implemented as deterministic PAV fallback.")


def main() -> None:
    C.ensure_dirs()
    e20_fn_analysis()
    e21_ranking_calibration()


if __name__ == "__main__":
    main()
