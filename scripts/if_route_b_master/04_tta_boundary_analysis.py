#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def e14_tta_table() -> None:
    metrics = C.load_tta_metrics()
    ci = C.read_csv(C.PATHS["tta_ci"])
    tests = C.read_csv(C.PATHS["tta_tests"])
    ci = ci if ci is not None else pd.DataFrame()
    tests = tests if tests is not None else pd.DataFrame()
    name_map = {
        "Step2.9 source-only DG ensemble": "Source-only (HyDRA-DG, Inductive)",
        "confidence filtered pseudo label no threshold from test labels": "Confidence-filtered Pseudo-label TTA",
        "target centre normalisation": "Target-centre Normalisation",
        "source free coral": "Source-free CORAL",
        "prototype distribution alignment no labels": "Prototype Distribution Alignment",
        "source free mmd": "Source-free MMD",
    }
    rows = []
    source_auc = float(metrics.loc[metrics["Method"].eq("Step2.9 source-only DG ensemble"), "AUC"].iloc[0])
    for _, m in metrics.iterrows():
        method = str(m["Method"])
        cirow = ci[ci["Method"].eq(method)] if not ci.empty and "Method" in ci.columns else pd.DataFrame()
        prow = tests[tests.get("comparison", pd.Series([], dtype=str)).astype(str).str.startswith(method + " vs")] if not tests.empty else pd.DataFrame()
        auc_ci = cirow["CIN2+ AUC (95% CI)"].iloc[0] if not cirow.empty and "CIN2+ AUC (95% CI)" in cirow else C.fmt(m["AUC"])
        interp = "Primary source-only reference." if method == "Step2.9 source-only DG ensemble" else "Score-level adaptation-boundary analysis; ranking was not materially repaired."
        rows.append(
            {
                "Method": name_map.get(method, method),
                "Track": m["Track"],
                "CIN2+ AUC (95% CI)": auc_ci,
                "Delta AUC vs Source": C.fmt(float(m["AUC"]) - source_auc),
                "CIN3+ Sensitivity": C.fmt(m["CIN3+ sensitivity"]),
                "CIN3+ FN": int(m["CIN3+ FN"]),
                "Screen-positive Rate": C.fmt(m["screen_positive_rate"]),
                "Adjusted p-value": C.fmt(prow["adjusted_p_value"].iloc[0]) if not prow.empty and "adjusted_p_value" in prow else "NA",
                "Interpretation": interp,
            }
        )
    for missing in ["Adaptive BatchNorm", "TENT"]:
        rows.append(
            {
                "Method": missing,
                "Track": "not_run",
                "CIN2+ AUC (95% CI)": "NOT AVAILABLE",
                "Delta AUC vs Source": "NA",
                "CIN3+ Sensitivity": "NA",
                "CIN3+ FN": "NA",
                "Screen-positive Rate": "NA",
                "Adjusted p-value": "NA",
                "Interpretation": "NOT AVAILABLE (no image-level checkpoint)",
            }
        )
    out = pd.DataFrame(rows)
    C.write_table(out, "Table_TTA_Comparison_IF")
    C.write_text(C.OUT / "paper_sections" / "sec_tta_comparison_summary.txt", "The best score-level TTA candidate reduced CIN3+ false negatives but did not improve pooled CIN2+ AUC or reduce the centre gap. Adaptive BatchNorm and TENT were not run because no image-level checkpoint was available.\n")
    C.append_manifest("E14", "TTA Comparison Table", "COMPLETED", ["tables/Table_TTA_Comparison_IF.csv", "paper_sections/sec_tta_comparison_summary.txt"])


def roc_curve(y, score):
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    rows = []
    for t in thresholds:
        pred = s >= t
        tp = ((pred) & (y == 1)).sum()
        fp = ((pred) & (y == 0)).sum()
        tn = ((~pred) & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum()
        rows.append({"threshold": t, "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)})
    return pd.DataFrame(rows)


def panel_roc(ax, src, tta, title):
    for label, df, color, ls in [("Source-only", src, C.SCI_PALETTE[0], "-"), ("Best TTA", tta, C.SCI_PALETTE[4], "--")]:
        r = roc_curve(df["pathology_cin2plus"], df["prob_cin2plus"])
        auc = C.auc_score(df["pathology_cin2plus"], df["prob_cin2plus"])
        ax.plot(r["fpr"], r["tpr"], label=f"{label} AUC={auc:.3f}", color=color, linestyle=ls, lw=2.0)
        if "threshold_cin3_safety95" in df.columns:
            t = pd.to_numeric(df["threshold_cin3_safety95"], errors="coerce").dropna()
            if not t.empty:
                th = float(t.iloc[0])
                pred = df["prob_cin2plus"] >= th
                y = df["pathology_cin2plus"].astype(int)
                fp = ((pred) & (y == 0)).sum()
                tn = ((~pred) & (y == 0)).sum()
                tp = ((pred) & (y == 1)).sum()
                fn = ((~pred) & (y == 1)).sum()
                ax.scatter([fp / max(fp + tn, 1)], [tp / max(tp + fn, 1)], color=color, s=42, marker="o", edgecolor="white", linewidth=0.8, zorder=5)
    ax.plot([0, 1], [0, 1], color=C.SCI_PALETTE[6], linestyle=":", lw=1.0)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(fontsize=8, frameon=False)


def e15_roc() -> None:
    C.setup_plot_style()
    src = C.load_source_preds()
    tta = C.best_tta_predictions()
    hard = "襄阳市中心医院"
    rows = []
    subsets = {
        "pooled": (src, tta),
        "xiangyang": (src[src["held_out_center"].eq(hard)], tta[tta["held_out_center"].eq(hard)]),
        "non_xiangyang": (src[~src["held_out_center"].eq(hard)], tta[~tta["held_out_center"].eq(hard)]),
    }
    for subset, (a, b) in subsets.items():
        rows.append({"subset": subset, "method": "source-only", "auc": C.auc_score(a["pathology_cin2plus"], a["prob_cin2plus"]), "n": len(a)})
        rows.append({"subset": subset, "method": "best_tta", "auc": C.auc_score(b["pathology_cin2plus"], b["prob_cin2plus"]), "n": len(b)})
    pd.DataFrame(rows).to_csv(C.OUT / "statistics" / "roc_tta_analysis.csv", index=False, encoding="utf-8-sig")
    fig, axes = C.plt.subplots(1, 3, figsize=(13.5, 4.2))
    panel_roc(axes[0], *subsets["pooled"], "Pooled five-centre")
    panel_roc(axes[1], *subsets["xiangyang"], "Xiangyang only")
    panel_roc(axes[2], *subsets["non_xiangyang"], "Non-Xiangyang pooled")
    caption = "Score-level TTA shifted the operating point but did not materially alter the ROC curve, indicating that the residual hardest-centre failure is not solved by score recalibration alone."
    C.save_fig(fig, "Figure_ROC_TTA_Analysis", caption)
    C.append_manifest("E15", "ROC TTA Analysis", "COMPLETED", ["figures/Figure_ROC_TTA_Analysis.pdf", "statistics/roc_tta_analysis.csv"])


def e16_pareto() -> None:
    C.setup_plot_style()
    metrics = C.load_tta_metrics()
    pts = metrics[["Method", "Track", "AUC", "CIN3+ sensitivity", "CIN3+ FN", "screen_positive_rate"]].copy()
    pts.to_csv(C.OUT / "statistics" / "tta_pareto_points.csv", index=False, encoding="utf-8-sig")
    fig, ax = C.plt.subplots(figsize=(7.5, 5.2))
    for i, (_, r) in enumerate(pts.iterrows()):
        marker = "s" if "inductive" in str(r["Track"]) else "o"
        ax.scatter(r["CIN3+ sensitivity"], r["AUC"], s=78, marker=marker, color=C.palette(len(pts))[i], edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(str(r["Method"])[:26], (r["CIN3+ sensitivity"], r["AUC"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.95, color=C.SCI_PALETTE[4], linestyle="--", lw=1.2, label="CIN3+ sensitivity target")
    ax.axhline(0.750, color=C.SCI_PALETTE[2], linestyle="--", lw=1.2, label="Route A AUC context")
    ax.set_xlabel("CIN3+ sensitivity")
    ax.set_ylabel("CIN2+ AUC")
    ax.set_title("TTA Pareto Boundary", weight="bold")
    ax.legend(frameon=False)
    C.save_fig(fig, "Figure_TTA_Pareto", "Pareto view of score-level TTA methods. No point jointly reached the AUC and CIN3+ sensitivity reference targets.")
    C.write_text(C.OUT / "paper_sections" / "sec_tta_pareto_summary.txt", "The Pareto analysis shows that sensitivity gains from score-level TTA were not accompanied by improved ranking performance. No candidate jointly reached the historical Route A AUC and CIN3+ sensitivity reference targets.\n")
    C.append_manifest("E16", "TTA Pareto Analysis", "COMPLETED", ["figures/Figure_TTA_Pareto.pdf", "statistics/tta_pareto_points.csv", "paper_sections/sec_tta_pareto_summary.txt"])


def main() -> None:
    C.ensure_dirs()
    e14_tta_table()
    e15_roc()
    e16_pareto()


if __name__ == "__main__":
    main()
