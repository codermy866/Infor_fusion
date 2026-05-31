#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import seaborn as sns

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

OUT = C.OUT / "abl01_module_ablation"
PKG = C.OUT / "vlm03_feature_package"


VARIANTS = [
    ("Full cached-adapter reliability fusion", "reliability_gated", ["clinical", "text", "colpo", "oct", "combined"], "full"),
    ("w/o VLM-enhanced Module A (clinical/text only)", "lr", ["clinical", "text"], "module_a_proxy"),
    ("w/o B1 reliability gating (simple concat)", "lr", ["combined"], "b1"),
    ("w/o B2 iterative evidence accumulation (image avg)", "reliability_gated", ["clinical", "colpo", "oct"], "b2"),
    ("w/o Module C guideline alignment (image only)", "reliability_gated", ["colpo", "oct"], "module_c_proxy"),
    ("w/o text/semantic branch", "reliability_gated", ["clinical", "colpo", "oct", "combined"], "text"),
    ("clinical + image simple fusion baseline", "lr", ["clinical", "colpo", "oct"], "simple_fusion"),
]


def main() -> None:
    C.ensure_dirs()
    C.setup_plot_style()
    OUT.mkdir(parents=True, exist_ok=True)
    all_agg = []
    all_centre = []
    all_pred = []
    for v_i, (name, mode, groups, _) in enumerate(VARIANTS):
        preds = []
        for f_i, fold_dir in enumerate(C.fold_dirs(PKG), start=1):
            res = C.run_fold_variant(fold_dir, name, mode, groups, seed=2026 + 13 * v_i + f_i)
            preds.append(res["predictions"])
        pred = pd.concat(preds, ignore_index=True)
        agg, centre = C.aggregate_predictions(pred, name)
        all_agg.append(agg)
        all_centre.append(centre)
        all_pred.append(pred)

    agg_df = pd.concat(all_agg, ignore_index=True)
    centre_df = pd.concat(all_centre, ignore_index=True)
    pred_df = pd.concat(all_pred, ignore_index=True)

    full = agg_df[agg_df["variant"].eq("Full cached-adapter reliability fusion")].iloc[0]
    contrib = []
    for name, _, _, module in VARIANTS[1:]:
        row = agg_df[agg_df["variant"].eq(name)].iloc[0]
        delta_auc = float(full["CIN2+ AUC"]) - float(row["CIN2+ AUC"])
        delta_fn = float(row["CIN3+ FN"]) - float(full["CIN3+ FN"])
        if module == "module_a_proxy":
            status = "NOT_TESTABLE_FOR_BIOMEDCLIP_LORA"
        elif delta_auc > 0.005 or delta_fn > 0:
            status = "SUPPORTED"
        elif abs(delta_auc) <= 0.005 and abs(delta_fn) <= 1:
            status = "MIXED"
        else:
            status = "NOT_SUPPORTED"
        contrib.append(
            {
                "module_or_comparison": module,
                "ablated_variant": name,
                "delta_auc_full_minus_ablation": delta_auc,
                "delta_fn_ablation_minus_full": delta_fn,
                "contribution_status": status,
            }
        )
    contrib_df = pd.DataFrame(contrib)
    agg_out = agg_df.merge(contrib_df[["ablated_variant", "contribution_status"]], left_on="variant", right_on="ablated_variant", how="left").drop(columns=["ablated_variant"])
    agg_out.loc[agg_out["variant"].eq("Full cached-adapter reliability fusion"), "contribution_status"] = "REFERENCE"

    C.write_csv(OUT / "abl01_aggregate_metrics.csv", agg_out)
    C.write_csv(OUT / "abl01_centre_level_metrics.csv", centre_df)
    C.write_csv(OUT / "abl01_patient_predictions.csv", pred_df)
    C.write_csv(OUT / "abl01_module_contribution_status.csv", contrib_df)
    C.write_latex_table(agg_out, OUT / "table_abl01_module_ablation.tex")

    fig, axes = C.plt.subplots(1, 3, figsize=(14, 4.4))
    sns.barplot(data=agg_out, y="variant", x="CIN2+ AUC", ax=axes[0], palette=C.SCI_PALETTE)
    axes[0].set_title("AUC", weight="bold")
    sns.barplot(data=agg_out, y="variant", x="CIN3+ FN", ax=axes[1], palette=C.SCI_PALETTE)
    axes[1].set_title("CIN3+ FN", weight="bold")
    sns.barplot(data=agg_out, y="variant", x="centre_gap", ax=axes[2], palette=C.SCI_PALETTE)
    axes[2].set_title("Centre Gap", weight="bold")
    for ax in axes:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7)
    C.save_fig(fig, OUT / "figure_abl01_module_ablation")

    report = [
        "# ABL01 Module-Level Ablation Report",
        "",
        "Status: `COMPLETED_WITH_CACHED_FEATURE_ADAPTER`",
        "",
        "This ablation evaluates a lightweight cached-feature approximation. It does not verify BioMedCLIP-LoRA Module A, but it provides auditable locked-LOCO comparisons for reliability fusion proxies.",
        "",
        "## Aggregate Metrics",
        "",
        C.md_table(agg_out),
        "",
        "## Module Contribution Status",
        "",
        C.md_table(contrib_df),
    ]
    C.write_text(OUT / "abl01_audit_report.md", "\n".join(report) + "\n")
    C.status_json(OUT / "status.json", "PASS", "ABL01 completed with cached adapter caveat.")
    C.file_manifest(OUT, OUT / "abl01_file_manifest.csv")


if __name__ == "__main__":
    main()
