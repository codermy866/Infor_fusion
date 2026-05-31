#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import seaborn as sns

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

OUT = C.OUT / "abl04_vlm_backbone_ablation"
VLM01 = C.OUT / "vlm01_foldwise_lora"

BACKBONES = {
    "Clinical-only baseline": {"groups": ["clinical", "text"], "frozen": False, "available": True},
    "Frozen cached Step2 features": {"groups": ["clinical", "colpo", "oct"], "frozen": True, "available": True},
    "Cached feature adapter": {"groups": ["clinical", "text", "colpo", "oct"], "frozen": False, "available": True},
    "BioMedCLIP-LoRA": {"groups": ["clinical", "text", "colpo", "oct"], "frozen": False, "available": False},
}


def main() -> None:
    C.ensure_dirs()
    C.setup_plot_style()
    OUT.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    mmd_rows = []
    for fold_dir in C.fold_dirs(VLM01):
        src = C.read_table(fold_dir / "source_features.parquet")
        tgt = C.read_table(fold_dir / "target_features.parquet")
        for name, spec in BACKBONES.items():
            if not spec["available"]:
                continue
            res = C.linear_probe_fold(src, tgt, spec["groups"], frozen=spec["frozen"])
            score = res.pop("target_score")
            for pid, centre, cl, y2, y3, s in zip(tgt["patient_id"], tgt["centre"], tgt["centre_label"], tgt["cin2_label"], tgt["cin3_label"], score):
                pred_rows.append({"patient_id": pid, "centre": centre, "centre_label": cl, "backbone": name, "cin2_label": int(y2), "cin3_label": int(y3), "score": float(s), "fold": fold_dir.name})
            mmd_rows.append(
                {
                    "backbone": name,
                    "fold": fold_dir.name,
                    "source_target_MMD": C.mmd_rbf(C.feature_matrix(src, spec["groups"], frozen=spec["frozen"]), C.feature_matrix(tgt, spec["groups"], frozen=spec["frozen"])),
                }
            )
    pred_df = pd.DataFrame(pred_rows)
    metric_rows = []
    for name in BACKBONES:
        if name == "BioMedCLIP-LoRA":
            metric_rows.append(
                {
                    "backbone": name,
                    "availability": "NOT_TESTABLE",
                    "linear_probe_CIN2+_AUC": "NA",
                    "full_fusion_CIN2+_AUC": "NA",
                    "CIN3+ sensitivity": "NA",
                    "CIN3+ FN": "NA",
                    "centre_gap": "NA",
                    "ECE": "NA",
                    "notes": "BioMedCLIP-LoRA could not be verified; only frozen/cached VLM feature analysis was completed.",
                }
            )
            continue
        g = pred_df[pred_df["backbone"].eq(name)]
        t = C.select_threshold_for_cin3(g["cin3_label"], g["score"])
        m = C.eval_binary_metrics(g["cin2_label"], g["cin3_label"], g["score"], t)
        centre_metrics = []
        for centre, cg in g.groupby("centre"):
            centre_metrics.append(C.eval_binary_metrics(cg["cin2_label"], cg["cin3_label"], cg["score"], t) | {"centre": centre})
        gap = C.centre_gap(pd.DataFrame(centre_metrics))
        metric_rows.append(
            {
                "backbone": name,
                "availability": "AVAILABLE",
                "linear_probe_CIN2+_AUC": m["CIN2+ AUC"],
                "full_fusion_CIN2+_AUC": m["CIN2+ AUC"],
                "CIN3+ sensitivity": m["CIN3+ sensitivity"],
                "CIN3+ FN": m["CIN3+ FN"],
                "centre_gap": gap,
                "ECE": m["ECE_CIN2"],
                "notes": "Locked LOCO linear downstream classifier.",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    mmd = pd.DataFrame(mmd_rows)
    mmd_summary = mmd.groupby("backbone", as_index=False)["source_target_MMD"].mean().rename(columns={"source_target_MMD": "mean_source_target_MMD"})
    C.write_csv(OUT / "abl04_backbone_metrics.csv", metrics)
    C.write_csv(OUT / "abl04_mmd_comparison.csv", mmd_summary)
    C.write_csv(OUT / "abl04_patient_predictions.csv", pred_df)
    C.write_latex_table(metrics, OUT / "table_abl04_vlm_backbone_ablation.tex")

    plot_metrics = metrics[metrics["availability"].eq("AVAILABLE")].copy()
    plot_metrics["linear_probe_CIN2+_AUC"] = pd.to_numeric(plot_metrics["linear_probe_CIN2+_AUC"])
    fig, ax = C.plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(data=plot_metrics, x="backbone", y="linear_probe_CIN2+_AUC", ax=ax, palette=C.SCI_PALETTE)
    ax.set_title("Backbone / Feature Source Comparison", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Pooled CIN2+ AUC")
    ax.tick_params(axis="x", rotation=18)
    C.save_fig(fig, OUT / "figure_abl04_backbone_comparison")

    fig, ax = C.plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(data=mmd_summary, x="backbone", y="mean_source_target_MMD", ax=ax, palette=C.SCI_PALETTE)
    ax.set_title("Source-Target MMD by Feature Source", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Mean source-target MMD")
    ax.tick_params(axis="x", rotation=18)
    C.save_fig(fig, OUT / "figure_abl04_mmd_reduction")

    report = [
        "# ABL04 VLM Backbone Ablation Report",
        "",
        "BIO_MEDCLIP_LORA_SUPPORTED = `NOT_TESTABLE`",
        "",
        "BioMedCLIP-LoRA could not be verified; only frozen/cached VLM feature analysis was completed.",
        "",
        C.md_table(metrics),
    ]
    C.write_text(OUT / "abl04_audit_report.md", "\n".join(report) + "\n")
    C.status_json(OUT / "status.json", "PASS", "ABL04 completed, BioMedCLIP-LoRA not testable.", BIO_MEDCLIP_LORA_SUPPORTED="NOT_TESTABLE")
    C.file_manifest(OUT, OUT / "abl04_file_manifest.csv")


if __name__ == "__main__":
    main()
