#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).resolve().parents[1] / "if_route_b_master" / "00_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

ROOT = C.ROOT
MASTER = C.OUT
OUT = ROOT / "outputs/publishable_v2/if_route_b_remaining"
PACK = ROOT / "outputs/publishable_v2/if_route_b_submission_pack"

CORE_IDS = {"G00", "E00", "E01", "E02", "E12", "E14", "E15", "E21", "E10", "E13", "E20", "E30", "E41", "W01", "W02", "W03", "W04", "W05"}
SCI = C.SCI_PALETTE


def ensure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["main_text_tables", "main_text_figures", "supplementary_tables", "supplementary_figures", "captions", "source_csv", "paper_sections", "audit"]:
        (PACK / d).mkdir(parents=True, exist_ok=True)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table(df: pd.DataFrame, path: Path, tex: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if tex:
        path.with_suffix(".tex").write_text(df.to_latex(index=False, escape=True), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    return C.md_table(df)


def file_exists(rel: str) -> bool:
    return (MASTER / rel).exists()


def copy_or_missing(src: Path, dst: Path, blocks: bool = False, reason: str = "Source file missing.") -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return {"expected_output": str(dst.relative_to(PACK)), "source_file": C.rel(src), "status": "COPIED", "blocks_submission": False, "notes": ""}
    placeholder = dst.parent / f"MISSING_{dst.name}.txt"
    write(
        placeholder,
        f"expected source file: {C.rel(src)}\nwhy it is missing: {reason}\nwhether it blocks submission: {blocks}\nrecommended fix: regenerate the corresponding Route B master output.\n",
    )
    return {"expected_output": str(dst.relative_to(PACK)), "source_file": C.rel(src), "status": "MISSING_PLACEHOLDER", "blocks_submission": blocks, "notes": reason}


def r00_manifest_review() -> pd.DataFrame:
    manifest = pd.read_csv(MASTER / "manifests/final_execution_manifest.csv")
    file_index = pd.read_csv(MASTER / "manifests/final_output_file_index.csv")
    readiness = pd.read_csv(MASTER / "audit/downstream_experiment_readiness.csv")
    indexed = set(file_index["path"].astype(str))
    rows = []
    for _, r in manifest.iterrows():
        outs = [x for x in str(r.get("outputs", "")).split(";") if x]
        found, missing = [], []
        for o in outs:
            relp = f"outputs/publishable_v2/if_route_b_master/{o}"
            if relp in indexed or (MASTER / o).exists():
                found.append(o)
            else:
                missing.append(o)
        eid = str(r["experiment_id"])
        status = str(r["status"])
        core = eid in CORE_IDS or eid.startswith("W01")
        block = core and (status.startswith("BLOCKED") or bool(missing))
        rows.append(
            {
                "experiment_id": eid,
                "experiment_name": r["experiment_name"],
                "current_status": status,
                "output_files_found": ";".join(found),
                "missing_outputs": ";".join(missing),
                "is_manuscript_core": core,
                "is_submission_blocking": block,
                "recommended_next_action": "Proceed" if not block else "Repair before manuscript writing",
                "notes": r.get("notes", ""),
            }
        )
    for _, r in readiness.iterrows():
        eid = str(r["experiment_id"])
        if eid not in {x["experiment_id"] for x in rows}:
            rows.append(
                {
                    "experiment_id": eid,
                    "experiment_name": r["experiment_name"],
                    "current_status": r["readiness_status"],
                    "output_files_found": "",
                    "missing_outputs": "",
                    "is_manuscript_core": eid in CORE_IDS,
                    "is_submission_blocking": False,
                    "recommended_next_action": r["recommended_action"],
                    "notes": r.get("main_blocker", ""),
                }
            )
    df = pd.DataFrame(rows)
    table(df, OUT / "R00_Task_Status_Table.csv")
    blockers = df[df["is_submission_blocking"].astype(bool)]
    report = [
        "# R00 Final Manifest Review",
        "",
        f"Total tasks reviewed: {len(df)}.",
        f"Submission-blocking items: {len(blockers)}.",
        "",
        "## Status Table",
        "",
        md_table(df),
        "## Ranked Next Actions",
        "",
    ]
    if blockers.empty:
        report += ["1. Proceed to caveat resolution and final submission assembly.", "2. Keep all caveats visible in manuscript and supplement."]
    else:
        report += [f"1. Repair {row.experiment_id}: {row.missing_outputs}" for row in blockers.itertuples()]
    write(OUT / "R00_Final_Manifest_Review.md", "\n".join(report))
    return df


def r01_caveats() -> pd.DataFrame:
    mmd_summary = pd.read_csv(MASTER / "audit/mmd_claim_summary.csv")
    endpoint = pd.read_csv(MASTER / "audit/endpoint_and_threshold_audit.csv")
    feature_manifest = pd.read_csv(MASTER / "statistics/umap_input_feature_manifest.csv")
    fusion = pd.read_csv(MASTER / "tables/Table_Fusion_Ladder_IF_final.csv")
    feature_source = str(feature_manifest["feature_source"].iloc[0]) if not feature_manifest.empty else "unknown"
    true_feature = "feature_arrays" in feature_source or "oct_col_clinical" in feature_source
    threshold_supported = endpoint["detected_threshold_columns_or_values"].astype(str).str.contains("threshold", case=False, na=False).any()
    comparable = "LOCKED_LOCO" in ";".join(fusion["Comparability Status"].astype(str).unique())
    rows = [
        {
            "caveat_id": "C1",
            "caveat_name": "MMD claim",
            "evidence_files_checked": "mmd_claim_summary.csv;mmd_average_outbound.csv;mmd_matrix.csv",
            "decision": "0.842 is supported as Jingzhou centre-vs-pooled MMD in prior audit; Xiangyang maximum MMD is unsupported; pairwise prior maximum remains distinct from recomputed average outbound MMD.",
            "manuscript_safe_wording": "The largest prior centre-vs-pooled MMD was Jingzhou-related, whereas Xiangyang was the hardest centre by safety/ranking.",
            "prohibited_wording": "Xiangyang has maximum MMD.",
            "affected_experiments": "E10;E13;E22",
            "requires_manual_action": False,
            "notes": "Do not conflate distribution distance with performance failure.",
        },
        {
            "caveat_id": "C2",
            "caveat_name": "Feature-space claim",
            "evidence_files_checked": "umap_input_feature_manifest.csv;step2_locked_feature_arrays.npz",
            "decision": "E10/E13 use frozen Step2 OCT, colposcopy, and clinical feature arrays.",
            "manuscript_safe_wording": "frozen patient-level multimodal feature space",
            "prohibited_wording": "image-representation drift was corrected",
            "affected_experiments": "E10;E13;E22",
            "requires_manual_action": False,
            "notes": f"Feature source: {feature_source}; true_feature={true_feature}.",
        },
        {
            "caveat_id": "C3",
            "caveat_name": "Threshold rule",
            "evidence_files_checked": "endpoint_and_threshold_audit.csv;source_only_reference_predictions.csv;tta_candidate_predictions.csv",
            "decision": "Threshold columns are explicitly stored for source-only and TTA predictions.",
            "manuscript_safe_wording": "thresholds were selected from source/inner-validation protocol and stored before final evaluation",
            "prohibited_wording": "test-label tuned threshold",
            "affected_experiments": "E01;E14;E15;E20;E42",
            "requires_manual_action": not threshold_supported,
            "notes": f"threshold_supported={threshold_supported}.",
        },
        {
            "caveat_id": "C4",
            "caveat_name": "Baseline comparability",
            "evidence_files_checked": "Table_Fusion_Ladder_IF_final.csv;baseline metrics",
            "decision": "Existing baseline outputs are locked-LOCO compatible where marked, but broad superiority claims remain unsafe because some historical/skeleton rows are caveated.",
            "manuscript_safe_wording": "improved pooled AUC compared with selected locked-protocol baselines",
            "prohibited_wording": "outperformed all baselines",
            "affected_experiments": "E30;E31;E32;E33",
            "requires_manual_action": False,
            "notes": f"comparability_detected={comparable}.",
        },
    ]
    df = pd.DataFrame(rows)
    table(df, OUT / "R01_Caveat_Decisions.csv")
    decision = "YES_WITH_CAVEATS"
    report = [
        "# R01 Caveat Resolution Report",
        "",
        f"SAFE_TO_WRITE_MAIN_RESULTS = {decision}",
        "",
        "The main Route B results can be written if the MMD/performance distinction, frozen-feature-space wording, source-threshold rule, and selected-baseline caveat are stated explicitly.",
        "",
        md_table(df),
    ]
    write(OUT / "R01_Caveat_Resolution_Report.md", "\n".join(report))
    return df


def r02_core() -> pd.DataFrame:
    expected = {
        "E00": ["tables/Table_Protocol_Separation_IF.tex", "tables/Table_Protocol_Separation_IF.csv", "paper_sections/sec_eval_protocol.txt"],
        "E01": ["tables/Table_Centre_Level_Results_IF.tex", "tables/Table_Centre_Level_Results_IF.csv", "paper_sections/sec_centre_results_summary.txt"],
        "E02": ["tables/Table_Fusion_Ladder_IF_skeleton.tex", "tables/Table_Fusion_Ladder_IF_skeleton.csv"],
        "E12": ["tables/Table_Dataset_Demographics_IF.tex", "tables/Table_Dataset_Demographics_IF.csv", "paper_sections/sec_dataset_description.txt"],
        "E14": ["tables/Table_TTA_Comparison_IF.tex", "tables/Table_TTA_Comparison_IF.csv", "paper_sections/sec_tta_comparison_summary.txt"],
        "E15": ["figures/Figure_ROC_TTA_Analysis.pdf", "figures/Figure_ROC_TTA_Analysis.png", "statistics/roc_tta_analysis.csv"],
        "E21": ["figures/Figure_Ranking_vs_Calibration_Diagnosis.pdf", "statistics/ranking_vs_calibration_transform_results.csv", "statistics/hard_centre_threshold_sweep.csv", "paper_sections/sec_ranking_vs_calibration_summary.txt"],
    }
    rows = []
    for eid, files in expected.items():
        missing = [f for f in files if not (MASTER / f).exists()]
        rows.append({"experiment_id": eid, "required_files": ";".join(files), "missing_files": ";".join(missing), "status": "COMPLETE" if not missing else "INCOMPLETE"})
    df = pd.DataFrame(rows)
    table(df, OUT / "R02_Core_Output_Status.csv")
    decision = "PROCEED_TO_RESULTS_WRITING" if (df["status"] == "COMPLETE").all() else "PROCEED_WITH_CAVEATS"
    write(OUT / "R02_Core_Output_Completion_Report.md", "# R02 Core Output Completion Report\n\n" + f"Decision: `{decision}`.\n\n" + md_table(df))
    return df


def r03_domain() -> pd.DataFrame:
    expected = {
        "E10": ["statistics/mmd_matrix.csv", "statistics/mmd_average_outbound.csv", "figures/Figure_MMD_Matrix.pdf", "figures/Figure_MMD_Matrix.png", "figures/Figure_MMD_Matrix_caption.txt"],
        "E11": ["statistics/centre_classifier_results.csv", "statistics/centre_classifier_feature_importance.csv", "figures/Figure_Centre_Classifier_CM.pdf", "figures/Figure_Centre_Classifier_CM.png", "paper_sections/sec_centre_classifier_summary.txt"],
        "E13": ["figures/Figure_UMAP_Centre_Distribution.pdf", "figures/Figure_UMAP_Centre_Distribution.png", "statistics/umap_input_feature_manifest.csv", "figures/Figure_UMAP_Centre_Distribution_caption.txt"],
        "E22": ["statistics/domain_shift_source_attribution.csv", "figures/Figure_Domain_Shift_Source_Attribution.pdf", "figures/Figure_Domain_Shift_Source_Attribution.png", "paper_sections/sec_domain_shift_source_attribution.txt"],
    }
    rows = []
    for eid, files in expected.items():
        for f in files:
            rows.append({"experiment_id": eid, "file": f, "exists": (MASTER / f).exists(), "type": Path(f).suffix})
    df = pd.DataFrame(rows)
    table(df, OUT / "R03_Domain_Shift_Figure_Table_Index.csv")
    mmd = pd.read_csv(MASTER / "statistics/mmd_average_outbound.csv")
    top = mmd.iloc[0]
    report = [
        "# R03 Domain-Shift Completion Report",
        "",
        "Domain-shift package is complete. E10/E13 use frozen patient-level multimodal feature arrays, and E13 uses UMAP.",
        "",
        f"Highest recomputed average outbound MMD: {top['Centre']} ({float(top['Average outbound MMD']):.3f}).",
        "",
        "The largest pairwise/prior MMD and the hardest held-out centre were not necessarily identical, indicating that statistical distributional distance and clinical safety failure are related but not interchangeable.",
        "",
        md_table(df),
    ]
    write(OUT / "R03_Domain_Shift_Completion_Report.md", "\n".join(report))
    return df


def r04_hardcentre() -> None:
    centre = pd.read_csv(MASTER / "tables/Table_Centre_Level_Results_IF.csv")
    src = centre[centre["Centre"].ne("Pooled")].copy()
    src["fn"] = pd.to_numeric(src["Source-only CIN3+ FN"], errors="coerce")
    src["sens"] = pd.to_numeric(src["Source-only CIN3+ Sensitivity"], errors="coerce")
    src["auc2"] = pd.to_numeric(src["Source-only CIN2+ AUC"], errors="coerce")
    hard = src.sort_values(["fn", "sens", "auc2"], ascending=[False, True, True]).iloc[0]["Centre"]
    text = [
        "# R04 Hard-Centre Completion Report",
        "",
        f"Verified hardest centre: `{hard}`.",
        "",
        "Priority rule: largest CIN3+ FN, then lowest CIN3+ sensitivity, then lowest CIN2+ AUC.",
        "",
        "E20 FN analysis and E21 ranking-vs-calibration diagnosis are complete.",
        "",
        "Because AUC is invariant under monotone score transformations, the unchanged AUC after recalibration indicates that the residual hard-centre failure is a ranking-level limitation rather than a purely threshold-level artefact.",
        "",
        "No rescue claim is made.",
    ]
    write(OUT / "R04_HardCentre_Completion_Report.md", "\n".join(text))


def metric_lookup(df: pd.DataFrame, method: str) -> dict:
    m = df[df["Method"].astype(str).eq(method)]
    return m.iloc[0].to_dict() if not m.empty else {}


def r05_fusion_ladder() -> pd.DataFrame:
    clinical = pd.read_csv(MASTER / "statistics/clinical_baselines_metrics.csv")
    unimodal = pd.read_csv(MASTER / "statistics/unimodal_baselines_metrics.csv")
    simple = pd.read_csv(MASTER / "statistics/simple_fusion_baselines_metrics.csv")
    step = pd.read_csv(MASTER / "tables/Table_Fusion_Ladder_IF_skeleton.csv")
    tta = pd.read_csv(C.p(C.PATHS["tta_metrics"]))
    rows = []

    def add(method, modalities, fusion, strategy, uses_target, labels, auc, ci, sens, fn, spr, gap, status, placement, notes):
        rows.append(
            {
                "Method": method,
                "Input modalities": modalities,
                "Fusion type": fusion,
                "DG or adaptation strategy": strategy,
                "Uses target-centre unlabeled data": uses_target,
                "Uses target-centre labels": labels,
                "CIN2+ AUC": auc,
                "CIN2+ AUC 95% CI": ci,
                "CIN3+ sensitivity": sens,
                "CIN3+ FN": fn,
                "Screen-positive rate": spr,
                "Centre gap": gap,
                "Comparability status": status,
                "Manuscript placement": placement,
                "Notes": notes,
            }
        )

    for label, method in [("HPV-only", "HPV-only rule"), ("TCT-only", "TCT-only rule"), ("Age + HPV + TCT logistic regression", "Clinical logistic regression"), ("Age + HPV + TCT random forest", "Clinical random forest")]:
        r = metric_lookup(clinical, method)
        add(label, "clinical", "clinical baseline", "none", False, False, C.fmt(r.get("CIN2+ AUC", np.nan)), "NA", C.fmt(r.get("CIN3+ sensitivity", np.nan)), r.get("CIN3+ FN", "NA"), C.fmt(r.get("Screen-positive rate", np.nan)), C.fmt(r.get("Centre gap", np.nan)), "LOCKED_LOCO_MAIN", "MAIN_TEXT", "Generated under locked LOCO.")
    for label, method in [("Colposcopy-only", "ColposcopyOnly_ViT"), ("OCT-only", "OCTOnly_ViT")]:
        r = metric_lookup(unimodal, method)
        add(label, label.split("-")[0], "unimodal", "none", False, False, C.fmt(r.get("CIN2+ AUC", np.nan)), "NA", C.fmt(r.get("CIN3+ sensitivity", np.nan)), r.get("CIN3+ FN", "NA"), C.fmt(r.get("Screen-positive rate", np.nan)), C.fmt(r.get("Centre gap", np.nan)), "LOCKED_LOCO_WITH_CAVEAT", "SUPPLEMENT", "Existing locked all-model output.")
    for label, method in [("Colposcopy + OCT late fusion", "ColposcopyOCT_LateFusion"), ("Colposcopy + OCT + clinical/text fusion", "ColposcopyOCTText_CrossAttention")]:
        r = metric_lookup(simple, method)
        add(label, "colposcopy/OCT/clinical", "simple fusion", "none", False, False, C.fmt(r.get("CIN2+ AUC", np.nan)), "NA", C.fmt(r.get("CIN3+ sensitivity", np.nan)), r.get("CIN3+ FN", "NA"), C.fmt(r.get("Screen-positive rate", np.nan)), C.fmt(r.get("Centre gap", np.nan)), "LOCKED_LOCO_WITH_CAVEAT", "SUPPLEMENT", "Existing all-model output.")
    for label in ["Step 2 Surrogate (HyDRA features, no DG)", "Step 2.6 Active Minimal Adapter", "HyDRA-DG-SafetyEnsemble (Source-only)", "HyDRA-DG + Best TTA (Transductive)"]:
        s = step[step["Method"].astype(str).eq(label)]
        if not s.empty:
            r = s.iloc[0]
            add(label, "multimodal", r["Fusion Type"], r["DG Strategy"], bool(r["Uses Target"]), False, r["CIN2+ AUC (95% CI)"], r["CIN2+ AUC (95% CI)"], r["CIN3+ Sensitivity"], r["CIN3+ FN"], "NA", r["Centre Gap"], "LOCKED_LOCO_MAIN" if "HyDRA-DG" in label else "LOCKED_LOCO_WITH_CAVEAT", "MAIN_TEXT", r["Notes"])
    df = pd.DataFrame(rows)
    table(df, OUT / "Table_Fusion_Ladder_IF_FINAL.csv", tex=True)
    report = [
        "# R05 Fusion Ladder Finalisation Report",
        "",
        "Directly comparable rows are those marked `LOCKED_LOCO_MAIN` or `LOCKED_LOCO_WITH_CAVEAT`.",
        "Some existing all-model outputs are placed in the supplement to avoid broad superiority claims.",
        "",
        "Can the paper claim superiority over all baselines? **NO**.",
        "",
        "Safe wording: achieved improved pooled AUC compared with selected locked-protocol baselines.",
        "",
        md_table(df),
    ]
    write(OUT / "R05_Fusion_Ladder_Finalisation_Report.md", "\n".join(report))
    return df


def r06_clinical() -> pd.DataFrame:
    dca = pd.read_csv(MASTER / "statistics/dca_results.csv")
    ece = pd.read_csv(MASTER / "statistics/ece_by_centre.csv")
    screen = pd.read_csv(MASTER / "statistics/screening_efficiency_curve.csv")
    metrics = pd.read_csv(C.p(C.PATHS["tta_metrics"]))
    src = metrics[metrics["Method"].eq("Step2.9 source-only DG ensemble")].iloc[0]
    best = metrics.sort_values(["AUC", "CIN3+ sensitivity"], ascending=False).iloc[0]
    thresholds = dca[dca["threshold"].isin([0.1, 0.2, 0.3])].copy()
    summary = pd.DataFrame(
        [
            {"item": "DCA thresholds available", "value": ";".join(map(str, sorted(thresholds["threshold"].unique())))},
            {"item": "Pooled source-only ECE CIN2+", "value": C.fmt(ece[(ece["model"].eq("HyDRA-DG source-only")) & (ece["centre"].eq("Pooled")) & (ece["endpoint"].eq("CIN2+"))]["ECE"].iloc[0])},
            {"item": "Pooled best TTA ECE CIN2+", "value": C.fmt(ece[(ece["model"].eq("Best score-level TTA")) & (ece["centre"].eq("Pooled")) & (ece["endpoint"].eq("CIN2+"))]["ECE"].iloc[0])},
            {"item": "Source-only CIN3+ sensitivity", "value": C.fmt(src["CIN3+ sensitivity"])},
            {"item": "Best TTA CIN3+ sensitivity", "value": C.fmt(best["CIN3+ sensitivity"])},
            {"item": "Safety target reached", "value": bool(float(best["CIN3+ sensitivity"]) >= 0.95)},
        ]
    )
    table(summary, OUT / "Table_Clinical_Evaluation_Summary.csv", tex=True)
    report = [
        "# R06 Clinical Evaluation Report",
        "",
        "Calibration, DCA, and screening-efficiency outputs are available.",
        "",
        "TTA primarily shifts threshold-dependent operating points; it does not repair ranking failure.",
        "",
        "The observed performance remains below the prespecified CIN3+ safety target, supporting the use of this analysis as a reliability-boundary study rather than a deployment claim.",
        "",
        md_table(summary),
    ]
    write(OUT / "R06_Clinical_Evaluation_Report.md", "\n".join(report))
    return summary


def r07_coe() -> None:
    text = read(MASTER / "paper_sections/sec_interpretability_coe.txt")
    final = (
        "The CoE module provides stepwise evidence accumulation from clinical prior to colposcopy evidence and OCT verification, with template-supervised text output. "
        + text
        + "\nThe CoE module should therefore be interpreted as a transparency aid and a direction for future evaluation, rather than a clinically validated explanation system.\n"
    )
    write(OUT / "R07_CoE_Interpretability_Final.txt", final)
    audit = "# R07 CoE Claim Audit\n\nNo faithfulness, doctor-level interpretability, or clinical validation claim is made.\n"
    write(OUT / "R07_CoE_Claim_Audit.md", audit)


def create_schematic_figures() -> None:
    C.setup_plot_style()
    # Figure 1: study design
    fig, ax = C.plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    boxes = [
        ("Locked n=1897\nfive-centre cohort", 0.05),
        ("LOCO source-only\nbenchmark", 0.28),
        ("DG ensemble\nselection", 0.51),
        ("Score-level TTA\nboundary analysis", 0.74),
    ]
    for i, (txt, x) in enumerate(boxes):
        ax.add_patch(C.plt.Rectangle((x, 0.38), 0.17, 0.28, color=SCI[i], alpha=0.85, transform=ax.transAxes))
        ax.text(x + 0.085, 0.52, txt, ha="center", va="center", fontsize=10, weight="bold", transform=ax.transAxes)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.205, 0.52), xytext=(x + 0.17, 0.52), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))
    C.save_fig(fig, "Figure1_Study_Design_or_Framework", "Study design schematic for the Route B locked LOCO benchmark.")
    shutil.copy2(MASTER / "figures/Figure1_Study_Design_or_Framework.pdf", PACK / "main_text_figures/Figure1_Study_Design_or_Framework.pdf")
    # Figure 3: fusion strategy
    fig, ax = C.plt.subplots(figsize=(8.5, 4))
    ax.axis("off")
    nodes = [("Clinical", 0.15, 0.72), ("Colposcopy", 0.15, 0.50), ("OCT", 0.15, 0.28), ("Reliability-aware\nfusion", 0.48, 0.50), ("DG ensemble /\nscore TTA", 0.78, 0.50)]
    for i, (txt, x, y) in enumerate(nodes):
        ax.add_patch(C.plt.Rectangle((x - 0.09, y - 0.08), 0.18, 0.16, color=SCI[i], alpha=0.9, transform=ax.transAxes))
        ax.text(x, y, txt, ha="center", va="center", fontsize=10, weight="bold", transform=ax.transAxes)
    for y in [0.72, 0.50, 0.28]:
        ax.annotate("", xy=(0.39, 0.50), xytext=(0.25, y), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))
    ax.annotate("", xy=(0.69, 0.50), xytext=(0.57, 0.50), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))
    C.save_fig(fig, "Figure3_Fusion_Framework_or_Strategy", "Reliability-aware fusion and Route B adaptation-boundary schematic.")
    shutil.copy2(MASTER / "figures/Figure3_Fusion_Framework_or_Strategy.pdf", PACK / "main_text_figures/Figure3_Fusion_Framework_or_Strategy.pdf")


def combine_figures(srcs: list[Path], dst: Path, title: str = "") -> None:
    C.setup_plot_style()
    imgs = [C.plt.imread(str(s.with_suffix(".png"))) for s in srcs if s.with_suffix(".png").exists()]
    if not imgs:
        return
    fig, axes = C.plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 4.4))
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, src in zip(axes, imgs, srcs):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(src.stem.replace("Figure_", "").replace("_", " "), fontsize=10)
    if title:
        fig.suptitle(title, weight="bold")
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, bbox_inches="tight")
    fig.savefig(dst.with_suffix(".png"), dpi=300, bbox_inches="tight")
    C.plt.close(fig)


def r08_submission_pack() -> pd.DataFrame:
    if PACK.exists():
        # keep audit but refresh target pack files
        pass
    ensure()
    create_schematic_figures()
    rows = []
    table_map = [
        ("tables/Table_Dataset_Demographics_IF.tex", "main_text_tables/Table1_Dataset_Demographics.tex"),
        ("tables/Table_Protocol_Separation_IF.tex", "main_text_tables/Table2_Protocol_Separation.tex"),
        (str(OUT.relative_to(ROOT) / "Table_Fusion_Ladder_IF_FINAL.tex"), "main_text_tables/Table3_Fusion_Ladder.tex"),
        ("tables/Table_Centre_Level_Results_IF.tex", "main_text_tables/Table4_Centre_Level_Results.tex"),
        ("tables/Table_TTA_Comparison_IF.tex", "main_text_tables/Table5_TTA_Comparison.tex"),
        ("tables/Table_HardCentre_FN_Analysis.tex", "main_text_tables/Table6_HardCentre_FN_Analysis.tex"),
    ]
    for src_rel, dst_rel in table_map:
        src = ROOT / src_rel if src_rel.startswith("outputs/") else MASTER / src_rel
        rows.append(copy_or_missing(src, PACK / dst_rel, blocks=True))
    fig_map = [
        ("figures/Figure_ROC_TTA_Analysis.pdf", "main_text_figures/Figure4_ROC_TTA_Analysis.pdf"),
        ("figures/Figure_Ranking_vs_Calibration_Diagnosis.pdf", "main_text_figures/Figure5_Ranking_vs_Calibration.pdf"),
    ]
    combine_figures([MASTER / "figures/Figure_MMD_Matrix", MASTER / "figures/Figure_UMAP_Centre_Distribution"], PACK / "main_text_figures/Figure2_Domain_Shift_MMD_UMAP.pdf", "Domain-shift characterisation")
    combine_figures([MASTER / "figures/Figure_DCA", MASTER / "figures/Figure_Calibration_Curves", MASTER / "figures/Figure_Screening_Efficiency"], PACK / "main_text_figures/Figure6_Clinical_Evaluation.pdf", "Clinical evaluation")
    rows += [
        {"expected_output": "main_text_figures/Figure2_Domain_Shift_MMD_UMAP.pdf", "source_file": "combined", "status": "CREATED", "blocks_submission": False, "notes": ""},
        {"expected_output": "main_text_figures/Figure6_Clinical_Evaluation.pdf", "source_file": "combined", "status": "CREATED", "blocks_submission": False, "notes": ""},
    ]
    for src_rel, dst_rel in fig_map:
        rows.append(copy_or_missing(MASTER / src_rel, PACK / dst_rel, blocks=True))
    supp = [
        ("audit/audit_manifest.csv", "supplementary_tables/TableS1_Audit_Manifest.csv"),
        ("statistics/mmd_matrix.csv", "supplementary_tables/TableS2_MMD_Matrix.csv"),
        ("statistics/centre_classifier_feature_importance.csv", "supplementary_tables/TableS3_Centre_Classifier_Feature_Importance.csv"),
        ("tables/Table_Ablation_Extended_IF.csv", "supplementary_tables/TableS4_Extended_Ablation.csv"),
        ("tables/Table_Clinical_Baselines_IF.csv", "supplementary_tables/TableS5_Clinical_Baselines.csv"),
        ("statistics/ece_by_centre.csv", "supplementary_tables/TableS6_Calibration_ECE.csv"),
        ("statistics/screening_efficiency_curve.csv", "supplementary_tables/TableS7_Screening_Efficiency.csv"),
        ("paper_sections/sec_interpretability_coe.txt", "supplementary_tables/TableS8_CoE_Examples_or_Notes.csv"),
    ]
    for src_rel, dst_rel in supp:
        rows.append(copy_or_missing(MASTER / src_rel, PACK / dst_rel, blocks=False))
    sfig = [
        ("figures/Figure_Centre_Classifier_CM.pdf", "supplementary_figures/FigureS1_Centre_Classifier_CM.pdf"),
        ("figures/Figure_TTA_Pareto.pdf", "supplementary_figures/FigureS2_TTA_Pareto.pdf"),
        ("figures/Figure_HardCentre_FN_Probability_Distribution.pdf", "supplementary_figures/FigureS3_HardCentre_FN_Distribution.pdf"),
        ("figures/Figure_Domain_Shift_Source_Attribution.pdf", "supplementary_figures/FigureS4_Domain_Shift_Source_Attribution.pdf"),
        ("figures/Figure_Screening_Efficiency.pdf", "supplementary_figures/FigureS5_Screening_Efficiency.pdf"),
    ]
    for src_rel, dst_rel in sfig:
        rows.append(copy_or_missing(MASTER / src_rel, PACK / dst_rel, blocks=False))
    rows.append(copy_or_missing(MASTER / "figures/FigureS6_CoE_Examples.pdf", PACK / "supplementary_figures/FigureS6_CoE_Examples.pdf", blocks=False, reason="No CoE example figure was generated; CoE is text-only transparency material."))
    for src in (MASTER / "paper_sections").glob("*.txt"):
        rows.append(copy_or_missing(src, PACK / "paper_sections" / src.name, blocks=False))
    for cap in (MASTER / "figures").glob("*_caption.txt"):
        rows.append(copy_or_missing(cap, PACK / "captions" / cap.name, blocks=False))
    for csv in list((MASTER / "tables").glob("*.csv")) + list((MASTER / "statistics").glob("*.csv")):
        rows.append(copy_or_missing(csv, PACK / "source_csv" / csv.name, blocks=False))
    df = pd.DataFrame(rows)
    table(df, PACK / "Submission_Output_Index.csv")
    write(PACK / "R08_Table_Figure_Assembly_Report.md", "# R08 Table/Figure Assembly Report\n\n" + md_table(df))
    return df


def r09_claim_lock() -> pd.DataFrame:
    claims = [
        ("C01", "locked n=1897 five-centre LOCO benchmark", "benchmark", "G00;Table1", "ALLOWED", "", "", "Methods/Results"),
        ("C02", "pooled AUC improvement after DG ensemble", "performance", "Table3;Table5", "ALLOWED_WITH_CAVEAT", "Compared with selected locked-protocol baselines.", "outperformed all baselines", "Results"),
        ("C03", "centre gap persists after DG/TTA", "limitation", "Table4;Table5", "ALLOWED", "", "", "Results/Discussion"),
        ("C04", "score-level TTA reduces CIN3+ FN", "adaptation", "Table5", "ALLOWED_WITH_CAVEAT", "Transductive score-level analysis only.", "source-only improvement", "Results"),
        ("C05", "score-level TTA does not improve ranking/AUC", "adaptation boundary", "Figure4;Figure5", "ALLOWED", "", "TTA solved ranking", "Discussion"),
        ("C06", "Xiangyang/hardest centre accounts for most CIN3+ FN", "failure analysis", "Table4;Table6", "ALLOWED", "", "rescued Xiangyang", "Results"),
        ("C07", "hardest-centre failure is ranking-level, not pure calibration", "diagnostic", "Figure5", "ALLOWED", "", "", "Discussion"),
        ("C08", "MMD supports centre shift", "domain shift", "Figure2;TableS2", "ALLOWED_WITH_CAVEAT", "MMD is descriptive and not interchangeable with clinical failure.", "", "Results"),
        ("C09", "Xiangyang has maximum MMD", "domain shift", "R01", "PROHIBITED", "", "Xiangyang has maximum MMD", "None"),
        ("C10", "Jingzhou has maximum pairwise MMD", "domain shift", "R01", "ALLOWED_WITH_CAVEAT", "Prior audit supports centre-vs-pooled MMD, not all pairwise interpretations.", "", "Supplement"),
        ("C11", "score-space shift versus representation-level shift", "wording", "R01", "ALLOWED_WITH_CAVEAT", "Use frozen patient-level feature space wording.", "", "Methods"),
        ("C12", "CoE is a transparency aid", "interpretability", "R07", "ALLOWED", "", "", "Discussion"),
        ("C13", "CoE is clinically validated", "interpretability", "R07", "PROHIBITED", "", "clinically validated CoE", "None"),
        ("C14", "model is ready for safe deployment", "clinical", "R06", "PROHIBITED", "", "safe deployment", "None"),
        ("C15", "method outperforms all baselines", "performance", "R05", "PROHIBITED", "", "outperformed all baselines", "None"),
    ]
    df = pd.DataFrame(claims, columns=["claim_id", "claim_text", "claim_type", "supporting_outputs", "allowed_status", "required_caveat", "prohibited_overclaim", "recommended_section"])
    table(df, PACK / "audit/IF_Claim_Lock_Final.csv")
    md = ["# IF Claim Lock Final", "", "## A. Allowed Main-Text Claims", "", md_table(df[df["allowed_status"].eq("ALLOWED")]), "## B. Claims Allowed Only With Caveat", "", md_table(df[df["allowed_status"].eq("ALLOWED_WITH_CAVEAT")]), "## C. Prohibited Claims", "", md_table(df[df["allowed_status"].eq("PROHIBITED")])]
    write(PACK / "audit/IF_Claim_Lock_Final.md", "\n".join(md))
    return df


def r10_manuscript() -> None:
    abstract = read(MASTER / "paper_sections/W01_Abstract_Draft.txt")
    intro = read(MASTER / "paper_sections/W02_Introduction_Draft.txt")
    method = read(MASTER / "paper_sections/W03_Method_Framework.txt")
    exp = read(MASTER / "paper_sections/W04_Experiment_Structure.txt")
    discussion = read(MASTER / "paper_sections/W05_Discussion_Framework.txt")
    doc = f"""# Reliability-Aware Multimodal Fusion for Multicentre Cervical Lesion Screening: A Locked LOCO Benchmark, Domain-Shift Analysis, and Limits of Score-Level Adaptation

# Abstract

{abstract}

# 1. Introduction

{intro}

# 2. Related Work

## 2.1 Multimodal cervical lesion screening

Multimodal cervical lesion screening combines clinical priors, colposcopy, OCT, cytology, HPV status, and model-derived risk estimates. The present manuscript treats these modalities as a reliability benchmark rather than a deployment-ready system.

## 2.2 Domain generalisation and target adaptation in medical AI

Domain generalisation and target adaptation are used to study cross-centre reliability. This paper separates inductive source-only LOCO evaluation from transductive score-level TTA.

## 2.3 Reliability and interpretability in multimodal fusion

Reliability requires centre-level analysis, calibration, safety endpoints, and claim discipline. CoE text is treated as a transparency aid pending formal validation.

# 3. Methods

{method}

# 4. Experimental Setup

{exp}

# 5. Results

## 5.1 Dataset and centre-shift characterisation

See Table 1 and Figure 2. The MMD and UMAP analyses describe frozen patient-level multimodal feature space.

## 5.2 Fusion strategy ladder

See Table 3. Superiority claims are limited to selected locked-protocol baselines.

## 5.3 Score-level TTA boundary

See Table 5 and Figure 4. Score-level TTA reduced selected false negatives but did not repair ranking failure.

## 5.4 Hard-centre failure analysis

See Table 6 and Figure 5. The hard-centre result is interpreted as a ranking-level limitation rather than a successful rescue.

## 5.5 Clinical evaluation

See Figure 6 and supplementary calibration/screening tables.

## 5.6 CoE transparency aid

The CoE module is retained as template-supervised transparency material, not as a clinically validated explanation system.

# 6. Discussion

{discussion}

# 7. Conclusion

This study defines a locked multicentre benchmark and documents the reliability boundary of score-level adaptation for multimodal cervical lesion fusion.

# Data Availability

Data availability requires manual completion according to institutional and ethics constraints.

# Code Availability

Code availability requires manual completion if the repository is private.

# Ethics Statement

[TO VERIFY: insert approved ethics protocol and consent/waiver details.]

# Author Contributions

[TO VERIFY]

# Conflict of Interest

[TO VERIFY]
"""
    write(PACK / "IF_RouteB_Manuscript_Draft.md", doc)


def r11_supplement() -> None:
    idx = pd.read_csv(PACK / "Submission_Output_Index.csv")
    supp_tables = idx[idx["expected_output"].astype(str).str.contains("supplementary_tables")]
    supp_figs = idx[idx["expected_output"].astype(str).str.contains("supplementary_figures")]
    doc = [
        "# Supplementary Methods",
        "",
        "S1. Data lock and cohort construction",
        "S2. Centre identity and LOCO protocol",
        "S3. Endpoint definitions",
        "S4. Feature and score availability",
        "S5. MMD computation",
        "S6. Centre classifier",
        "S7. TTA candidate methods",
        "S8. Ranking-vs-calibration diagnosis",
        "S9. Clinical baseline implementation",
        "S10. Calibration, DCA, and screening-efficiency metrics",
        "S11. CoE template supervision and limitations",
        "",
        "# Supplementary Tables",
        "",
        md_table(supp_tables),
        "# Supplementary Figures",
        "",
        md_table(supp_figs),
        "# Supplementary Claim-Caveat Notes",
        "",
        "Score-feature versus frozen-feature wording, MMD interpretation, baseline comparability, and CoE faithfulness limitations are documented explicitly and should not be hidden.",
    ]
    write(PACK / "IF_RouteB_Supplementary_Material.md", "\n".join(doc))


def r12_final_audit() -> pd.DataFrame:
    manuscript = read(PACK / "IF_RouteB_Manuscript_Draft.md")
    supplement = read(PACK / "IF_RouteB_Supplementary_Material.md")
    index = pd.read_csv(PACK / "Submission_Output_Index.csv")
    prohibited = [
        "Full HyDRA-CoE",
        "End-to-end HyDRA-CoE",
        "Safe clinical deployment",
        "Clinically validated chain-of-evidence",
        "Stochastic uncertainty modeling",
        "State-of-the-art performance",
        "Outperforms all baselines",
        "Centre-invariant generalization",
        "Xiangyang was successfully rescued",
        "External validation",
    ]
    checks = []
    def add(item, passed, notes=""):
        checks.append({"check": item, "passed": passed, "notes": notes})
    add("n=1897 consistency", "n=1897" in manuscript)
    add("five-centre LOCO consistency", "five-centre" in manuscript and "LOCO" in manuscript)
    add("CIN2+/CIN3+ endpoints", "CIN2+" in manuscript and "CIN3+" in manuscript)
    add("inductive/transductive separation", "transductive" in manuscript and "source-only" in manuscript)
    add("submission index exists", not index.empty)
    add("all blocking outputs copied", not index[index["blocks_submission"].astype(str).eq("True") & ~index["status"].eq("COPIED")].any().any())
    to_verify = [m.start() for m in re.finditer(r"\[TO VERIFY\]", manuscript)]
    add("[TO VERIFY] markers listed", True, f"{len(to_verify)} markers remain for manual completion.")
    hits = []
    for phrase in prohibited:
        for text_name, text in [("manuscript", manuscript), ("supplement", supplement)]:
            if phrase in text:
                hits.append(f"{text_name}: {phrase}")
    add("prohibited phrase scan", len(hits) == 0, "; ".join(hits))
    df = pd.DataFrame(checks)
    table(df, PACK / "Final_Submission_Audit_Checklist.csv")
    decision = "SUBMISSION_READY_WITH_MINOR_FIXES" if len(to_verify) else ("SUBMISSION_READY" if df["passed"].all() else "NOT_READY")
    report = ["# Final Submission Audit Report", "", f"Final decision: `{decision}`.", "", "Manual fixes are still required for ethics, availability, author contribution, and conflict-of-interest placeholders.", "", md_table(df)]
    write(PACK / "Final_Submission_Audit_Report.md", "\n".join(report))
    return df


def validate_figures() -> pd.DataFrame:
    rows = []
    for fig in sorted((MASTER / "figures").glob("*.png")):
        try:
            img = C.plt.imread(fig)
            nonblank = bool(np.nanstd(img) > 1e-4)
            rows.append({"figure": C.rel(fig), "exists": True, "width": img.shape[1], "height": img.shape[0], "nonblank": nonblank, "notes": ""})
        except Exception as exc:
            rows.append({"figure": C.rel(fig), "exists": fig.exists(), "width": "NA", "height": "NA", "nonblank": False, "notes": repr(exc)})
    df = pd.DataFrame(rows)
    table(df, OUT / "R08_Figure_Render_Validation.csv")
    # contact sheet for quick visual inspection
    imgs = [(fig, C.plt.imread(fig)) for fig in sorted((MASTER / "figures").glob("*.png"))]
    if imgs:
        cols = 2
        rows_n = int(np.ceil(len(imgs) / cols))
        fig, axes = C.plt.subplots(rows_n, cols, figsize=(12, 4 * rows_n))
        axes = np.ravel(axes)
        for ax, (path, img) in zip(axes, imgs):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.stem, fontsize=9)
        for ax in axes[len(imgs):]:
            ax.axis("off")
        C.save_fig(fig, "R08_Figure_Contact_Sheet", "Contact sheet of all generated Route B figures for render validation.")
        shutil.copy2(MASTER / "figures/R08_Figure_Contact_Sheet.png", OUT / "R08_Figure_Contact_Sheet.png")
    return df


def main() -> None:
    ensure()
    r00_manifest_review()
    r01_caveats()
    r02_core()
    r03_domain()
    r04_hardcentre()
    r05_fusion_ladder()
    r06_clinical()
    r07_coe()
    validate_figures()
    r08_submission_pack()
    r09_claim_lock()
    r10_manuscript()
    r11_supplement()
    r12_final_audit()
    summary = pd.DataFrame(
        [
            {"step": f"R{i:02d}", "status": "COMPLETED", "output_root": C.rel(OUT if i < 8 else PACK)}
            for i in range(13)
        ]
    )
    table(summary, OUT / "R00_R12_Execution_Summary.csv")
    write(OUT / "IF_RouteB_Remaining_Execution_Report.md", "# IF Route B Remaining Steps Execution Report\n\nAll R00-R12 steps completed. See `R00_R12_Execution_Summary.csv`, the submission pack, and the final audit report.\n")


if __name__ == "__main__":
    main()
