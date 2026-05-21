#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect final no-report HyDRA-CoE execution outputs into one folder."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "paper_revision" / "results" / "final_no_report_execution_bundle"


FILES = [
    "paper_revision/EXPERIMENT_SCOPE.md",
    "paper_revision/splits/full_multimodal_resplit/README.md",
    "paper_revision/splits/SPLIT_POLICY.md",
    "paper_revision/tables/cohort_flow_3010_to_1897.csv",
    "paper_revision/tables/image_volume_summary.csv",
    "paper_revision/tables/centerwise_aligned_case_summary.csv",
    "paper_revision/tables/clinical_variable_mapping_audit.csv",
    "paper_revision/tables/split_summary_1897.csv",
    "paper_revision/tables/loco_split_summary.csv",
    "paper_revision/tables/guideline_prototype_audit.csv",
    "paper_revision/results/config_sanity/hydra_coe_config_sanity.json",
    "paper_revision/results/tables/locked_thresholds_by_run.csv",
    "paper_revision/results/tables/internal_validation_metrics_locked_threshold.csv",
    "paper_revision/results/tables/external_test_metrics_locked_threshold.csv",
    "paper_revision/results/tables/formatted_locked_threshold_main_table.csv",
    "paper_revision/results/tables/missing_modality_robustness_metrics.csv",
    "paper_revision/results/tables/missing_modality_robustness_metrics_formatted.csv",
    "paper_revision/results/tables/input_corruption_robustness_metrics.csv",
    "paper_revision/results/tables/input_corruption_robustness_metrics_formatted.csv",
    "paper_revision/results/tables/label_noise_metrics.csv",
    "paper_revision/results/tables/label_noise_degradation_vs_clean.csv",
    "paper_revision/results/tables/formatted_label_noise_table.csv",
    "paper_revision/results/tables/loco_metrics_by_center.csv",
    "paper_revision/results/tables/loco_formatted_table.csv",
    "paper_revision/results/tables/centerwise_ece_brier_table.csv",
    "paper_revision/results/tables/centerwise_reliability_summary.csv",
    "paper_revision/results/tables/coe_faithfulness_proxy_metrics.csv",
    "paper_revision/results/tables/coe_failure_case_mining_top100.csv",
    "paper_revision/results/clinical_decision/clinical_decision_per_1000_table.csv",
    "paper_revision/results/clinical_decision/decision_curve_points.csv",
    "paper_revision/results/clinical_decision/formatted_clinical_utility_table.csv",
    "paper_revision/results/final_pipeline_manifest.csv",
]


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, encoding="utf-8-sig")


def key_results() -> pd.DataFrame:
    rows = []
    final_idx = ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "final_1897_case_index.csv"
    if final_idx.exists():
        df = pd.read_csv(final_idx, encoding="utf-8-sig")
        rows.append({"section": "cohort", "metric": "final_aligned_cases", "value": len(df)})
        rows.append({"section": "cohort", "metric": "positive_cases", "value": int(df["label"].sum())})
        rows.append({"section": "cohort", "metric": "negative_cases", "value": int((df["label"] == 0).sum())})
    image_summary = ROOT / "paper_revision" / "tables" / "image_volume_summary.csv"
    if image_summary.exists():
        img = pd.read_csv(image_summary)
        total = img[img["modality"].eq("total")].iloc[0]
        rows.append({"section": "image_volume", "metric": "total_images_or_bscans", "value": int(total["total_images_or_bscans"])})
    split_summary = ROOT / "paper_revision" / "tables" / "split_summary_1897.csv"
    if split_summary.exists():
        for _, row in pd.read_csv(split_summary).iterrows():
            rows.append({"section": "split", "metric": f"{row['subset']}_cases", "value": int(row["n_cases"])})
    for path, section in [
        ("paper_revision/results/tables/formatted_locked_threshold_main_table.csv", "locked_threshold_external"),
        ("paper_revision/results/tables/missing_modality_robustness_metrics.csv", "missing_modality"),
        ("paper_revision/results/tables/input_corruption_robustness_metrics.csv", "input_corruption"),
        ("paper_revision/results/tables/loco_metrics_by_center.csv", "loco_centerwise"),
        ("paper_revision/results/clinical_decision/clinical_decision_per_1000_table.csv", "clinical_decision"),
    ]:
        full = ROOT / path
        if full.exists():
            table = pd.read_csv(full)
            rows.append({"section": section, "metric": "rows", "value": len(table)})
            if "auc" in table and len(table):
                auc = pd.to_numeric(table["auc"], errors="coerce").dropna()
                if len(auc):
                    rows.append({"section": section, "metric": "auc_median", "value": round(float(auc.median()), 4)})
    return pd.DataFrame(rows)


def acceptance_checklist() -> pd.DataFrame:
    checks = {
        "final_1897_case_index_exists": (ROOT / "paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv").exists(),
        "cohort_flow_exists": (ROOT / "paper_revision/tables/cohort_flow_3010_to_1897.csv").exists(),
        "image_volume_summary_exists": (ROOT / "paper_revision/tables/image_volume_summary.csv").exists(),
        "clinical_mapping_audit_exists": (ROOT / "paper_revision/tables/clinical_variable_mapping_audit.csv").exists(),
        "all_center_split_exists": (ROOT / "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/train_labels.csv").exists(),
        "loco_splits_exist": (ROOT / "paper_revision/splits/loco").exists(),
        "guideline_prototype_audit_exists": (ROOT / "paper_revision/tables/guideline_prototype_audit.csv").exists(),
        "locked_threshold_tables_exist": (ROOT / "paper_revision/results/tables/locked_thresholds_by_run.csv").exists(),
        "robustness_tables_exist": (ROOT / "paper_revision/results/tables/missing_modality_robustness_metrics.csv").exists()
        and (ROOT / "paper_revision/results/tables/input_corruption_robustness_metrics.csv").exists(),
        "label_noise_outputs_exist": (ROOT / "paper_revision/results/tables/label_noise_metrics.csv").exists(),
        "coe_proxy_outputs_exist": (ROOT / "paper_revision/results/tables/coe_faithfulness_proxy_metrics.csv").exists(),
        "clinical_decision_outputs_exist": (ROOT / "paper_revision/results/clinical_decision/clinical_decision_per_1000_table.csv").exists(),
        "pipeline_manifest_exists": (ROOT / "paper_revision/results/final_pipeline_manifest.csv").exists(),
    }
    return pd.DataFrame([{"check": key, "status": "pass" if value else "missing"} for key, value in checks.items()])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "tables").mkdir(parents=True)
    (BUNDLE / "docs").mkdir(parents=True)
    manifest_rows = []
    for relative in FILES:
        src = ROOT / relative
        if not src.exists():
            manifest_rows.append({"source": relative, "copied_to": "", "status": "missing"})
            continue
        subdir = "docs" if src.suffix.lower() == ".md" else "tables"
        dst = BUNDLE / subdir / src.name
        shutil.copy2(src, dst)
        manifest_rows.append({"source": relative, "copied_to": str(dst.relative_to(BUNDLE)), "status": "copied"})

    key = key_results()
    checks = acceptance_checklist()
    key.to_csv(BUNDLE / "key_results.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(BUNDLE / "acceptance_checklist.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(BUNDLE / "result_manifest.csv", index=False, encoding="utf-8-sig")

    summary = [
        "# HyDRA-CoE No-Report Final Execution Bundle",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This bundle centralizes outputs generated from the locked 1,897-case no-report tri-modal cohort.",
        "",
        "## Key Results",
        "",
        markdown_table(key) if not key.empty else "No key results available.",
        "",
        "## Acceptance Checklist",
        "",
        markdown_table(checks),
        "",
        "## Notes",
        "",
        "- Current Python environment does not provide torch, so GPU training/checkpoint inference stages are represented by reproducible scripts and dry-run checks.",
        "- Existing prediction CSVs under paper_revision/results/predictions were used to build locked-threshold, robustness, LOCO, CoE proxy, and clinical decision tables.",
        "- Examination reports, diagnosis reports, generated reports, and VLM evidence caches are excluded from current main experiment inputs.",
    ]
    (BUNDLE / "FINAL_EXECUTION_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote final execution bundle: {BUNDLE}")


if __name__ == "__main__":
    main()
