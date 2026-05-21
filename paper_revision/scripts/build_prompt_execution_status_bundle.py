#!/usr/bin/env python3
"""Build a compact implementation-status bundle for the no-report prompt plan.

The bundle is intentionally paper-facing: summary tables, metrics, figures, and
audit outputs are copied; case-level files that may contain patient identifiers
or source paths are referenced in a manifest but are not duplicated.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = ROOT / "HyDRA_CoE_Final_Experimental_Execution_Prompt_No_Report.md"
OUT_DIR = ROOT / "HyDRA_CoE_No_Report_Execution_Status_Bundle"


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def read_rows(rel: str) -> int | None:
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(rel: str, dest_subdir: str, copied: list[dict], missing: list[dict]) -> None:
    src = ROOT / rel
    if not src.exists():
        missing.append({"source": rel, "reason": "missing"})
        return
    dest = OUT_DIR / dest_subdir / Path(rel).name
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    copied.append(
        {
            "source": str(src),
            "bundle_path": str(dest),
            "bytes": src.stat().st_size,
        }
    )


def copy_tree_files(
    rel_dir: str,
    dest_subdir: str,
    copied: list[dict],
    missing: list[dict],
    patterns: Iterable[str] = ("*.csv", "*.json", "*.md", "*.png", "*.log"),
) -> None:
    src_dir = ROOT / rel_dir
    if not src_dir.exists():
        missing.append({"source": rel_dir, "reason": "missing directory"})
        return
    for pattern in patterns:
        for src in src_dir.rglob(pattern):
            rel = src.relative_to(ROOT)
            # Avoid duplicating per-case prediction CSVs in the paper-facing bundle.
            if "/predictions/" in str(rel):
                continue
            dest = OUT_DIR / dest_subdir / src.relative_to(src_dir)
            ensure_dir(dest.parent)
            shutil.copy2(src, dest)
            copied.append(
                {
                    "source": str(src),
                    "bundle_path": str(dest),
                    "bytes": src.stat().st_size,
                }
            )


def build_prompt_status() -> list[dict]:
    final_rows = read_rows("paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv")
    train_rows = read_rows(
        "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/train_labels.csv"
    )
    val_rows = read_rows(
        "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/val_labels.csv"
    )
    test_rows = read_rows(
        "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/external_test_labels.csv"
    )
    corrected_results = exists("paper_revision/results/real_50epoch_5center_corrected/tables/key_results_corrected_5center.csv")
    stage1_ckpt = exists(
        "paper_revision/results/stage1_clinical_semantic_adapter/checkpoints/stage1_clinical_semantic_adapter_best.pth"
    )

    return [
        {
            "prompt": "A",
            "title": "Data alignment audit and cohort locking",
            "status": "已实现",
            "evidence": f"final_1897_case_index rows={final_rows}; corrected five-centre audit and image-volume tables exist.",
            "pending": "无；case-level index contains source paths and is referenced rather than duplicated in the bundle.",
        },
        {
            "prompt": "B",
            "title": "No-report/VLM scope cleanup",
            "status": "已实现",
            "evidence": "EXPERIMENT_SCOPE.md exists; main configs set no_report_mode=True, use_vlm_retriever=False, vlm_json_path=None.",
            "pending": "仅保留 backward-compatible legacy comments/disabled code paths.",
        },
        {
            "prompt": "C",
            "title": "Clinical variable normalization for HPV/TCT/Age",
            "status": "已实现",
            "evidence": "clinical_variable_mapping.py, dataset updates, audit CSV, and tests exist.",
            "pending": "无。",
        },
        {
            "prompt": "D",
            "title": "Patient-level and LOCO splits from locked 1897 cohort",
            "status": "已实现",
            "evidence": f"all-center split rows train/val/test={train_rows}/{val_rows}/{test_rows}; LOCO split summary exists.",
            "pending": "无；current all-center external test is n=403 across five centres.",
        },
        {
            "prompt": "E",
            "title": "Dataset/config alignment to locked tri-modal cohort",
            "status": "已实现",
            "evidence": "all_center configs, feature cache summary, and config sanity JSON exist; feature NPZs rebuilt from locked cache.",
            "pending": "无。",
        },
        {
            "prompt": "F",
            "title": "Clinical semantic adapter without report supervision",
            "status": "部分实现",
            "evidence": "Stage-1 config and training script exist and enforce no-report/no-label Stage-1 scope.",
            "pending": "stage1_clinical_semantic_adapter_best.pth not found; needs actual Stage-1 training if this module is enabled."
            if not stage1_ckpt
            else "无。",
        },
        {
            "prompt": "G",
            "title": "Guideline prototype prior without pathology leakage",
            "status": "已实现",
            "evidence": "guideline_clinical_prototypes.json, leakage checker, and guideline_prototype_audit.csv exist.",
            "pending": "无。",
        },
        {
            "prompt": "H",
            "title": "Reliability posterior outputs and interpretability fields",
            "status": "已实现",
            "evidence": "evaluate_checkpoint_predictions.py exports reliability, precision, guideline, CoE, z_causal, and z_noise fields; corrected prediction CSV has all requested columns.",
            "pending": "Per-case prediction CSVs are not duplicated in the bundle for privacy; source paths are listed.",
        },
        {
            "prompt": "I",
            "title": "Locked-threshold evaluation",
            "status": "已实现",
            "evidence": "locked_thresholds_by_run.csv and internal/external locked-threshold metric tables exist; corrected checkpoint threshold selected on internal validation.",
            "pending": "无。",
        },
        {
            "prompt": "J",
            "title": "Requirement-level ablation configs",
            "status": "部分实现",
            "evidence": "Ablation configs, runner, and requirement_ablation_metrics tables exist.",
            "pending": "Need full corrected five-centre 50-epoch multi-seed rerun for every ablation before treating as final paper numbers.",
        },
        {
            "prompt": "K",
            "title": "Missing-modality and corruption robustness",
            "status": "部分实现",
            "evidence": "Robustness scripts and missing/corruption tables exist.",
            "pending": "Need corrected five-centre checkpoint-based rerun under the latest 403-case external test before final submission.",
        },
        {
            "prompt": "L",
            "title": "Label-noise stress test",
            "status": "部分实现",
            "evidence": "Noise split/config scripts and label-noise result tables exist.",
            "pending": "Need corrected five-centre multi-seed rerun if these numbers are used as final paper evidence.",
        },
        {
            "prompt": "M",
            "title": "LOCO and center-wise calibration",
            "status": "部分实现",
            "evidence": "Corrected five-centre LOCO splits exist; LOCO/calibration table builders and previous metric tables exist.",
            "pending": "Need full corrected LOCO model training/evaluation for all five held-out centres.",
        },
        {
            "prompt": "N",
            "title": "CoE faithfulness proxy",
            "status": "部分实现",
            "evidence": "CoE faithfulness scripts and proxy metric tables exist; analysis is labelled as proxy, not clinical explanation validation.",
            "pending": "Need corrected checkpoint rerun if using the latest five-centre split as the sole final evidence source.",
        },
        {
            "prompt": "O",
            "title": "Clinical decision quality per 1000 women",
            "status": "部分实现",
            "evidence": "Clinical decision script and per-1000/decision-curve outputs exist.",
            "pending": "Need rebuild from corrected locked-threshold predictions if final table must match the latest corrected checkpoint exactly.",
        },
        {
            "prompt": "P",
            "title": "Final reproducible pipeline",
            "status": "部分实现",
            "evidence": "run_final_hydra_coe_pipeline.sh and final_pipeline_manifest.csv exist; corrected 50-epoch checkpoint/feature-space runs were executed separately.",
            "pending": "Need one clean end-to-end pipeline rerun after corrected five-centre mapping if a single reproducibility manifest is required.",
        },
        {
            "prompt": "Q",
            "title": "Paper-facing naming cleanup",
            "status": "部分实现",
            "evidence": "create_hydra_coe alias exists; current corrected outputs use HyDRA-CoE naming.",
            "pending": "Some legacy files/tables may still contain backward-compatible Bio-COT names and need a final paper-facing text sweep.",
        },
    ]


def build_acceptance_status() -> list[dict]:
    return [
        {
            "item": "final_1897_case_index.csv exists and has 1897 patients",
            "status": "已实现" if read_rows("paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv") == 1897 else "待实现",
            "evidence": "paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv",
        },
        {
            "item": "cohort_flow_3010_to_1897.csv exists",
            "status": "已实现" if exists("paper_revision/tables/cohort_flow_3010_to_1897.csv") else "待实现",
            "evidence": "paper_revision/tables/cohort_flow_3010_to_1897.csv",
        },
        {
            "item": "image_volume_summary.csv reports total image count",
            "status": "已实现" if exists("paper_revision/tables/image_volume_summary.csv") else "待实现",
            "evidence": "corrected audit reports total images/B-scans = 137294.",
        },
        {
            "item": "Dataset returns colposcopy, OCT, HPV/TCT/Age clinical text, center, and label only",
            "status": "已实现",
            "evidence": "cached_patch_dataset.py and dataset_v3_2.py use clinical_info/clinical_features and no active report fields.",
        },
        {
            "item": "No report fields are active in training/evaluation",
            "status": "已实现",
            "evidence": "Scope doc and corrected prediction/audit outputs contain no report columns.",
        },
        {
            "item": "Main config has use_vlm_retriever=False",
            "status": "已实现",
            "evidence": "config.py and paper_revision configs set use_vlm_retriever=False.",
        },
        {
            "item": "Clinical variable mapping is audited",
            "status": "已实现" if exists("paper_revision/tables/clinical_variable_mapping_audit.csv") else "待实现",
            "evidence": "paper_revision/tables/clinical_variable_mapping_audit.csv",
        },
        {
            "item": "All-center patient holdout split is generated from locked 1897 cohort",
            "status": "已实现",
            "evidence": "train/val/external_test rows sum to 1897.",
        },
        {
            "item": "LOCO splits are generated from locked 1897 cohort",
            "status": "已实现" if exists("paper_revision/tables/loco_split_summary.csv") else "待实现",
            "evidence": "paper_revision/splits/loco/<5 centres>/ and loco_split_summary.csv.",
        },
        {
            "item": "Stage-1 clinical semantic adapter does not use labels or reports",
            "status": "部分实现",
            "evidence": "Stage-1 code/config exist; final adapter checkpoint is not present.",
        },
        {
            "item": "Guideline prototypes contain no pathology leakage",
            "status": "已实现" if exists("paper_revision/tables/guideline_prototype_audit.csv") else "待实现",
            "evidence": "paper_revision/tables/guideline_prototype_audit.csv.",
        },
        {
            "item": "Prediction CSV exports reliability and interpretability fields",
            "status": "已实现",
            "evidence": "Corrected external prediction CSV contains all requested interpretability columns.",
        },
        {
            "item": "Locked threshold is selected from internal_validation only",
            "status": "已实现" if exists("paper_revision/results/real_50epoch_5center_corrected/tables/locked_thresholds_by_run.csv") else "待实现",
            "evidence": "Corrected locked-threshold tables.",
        },
        {
            "item": "Requirement-level ablations correspond to scientific questions",
            "status": "部分实现",
            "evidence": "Configs and result tables exist; corrected full multi-seed rerun is pending.",
        },
        {
            "item": "Robustness experiments use locked thresholds",
            "status": "部分实现",
            "evidence": "Scripts/tables exist; latest corrected checkpoint rerun is pending.",
        },
        {
            "item": "Label noise perturbs training labels only",
            "status": "部分实现",
            "evidence": "Noise split and table scripts exist; corrected rerun is pending.",
        },
        {
            "item": "CoE faithfulness is marked as proxy analysis",
            "status": "已实现",
            "evidence": "CoE faithfulness outputs are described as automatic/proxy metrics.",
        },
        {
            "item": "Clinical decision quality is reported per 1000 women",
            "status": "部分实现",
            "evidence": "Clinical decision table exists; rebuild from corrected checkpoint predictions is recommended.",
        },
        {
            "item": "Final pipeline writes a manifest and continues after failed runs",
            "status": "部分实现" if exists("paper_revision/results/final_pipeline_manifest.csv") else "待实现",
            "evidence": "final_pipeline_manifest.csv exists; clean corrected end-to-end rerun is pending.",
        },
        {
            "item": "Paper-facing outputs use HyDRA-CoE naming",
            "status": "部分实现",
            "evidence": "Corrected outputs use HyDRA-CoE; legacy references remain for backward compatibility.",
        },
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(status_rows: list[dict], acceptance_rows: list[dict], copied_rows: list[dict], skipped_rows: list[dict]) -> None:
    status_counts = pd.Series([r["status"] for r in status_rows]).value_counts().to_dict()
    acceptance_counts = pd.Series([r["status"] for r in acceptance_rows]).value_counts().to_dict()

    lines = [
        "# HyDRA-CoE No-Report Prompt Execution Status",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Latest Corrected Cohort Snapshot",
        "",
        "- Raw registry cohort: 3010 cases.",
        "- Locked tri-modal aligned cohort: 1897 cases.",
        "- Five centres only: C01_WUHAN_RENMIN, C02_ENSHI, C03_XIANGYANG, C04_SHIYAN, C05_JINGZHOU.",
        "- Corrected all-center split: train 1317, validation 177, external test 403.",
        "- Image/B-scan volume: colposcopy 8216, OCT/B-scans 129078, total 137294.",
        "- Current experiment scope: no examination reports, no diagnosis reports, no generated reports, no VLM evidence cache.",
        "",
        "## Prompt A-Q Status",
        "",
        f"Status counts: {json.dumps(status_counts, ensure_ascii=False)}",
        "",
        "| Prompt | Status | Evidence | Pending / Notes |",
        "|---|---|---|---|",
    ]
    for row in status_rows:
        lines.append(
            f"| {row['prompt']}. {row['title']} | {row['status']} | {row['evidence']} | {row['pending']} |"
        )

    lines.extend(
        [
            "",
            "## Final Acceptance Checklist",
            "",
            f"Status counts: {json.dumps(acceptance_counts, ensure_ascii=False)}",
            "",
            "| Item | Status | Evidence |",
            "|---|---|---|",
        ]
    )
    for row in acceptance_rows:
        lines.append(f"| {row['item']} | {row['status']} | {row['evidence']} |")

    lines.extend(
        [
            "",
            "## Bundle Contents",
            "",
            f"- Copied paper-facing files: {len(copied_rows)}",
            f"- Sensitive or case-level files referenced but not copied: {len(skipped_rows)}",
            "",
            "Case-level split/index/prediction files may include hospital IDs or source file paths. They are therefore listed in `sensitive_or_case_level_sources_not_copied.csv` instead of being duplicated here.",
            "",
        ]
    )
    (OUT_DIR / "00_status" / "IMPLEMENTED_PENDING_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUT_DIR)
    for sub in [
        "00_status",
        "01_latest_corrected_5center_results",
        "02_core_audit_and_splits",
        "03_scope_configs_and_docs",
        "04_supporting_prompt_outputs",
        "05_visualizations",
    ]:
        ensure_dir(OUT_DIR / sub)

    copied: list[dict] = []
    missing: list[dict] = []
    skipped_sensitive: list[dict] = []

    # Prompt file and status docs.
    copy_file("HyDRA_CoE_Final_Experimental_Execution_Prompt_No_Report.md", "00_status", copied, missing)

    # Latest corrected five-centre result bundle: summary metrics, figures, logs, and manifests only.
    copy_tree_files(
        "paper_revision/results/real_50epoch_5center_corrected",
        "01_latest_corrected_5center_results",
        copied,
        missing,
        patterns=("*.csv", "*.json", "*.md", "*.png", "*.log"),
    )

    # Core cohort/split summaries.
    for rel in [
        "paper_revision/tables/cohort_flow_3010_to_1897.csv",
        "paper_revision/tables/image_volume_summary.csv",
        "paper_revision/tables/centerwise_aligned_case_summary.csv",
        "paper_revision/tables/center_code_mapping_audit.csv",
        "paper_revision/tables/split_summary_1897.csv",
        "paper_revision/tables/loco_split_summary.csv",
        "paper_revision/splits/full_multimodal_resplit/README.md",
        "paper_revision/splits/SPLIT_POLICY.md",
    ]:
        copy_file(rel, "02_core_audit_and_splits", copied, missing)

    # Scope and config docs.
    for rel in [
        "paper_revision/EXPERIMENT_SCOPE.md",
        "paper_revision/README.md",
        "paper_revision/configs/all_center_patient_holdout_config.py",
        "paper_revision/configs/all_center_elbo_structured_prior_config.py",
        "paper_revision/configs/guideline_clinical_prototypes.json",
        "paper_revision/results/config_sanity/hydra_coe_config_sanity.json",
    ]:
        copy_file(rel, "03_scope_configs_and_docs", copied, missing)

    # Supporting prompt-output tables.
    for rel in [
        "paper_revision/tables/clinical_variable_mapping_audit.csv",
        "paper_revision/tables/guideline_prototype_audit.csv",
        "paper_revision/tables/requirement_ablation_metrics.csv",
        "paper_revision/tables/requirement_ablation_metrics_formatted.csv",
        "paper_revision/tables/missing_modality_robustness_metrics.csv",
        "paper_revision/tables/missing_modality_robustness_metrics_formatted.csv",
        "paper_revision/tables/input_corruption_robustness_metrics.csv",
        "paper_revision/tables/input_corruption_robustness_metrics_formatted.csv",
        "paper_revision/tables/label_noise_stress_metrics.csv",
        "paper_revision/tables/label_noise_stress_metrics_formatted.csv",
        "paper_revision/results/tables/loco_metrics_by_center.csv",
        "paper_revision/results/tables/loco_formatted_table.csv",
        "paper_revision/results/tables/centerwise_ece_brier_table.csv",
        "paper_revision/results/tables/centerwise_reliability_summary.csv",
        "paper_revision/tables/coe_faithfulness_automatic_metrics.csv",
        "paper_revision/results/tables/coe_faithfulness_proxy_metrics.csv",
        "paper_revision/results/clinical_decision/clinical_decision_per_1000_table.csv",
        "paper_revision/results/clinical_decision/decision_curve_points.csv",
        "paper_revision/results/clinical_decision/formatted_clinical_utility_table.csv",
        "paper_revision/results/final_pipeline_manifest.csv",
    ]:
        copy_file(rel, "04_supporting_prompt_outputs", copied, missing)

    # Existing visualizations useful for manuscript review.
    for rel in [
        "paper_revision/figures/decision_curve_external.png",
        "paper_revision/figures/resplit_feature_official_jingzhou_shiyan_external_decision_curve.png",
        "paper_revision/figures/resplit_feature_recommended_enshi_external_decision_curve.png",
    ]:
        copy_file(rel, "05_visualizations", copied, missing)

    # Deliberately do not duplicate case-level files that may contain identifiers or source paths.
    for rel, reason in [
        ("paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv", "case-level index with source paths"),
        ("paper_revision/splits/full_multimodal_resplit/full_multimodal_all_cases_audit.csv", "case-level audit with source paths"),
        (
            "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/train_labels.csv",
            "case-level split file",
        ),
        (
            "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/val_labels.csv",
            "case-level split file",
        ),
        (
            "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/external_test_labels.csv",
            "case-level split file",
        ),
        (
            "paper_revision/results/real_50epoch_5center_corrected/predictions/HyDRA-CoE_runreal50_corrected_seed2026_internal_validation_full.csv",
            "case-level prediction file",
        ),
        (
            "paper_revision/results/real_50epoch_5center_corrected/predictions/HyDRA-CoE_runreal50_corrected_seed2026_external_test_full.csv",
            "case-level prediction file",
        ),
    ]:
        src = ROOT / rel
        skipped_sensitive.append(
            {
                "source": str(src),
                "exists": src.exists(),
                "reason": reason,
            }
        )

    status_rows = build_prompt_status()
    acceptance_rows = build_acceptance_status()

    write_csv(OUT_DIR / "00_status" / "prompt_execution_status.csv", status_rows)
    write_csv(OUT_DIR / "00_status" / "final_acceptance_checklist.csv", acceptance_rows)
    write_csv(OUT_DIR / "00_status" / "copied_result_manifest.csv", copied)
    write_csv(OUT_DIR / "00_status" / "missing_expected_sources.csv", missing)
    write_csv(OUT_DIR / "00_status" / "sensitive_or_case_level_sources_not_copied.csv", skipped_sensitive)
    write_markdown(status_rows, acceptance_rows, copied, skipped_sensitive)

    readme = [
        "# HyDRA-CoE No-Report Execution Status Bundle",
        "",
        "This folder centralizes the implemented no-report experiment outputs that are safe to view as paper-facing summaries.",
        "",
        "Start here:",
        "",
        "- `00_status/IMPLEMENTED_PENDING_CHECKLIST.md`",
        "- `00_status/prompt_execution_status.csv`",
        "- `01_latest_corrected_5center_results/CORRECTED_5CENTER_RESULTS_SUMMARY.md`",
        "",
        "The current main experiment uses 1897 aligned patients from the original 3010-case registry, with colposcopy images, OCT images/B-scans, and HPV/TCT/Age clinical text variables only. Examination reports and VLM evidence caches are not used.",
        "",
        "Case-level files are referenced, not copied, in order to avoid duplicating identifiers or source paths.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Bundle written to: {OUT_DIR}")
    print("Prompt status counts:", dict(pd.Series([r["status"] for r in status_rows]).value_counts()))
    print("Acceptance status counts:", dict(pd.Series([r["status"] for r in acceptance_rows]).value_counts()))
    print(f"Copied files: {len(copied)}")
    print(f"Skipped case-level/sensitive sources: {len(skipped_sensitive)}")


if __name__ == "__main__":
    main()
