#!/usr/bin/env python3
"""Run the Information Fusion supplementary experiment package.

This script implements the executable parts of prompts P00-P14 from
``Codex Sequential Prompts for Information Fusion Supplementary Experiments``.
It reuses locked patient-level predictions and cached feature arrays, writes
all new outputs under ``paper_revisions/if_supplementary_experiments``, and
records unavailable experiments explicitly instead of fabricating results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional for tabular outputs
    sns = None

from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_revisions/if_supplementary_experiments"
LOG_DIR = OUT / "logs"

PATHS = {
    "data_lock": ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv",
    "table1": ROOT / "outputs/publishable_v2/tables/table1_cohort_baseline_n1897.csv",
    "test_predictions": ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_all_models.csv",
    "validation_predictions": ROOT / "outputs/publishable_v2/step2_main_loco/predictions/validation_predictions_all_models.csv",
    "hydra_predictions": ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_hydra_full.csv",
    "feature_npz": ROOT / "outputs/publishable_v2/step2_main_loco/audit/step2_locked_feature_arrays.npz",
    "loco_folds": ROOT / "outputs/publishable_v2/splits/loco_folds_v2.json",
    "fixed_split": ROOT / "outputs/publishable_v2/splits/fixed_external_split_v2.json",
    "split_manifest": ROOT / "outputs/publishable_v2/splits/split_manifest_v2.csv",
    "missing_modality": ROOT / "paper_revision/final_outputs/infofusion_revision_package/05_robustness/missing_modality_robustness_metrics.csv",
    "input_corruption": ROOT / "paper_revision/final_outputs/infofusion_revision_package/05_robustness/input_corruption_robustness_metrics.csv",
    "label_noise": ROOT / "paper_revision/final_outputs/infofusion_revision_package/05_robustness/label_noise_stress_metrics.csv",
    "failure_cases": ROOT / "paper_revision/final_outputs/infofusion_revision_package/08_coe_faithfulness/failure_cases_automatic.csv",
    "umap_manifest": ROOT / "outputs/publishable_v2/if_route_b_submission_pack/source_csv/umap_input_feature_manifest.csv",
    "shared_lora_ablation_root": ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897/ablations",
}

MODEL_DISPLAY = {
    "HyDRA_CoE_Full": "HyDRA-CoE full",
    "BioMedCLIP_Finetuned": "BioMedCLIP",
    "ColposcopyOCTText_CrossAttention": "Cross-attention fusion",
    "ColposcopyOCT_LateFusion": "Late score fusion",
    "ColposcopyOCT_EarlyConcat": "Early concat",
    "OCTOnly_ViT": "OCT only",
    "ColposcopyOnly_ViT": "Colposcopy only",
    "ClinicalOnly_Logistic": "Clinical logistic",
    "ClinicalOnly_XGBoost": "Clinical XGBoost",
}

CENTER_DISPLAY = {
    "武大人民医院": "Wuhan Renmin",
    "恩施州中心医院": "Enshi",
    "襄阳市中心医院": "Xiangyang",
    "十堰市人民医院": "Shiyan",
    "荆州市第一人民医院": "Jingzhou",
}

PALETTE = [
    "#8b98b3",
    "#abb8cc",
    "#dbb98c",
    "#edd6b8",
    "#b57979",
    "#dea3a2",
    "#b3b0b0",
    "#d9d8d8",
]

PRIMARY_MODEL = "HyDRA_CoE_Full"
BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 2026
HASH_SALT = "if_supplementary_experiments_2026"


@dataclass
class StepResult:
    step_id: str
    script: str = "scripts/if_supplementary/run_all_if_supplementary_experiments.py"
    status: str = "PASS"
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SupplementaryRunner:
    def __init__(self) -> None:
        self.step_results: list[StepResult] = []
        self.missing: list[dict[str, str]] = []
        self.protocol: pd.DataFrame | None = None
        self.test_raw: pd.DataFrame | None = None
        self.val_raw: pd.DataFrame | None = None
        self.test_agg: pd.DataFrame | None = None
        self.val_agg: pd.DataFrame | None = None
        self.thresholds: pd.DataFrame | None = None
        self.cin3_safety: pd.DataFrame | None = None
        self.pooled_ci: pd.DataFrame | None = None
        self.center_ci: pd.DataFrame | None = None

    def run(self) -> None:
        setup_style()
        make_dirs()
        steps = [
            self.p00_repository_data_audit,
            self.p01_protocol_lock,
            self.p02_prediction_registry,
            self.p03_cin3_safety,
            self.p04_patient_level_statistics,
            self.p05_same_backbone_baselines,
            self.p06_modality_and_missingness,
            self.p07_calibration_ablation,
            self.p08_center_shift,
            self.p09_clinical_utility,
            self.p10_reliability_validation,
            self.p11_coe_faithfulness,
            self.p12_subgroup_failure_audit,
            self.p13_final_tables_figures_claim_lock,
            self.p14_reproducibility_package,
        ]
        for fn in steps:
            self._run_step(fn)
        self.write_missing_requirements()
        self.write_runtime_manifest()

    def _run_step(self, fn) -> None:
        result = StepResult(step_id=fn.__name__.split("_", 1)[0].upper())
        result.start_time = now()
        start = time.time()
        try:
            outputs, warns = fn()
            result.output_files.extend(str_rel(x) for x in outputs)
            result.warnings.extend(warns)
            if any("NOT_EXECUTABLE" in w for w in warns):
                result.status = "PARTIAL"
        except Exception as exc:  # keep later prompts executable
            result.status = "FAIL"
            result.errors.append(repr(exc))
            report_dir = OUT / "13_submission_audit"
            report_dir.mkdir(parents=True, exist_ok=True)
            append_text(report_dir / "ORCHESTRATOR_ERRORS.log", f"{result.step_id}: {repr(exc)}\n")
        result.end_time = now()
        result.duration_seconds = round(time.time() - start, 3)
        self.step_results.append(result)
        print(f"[{result.status}] {result.step_id} {result.duration_seconds:.1f}s")

    # P00
    def p00_repository_data_audit(self) -> tuple[list[Path], list[str]]:
        out = OUT / "00_protocol_lock"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        inventory_rows: list[dict[str, object]] = []

        candidates = {
            "analytic_cohort": [PATHS["data_lock"], PATHS["table1"]],
            "split_files": [PATHS["loco_folds"], PATHS["fixed_split"], PATHS["split_manifest"]],
            "prediction_files": [
                PATHS["test_predictions"],
                PATHS["validation_predictions"],
                PATHS["hydra_predictions"],
                *sorted((ROOT / "outputs/publishable_v2").glob("**/*predictions*.csv"))[:80],
            ],
            "feature_caches": [
                PATHS["feature_npz"],
                *sorted((ROOT / "outputs/publishable_v2").glob("**/*feature*.npz"))[:50],
                *sorted((ROOT / "outputs/publishable_v2").glob("**/*feature*.csv"))[:80],
            ],
            "checkpoints": sorted((ROOT / "outputs/publishable_v2").glob("**/checkpoints/**"))[:120],
            "scripts": sorted((ROOT / "scripts").glob("**/*.py"))[:250],
            "robustness_outputs": [
                PATHS["missing_modality"],
                PATHS["input_corruption"],
                PATHS["label_noise"],
                PATHS["failure_cases"],
            ],
        }
        seen: set[Path] = set()
        for category, paths in candidates.items():
            for path in paths:
                path = Path(path)
                if path in seen:
                    continue
                seen.add(path)
                inventory_rows.append(file_inventory_row(category, path))

        inv = pd.DataFrame(inventory_rows)
        inv_path = out / "repository_inventory.csv"
        inv.to_csv(inv_path, index=False, encoding="utf-8-sig")
        outputs.append(inv_path)

        lock = read_csv(PATHS["data_lock"])
        center_summary = summarize_centers(lock)
        center_path = out / "center_and_endpoint_summary.csv"
        center_summary.to_csv(center_path, index=False, encoding="utf-8-sig")
        outputs.append(center_path)

        config = {
            "created_at": now(),
            "root": str(ROOT),
            "paths": {k: str(v) for k, v in PATHS.items() if isinstance(v, Path)},
            "columns": {
                "patient_id": "patient_id",
                "case_id": "case_id",
                "center": "center_name",
                "cin2_label": "pathology_cin2plus",
                "cin3_label": "pathology_cin3plus",
                "score": "prob_cin2plus",
            },
            "primary_protocol": "strict five-fold LOCO",
            "can_reconstruct_strict_loco": bool(PATHS["test_predictions"].exists() and PATHS["validation_predictions"].exists()),
            "invasive_cancer_label_available": False,
        }
        cfg_path = out / "if_supplementary_paths.json"
        write_json(cfg_path, config)
        outputs.append(cfg_path)

        report = [
            "# Repository and Data Audit",
            "",
            f"Created at: `{now()}`",
            "",
            "## Analytic Cohort",
            f"- Candidate analytic cohort size: `{len(lock)}` patients/cases from `{str_rel(PATHS['data_lock'])}`.",
            f"- Available centers: {', '.join(CENTER_DISPLAY.get(c, c) for c in sorted(lock['center_name'].dropna().unique()))}.",
            "- Patient ID column: `patient_id`; case ID column: `case_id`.",
            "- Endpoint labels: `pathology_cin2plus`, `pathology_cin3plus`.",
            "- Invasive cancer as a separate patient-level label: `not available`; CIN3+ is evaluated from `pathology_cin3plus`.",
            "",
            "## Modality Availability",
            md_table(center_summary),
            "",
            "## Protocol Status",
            "- Primary strict five-fold LOCO can be reconstructed from current patient-level prediction files.",
            "- Fixed external split files exist and are preserved as secondary supplementary material only.",
            "- Validation predictions contain source-center validation rows; this package locks one hard validation center per LOCO fold for thresholds.",
            "",
            "## Blockers",
            "- True invasive-cancer-only endpoint cannot be computed without a separate invasive cancer label.",
            "- Raw intervention-based CoE faithfulness cannot be computed without saved intervention logits or an inference hook exporting intervened states.",
            "- Image quality annotations are not available; reliability quality analyses use proxy variables only.",
            "- Random 10/30/50% modality dropout patient-level predictions are not available in the locked output tree.",
        ]
        report_path = out / "REPOSITORY_DATA_AUDIT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P01
    def p01_protocol_lock(self) -> tuple[list[Path], list[str]]:
        out = OUT / "00_protocol_lock"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        lock = read_csv(PATHS["data_lock"])
        centers = sorted(lock["center_name"].dropna().unique())
        rows = []
        for test_center in centers:
            val_center = choose_validation_center(lock, test_center)
            train_centers = [c for c in centers if c not in {test_center, val_center}]
            row = {
                "fold_id": f"loco_{test_center}",
                "held_out_test_center": test_center,
                "validation_center": val_center,
                "training_centers": ";".join(train_centers),
                "target_labels_used_for_training": False,
                "target_labels_used_for_threshold_selection": False,
                "target_unlabeled_predictions_used_for_calibration": True,
                "notes": "Primary strict LOCO; target labels excluded from training and threshold selection.",
            }
            for role, selected in [
                ("train", train_centers),
                ("val", [val_center]),
                ("test", [test_center]),
            ]:
                sub = lock[lock["center_name"].isin(selected)]
                row[f"n_{role}"] = int(len(sub))
                row[f"cin2_pos_{role}"] = int(pd.to_numeric(sub["pathology_cin2plus"], errors="coerce").fillna(0).sum())
                row[f"cin3_pos_{role}"] = int(pd.to_numeric(sub["pathology_cin3plus"], errors="coerce").fillna(0).sum())
                row[f"invasive_cancer_{role}"] = np.nan
                row[f"oct_bscan_count_{role}"] = int(pd.to_numeric(sub.get("oct_num_bscans", 0), errors="coerce").fillna(0).sum())
                row[f"colposcopy_image_count_{role}"] = int(pd.to_numeric(sub.get("colposcopy_num_images", 0), errors="coerce").fillna(0).sum())
                row[f"missing_hpv_{role}"] = int(sub.get("hpv_status_harmonized", pd.Series(index=sub.index, dtype=object)).isna().sum())
                row[f"missing_tct_{role}"] = int(sub.get("tct_status_harmonized", pd.Series(index=sub.index, dtype=object)).isna().sum())
            if lock.loc[lock["center_name"].eq(test_center), "pathology_cin2plus"].nunique(dropna=True) < 2:
                row["notes"] += " One-class CIN2+ test center; AUROC undefined."
            rows.append(row)
        protocol = pd.DataFrame(rows)
        self.protocol = protocol
        protocol_path = out / "loco_protocol_lock.csv"
        protocol.to_csv(protocol_path, index=False, encoding="utf-8-sig")
        outputs.append(protocol_path)

        if PATHS["fixed_split"].exists():
            fixed_path = out / "fixed_external_protocol_lock.csv"
            fixed_rows = [{"source_file": str_rel(PATHS["fixed_split"]), "status": "PRESERVED_AS_SECONDARY_SUPPLEMENT_ONLY"}]
            pd.DataFrame(fixed_rows).to_csv(fixed_path, index=False, encoding="utf-8-sig")
            outputs.append(fixed_path)

        fig_path = out / "figure_protocol_split_flow"
        fig_split_flow(protocol, fig_path)
        outputs.extend([fig_path.with_suffix(".png"), fig_path.with_suffix(".pdf")])

        report = [
            "# Protocol Lock",
            "",
            "Primary analysis is strict five-fold leave-one-center-out (LOCO). For each fold, the held-out center is used only for testing.",
            "",
            "Validation thresholds are selected from one deterministic hard validation center among the remaining source centers. Target labels are not used for training or threshold selection.",
            "",
            "Target-label-free logit median matching, when used, is transductive calibration because it uses the unlabeled target prediction distribution.",
            "",
            "A fixed external split exists and is treated as optional supplementary material only; it is not mixed into the primary LOCO tables.",
            "",
            "## Center Class Check",
            md_table(class_check(lock)),
        ]
        report_path = out / "PROTOCOL_LOCK.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P02
    def p02_prediction_registry(self) -> tuple[list[Path], list[str]]:
        out = OUT / "01_prediction_registry"
        std_dir = out / "standardized_predictions"
        out.mkdir(parents=True, exist_ok=True)
        std_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []

        test = self._load_test_predictions()
        val = self._load_validation_predictions()
        registry_rows: list[dict[str, object]] = []

        for split_name, df, src in [
            ("test", test, PATHS["test_predictions"]),
            ("validation", val, PATHS["validation_predictions"]),
        ]:
            for model, g in df.groupby("model_name", dropna=False):
                std = standardize_step2_predictions(g, split_name, src)
                path = std_dir / f"{safe_name(model)}_{split_name}.csv"
                std.to_csv(path, index=False, encoding="utf-8-sig")
                outputs.append(path)
                registry_rows.append(
                    {
                        "model_name": model,
                        "protocol": "strict_loco",
                        "fold_id": "all_loco_folds",
                        "center": "all_available_centers",
                        "seed": "all_available_seeds",
                        "prediction_path": str_rel(path),
                        "n_patients": int(std["patient_id"].nunique()),
                        "has_cin2": std["y_cin2"].notna().any(),
                        "has_cin3": std["y_cin3"].notna().any(),
                        "has_logits": std["pred_logit"].notna().any(),
                        "has_reliability_weights": bool({"alpha_colposcopy", "alpha_oct", "alpha_semantic"}.issubset(set(g.columns))),
                        "has_latent_states": bool({"delta_prior_to_semantic", "delta_semantic_to_colposcopy", "delta_colposcopy_to_oct"}.issubset(set(g.columns))),
                        "usable_for_main_table": split_name == "test",
                        "notes": f"Standardized from {str_rel(src)}; split={split_name}.",
                    }
                )

        shared_rows, shared_outputs = self._register_shared_lora_predictions(std_dir)
        registry_rows.extend(shared_rows)
        outputs.extend(shared_outputs)

        reg = pd.DataFrame(registry_rows)
        reg_path = out / "prediction_registry.csv"
        reg.to_csv(reg_path, index=False, encoding="utf-8-sig")
        outputs.append(reg_path)

        self.test_agg = aggregate_step2_predictions(test)
        self.val_agg = aggregate_step2_predictions(val)
        agg_test_path = out / "standardized_patient_mean_test_predictions.csv"
        agg_val_path = out / "standardized_patient_mean_validation_predictions.csv"
        self.test_agg.to_csv(agg_test_path, index=False, encoding="utf-8-sig")
        self.val_agg.to_csv(agg_val_path, index=False, encoding="utf-8-sig")
        outputs.extend([agg_test_path, agg_val_path])

        report = [
            "# Prediction Registry Audit",
            "",
            f"Standardized prediction files were created for `{reg['model_name'].nunique()}` main models across validation and test splits.",
            "",
            "- All later analyses in this package use patient-level seed-averaged predictions from `standardized_patient_mean_*_predictions.csv`.",
            "- Patient and case identifiers are hashed in standardized files.",
            "- Undefined metrics are returned as blank/NaN for one-class groups.",
            "",
            "## Registry Summary",
            md_table(reg.groupby(["protocol", "usable_for_main_table"]).size().reset_index(name="n_files")),
        ]
        report_path = out / "PREDICTION_REGISTRY_AUDIT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P03
    def p03_cin3_safety(self) -> tuple[list[Path], list[str]]:
        out = OUT / "02_cin3_safety"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        test = self._get_test_agg()
        val = self._get_val_agg()
        protocol = self._get_protocol()

        rows = []
        threshold_rows = []
        for (model, fold), gtest in test.groupby(["model_name", "fold_id"], dropna=False):
            prow = protocol[protocol["fold_id"].eq(fold)]
            val_center = str(prow["validation_center"].iloc[0]) if not prow.empty else None
            gval = val[(val["model_name"].eq(model)) & (val["fold_id"].eq(fold))]
            if val_center:
                hard_val = gval[gval["center"].eq(val_center)]
                if len(hard_val) > 0:
                    gval = hard_val
            t_cin2, _ = select_threshold_f1(gval["y_cin2"].to_numpy(), gval["score"].to_numpy())
            t_cin3, safety_met = select_threshold_safety(
                gval["y_cin3"].to_numpy(),
                gval["score"].to_numpy(),
                sensitivity_floor=0.95,
            )
            threshold_rows.append(
                {
                    "model_name": model,
                    "fold_id": fold,
                    "validation_center": val_center,
                    "threshold_cin2_f1_val": t_cin2,
                    "threshold_cin3_safety_val": t_cin3,
                    "safety_floor_met_on_validation": safety_met,
                    "n_validation": len(gval),
                }
            )
            m2 = binary_metrics(gtest["y_cin2"], gtest["score"], t_cin2)
            m3 = binary_metrics(gtest["y_cin3"], gtest["score"], t_cin3)
            rows.append(
                {
                    "model_name": model,
                    "protocol": "strict_loco",
                    "fold_id": fold,
                    "test_center": first_value(gtest["center"]),
                    "validation_center": val_center,
                    "n_test": int(len(gtest)),
                    "cin3_pos_test": int(np.nansum(gtest["y_cin3"])),
                    "cin2_pos_test": int(np.nansum(gtest["y_cin2"])),
                    "threshold_cin2_f1_val": t_cin2,
                    "threshold_cin3_safety_val": t_cin3,
                    "safety_floor_target": 0.95,
                    "safety_floor_met_on_validation": safety_met,
                    "cin3_sensitivity_test": m3["sensitivity"],
                    "cin3_specificity_test": m3["specificity"],
                    "cin3_ppv_test": m3["ppv"],
                    "cin3_npv_test": m3["npv"],
                    "cin3_false_negatives_test": m3["fn"],
                    "cin3_false_positives_test": m3["fp"],
                    "cin2_sensitivity_test": m2["sensitivity"],
                    "cin2_specificity_test": m2["specificity"],
                    "cin2_false_negatives_test": m2["fn"],
                    "referral_rate_test": m2["referral_rate"],
                    "notes": metric_notes(gtest["y_cin3"]),
                }
            )

        safety = pd.DataFrame(rows)
        thresholds = pd.DataFrame(threshold_rows)
        self.thresholds = thresholds
        self.cin3_safety = safety
        safety_path = out / "cin3_safety_by_center.csv"
        summary_path = out / "cin3_safety_summary.csv"
        thresholds_path = out / "validation_locked_thresholds.csv"
        safety.to_csv(safety_path, index=False, encoding="utf-8-sig")
        thresholds.to_csv(thresholds_path, index=False, encoding="utf-8-sig")
        summary = safety.groupby("model_name", dropna=False).agg(
            n_test=("n_test", "sum"),
            cin3_pos_test=("cin3_pos_test", "sum"),
            cin3_false_negatives_test=("cin3_false_negatives_test", "sum"),
            mean_cin3_sensitivity_test=("cin3_sensitivity_test", "mean"),
            mean_cin3_specificity_test=("cin3_specificity_test", "mean"),
            mean_referral_rate_test=("referral_rate_test", "mean"),
            folds_meeting_validation_floor=("safety_floor_met_on_validation", "sum"),
        ).reset_index()
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        outputs.extend([safety_path, summary_path, thresholds_path])

        fig_cin3_false_negatives(safety, out / "figure_cin3_false_negatives_by_center")
        fig_cin3_tradeoff(safety, out / "figure_cin3_sensitivity_specificity_tradeoff")
        fig_locked_thresholds(thresholds, out / "figure_locked_thresholds_by_fold")
        outputs.extend(fig_pair_paths(out, "figure_cin3_false_negatives_by_center"))
        outputs.extend(fig_pair_paths(out, "figure_cin3_sensitivity_specificity_tradeoff"))
        outputs.extend(fig_pair_paths(out, "figure_locked_thresholds_by_fold"))

        hydra = safety[safety["model_name"].eq(PRIMARY_MODEL)]
        floor_count = int(hydra["safety_floor_met_on_validation"].sum()) if len(hydra) else 0
        report = [
            "# CIN3+ Safety Report",
            "",
            "Thresholds were selected on the locked validation center only, then applied to the held-out LOCO test center.",
            "",
            f"- HyDRA validation folds meeting CIN3+ sensitivity floor: `{floor_count}/{len(hydra)}`.",
            f"- HyDRA total CIN3+ false negatives on test: `{int(hydra['cin3_false_negatives_test'].sum()) if len(hydra) else 'NA'}`.",
            "- Separate invasive cancer label is unavailable; the report uses `pathology_cin3plus` as the CIN3+ safety endpoint.",
            "",
            "Safe wording: validation-locked thresholds characterize the observed CIN3+ sensitivity/referral trade-off under LOCO.",
            "",
            "Unsafe wording: do not claim clinical safety or deployment readiness unless all centers meet the prespecified sensitivity floor with acceptable referral burden.",
        ]
        report_path = out / "CIN3_SAFETY_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P04
    def p04_patient_level_statistics(self) -> tuple[list[Path], list[str]]:
        out = OUT / "03_patient_level_statistics"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        test = self._get_test_agg()
        thresholds = self._get_thresholds()
        test = test.merge(thresholds[["model_name", "fold_id", "threshold_cin2_f1_val"]], on=["model_name", "fold_id"], how="left")
        test["threshold"] = test["threshold_cin2_f1_val"].fillna(0.5)

        center_rows = []
        metric_names = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "brier", "ece", "referral_rate"]

        for (model, center), g in test.groupby(["model_name", "center"], dropna=False):
            row = {"model_name": model, "center": center, "protocol": "strict_loco", "n": int(len(g))}
            base = metrics_for_ci(g, g["threshold"].iloc[0])
            row.update(flatten_ci(base, {m: (np.nan, np.nan) for m in metric_names}))
            row["ci_source"] = "not_available_in_existing_step2_center_level_bootstrap"
            center_rows.append(row)

        center_ci = pd.DataFrame(center_rows)
        pooled_ci = load_existing_pooled_bootstrap_ci(test)
        self.center_ci = center_ci
        self.pooled_ci = pooled_ci
        center_path = out / "main_metrics_with_ci_by_center.csv"
        pooled_path = out / "pooled_metrics_with_ci.csv"
        center_ci.to_csv(center_path, index=False, encoding="utf-8-sig")
        pooled_ci.to_csv(pooled_path, index=False, encoding="utf-8-sig")
        outputs.extend([center_path, pooled_path])

        pair = load_existing_paired_tests()
        pair_path = out / "paired_model_comparison.csv"
        pair.to_csv(pair_path, index=False, encoding="utf-8-sig")
        outputs.append(pair_path)

        fig_auc_ci_forest(center_ci, out / "figure_auc_ci_forest_by_center")
        fig_sens_spec_ci(pooled_ci, out / "figure_sensitivity_specificity_ci_by_model")
        fig_paired_auc_diff(pair, out / "figure_paired_auc_difference_vs_hydra")
        outputs.extend(fig_pair_paths(out, "figure_auc_ci_forest_by_center"))
        outputs.extend(fig_pair_paths(out, "figure_sensitivity_specificity_ci_by_model"))
        outputs.extend(fig_pair_paths(out, "figure_paired_auc_difference_vs_hydra"))

        hydra = pooled_ci[pooled_ci["model_name"].eq(PRIMARY_MODEL)]
        report = [
            "# Patient-Level Statistics Report",
            "",
            f"Pooled bootstrap confidence intervals and paired tests are reused from the locked Step2 output generated with `{BOOTSTRAP_ITERATIONS}` patient-level resamples.",
            "Center-level rows contain point estimates because the existing locked Step2 file does not store center-level bootstrap CIs.",
            "",
            "Recommended Table 2 replacement: `pooled_metrics_with_ci.csv` plus center-level details from `main_metrics_with_ci_by_center.csv`.",
            "",
            "Statistical superiority should only be claimed for comparisons whose paired bootstrap CI excludes zero and p-value supports the direction.",
            "",
            "HyDRA pooled metrics:",
            md_table(hydra) if len(hydra) else "HyDRA row not found.",
        ]
        report_path = out / "PATIENT_LEVEL_STATISTICS_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P05
    def p05_same_backbone_baselines(self) -> tuple[list[Path], list[str]]:
        out = OUT / "04_same_backbone_fusion_baselines"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        warns: list[str] = []
        test = self._get_test_agg()
        thresholds = self._get_thresholds()
        merged = test.merge(thresholds[["model_name", "fold_id", "threshold_cin2_f1_val", "threshold_cin3_safety_val"]], on=["model_name", "fold_id"], how="left")
        merged["variant"] = merged["model_name"].map(MAIN_TO_SAME_BACKBONE).fillna(merged["model_name"])
        by_center = compute_group_metrics(merged, ["variant", "model_name", "center"], "threshold_cin2_f1_val")
        pooled = compute_group_metrics(merged, ["variant", "model_name"], "threshold_cin2_f1_val")

        shared = load_shared_lora_ablation_metrics()
        if len(shared):
            shared_path = out / "shared_lora_available_ablation_metrics.csv"
            shared.to_csv(shared_path, index=False, encoding="utf-8-sig")
            outputs.append(shared_path)
        else:
            warns.append("NOT_EXECUTABLE: shared-lora ablation prediction files were not found.")

        expected = [
            "same_backbone_clinical_only",
            "same_backbone_colpo_only",
            "same_backbone_oct_only",
            "same_backbone_early_concat",
            "same_backbone_late_score_fusion",
            "same_backbone_cross_attention",
            "same_backbone_gated_fusion",
            "same_backbone_reliability_without_coe",
            "same_backbone_coe_without_reliability",
            "same_backbone_hydra_full",
        ]
        available = set(by_center["variant"].dropna().unique()) | set(shared["variant"].dropna().unique() if len(shared) else [])
        ablation_rows = []
        for variant in expected:
            status = "AVAILABLE" if variant in available else "NOT_EXECUTABLE"
            source = "locked_step2_predictions" if variant in set(by_center["variant"].unique()) else ""
            if len(shared) and variant in set(shared["variant"].unique()):
                source = "shared_lora_improved_1897_ablation_predictions"
            if status == "NOT_EXECUTABLE":
                self.missing.append({"step": "P05", "item": variant, "reason": "No matching patient-level same-backbone prediction file found."})
            ablation_rows.append({"variant": variant, "status": status, "source": source, "notes": "" if status == "AVAILABLE" else "Requires raw/frozen-feature training for this exact control."})
        ablation = pd.DataFrame(ablation_rows)

        by_center_path = out / "same_backbone_metrics_by_center.csv"
        pooled_path = out / "same_backbone_pooled_metrics.csv"
        ablation_path = out / "same_backbone_ablation_table.csv"
        by_center.to_csv(by_center_path, index=False, encoding="utf-8-sig")
        pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
        ablation.to_csv(ablation_path, index=False, encoding="utf-8-sig")
        outputs.extend([by_center_path, pooled_path, ablation_path])

        fig_same_backbone_auc(pooled, out / "figure_same_backbone_auc_by_variant")
        fig_same_backbone_safety(merged, out / "figure_same_backbone_cin3_safety_by_variant")
        fig_same_backbone_calibration(merged, out / "figure_same_backbone_calibration_by_variant")
        outputs.extend(fig_pair_paths(out, "figure_same_backbone_auc_by_variant"))
        outputs.extend(fig_pair_paths(out, "figure_same_backbone_cin3_safety_by_variant"))
        outputs.extend(fig_pair_paths(out, "figure_same_backbone_calibration_by_variant"))

        report = [
            "# Same-Backbone Fusion Baseline Report",
            "",
            "Available patient-level controls were evaluated from locked Step2 LOCO predictions and available Shared-LoRA ablation predictions.",
            "",
            "The exact ten-variant same-backbone design is only partially available in the current repository. Missing variants are listed in `same_backbone_ablation_table.csv` and `MISSING_REQUIREMENTS.md`.",
            "",
            "Do not state that the full gain is isolated from backbone/pretraining unless all same-backbone variants are trained on the same cohort with the same encoder and schedule.",
        ]
        report_path = out / "SAME_BACKBONE_BASELINE_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        if (ablation["status"] == "NOT_EXECUTABLE").any():
            warns.append("NOT_EXECUTABLE: one or more required same-backbone variants are missing.")
        return outputs, warns

    # P06
    def p06_modality_and_missingness(self) -> tuple[list[Path], list[str]]:
        out = OUT / "05_modality_ablation_and_missingness"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        warns: list[str] = []
        test = self._get_test_agg()
        thresholds = self._get_thresholds()
        merged = test.merge(thresholds[["model_name", "fold_id", "threshold_cin2_f1_val", "threshold_cin3_safety_val"]], on=["model_name", "fold_id"], how="left")
        merged["variant"] = merged["model_name"].map(MODALITY_VARIANT).fillna(merged["model_name"])
        contrib = compute_group_metrics(merged, ["variant", "model_name"], "threshold_cin2_f1_val")
        contrib_path = out / "modality_contribution_table.csv"
        contrib.to_csv(contrib_path, index=False, encoding="utf-8-sig")
        outputs.append(contrib_path)

        if PATHS["missing_modality"].exists():
            removal = read_csv(PATHS["missing_modality"])
            removal["source"] = str_rel(PATHS["missing_modality"])
            removal["cin3_false_negatives"] = np.nan
        else:
            removal = pd.DataFrame()
            warns.append("NOT_EXECUTABLE: complete modality removal metrics not found.")
        removal_path = out / "complete_modality_removal_table.csv"
        removal.to_csv(removal_path, index=False, encoding="utf-8-sig")
        outputs.append(removal_path)

        dropout_rows = []
        for rate in [0.10, 0.30, 0.50]:
            for modality in ["clinical", "colposcopy", "oct", "random_modality"]:
                dropout_rows.append(
                    {
                        "dropout_rate": rate,
                        "dropout_modality": modality,
                        "status": "NOT_EXECUTABLE",
                        "auc": np.nan,
                        "sensitivity": np.nan,
                        "specificity": np.nan,
                        "ppv": np.nan,
                        "npv": np.nan,
                        "f1": np.nan,
                        "brier": np.nan,
                        "ece": np.nan,
                        "referral_rate": np.nan,
                        "cin3_false_negatives": np.nan,
                        "notes": "Patient-level random modality dropout predictions at this rate are not available.",
                    }
                )
        dropout = pd.DataFrame(dropout_rows)
        dropout_path = out / "random_dropout_stress_table.csv"
        dropout.to_csv(dropout_path, index=False, encoding="utf-8-sig")
        outputs.append(dropout_path)
        self.missing.append({"step": "P06", "item": "random modality dropout 10/30/50%", "reason": "No patient-level dropout-rate prediction files found."})
        warns.append("NOT_EXECUTABLE: random 10/30/50% modality dropout stress test requires new predictions.")

        fig_modality_auc(contrib, out / "figure_modality_contribution_auc")
        fig_removal_auc_npv(removal, out / "figure_complete_modality_removal_auc_npv")
        fig_dropout_placeholder(dropout, out / "figure_random_dropout_rate_auc_npv")
        fig_missing_cin3_fn(removal, out / "figure_missing_modality_cin3_fn")
        outputs.extend(fig_pair_paths(out, "figure_modality_contribution_auc"))
        outputs.extend(fig_pair_paths(out, "figure_complete_modality_removal_auc_npv"))
        outputs.extend(fig_pair_paths(out, "figure_random_dropout_rate_auc_npv"))
        outputs.extend(fig_pair_paths(out, "figure_missing_modality_cin3_fn"))

        report = [
            "# Modality Contribution and Missingness Report",
            "",
            "Part A uses locked patient-level LOCO predictions for available modality/fusion variants.",
            "Part B reuses existing complete modality-removal and random one/two modality-removal metrics.",
            "Part C is not executable from current outputs because patient-level 10/30/50% modality dropout predictions are absent.",
            "",
            "Do not describe complete modality removal as random dropout-rate stress testing.",
        ]
        report_path = out / "MODALITY_AND_MISSINGNESS_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, warns

    # P07
    def p07_calibration_ablation(self) -> tuple[list[Path], list[str]]:
        out = OUT / "06_calibration_ablation"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        test = self._get_test_agg()
        val = self._get_val_agg()
        thresholds = self._get_thresholds()

        rows = []
        calibrated_frames = []
        for (model, fold), gtest in test.groupby(["model_name", "fold_id"], dropna=False):
            gval = val[(val["model_name"].eq(model)) & (val["fold_id"].eq(fold))]
            th = thresholds[(thresholds["model_name"].eq(model)) & (thresholds["fold_id"].eq(fold))]
            t_cin2 = float(th["threshold_cin2_f1_val"].iloc[0]) if len(th) else 0.5
            t_cin3 = float(th["threshold_cin3_safety_val"].iloc[0]) if len(th) else 0.5
            temp = fit_temperature(gval["y_cin2"].to_numpy(), gval["score"].to_numpy()) if len(gval) else 1.0
            val_med = np.nanmedian(logit_np(gval["score"].to_numpy())) if len(gval) else 0.0
            test_med = np.nanmedian(logit_np(gtest["score"].to_numpy()))
            offset = val_med - test_med
            variants = {
                "no_calibration": (gtest["score"].to_numpy(), 0.5, 0.5, False),
                "threshold_lock_only": (gtest["score"].to_numpy(), t_cin2, t_cin3, False),
                "temperature_scaling_val_only": (sigmoid(logit_np(gtest["score"].to_numpy()) / temp), t_cin2, t_cin3, False),
                "logit_median_matching_target_unlabeled": (sigmoid(logit_np(gtest["score"].to_numpy()) + offset), t_cin2, t_cin3, True),
                "full_hydra_calibration" if model == PRIMARY_MODEL else "full_model_calibration": (sigmoid(logit_np(gtest["score"].to_numpy()) + offset), t_cin2, t_cin3, True),
            }
            for variant, (score, th2, th3, uses_target_dist) in variants.items():
                tmp = gtest.copy()
                tmp["calibration_variant"] = variant
                tmp["calibrated_score"] = score
                tmp["uses_unlabeled_target_distribution"] = uses_target_dist
                calibrated_frames.append(tmp)
                m2 = binary_metrics(tmp["y_cin2"], tmp["calibrated_score"], th2)
                m3 = binary_metrics(tmp["y_cin3"], tmp["calibrated_score"], th3)
                rows.append(
                    {
                        "model_name": model,
                        "fold_id": fold,
                        "center": first_value(gtest["center"]),
                        "calibration_variant": variant,
                        "uses_unlabeled_target_distribution": uses_target_dist,
                        "temperature": temp if variant == "temperature_scaling_val_only" else np.nan,
                        "median_offset": offset if uses_target_dist else np.nan,
                        "auroc": safe_auc(tmp["y_cin2"], tmp["calibrated_score"]),
                        "brier": safe_brier(tmp["y_cin2"], tmp["calibrated_score"]),
                        "ece": expected_calibration_error(tmp["y_cin2"], tmp["calibrated_score"]),
                        "sensitivity": m2["sensitivity"],
                        "specificity": m2["specificity"],
                        "ppv": m2["ppv"],
                        "npv": m2["npv"],
                        "cin3_sensitivity": m3["sensitivity"],
                        "cin3_false_negatives": m3["fn"],
                        "referral_rate": m2["referral_rate"],
                    }
                )

        by_center = pd.DataFrame(rows)
        pooled = by_center.groupby(["model_name", "calibration_variant", "uses_unlabeled_target_distribution"], dropna=False).agg(
            auroc=("auroc", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            ppv=("ppv", "mean"),
            npv=("npv", "mean"),
            cin3_sensitivity=("cin3_sensitivity", "mean"),
            cin3_false_negatives=("cin3_false_negatives", "sum"),
            referral_rate=("referral_rate", "mean"),
        ).reset_index()
        by_center_path = out / "calibration_ablation_by_center.csv"
        pooled_path = out / "calibration_ablation_pooled.csv"
        by_center.to_csv(by_center_path, index=False, encoding="utf-8-sig")
        pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
        outputs.extend([by_center_path, pooled_path])

        cal_all = pd.concat(calibrated_frames, ignore_index=True)
        cal_path = out / "calibration_variant_patient_scores.csv"
        cal_all[["patient_id_hash", "case_id_hash", "model_name", "fold_id", "center", "y_cin2", "y_cin3", "score", "calibration_variant", "calibrated_score", "uses_unlabeled_target_distribution"]].to_csv(cal_path, index=False, encoding="utf-8-sig")
        outputs.append(cal_path)

        fig_calibration_ece(by_center, out / "figure_calibration_ablation_ece_by_center")
        fig_reliability_curves(cal_all, out / "figure_calibration_reliability_curves")
        fig_probability_density(cal_all, out / "figure_calibration_probability_density")
        fig_calibration_cin3_safety(by_center, out / "figure_calibration_effect_on_cin3_safety")
        outputs.extend(fig_pair_paths(out, "figure_calibration_ablation_ece_by_center"))
        outputs.extend(fig_pair_paths(out, "figure_calibration_reliability_curves"))
        outputs.extend(fig_pair_paths(out, "figure_calibration_probability_density"))
        outputs.extend(fig_pair_paths(out, "figure_calibration_effect_on_cin3_safety"))

        log_path = out / "target_information_used_by_calibration.csv"
        pd.DataFrame(
            [
                {"calibration_variant": "no_calibration", "uses_target_labels": False, "uses_unlabeled_target_distribution": False},
                {"calibration_variant": "threshold_lock_only", "uses_target_labels": False, "uses_unlabeled_target_distribution": False},
                {"calibration_variant": "temperature_scaling_val_only", "uses_target_labels": False, "uses_unlabeled_target_distribution": False},
                {"calibration_variant": "logit_median_matching_target_unlabeled", "uses_target_labels": False, "uses_unlabeled_target_distribution": True},
                {"calibration_variant": "full_hydra_calibration", "uses_target_labels": False, "uses_unlabeled_target_distribution": True},
            ]
        ).to_csv(log_path, index=False, encoding="utf-8-sig")
        outputs.append(log_path)

        report = [
            "# Calibration Ablation Report",
            "",
            "Use the wording `target-label-free transductive calibration` for logit median matching. Do not call it source-only.",
            "",
            "Calibration variants use validation labels only for validation-center threshold/temperature selection. Median matching uses the unlabeled target prediction distribution and is flagged accordingly.",
        ]
        report_path = out / "CALIBRATION_ABLATION_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P08
    def p08_center_shift(self) -> tuple[list[Path], list[str]]:
        out = OUT / "07_center_shift"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        warns: list[str] = []
        lock = read_csv(PATHS["data_lock"])[["case_id", "center_name", "pathology_cin2plus"]]
        reps = load_feature_representations(lock)
        if not reps:
            warns.append("NOT_EXECUTABLE: no feature representations available for center shift.")
            self.missing.append({"step": "P08", "item": "feature representations", "reason": "No compatible feature cache found."})
            return outputs, warns

        classifier_rows = []
        confs = {}
        for name, frame in reps.items():
            X = frame["X"]
            y = frame["center"].astype(str).to_numpy()
            pred, acc, macro_f1 = center_classifier_cv(X, y)
            classifier_rows.append({"representation": name, "center_classifier_accuracy": acc, "center_classifier_macro_f1": macro_f1, "n": len(y)})
            labels = sorted(np.unique(y))
            confs[name] = (labels, confusion_matrix(y, pred, labels=labels))
        classifier = pd.DataFrame(classifier_rows)
        classifier_path = out / "center_classifier_results.csv"
        classifier.to_csv(classifier_path, index=False, encoding="utf-8-sig")
        outputs.append(classifier_path)

        mmd = pairwise_distance_table(reps, "mmd")
        coral = pairwise_distance_table(reps, "coral")
        mmd_path = out / "mmd_pairwise_by_representation.csv"
        coral_path = out / "coral_pairwise_by_representation.csv"
        mmd.to_csv(mmd_path, index=False, encoding="utf-8-sig")
        coral.to_csv(coral_path, index=False, encoding="utf-8-sig")
        outputs.extend([mmd_path, coral_path])

        perf_corr = shift_performance_correlation(mmd, coral, self._get_test_agg(), self._get_thresholds())
        corr_path = out / "shift_performance_correlation.csv"
        perf_corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
        outputs.append(corr_path)

        first_rep = next(iter(reps))
        labels, conf = confs[first_rep]
        fig_center_confusion(labels, conf, out / "figure_center_classifier_confusion")
        fig_distance_heatmap(mmd, out / "figure_mmd_heatmap", "MMD")
        fig_distance_heatmap(coral, out / "figure_coral_heatmap", "CORAL")
        fig_umap_proxy(reps.get("visual_before_fusion_proxy", reps[first_rep]), out / "figure_umap_before_fusion_center_label", "Before-fusion feature proxy")
        fig_umap_proxy(reps.get("feature_concat_proxy", reps[first_rep]), out / "figure_umap_after_hydra_center_label", "After-HyDRA latent proxy")
        fig_shift_scatter(perf_corr, "auc_drop_vs_best", out / "figure_shift_vs_auc_drop")
        fig_shift_scatter(perf_corr, "ece", out / "figure_shift_vs_ece")
        for name in [
            "figure_center_classifier_confusion",
            "figure_mmd_heatmap",
            "figure_coral_heatmap",
            "figure_umap_before_fusion_center_label",
            "figure_umap_after_hydra_center_label",
            "figure_shift_vs_auc_drop",
            "figure_shift_vs_ece",
        ]:
            outputs.extend(fig_pair_paths(out, name))

        hardest = perf_corr.sort_values("auroc", na_position="first").head(1)["center"].iloc[0] if len(perf_corr) else "NA"
        largest_shift = perf_corr.sort_values("mean_shift_distance", ascending=False).head(1)["center"].iloc[0] if len(perf_corr) else "NA"
        report = [
            "# Center-Shift Report",
            "",
            f"Hardest center by HyDRA AUROC proxy: `{hardest}`.",
            f"Largest average shift center: `{largest_shift}`.",
            "",
            "The available representations are cached feature proxies (`oct`, `col`, `clinical`, and concatenated feature cache). True saved causal/nuisance latent states are not available for all cases.",
            "Do not claim center invariance if center classifier accuracy remains high.",
        ]
        report_path = out / "CENTER_SHIFT_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, warns

    # P09
    def p09_clinical_utility(self) -> tuple[list[Path], list[str]]:
        out = OUT / "08_clinical_utility"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        test = self._get_test_agg()
        thresholds = self._get_thresholds()
        merged = test.merge(thresholds[["model_name", "fold_id", "threshold_cin2_f1_val", "threshold_cin3_safety_val"]], on=["model_name", "fold_id"], how="left")
        rows = []
        for (model, center), g in merged.groupby(["model_name", "center"], dropna=False):
            th2 = float(g["threshold_cin2_f1_val"].median()) if g["threshold_cin2_f1_val"].notna().any() else 0.5
            th3 = float(g["threshold_cin3_safety_val"].median()) if g["threshold_cin3_safety_val"].notna().any() else 0.5
            pred2 = g["score"].to_numpy() >= th2
            pred3 = g["score"].to_numpy() >= th3
            y2 = g["y_cin2"].to_numpy().astype(int)
            y3 = g["y_cin3"].to_numpy().astype(int)
            m2 = binary_metrics(y2, g["score"], th2)
            m3 = binary_metrics(y3, g["score"], th3)
            n = len(g)
            rows.append(
                {
                    "model_name": model,
                    "center": center,
                    "n": n,
                    "number_referred_cin2_threshold": int(pred2.sum()),
                    "referral_rate": float(pred2.mean()),
                    "cin2_sensitivity": m2["sensitivity"],
                    "cin3_sensitivity": m3["sensitivity"],
                    "cin2_false_negatives": m2["fn"],
                    "cin3_false_negatives": m3["fn"],
                    "specificity": m2["specificity"],
                    "npv": m2["npv"],
                    "ppv": m2["ppv"],
                    "unnecessary_model_positive_referrals": int(((pred2 == 1) & (y2 == 0)).sum()),
                    "unnecessary_referrals_avoided_vs_treat_all": int((y2 == 0).sum() - ((pred2 == 1) & (y2 == 0)).sum()),
                    "missed_cin2_per_1000": per_1000(m2["fn"], n),
                    "missed_cin3_per_1000": per_1000(m3["fn"], n),
                    "threshold_cin2": th2,
                    "threshold_cin3": th3,
                }
            )
        utility = pd.DataFrame(rows)
        util_path = out / "clinical_utility_by_center.csv"
        per1000_path = out / "clinical_utility_per_1000.csv"
        utility.to_csv(util_path, index=False, encoding="utf-8-sig")
        utility.assign(
            referred_per_1000=lambda d: d["referral_rate"] * 1000,
            false_positive_referrals_per_1000=lambda d: d["unnecessary_model_positive_referrals"] / d["n"] * 1000,
        ).to_csv(per1000_path, index=False, encoding="utf-8-sig")
        outputs.extend([util_path, per1000_path])

        dca = decision_curve(merged)
        dca_path = out / "decision_curve_net_benefit.csv"
        dca.to_csv(dca_path, index=False, encoding="utf-8-sig")
        outputs.append(dca_path)

        fig_decision_curve(dca, out / "figure_decision_curve_analysis")
        fig_referral_vs_missed(utility, out / "figure_referral_reduction_vs_cin3_missed")
        fig_per1000(utility, out / "figure_per_1000_patient_utility")
        fig_locked_tradeoff(utility, out / "figure_locked_threshold_clinical_tradeoff")
        for name in [
            "figure_decision_curve_analysis",
            "figure_referral_reduction_vs_cin3_missed",
            "figure_per_1000_patient_utility",
            "figure_locked_threshold_clinical_tradeoff",
        ]:
            outputs.extend(fig_pair_paths(out, name))

        hydra = utility[utility["model_name"].eq(PRIMARY_MODEL)]
        report = [
            "# Clinical Utility Report",
            "",
            "The dataset does not contain actual biopsy/referral decisions; therefore outputs use `model-positive referrals`, not biopsy reduction.",
            "",
            f"HyDRA missed CIN3+ per 1,000 patients by center: {hydra[['center', 'missed_cin3_per_1000']].to_dict('records') if len(hydra) else 'NA'}",
            "",
            "Clinical acceptability is not claimed; the report quantifies the observed referral/safety trade-off.",
        ]
        report_path = out / "CLINICAL_UTILITY_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P10
    def p10_reliability_validation(self) -> tuple[list[Path], list[str]]:
        out = OUT / "09_reliability_validation"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        warns: list[str] = []
        hydra = read_csv(PATHS["hydra_predictions"], low_memory=False)
        thresholds = self._get_thresholds()
        rel = aggregate_hydra_reliability(hydra)
        rel = rel.merge(thresholds[thresholds["model_name"].eq(PRIMARY_MODEL)][["fold_id", "threshold_cin2_f1_val"]], on="fold_id", how="left")
        rel["threshold"] = rel["threshold_cin2_f1_val"].fillna(0.5)
        rel["prediction_correct"] = (rel["score"].ge(rel["threshold"]).astype(int) == rel["y_cin2"].astype(int))
        rel["error_type"] = np.select(
            [
                rel["prediction_correct"],
                rel["score"].ge(rel["threshold"]) & rel["y_cin2"].eq(0),
                rel["score"].lt(rel["threshold"]) & rel["y_cin2"].eq(1),
            ],
            ["correct", "false_positive", "false_negative"],
            default="unknown",
        )
        rel["hpv_missing"] = rel["hpv_status_harmonized"].isna()
        rel["tct_missing"] = rel["tct_status_harmonized"].isna()
        rel["reliability_entropy"] = reliability_entropy(rel[["alpha_clinical", "alpha_colposcopy", "alpha_oct"]].to_numpy())
        rel_path = out / "reliability_weights_patient_level.csv"
        rel.to_csv(rel_path, index=False, encoding="utf-8-sig")
        outputs.append(rel_path)

        summary = reliability_distribution_summary(rel)
        summary_path = out / "reliability_distribution_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        outputs.append(summary_path)

        corruption = reliability_corruption_response()
        corr_resp_path = out / "reliability_corruption_response.csv"
        corruption.to_csv(corr_resp_path, index=False, encoding="utf-8-sig")
        outputs.append(corr_resp_path)
        warns.append("NOT_EXECUTABLE: corruption-response reliability weights were not exported; table uses aggregate metric response where available.")

        quality = reliability_quality_corr(rel)
        quality_path = out / "reliability_quality_correlation.csv"
        quality.to_csv(quality_path, index=False, encoding="utf-8-sig")
        outputs.append(quality_path)

        examples = rel.sort_values(["prediction_correct", "reliability_entropy"], ascending=[False, True]).head(40)
        examples_path = out / "reliability_case_examples.csv"
        examples.to_csv(examples_path, index=False, encoding="utf-8-sig")
        outputs.append(examples_path)

        fig_reliability_distribution(rel, out / "figure_reliability_weight_distribution_by_modality")
        fig_reliability_by_center(rel, out / "figure_reliability_by_center")
        fig_reliability_corruption(corruption, out / "figure_reliability_corruption_response")
        fig_quality_scatter(rel, out / "figure_quality_vs_reliability_scatter")
        fig_entropy_error(rel, out / "figure_reliability_entropy_vs_error")
        fig_reliability_simplex(rel, out / "figure_reliability_simplex")
        for name in [
            "figure_reliability_weight_distribution_by_modality",
            "figure_reliability_by_center",
            "figure_reliability_corruption_response",
            "figure_quality_vs_reliability_scatter",
            "figure_reliability_entropy_vs_error",
            "figure_reliability_simplex",
        ]:
            outputs.extend(fig_pair_paths(out, name))

        report = [
            "# Reliability Validation Report",
            "",
            "Reliability weights are extracted from saved HyDRA patient-level outputs.",
            "",
            "- Image quality scores are unavailable; quality correlations are proxy-only.",
            "- Corruption-response reliability weights are unavailable; aggregate corruption metrics are reported but cannot prove modality-specific weight down-regulation.",
            "- Treat reliability weights as internal model diagnostics unless external quality/perturbation correlations support interpretation.",
        ]
        report_path = out / "RELIABILITY_VALIDATION_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        self.missing.append({"step": "P10", "item": "corruption-response reliability weights", "reason": "Inference-time reliability weights under image/clinical perturbations were not exported."})
        return outputs, warns

    # P11
    def p11_coe_faithfulness(self) -> tuple[list[Path], list[str]]:
        out = OUT / "10_coe_faithfulness"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        warns: list[str] = ["NOT_EXECUTABLE: true CoE intervention logits were not exported; proxy CoE delta summaries were generated."]
        hydra = read_csv(PATHS["hydra_predictions"], low_memory=False)
        coe = aggregate_coe_proxy(hydra)
        intervention = coe.copy()
        for col in ["intervened_pred_score", "intervened_z1_logit", "intervened_z2_logit", "intervened_z3_logit", "delta_pred_score", "delta_z1_logit", "delta_z2_logit", "delta_z3_logit"]:
            intervention[col] = np.nan
        intervention["intervention_type"] = "not_executed"
        intervention["intervention_detail"] = "No saved intervention inference outputs; original CoE proxy states only."
        inter_path = out / "coe_intervention_patient_level.csv"
        intervention.to_csv(inter_path, index=False, encoding="utf-8-sig")
        outputs.append(inter_path)

        summary = coe_proxy_summary(coe)
        summary_path = out / "coe_faithfulness_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        outputs.append(summary_path)
        random_ctrl = pd.DataFrame([{"control": "random_clinical_masking", "status": "NOT_EXECUTABLE", "reason": "No intervention logits exported."}])
        random_path = out / "coe_random_control_summary.csv"
        random_ctrl.to_csv(random_path, index=False, encoding="utf-8-sig")
        outputs.append(random_path)
        mono = pd.DataFrame([{"check": "high_risk_to_low_risk_prior_should_reduce_score", "status": "NOT_EXECUTABLE", "reason": "No counterfactual prior-swap inference output."}])
        mono_path = out / "coe_monotonicity_check.csv"
        mono.to_csv(mono_path, index=False, encoding="utf-8-sig")
        outputs.append(mono_path)

        fig_coe_placeholder(summary, out / "figure_coe_targeted_vs_random_masking", "Targeted vs random masking not executable")
        fig_coe_placeholder(summary, out / "figure_coe_counterfactual_prior_swap", "Counterfactual prior swap not executable")
        fig_coe_placeholder(summary, out / "figure_coe_visual_masking_response", "Visual masking response not executable")
        fig_coe_delta_heatmap(coe, out / "figure_coe_stepwise_delta_heatmap")
        for name in [
            "figure_coe_targeted_vs_random_masking",
            "figure_coe_counterfactual_prior_swap",
            "figure_coe_visual_masking_response",
            "figure_coe_stepwise_delta_heatmap",
        ]:
            outputs.extend(fig_pair_paths(out, name))

        report = [
            "# CoE Faithfulness Report",
            "",
            "Current saved outputs contain CoE templates and stepwise delta proxies, but not intervention logits or intervened latent states.",
            "",
            "Therefore CoE should be described as a `latent evidence summary` or `stepwise evidence trajectory`, not as a validated causal explanation.",
            "",
            "Faithfulness is NOT established until targeted interventions produce larger and clinically sensible score/state changes than random controls.",
        ]
        report_path = out / "COE_FAITHFULNESS_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        self.missing.append({"step": "P11", "item": "CoE targeted/random/counterfactual/visual interventions", "reason": "No intervention inference outputs or saliency masks are available."})
        return outputs, warns

    # P12
    def p12_subgroup_failure_audit(self) -> tuple[list[Path], list[str]]:
        out = OUT / "11_subgroup_and_failure_audit"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        test = self._get_test_agg()
        thresholds = self._get_thresholds()
        hydra = test[test["model_name"].eq(PRIMARY_MODEL)].copy()
        hydra = hydra.merge(thresholds[thresholds["model_name"].eq(PRIMARY_MODEL)][["fold_id", "threshold_cin2_f1_val"]], on="fold_id", how="left")
        hydra["threshold"] = hydra["threshold_cin2_f1_val"].fillna(0.5)
        lock = read_csv(PATHS["data_lock"], low_memory=False)
        merged = hydra.merge(lock.drop(columns=["patient_id"], errors="ignore"), on="case_id", how="left", suffixes=("", "_lock"))
        merged["age_bin"] = pd.cut(pd.to_numeric(merged["age"], errors="coerce"), bins=[0, 29, 39, 49, 59, 200], labels=["<30", "30-39", "40-49", "50-59", ">=60"])
        merged["histology_stratum"] = merged.get("pathology_grade_harmonized", pd.Series(index=merged.index, dtype=object)).fillna("unknown")
        merged["hpv_stratum"] = merged.get("hpv_status_harmonized", pd.Series(index=merged.index, dtype=object)).fillna("missing")
        merged["tct_stratum"] = merged.get("tct_status_harmonized", pd.Series(index=merged.index, dtype=object)).fillna("missing")

        strata = []
        for typ, col in [("histology", "histology_stratum"), ("hpv", "hpv_stratum"), ("tct", "tct_stratum"), ("age", "age_bin"), ("center", "center")]:
            for val_name, g in merged.groupby(col, dropna=False):
                m = binary_metrics(g["y_cin2"], g["score"], float(g["threshold"].median()))
                m3 = binary_metrics(g["y_cin3"], g["score"], float(g["threshold"].median()))
                row = {
                    "stratum_type": typ,
                    "stratum": str(val_name),
                    "n": int(len(g)),
                    "auroc": safe_auc(g["y_cin2"], g["score"]),
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "ppv": m["ppv"],
                    "npv": m["npv"],
                    "false_negatives": m["fn"],
                    "false_positives": m["fp"],
                    "referral_rate": m["referral_rate"],
                    "cin3_false_negatives": m3["fn"],
                    "notes": metric_notes(g["y_cin2"]),
                }
                strata.append(row)
        strat = pd.DataFrame(strata)
        strat_path = out / "stratified_performance.csv"
        strat.to_csv(strat_path, index=False, encoding="utf-8-sig")
        outputs.append(strat_path)

        failures = failure_mode_table(merged)
        fail_path = out / "failure_mode_patient_level.csv"
        failures.to_csv(fail_path, index=False, encoding="utf-8-sig")
        outputs.append(fail_path)
        fail_summary = failures.groupby(["error_type", "failure_mode"], dropna=False).size().reset_index(name="n")
        fail_summary_path = out / "failure_mode_summary.csv"
        fail_summary.to_csv(fail_summary_path, index=False, encoding="utf-8-sig")
        outputs.append(fail_summary_path)
        rep_path = out / "representative_failure_cases.csv"
        failures.sort_values("score", ascending=False).head(50).to_csv(rep_path, index=False, encoding="utf-8-sig")
        outputs.append(rep_path)

        fig_stratified(strat, out / "figure_stratified_sensitivity_npv")
        fig_histology_score(merged, out / "figure_histology_grade_score_distribution")
        fig_failure_counts(fail_summary, out / "figure_failure_mode_counts")
        fig_false_negative_center_grade(failures, out / "figure_false_negative_by_center_and_grade")
        for name in [
            "figure_stratified_sensitivity_npv",
            "figure_histology_grade_score_distribution",
            "figure_failure_mode_counts",
            "figure_false_negative_by_center_and_grade",
        ]:
            outputs.extend(fig_pair_paths(out, name))

        report = [
            "# Subgroup and Failure Audit Report",
            "",
            "All subgroup rows include `n`; undefined metrics are left blank/NaN for invalid one-class strata.",
            "",
            "Failure-mode categories are proxy-based because explicit annotations for transformation zone type, inflammation, visibility, OCT artifact, and image quality are not available.",
        ]
        report_path = out / "SUBGROUP_FAILURE_AUDIT_REPORT.md"
        write_text(report_path, "\n".join(report) + "\n")
        outputs.append(report_path)
        return outputs, []

    # P13
    def p13_final_tables_figures_claim_lock(self) -> tuple[list[Path], list[str]]:
        out = OUT / "12_final_tables_figures"
        audit = OUT / "13_submission_audit"
        out.mkdir(parents=True, exist_ok=True)
        audit.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []

        table_map = {
            "Table_1_protocol_and_cohort_lock.csv": OUT / "00_protocol_lock/loco_protocol_lock.csv",
            "Table_2_main_loco_performance_with_ci.csv": OUT / "03_patient_level_statistics/pooled_metrics_with_ci.csv",
            "Table_3_cin3_safety_by_center.csv": OUT / "02_cin3_safety/cin3_safety_by_center.csv",
            "Table_4_same_backbone_fusion_ablation.csv": OUT / "04_same_backbone_fusion_baselines/same_backbone_ablation_table.csv",
            "Table_5_modality_and_missingness.csv": OUT / "05_modality_ablation_and_missingness/modality_contribution_table.csv",
            "Table_6_calibration_ablation.csv": OUT / "06_calibration_ablation/calibration_ablation_pooled.csv",
            "Table_7_center_shift_and_calibration.csv": OUT / "07_center_shift/shift_performance_correlation.csv",
            "Table_8_clinical_utility_per_1000.csv": OUT / "08_clinical_utility/clinical_utility_per_1000.csv",
            "Table_9_reliability_validation.csv": OUT / "09_reliability_validation/reliability_distribution_summary.csv",
            "Table_10_coe_faithfulness_controls.csv": OUT / "10_coe_faithfulness/coe_faithfulness_summary.csv",
            "Table_11_subgroup_failure_audit.csv": OUT / "11_subgroup_and_failure_audit/failure_mode_summary.csv",
        }
        for dst, src in table_map.items():
            path = out / dst
            copy_if_exists(src, path)
            outputs.append(path)

        fig_copy = {
            "Figure_1_protocol_flow": OUT / "00_protocol_lock/figure_protocol_split_flow",
            "Figure_3_cin3_safety_and_referral_tradeoff": OUT / "08_clinical_utility/figure_locked_threshold_clinical_tradeoff",
            "Figure_4_same_backbone_ablation": OUT / "04_same_backbone_fusion_baselines/figure_same_backbone_auc_by_variant",
            "Figure_5_modality_missingness_stress": OUT / "05_modality_ablation_and_missingness/figure_complete_modality_removal_auc_npv",
            "Figure_7_reliability_weight_validation": OUT / "09_reliability_validation/figure_reliability_weight_distribution_by_modality",
            "Figure_8_coe_faithfulness_controls": OUT / "10_coe_faithfulness/figure_coe_stepwise_delta_heatmap",
            "Figure_9_failure_mode_audit": OUT / "11_subgroup_and_failure_audit/figure_failure_mode_counts",
        }
        for dst_stem, src_stem in fig_copy.items():
            for suffix in [".png", ".pdf"]:
                src = src_stem.with_suffix(suffix)
                dst = (out / dst_stem).with_suffix(suffix)
                copy_if_exists(src, dst)
                outputs.append(dst)
        fig_main_roc_dca_calibration(self._get_test_agg(), self._get_thresholds(), out / "Figure_2_main_roc_dca_calibration")
        fig_center_shift_combo(out / "Figure_6_center_shift_umap_mmd_coral")
        outputs.extend(fig_pair_paths(out, "Figure_2_main_roc_dca_calibration"))
        outputs.extend(fig_pair_paths(out, "Figure_6_center_shift_umap_mmd_coral"))

        map_rows = []
        for path in sorted(out.glob("Table_*.csv")) + sorted(out.glob("Figure_*.*")):
            map_rows.append({"artifact": path.name, "path": str_rel(path), "type": "table" if path.suffix == ".csv" else "figure"})
        map_path = out / "FIGURE_TABLE_MAP.csv"
        pd.DataFrame(map_rows).to_csv(map_path, index=False, encoding="utf-8-sig")
        outputs.append(map_path)

        summary_path = out / "RESULTS_SUMMARY_FOR_MANUSCRIPT.md"
        supplement_path = out / "SUPPLEMENTARY_RESULTS_SUMMARY.md"
        summary = make_results_summary()
        write_text(summary_path, summary)
        write_text(supplement_path, summary)
        outputs.extend([summary_path, supplement_path])

        claim_path = audit / "IF_FINAL_EXPERIMENT_CLAIM_LOCK.md"
        audit_path = audit / "IF_SUPPLEMENTARY_EXPERIMENT_AUDIT.md"
        write_text(claim_path, self.make_claim_lock())
        write_text(audit_path, self.make_audit_report())
        outputs.extend([claim_path, audit_path])
        return outputs, []

    # P14
    def p14_reproducibility_package(self) -> tuple[list[Path], list[str]]:
        out = OUT / "13_submission_audit"
        out.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        env_path = out / "ENVIRONMENT_SNAPSHOT.txt"
        write_text(env_path, environment_snapshot())
        outputs.append(env_path)
        self.write_runtime_manifest()
        runtime_path = out / "RUNTIME_MANIFEST.csv"
        outputs.append(runtime_path)

        package = OUT / "IF_SUPPLEMENTARY_EXPERIMENTS_PACKAGE.zip"
        make_zip_package(package)
        outputs.append(package)
        return outputs, []

    def _load_test_predictions(self) -> pd.DataFrame:
        if self.test_raw is None:
            self.test_raw = read_csv(PATHS["test_predictions"], low_memory=False)
        return self.test_raw.copy()

    def _load_validation_predictions(self) -> pd.DataFrame:
        if self.val_raw is None:
            self.val_raw = read_csv(PATHS["validation_predictions"], low_memory=False)
        return self.val_raw.copy()

    def _get_protocol(self) -> pd.DataFrame:
        if self.protocol is None:
            path = OUT / "00_protocol_lock/loco_protocol_lock.csv"
            self.protocol = read_csv(path)
        return self.protocol.copy()

    def _get_test_agg(self) -> pd.DataFrame:
        if self.test_agg is None:
            path = OUT / "01_prediction_registry/standardized_patient_mean_test_predictions.csv"
            self.test_agg = read_csv(path, low_memory=False) if path.exists() else aggregate_step2_predictions(self._load_test_predictions())
        return self.test_agg.copy()

    def _get_val_agg(self) -> pd.DataFrame:
        if self.val_agg is None:
            path = OUT / "01_prediction_registry/standardized_patient_mean_validation_predictions.csv"
            self.val_agg = read_csv(path, low_memory=False) if path.exists() else aggregate_step2_predictions(self._load_validation_predictions())
        return self.val_agg.copy()

    def _get_thresholds(self) -> pd.DataFrame:
        if self.thresholds is None:
            path = OUT / "02_cin3_safety/validation_locked_thresholds.csv"
            self.thresholds = read_csv(path) if path.exists() else pd.DataFrame()
        return self.thresholds.copy()

    def _register_shared_lora_predictions(self, std_dir: Path) -> tuple[list[dict[str, object]], list[Path]]:
        rows = []
        outputs = []
        root = PATHS["shared_lora_ablation_root"]
        if not root.exists():
            self.missing.append({"step": "P02", "item": "shared_lora ablation predictions", "reason": "Ablation root not found."})
            return rows, outputs
        for pred in sorted(root.glob("*/predictions/Improved1897_LOCO_All_Patient_Predictions.csv")):
            variant = pred.parents[1].name
            df = read_csv(pred, low_memory=False)
            std = standardize_shared_lora_predictions(df, pred, variant)
            path = std_dir / f"shared_lora_{safe_name(variant)}_test.csv"
            std.to_csv(path, index=False, encoding="utf-8-sig")
            outputs.append(path)
            rows.append(
                {
                    "model_name": f"SharedLoRA_{variant}",
                    "protocol": "strict_loco_shared_lora_supplement",
                    "fold_id": "all_loco_folds",
                    "center": "all_available_centers",
                    "seed": "2026",
                    "prediction_path": str_rel(path),
                    "n_patients": int(std["patient_id"].nunique()),
                    "has_cin2": std["y_cin2"].notna().any(),
                    "has_cin3": std["y_cin3"].notna().any(),
                    "has_logits": std["pred_logit"].notna().any(),
                    "has_reliability_weights": {"reliability_oct", "reliability_colposcopy", "reliability_clinical_text"}.issubset(set(df.columns)),
                    "has_latent_states": {"coe_delta_clinical", "coe_delta_colposcopy", "coe_delta_oct"}.issubset(set(df.columns)),
                    "usable_for_main_table": False,
                    "notes": f"Supplementary Shared-LoRA ablation from {str_rel(pred)}.",
                }
            )
        return rows, outputs

    def write_missing_requirements(self) -> None:
        out = OUT / "MISSING_REQUIREMENTS.md"
        lines = [
            "# Missing Requirements",
            "",
            "This file lists requested analyses that could not be executed from the currently available locked predictions, feature caches, or labels.",
            "",
        ]
        if not self.missing:
            lines.append("No missing requirements were recorded.")
        else:
            for item in self.missing:
                lines.append(f"- `{item.get('step')}` / `{item.get('item')}`: {item.get('reason')}")
        write_text(out, "\n".join(lines) + "\n")

    def write_runtime_manifest(self) -> None:
        out = OUT / "13_submission_audit"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in self.step_results:
            rows.append(
                {
                    "step_id": r.step_id,
                    "script": r.script,
                    "status": r.status,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "duration_seconds": r.duration_seconds,
                    "output_files": ";".join(r.output_files),
                    "warnings": " | ".join(r.warnings),
                    "errors": " | ".join(r.errors),
                }
            )
        pd.DataFrame(rows).to_csv(out / "RUNTIME_MANIFEST.csv", index=False, encoding="utf-8-sig")

    def make_claim_lock(self) -> str:
        lock = read_csv(PATHS["data_lock"])
        centers = sorted(lock["center_name"].dropna().unique())
        safety_path = OUT / "02_cin3_safety/cin3_safety_by_center.csv"
        pooled_path = OUT / "03_patient_level_statistics/pooled_metrics_with_ci.csv"
        safety = read_csv(safety_path) if safety_path.exists() else pd.DataFrame()
        pooled = read_csv(pooled_path) if pooled_path.exists() else pd.DataFrame()
        hydra_pooled = pooled[pooled["model_name"].eq(PRIMARY_MODEL)] if len(pooled) else pd.DataFrame()
        hydra_safety = safety[safety["model_name"].eq(PRIMARY_MODEL)] if len(safety) else pd.DataFrame()
        return "\n".join(
            [
                "# IF Final Experiment Claim Lock",
                "",
                f"1. Final analytic cohort size: `{len(lock)}`.",
                f"2. Final center list: {', '.join(CENTER_DISPLAY.get(c, c) for c in centers)}.",
                "3. Endpoint definitions: CIN2+ = `pathology_cin2plus`; CIN3+ = `pathology_cin3plus`; invasive-cancer-only endpoint unavailable.",
                "4. Primary evaluation protocol: strict five-fold LOCO.",
                "5. Secondary fixed external split: preserved only as supplementary if used.",
                "6. Target-label-free transductive calibration: allowed only when explicitly labelled as transductive.",
                f"7. Final main HyDRA result: {hydra_pooled.to_dict('records') if len(hydra_pooled) else 'not available'}.",
                f"8. Final best baseline result: {best_baseline_summary(pooled)}.",
                f"9. CIN3+ safety by center: {hydra_safety[['test_center','cin3_sensitivity_test','cin3_false_negatives_test']].to_dict('records') if len(hydra_safety) else 'not available'}.",
                f"10. 0.95 CIN3+ validation safety floor achieved: `{int(hydra_safety['safety_floor_met_on_validation'].sum()) if len(hydra_safety) else 0}/{len(hydra_safety)}` folds.",
                "11. Patient-level statistical superiority: use paired bootstrap table; do not claim superiority where CI crosses zero.",
                "12. Same-backbone fusion contribution: PARTIAL because exact ten-variant same-backbone controls are not all available.",
                "13. Reliability interpretability: PARTIAL/proxy-only; perturbation-specific weight exports missing.",
                "14. CoE faithfulness: NOT established; intervention controls are not executable from saved outputs.",
                "",
                "## Allowed Claims",
                "- Locked n=1897 multicenter LOCO benchmark results.",
                "- Validation-locked CIN3+ safety/referral trade-off.",
                "- Target-label-free median matching as transductive calibration when clearly labelled.",
                "- Reliability weights and CoE trajectories as internal diagnostics/transparency aids.",
                "",
                "## Claims To Remove Or Soften",
                "- Clinical deployment safety.",
                "- Source-only performance after target-distribution calibration.",
                "- Causal/faithful explanation claims for CoE.",
                "- Complete same-backbone superiority unless missing variants are trained.",
                "",
            ]
        )

    def make_audit_report(self) -> str:
        rows = []
        for r in self.step_results:
            rows.append({"step": r.step_id, "status": r.status, "warnings": " | ".join(r.warnings), "errors": " | ".join(r.errors)})
        df = pd.DataFrame(rows)
        return "# IF Supplementary Experiment Audit\n\n" + md_table(df) + "\n"


def make_dirs() -> None:
    for rel in [
        "00_protocol_lock",
        "01_prediction_registry/standardized_predictions",
        "02_cin3_safety",
        "03_patient_level_statistics",
        "04_same_backbone_fusion_baselines",
        "05_modality_ablation_and_missingness",
        "06_calibration_ablation",
        "07_center_shift",
        "08_clinical_utility",
        "09_reliability_validation",
        "10_coe_faithfulness",
        "11_subgroup_and_failure_audit",
        "12_final_tables_figures",
        "13_submission_audit",
        "logs",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def preferred_font_family() -> str:
    for family in ("Arial", "Times New Roman", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk", palette=PALETTE)
    plt.rcParams.update(
        {
            "font.family": preferred_font_family(),
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#30335f",
            "axes.titlecolor": "#30335f",
        }
    )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def str_rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_name(name: object) -> str:
    s = str(name)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render a compact markdown table without requiring tabulate."""
    if df is None or len(df) == 0:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = [str(c) for c in view.columns]
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        vals = []
        for c in view.columns:
            val = row[c]
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.4g}")
            else:
                vals.append(str(val).replace("\n", " "))
        rows.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        rows.append(f"| ... | showing first {max_rows} of {len(df)} rows |" + " |" * max(0, len(cols) - 2))
    return "\n".join(rows)


def hash_id(value: object) -> str:
    raw = f"{HASH_SALT}:{value}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def file_inventory_row(category: str, path: Path) -> dict[str, object]:
    exists = path.exists()
    row = {
        "category": category,
        "path": str_rel(path),
        "exists": exists,
        "row_count_or_file_count": np.nan,
        "key_columns": "",
        "last_modified": "",
        "notes": "",
    }
    if not exists:
        row["notes"] = "missing"
        return row
    try:
        row["last_modified"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if path.is_dir():
            row["row_count_or_file_count"] = sum(1 for _ in path.rglob("*"))
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5)
            row["key_columns"] = ";".join(df.columns[:20])
            try:
                row["row_count_or_file_count"] = sum(1 for _ in path.open("rb")) - 1
            except Exception:
                row["row_count_or_file_count"] = np.nan
        elif path.suffix.lower() in {".json", ".md", ".txt", ".py", ".npz"}:
            row["row_count_or_file_count"] = path.stat().st_size
    except Exception as exc:
        row["notes"] = f"audit_read_error: {exc}"
    return row


def summarize_centers(lock: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center, g in lock.groupby("center_name", dropna=False):
        rows.append(
            {
                "center": center,
                "center_display": CENTER_DISPLAY.get(center, center),
                "n": int(len(g)),
                "cin2_pos": int(pd.to_numeric(g["pathology_cin2plus"], errors="coerce").fillna(0).sum()),
                "cin3_pos": int(pd.to_numeric(g["pathology_cin3plus"], errors="coerce").fillna(0).sum()),
                "oct_available_n": int(pd.to_numeric(g.get("oct_available", 0), errors="coerce").fillna(0).sum()),
                "colposcopy_available_n": int(pd.to_numeric(g.get("colposcopy_available", 0), errors="coerce").fillna(0).sum()),
                "clinical_prior_available_n": int(pd.to_numeric(g.get("clinical_prior_available", 0), errors="coerce").fillna(0).sum()),
                "cin2_prevalence": safe_div(pd.to_numeric(g["pathology_cin2plus"], errors="coerce").fillna(0).sum(), len(g)),
                "cin3_prevalence": safe_div(pd.to_numeric(g["pathology_cin3plus"], errors="coerce").fillna(0).sum(), len(g)),
            }
        )
    return pd.DataFrame(rows)


def class_check(lock: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center, g in lock.groupby("center_name", dropna=False):
        cin2_unique = sorted(pd.to_numeric(g["pathology_cin2plus"], errors="coerce").dropna().unique().tolist())
        cin3_unique = sorted(pd.to_numeric(g["pathology_cin3plus"], errors="coerce").dropna().unique().tolist())
        rows.append(
            {
                "center": center,
                "n": len(g),
                "cin2_classes": ",".join(map(str, cin2_unique)),
                "cin3_classes": ",".join(map(str, cin3_unique)),
                "one_class_cin2": len(cin2_unique) < 2,
                "one_class_cin3": len(cin3_unique) < 2,
            }
        )
    return pd.DataFrame(rows)


def choose_validation_center(lock: pd.DataFrame, test_center: str) -> str:
    centers = sorted(c for c in lock["center_name"].dropna().unique() if c != test_center)
    if not centers:
        return ""
    rows = []
    for candidate in centers:
        train = lock[~lock["center_name"].isin([test_center, candidate])]
        val = lock[lock["center_name"].eq(candidate)]
        train_prev = pd.to_numeric(train["pathology_cin2plus"], errors="coerce").mean()
        val_prev = pd.to_numeric(val["pathology_cin2plus"], errors="coerce").mean()
        rows.append((abs(float(val_prev) - float(train_prev)), candidate))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return rows[0][1]


def fig_split_flow(protocol: pd.DataFrame, stem: Path) -> None:
    centers = sorted(set(protocol["held_out_test_center"]).union(set(protocol["validation_center"])))
    matrix = []
    for _, row in protocol.iterrows():
        train = set(str(row["training_centers"]).split(";"))
        vals = []
        for c in centers:
            if c == row["held_out_test_center"]:
                vals.append(2)
            elif c == row["validation_center"]:
                vals.append(1)
            elif c in train:
                vals.append(0)
            else:
                vals.append(np.nan)
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = LinearSegmentedColormap.from_list("split", ["#abb8cc", "#dbb98c", "#b57979"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([CENTER_DISPLAY.get(c, c) for c in centers], rotation=35, ha="right")
    ax.set_yticks(range(len(protocol)))
    ax.set_yticklabels([CENTER_DISPLAY.get(c, c) for c in protocol["held_out_test_center"]])
    ax.set_title("Strict Five-Fold LOCO Protocol")
    ax.set_xlabel("Center")
    ax.set_ylabel("Held-out fold")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Train", "Validation", "Test"])
    fig.tight_layout()
    save_figure(fig, stem)


def standardize_step2_predictions(df: pd.DataFrame, split_name: str, source: Path) -> pd.DataFrame:
    out = pd.DataFrame()
    out["patient_id"] = df["patient_id"].map(hash_id)
    out["case_id_hash"] = df["case_id"].map(hash_id)
    out["center"] = df["center_name"]
    out["fold_id"] = df["fold_id"]
    out["protocol"] = "strict_loco"
    out["split"] = split_name
    out["model_name"] = df["model_name"]
    out["y_cin2"] = pd.to_numeric(df["pathology_cin2plus"], errors="coerce")
    out["y_cin3"] = pd.to_numeric(df["pathology_cin3plus"], errors="coerce")
    out["y_invasive_cancer"] = np.nan
    out["pred_cin2_score"] = pd.to_numeric(df["prob_cin2plus"], errors="coerce")
    out["pred_logit"] = logit_np(out["pred_cin2_score"].to_numpy())
    out["threshold_cin2_locked"] = pd.to_numeric(df.get("threshold_youden", np.nan), errors="coerce")
    out["threshold_cin3_locked"] = pd.to_numeric(df.get("threshold_safety95", np.nan), errors="coerce")
    out["pred_cin2_binary"] = pd.to_numeric(df.get("pred_t_youden", np.nan), errors="coerce")
    out["pred_cin3_binary"] = pd.to_numeric(df.get("pred_t_safety95", np.nan), errors="coerce")
    out["source_prediction_file"] = str_rel(source)
    out["seed"] = df.get("seed", np.nan)
    return out


def standardize_shared_lora_predictions(df: pd.DataFrame, source: Path, variant: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["patient_id"] = df["case_id"].map(hash_id)
    out["case_id_hash"] = df["case_id"].map(hash_id)
    out["center"] = df.get("center_name", df.get("center", ""))
    out["fold_id"] = df.get("fold_id", "")
    out["protocol"] = "strict_loco_shared_lora_supplement"
    out["split"] = df.get("split", "external_test")
    out["model_name"] = f"SharedLoRA_{variant}"
    out["y_cin2"] = pd.to_numeric(df.get("y_true", np.nan), errors="coerce")
    out["y_cin3"] = pd.to_numeric(df.get("pathology_cin3plus", np.nan), errors="coerce")
    out["y_invasive_cancer"] = np.nan
    out["pred_cin2_score"] = pd.to_numeric(df.get("y_prob", np.nan), errors="coerce")
    out["pred_logit"] = logit_np(out["pred_cin2_score"].to_numpy())
    out["threshold_cin2_locked"] = pd.to_numeric(df.get("cin2_threshold_from_val", np.nan), errors="coerce")
    out["threshold_cin3_locked"] = pd.to_numeric(df.get("cin3_safety_threshold_from_val", np.nan), errors="coerce")
    out["pred_cin2_binary"] = (out["pred_cin2_score"] >= out["threshold_cin2_locked"]).astype(float)
    out["pred_cin3_binary"] = (out["pred_cin2_score"] >= out["threshold_cin3_locked"]).astype(float)
    out["source_prediction_file"] = str_rel(source)
    out["seed"] = df.get("seed", 2026)
    return out


def aggregate_step2_predictions(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "case_id",
        "patient_id",
        "center_name",
        "fold_id",
        "held_out_center",
        "model_name",
        "pathology_cin2plus",
        "pathology_cin3plus",
        "age",
        "hpv_status_harmonized",
        "hpv16_18_status",
        "tct_status_harmonized",
        "oct_available",
        "colposcopy_available",
        "clinical_prior_available",
    ]
    present = [c for c in base_cols if c in df.columns]
    agg = df.groupby(present, dropna=False).agg(
        score=("prob_cin2plus", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    rename = {
        "center_name": "center",
        "pathology_cin2plus": "y_cin2",
        "pathology_cin3plus": "y_cin3",
    }
    agg = agg.rename(columns=rename)
    agg["patient_id_hash"] = agg["patient_id"].map(hash_id)
    agg["case_id_hash"] = agg["case_id"].map(hash_id)
    for c in ["y_cin2", "y_cin3"]:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").astype("Int64")
    return agg


def first_value(s: pd.Series) -> object:
    return s.dropna().iloc[0] if s.dropna().size else np.nan


def logit_np(p: Iterable[float]) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    return np.log(arr / (1 - arr))


def sigmoid(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else np.nan


def safe_auc(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def safe_auprc(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return np.nan


def safe_brier(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = np.clip(score[mask], 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return np.nan
    try:
        return float(brier_score_loss(y, score))
    except Exception:
        return np.nan


def expected_calibration_error(y: Iterable[int], score: Iterable[float], n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = np.clip(score[mask], 0, 1)
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if not idx.any():
            continue
        ece += idx.mean() * abs(y[idx].mean() - score[idx].mean())
    return float(ece)


def binary_metrics(y: Iterable[int], score: Iterable[float], threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask].astype(int)
    score = score[mask]
    if len(y) == 0:
        return {k: np.nan for k in ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "referral_rate", "tp", "fp", "tn", "fn"]}
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_values(y, pred)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn) if (tp + fn) else np.nan,
        "specificity": safe_div(tn, tn + fp) if (tn + fp) else np.nan,
        "ppv": safe_div(tp, tp + fp) if (tp + fp) else np.nan,
        "npv": safe_div(tn, tn + fn) if (tn + fn) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "referral_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def confusion_values(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def select_threshold_f1(y: Iterable[int], score: Iterable[float]) -> tuple[float, bool]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask].astype(int)
    score = score[mask]
    if len(y) == 0:
        return 0.5, False
    stats = threshold_stats(y, score)
    f1 = 2 * stats["tp"] / np.maximum(2 * stats["tp"] + stats["fp"] + stats["fn"], 1)
    idx = int(np.nanargmax(f1))
    return float(stats["threshold"][idx]), True


def select_threshold_safety(y: Iterable[int], score: Iterable[float], sensitivity_floor: float = 0.95) -> tuple[float, bool]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask].astype(int)
    score = score[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5, False
    stats = threshold_stats(y, score)
    sens = stats["tp"] / np.maximum(stats["tp"] + stats["fn"], 1)
    spec = stats["tn"] / np.maximum(stats["tn"] + stats["fp"], 1)
    referral = (stats["tp"] + stats["fp"]) / len(y)
    valid = np.where(sens >= sensitivity_floor)[0]
    if len(valid):
        order = np.lexsort((referral[valid], -spec[valid]))
        return float(stats["threshold"][valid[order[0]]]), True
    order = np.lexsort((referral, -spec, -sens))
    return float(stats["threshold"][order[0]]), False


def threshold_stats(y: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-score)
    s = score[order]
    yy = y[order].astype(int)
    change = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(yy)[change].astype(float)
    fp = np.cumsum(1 - yy)[change].astype(float)
    pos = float(yy.sum())
    neg = float(len(yy) - yy.sum())
    fn = pos - tp
    tn = neg - fp
    return {"threshold": s[change], "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metric_notes(y: Iterable[int]) -> str:
    vals = pd.Series(y).dropna().unique()
    if len(vals) < 2:
        return "One-class group; AUROC/AUPRC or class-conditional metrics may be undefined."
    return ""


def metrics_for_ci(g: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = g["y_cin2"].astype(float).to_numpy()
    score = g["score"].astype(float).to_numpy()
    return metrics_for_ci_arrays(y, score, threshold)


def bootstrap_metric_cis(g: pd.DataFrame, threshold: float, metrics: list[str], n_boot: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    y = g["y_cin2"].astype(int).to_numpy()
    score = g["score"].astype(float).to_numpy()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        if len(pos) > 0 and len(neg) > 0:
            idx = np.r_[rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
        else:
            idx = rng.choice(np.arange(len(g)), size=len(g), replace=True)
        met = metrics_for_ci_arrays(y[idx], score[idx], threshold)
        for m in metrics:
            if np.isfinite(met[m]):
                vals[m].append(met[m])
    return {m: ci(vals[m]) for m in metrics}


def metrics_for_ci_arrays(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask].astype(int)
    score = np.clip(score[mask], 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {k: np.nan for k in ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "brier", "ece", "referral_rate"]}
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_values(y, pred)
    sens = safe_div(tp, tp + fn) if (tp + fn) else np.nan
    spec = safe_div(tn, tn + fp) if (tn + fp) else np.nan
    ppv = safe_div(tp, tp + fp) if (tp + fp) else np.nan
    npv = safe_div(tn, tn + fn) if (tn + fn) else np.nan
    f1 = safe_div(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "auroc": fast_auc_score(y, score),
        "auprc": fast_average_precision(y, score),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": np.nanmean([sens, spec]) if np.isfinite(sens) or np.isfinite(spec) else np.nan,
        "brier": float(np.mean((score - y) ** 2)),
        "ece": expected_calibration_error(y, score),
        "referral_rate": float(pred.mean()),
    }


def fast_auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    denom = np.arange(1, len(y_sorted) + 1)
    precision = tp / denom
    return float((precision * y_sorted).sum() / n_pos)


def ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return np.nan, np.nan
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def flatten_ci(base: dict[str, float], cis: dict[str, tuple[float, float]]) -> dict[str, float]:
    out = {}
    for k, v in base.items():
        lo, hi = cis.get(k, (np.nan, np.nan))
        out[k] = v
        out[f"{k}_ci_low"] = lo
        out[f"{k}_ci_high"] = hi
    return out


def paired_auc_differences(test: pd.DataFrame, primary: str, n_boot: int) -> pd.DataFrame:
    hydra = test[test["model_name"].eq(primary)][["case_id", "y_cin2", "score"]].rename(columns={"score": "score_hydra"})
    rows = []
    rng = np.random.default_rng(RANDOM_SEED + 10)
    for model, g in test[~test["model_name"].eq(primary)].groupby("model_name"):
        comp = hydra.merge(g[["case_id", "score"]].rename(columns={"score": "score_baseline"}), on="case_id", how="inner")
        if len(comp) == 0:
            continue
        y = comp["y_cin2"].astype(int).to_numpy()
        diff = safe_auc(y, comp["score_hydra"]) - safe_auc(y, comp["score_baseline"])
        boot = []
        idx_all = np.arange(len(comp))
        pos = idx_all[y == 1]
        neg = idx_all[y == 0]
        for _ in range(n_boot):
            idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)] if len(pos) and len(neg) else rng.choice(idx_all, len(idx_all), replace=True)
            yy = y[idx]
            d = safe_auc(yy, comp["score_hydra"].to_numpy()[idx]) - safe_auc(yy, comp["score_baseline"].to_numpy()[idx])
            if np.isfinite(d):
                boot.append(d)
        lo, hi = ci(boot)
        p = 2 * min(np.mean(np.asarray(boot) <= 0), np.mean(np.asarray(boot) >= 0)) if boot else np.nan
        rows.append({"baseline_model": model, "comparison": f"{primary} minus {model}", "auc_difference": diff, "ci_low": lo, "ci_high": hi, "paired_bootstrap_p": p, "n": len(comp)})
    return pd.DataFrame(rows)


def load_existing_pooled_bootstrap_ci(test: pd.DataFrame) -> pd.DataFrame:
    src = ROOT / "outputs/publishable_v2/step2_main_loco/statistics/bootstrap_ci_all_metrics.csv"
    if not src.exists():
        rows = []
        for model, g in test.groupby("model_name", dropna=False):
            rows.append({"model_name": model, "protocol": "strict_loco", "n": int(len(g)), "aggregation": "pooled_micro", **metrics_for_ci(g, float(g["threshold"].median() if "threshold" in g else 0.5)), "ci_source": "point_estimate_only_fallback"})
        return pd.DataFrame(rows)
    old = read_csv(src)
    rows = []
    for _, r in old.iterrows():
        g = test[test["model_name"].eq(r["Method"])]
        point_extra = metrics_for_ci(g, float(g["threshold"].median())) if len(g) and "threshold" in g else {}
        rows.append(
            {
                "model_name": r["Method"],
                "protocol": "strict_loco",
                "n": int(len(g)) if len(g) else np.nan,
                "aggregation": "pooled_micro",
                "operating_point": r.get("Operating point", "t_safety95"),
                "auroc": r.get("auc", np.nan),
                "auroc_ci_low": r.get("auc_ci_low", np.nan),
                "auroc_ci_high": r.get("auc_ci_high", np.nan),
                "auprc": r.get("average_precision", np.nan),
                "auprc_ci_low": r.get("average_precision_ci_low", np.nan),
                "auprc_ci_high": r.get("average_precision_ci_high", np.nan),
                "sensitivity": r.get("sensitivity", np.nan),
                "sensitivity_ci_low": r.get("sensitivity_ci_low", np.nan),
                "sensitivity_ci_high": r.get("sensitivity_ci_high", np.nan),
                "specificity": r.get("specificity", np.nan),
                "specificity_ci_low": r.get("specificity_ci_low", np.nan),
                "specificity_ci_high": r.get("specificity_ci_high", np.nan),
                "ppv": r.get("ppv", np.nan),
                "ppv_ci_low": r.get("ppv_ci_low", np.nan),
                "ppv_ci_high": r.get("ppv_ci_high", np.nan),
                "npv": r.get("npv", np.nan),
                "npv_ci_low": r.get("npv_ci_low", np.nan),
                "npv_ci_high": r.get("npv_ci_high", np.nan),
                "f1": r.get("f1", np.nan),
                "f1_ci_low": r.get("f1_ci_low", np.nan),
                "f1_ci_high": r.get("f1_ci_high", np.nan),
                "referral_rate": r.get("screen_positive_rate", np.nan),
                "referral_rate_ci_low": r.get("screen_positive_rate_ci_low", np.nan),
                "referral_rate_ci_high": r.get("screen_positive_rate_ci_high", np.nan),
                "balanced_accuracy": point_extra.get("balanced_accuracy", np.nan),
                "brier": point_extra.get("brier", np.nan),
                "ece": point_extra.get("ece", np.nan),
                "ci_source": str_rel(src),
            }
        )
    return pd.DataFrame(rows)


def load_existing_paired_tests() -> pd.DataFrame:
    src = ROOT / "outputs/publishable_v2/step2_main_loco/statistics/paired_tests_vs_hydra.csv"
    if not src.exists():
        return pd.DataFrame([{"status": "NOT_EXECUTABLE", "reason": "Existing paired bootstrap file not found."}])
    old = read_csv(src)
    out = old.rename(
        columns={
            "Baseline method": "baseline_model",
            "Metric difference": "auc_difference",
            "95% CI for difference": "auc_difference_ci",
            "Raw P": "paired_bootstrap_p_raw",
            "Holm-adjusted P": "paired_bootstrap_p_holm",
        }
    )
    ci_vals = out["auc_difference_ci"].map(parse_ci_string) if "auc_difference_ci" in out else pd.Series([(np.nan, np.nan)] * len(out))
    out["ci_low"] = [x[0] for x in ci_vals]
    out["ci_high"] = [x[1] for x in ci_vals]
    out["paired_bootstrap_p"] = out.get("paired_bootstrap_p_holm", out.get("paired_bootstrap_p_raw", np.nan))
    out["comparison"] = PRIMARY_MODEL + " minus " + out["baseline_model"].astype(str)
    out["source"] = str_rel(src)
    return out


def parse_ci_string(value: object) -> tuple[float, float]:
    import re

    s = str(value).strip()
    if "--" in s:
        left, right = s.split("--", 1)
        try:
            return float(left), -float(right)
        except ValueError:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        lo, hi = float(nums[0]), float(nums[1])
        if lo >= 0 and hi < 0 and "-" in s:
            hi = abs(hi)
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    return np.nan, np.nan


MAIN_TO_SAME_BACKBONE = {
    "ClinicalOnly_Logistic": "same_backbone_clinical_only",
    "ClinicalOnly_XGBoost": "same_backbone_clinical_only",
    "ColposcopyOnly_ViT": "same_backbone_colpo_only",
    "OCTOnly_ViT": "same_backbone_oct_only",
    "ColposcopyOCT_EarlyConcat": "same_backbone_early_concat",
    "ColposcopyOCT_LateFusion": "same_backbone_late_score_fusion",
    "ColposcopyOCTText_CrossAttention": "same_backbone_cross_attention",
    "HyDRA_CoE_Full": "same_backbone_hydra_full",
}

MODALITY_VARIANT = {
    "ClinicalOnly_Logistic": "clinical_only",
    "ClinicalOnly_XGBoost": "clinical_only",
    "ColposcopyOnly_ViT": "colposcopy_only",
    "OCTOnly_ViT": "oct_only",
    "ColposcopyOCT_EarlyConcat": "oct_colposcopy",
    "ColposcopyOCT_LateFusion": "oct_colposcopy",
    "ColposcopyOCTText_CrossAttention": "oct_colposcopy_clinical",
    "HyDRA_CoE_Full": "hydra_full",
    "BioMedCLIP_Finetuned": "image_text_feature_cache",
}

SHARED_VARIANT_MAP = {
    "g1": "same_backbone_hydra_full",
    "best": "same_backbone_hydra_full",
    "no_coe": "same_backbone_reliability_without_coe",
    "no_variational": "same_backbone_coe_without_reliability",
    "no_center_reliability": "same_backbone_reliability_without_coe",
    "no_cross_attn": "same_backbone_gated_fusion",
}


def compute_group_metrics(df: pd.DataFrame, groups: list[str], threshold_col: str) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        th = float(g[threshold_col].median()) if threshold_col in g and g[threshold_col].notna().any() else 0.5
        row.update({"n": len(g), "auroc": safe_auc(g["y_cin2"], g["score"]), "auprc": safe_auprc(g["y_cin2"], g["score"]), "brier": safe_brier(g["y_cin2"], g["score"]), "ece": expected_calibration_error(g["y_cin2"], g["score"])})
        row.update(binary_metrics(g["y_cin2"], g["score"], th))
        rows.append(row)
    return pd.DataFrame(rows)


def load_shared_lora_ablation_metrics() -> pd.DataFrame:
    root = PATHS["shared_lora_ablation_root"]
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for pred in sorted(root.glob("*/predictions/Improved1897_LOCO_All_Patient_Predictions.csv")):
        variant_dir = pred.parents[1].name
        mapped = SHARED_VARIANT_MAP.get(variant_dir, f"shared_lora_{variant_dir}")
        df = read_csv(pred, low_memory=False)
        df = df[df.get("split", "").eq("external_test")] if "split" in df else df
        for center, g in df.groupby("center_name" if "center_name" in df else "center"):
            th = pd.to_numeric(g.get("cin2_threshold_from_val", 0.5), errors="coerce").median()
            m = binary_metrics(g["y_true"], g["y_prob"], float(th) if np.isfinite(th) else 0.5)
            rows.append({"variant": mapped, "source_variant": variant_dir, "center": center, "n": len(g), "auroc": safe_auc(g["y_true"], g["y_prob"]), "auprc": safe_auprc(g["y_true"], g["y_prob"]), **m})
    return pd.DataFrame(rows)


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    logits = logit_np(p)
    mask = np.isfinite(y) & np.isfinite(logits)
    y = y[mask]
    logits = logits[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 1.0
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.25, 5.0, 96):
        prob = sigmoid(logits / t)
        loss = log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])
        if loss < best_loss:
            best_t, best_loss = float(t), float(loss)
    return best_t


def load_feature_representations(lock: pd.DataFrame) -> dict[str, dict[str, object]]:
    reps: dict[str, dict[str, object]] = {}
    p = PATHS["feature_npz"]
    if p.exists():
        z = np.load(p, allow_pickle=True)
        cases = pd.Series(z["case_id"].astype(str), name="case_id")
        meta = pd.DataFrame({"case_id": cases}).merge(lock, on="case_id", how="left")
        for key, name in [("oct", "oct_feature_cache"), ("col", "colposcopy_feature_cache"), ("clinical", "clinical_feature_cache")]:
            if key in z.files:
                reps[name] = {"X": np.asarray(z[key], dtype=float), "center": meta["center_name"].fillna("unknown"), "label": meta["pathology_cin2plus"].fillna(0)}
        if {"oct", "col"}.issubset(set(z.files)):
            reps["visual_before_fusion_proxy"] = {"X": np.c_[z["oct"], z["col"]], "center": meta["center_name"].fillna("unknown"), "label": meta["pathology_cin2plus"].fillna(0)}
        if {"oct", "col", "clinical"}.issubset(set(z.files)):
            reps["feature_concat_proxy"] = {"X": np.c_[z["oct"], z["col"], z["clinical"]], "center": meta["center_name"].fillna("unknown"), "label": meta["pathology_cin2plus"].fillna(0)}
    return reps


def center_classifier_cv(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    Xr = reduce_features(X, 64)
    min_count = pd.Series(y).value_counts().min()
    n_splits = int(max(2, min(5, min_count)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"))
    pred = cross_val_predict(clf, Xr, y, cv=cv)
    return pred, float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))


def reduce_features(X: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X)
    if X.shape[1] <= n_components:
        return X
    return PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(StandardScaler().fit_transform(X))


def pairwise_distance_table(reps: dict[str, dict[str, object]], metric: str) -> pd.DataFrame:
    rows = []
    for rep, data in reps.items():
        X = reduce_features(data["X"], 32)
        centers = pd.Series(data["center"]).astype(str).to_numpy()
        for a in sorted(np.unique(centers)):
            for b in sorted(np.unique(centers)):
                xa = X[centers == a]
                xb = X[centers == b]
                if metric == "mmd":
                    dist = mmd_rbf(xa, xb)
                else:
                    dist = coral_distance(xa, xb)
                rows.append({"representation": rep, "center_a": a, "center_b": b, f"{metric}_distance": dist})
    return pd.DataFrame(rows)


def mmd_rbf(xa: np.ndarray, xb: np.ndarray, max_n: int = 250) -> float:
    rng = np.random.default_rng(RANDOM_SEED)
    if len(xa) > max_n:
        xa = xa[rng.choice(len(xa), max_n, replace=False)]
    if len(xb) > max_n:
        xb = xb[rng.choice(len(xb), max_n, replace=False)]
    x = np.vstack([xa, xb])
    if len(x) < 2:
        return np.nan
    d2 = pairwise_sq_dists(x, x)
    sigma2 = np.nanmedian(d2[d2 > 0])
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1.0
    kxx = np.exp(-pairwise_sq_dists(xa, xa) / (2 * sigma2)).mean()
    kyy = np.exp(-pairwise_sq_dists(xb, xb) / (2 * sigma2)).mean()
    kxy = np.exp(-pairwise_sq_dists(xa, xb) / (2 * sigma2)).mean()
    return float(kxx + kyy - 2 * kxy)


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)


def coral_distance(xa: np.ndarray, xb: np.ndarray) -> float:
    if len(xa) < 2 or len(xb) < 2:
        return np.nan
    ca = np.cov(xa, rowvar=False)
    cb = np.cov(xb, rowvar=False)
    return float(np.linalg.norm(ca - cb, ord="fro") / (4 * xa.shape[1] * xa.shape[1]))


def shift_performance_correlation(mmd: pd.DataFrame, coral: pd.DataFrame, test: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    dist = mmd[mmd["representation"].eq("feature_concat_proxy")].copy()
    if len(dist) == 0:
        dist = mmd.copy()
    dist = dist[dist["center_a"].ne(dist["center_b"])].groupby("center_a").agg(mean_mmd=("mmd_distance", "mean")).reset_index().rename(columns={"center_a": "center"})
    cdist = coral[coral["representation"].eq("feature_concat_proxy")]
    cdist = cdist[cdist["center_a"].ne(cdist["center_b"])].groupby("center_a").agg(mean_coral=("coral_distance", "mean")).reset_index().rename(columns={"center_a": "center"})
    hydra = test[test["model_name"].eq(PRIMARY_MODEL)].merge(thresholds[thresholds["model_name"].eq(PRIMARY_MODEL)][["fold_id", "threshold_cin2_f1_val"]], on="fold_id", how="left")
    rows = []
    for center, g in hydra.groupby("center"):
        th = float(g["threshold_cin2_f1_val"].median()) if g["threshold_cin2_f1_val"].notna().any() else 0.5
        m = binary_metrics(g["y_cin2"], g["score"], th)
        m3 = binary_metrics(g["y_cin3"], g["score"], th)
        rows.append({"center": center, "auroc": safe_auc(g["y_cin2"], g["score"]), "ece": expected_calibration_error(g["y_cin2"], g["score"]), "cin3_false_negatives": m3["fn"], "referral_rate": m["referral_rate"]})
    perf = pd.DataFrame(rows)
    best_auc = np.nanmax(perf["auroc"]) if len(perf) else np.nan
    perf["auc_drop_vs_best"] = best_auc - perf["auroc"]
    out = perf.merge(dist, on="center", how="left").merge(cdist, on="center", how="left")
    out["mean_shift_distance"] = out[["mean_mmd", "mean_coral"]].mean(axis=1)
    return out


def decision_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(0.05, 0.8, 31)
    for model, g in df.groupby("model_name"):
        y = g["y_cin2"].astype(int).to_numpy()
        score = g["score"].to_numpy()
        n = len(g)
        prevalence = y.mean()
        for pt in thresholds:
            pred = score >= pt
            tn, fp, fn, tp = confusion_values(y, pred.astype(int))
            nb = tp / n - fp / n * (pt / (1 - pt))
            treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
            rows.append({"model_name": model, "threshold_probability": pt, "net_benefit": nb, "treat_all_net_benefit": treat_all, "treat_none_net_benefit": 0.0})
    return pd.DataFrame(rows)


def per_1000(count: float, n: int) -> float:
    return float(count) / n * 1000 if n else np.nan


def aggregate_hydra_reliability(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "patient_id",
        "center_name",
        "fold_id",
        "pathology_cin2plus",
        "pathology_cin3plus",
        "hpv_status_harmonized",
        "tct_status_harmonized",
    ]
    agg = df.groupby(cols, dropna=False).agg(
        score=("prob_cin2plus", "mean"),
        alpha_colposcopy=("alpha_colposcopy", "mean"),
        alpha_oct=("alpha_oct", "mean"),
        alpha_clinical=("alpha_semantic", "mean"),
        logvar_colposcopy=("uncertainty_colposcopy", "mean"),
        logvar_oct=("uncertainty_oct", "mean"),
        logvar_clinical=("uncertainty_semantic", "mean"),
    ).reset_index()
    agg = agg.rename(columns={"center_name": "center", "pathology_cin2plus": "y_cin2", "pathology_cin3plus": "y_cin3"})
    agg["patient_id"] = agg["patient_id"].map(hash_id)
    agg["case_id_hash"] = agg["case_id"].map(hash_id)
    agg = agg.drop(columns=["case_id"])
    return agg


def reliability_entropy(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum(axis=1, keepdims=True)
    return -(w * np.log(w)).sum(axis=1) / np.log(w.shape[1])


def reliability_distribution_summary(rel: pd.DataFrame) -> pd.DataFrame:
    long = rel.melt(
        id_vars=["center", "y_cin2", "prediction_correct", "error_type"],
        value_vars=["alpha_clinical", "alpha_colposcopy", "alpha_oct"],
        var_name="modality",
        value_name="weight",
    )
    return long.groupby(["modality", "center", "y_cin2", "prediction_correct"], dropna=False).agg(n=("weight", "size"), mean_weight=("weight", "mean"), sd_weight=("weight", "std"), median_weight=("weight", "median")).reset_index()


def reliability_corruption_response() -> pd.DataFrame:
    if not PATHS["input_corruption"].exists():
        return pd.DataFrame([{"status": "NOT_EXECUTABLE", "reason": "Input corruption metrics not found."}])
    df = read_csv(PATHS["input_corruption"])
    return df.assign(status="METRIC_ONLY_NO_RELIABILITY_WEIGHT_EXPORT", source=str_rel(PATHS["input_corruption"]))


def reliability_quality_corr(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flag, weight in [("hpv_missing", "alpha_clinical"), ("tct_missing", "alpha_clinical")]:
        x = rel[flag].astype(float)
        y = rel[weight].astype(float)
        rows.append({"proxy_quality_variable": flag, "reliability_weight": weight, "correlation": x.corr(y), "status": "proxy_only"})
    for unc, weight in [("logvar_oct", "alpha_oct"), ("logvar_colposcopy", "alpha_colposcopy"), ("logvar_clinical", "alpha_clinical")]:
        rows.append({"proxy_quality_variable": unc, "reliability_weight": weight, "correlation": rel[unc].corr(rel[weight]), "status": "proxy_uncertainty"})
    rows.append({"proxy_quality_variable": "reliability_entropy", "reliability_weight": "prediction_error", "correlation": rel["reliability_entropy"].corr((~rel["prediction_correct"]).astype(float)), "status": "internal_diagnostic"})
    return pd.DataFrame(rows)


def aggregate_coe_proxy(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["case_id", "patient_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]
    agg = df.groupby(cols, dropna=False).agg(
        original_pred_score=("prob_cin2plus", "mean"),
        original_z1_logit=("delta_prior_to_semantic", "mean"),
        original_z2_logit=("delta_semantic_to_colposcopy", "mean"),
        original_z3_logit=("delta_colposcopy_to_oct", "mean"),
    ).reset_index()
    agg = agg.rename(columns={"center_name": "center", "pathology_cin2plus": "y_cin2", "pathology_cin3plus": "y_cin3"})
    agg["patient_id"] = agg["patient_id"].map(hash_id)
    agg["case_id_hash"] = agg["case_id"].map(hash_id)
    return agg.drop(columns=["case_id"])


def coe_proxy_summary(coe: pd.DataFrame) -> pd.DataFrame:
    cols = ["original_z1_logit", "original_z2_logit", "original_z3_logit"]
    return coe.groupby("center", dropna=False)[cols].agg(["mean", "std", "median"]).reset_index()


def failure_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    pred = df["score"] >= df["threshold"]
    y = df["y_cin2"].astype(int)
    out = df.copy()
    out["pred_binary"] = pred.astype(int)
    out["error_type"] = np.select(
        [pred & y.eq(0), (~pred) & y.eq(1)],
        ["false_positive", "false_negative"],
        default="not_error",
    )
    hpv = out.get("hpv_status_harmonized", pd.Series(index=out.index, dtype=object)).astype(str).str.lower()
    tct = out.get("tct_status_harmonized", pd.Series(index=out.index, dtype=object)).astype(str).str.lower()
    high_prior = hpv.str.contains("16|18|positive|阳", regex=True, na=False) | tct.str.contains("hsil|asc-h|lsil|高级|阳", regex=True, na=False)
    out["failure_mode"] = "not_error"
    out.loc[out["error_type"].eq("false_positive") & high_prior, "failure_mode"] = "high-risk clinical prior but benign histology"
    out.loc[out["error_type"].eq("false_positive") & ~high_prior, "failure_mode"] = "unknown false-positive proxy"
    out.loc[out["error_type"].eq("false_negative") & ~high_prior, "failure_mode"] = "missing or low-risk clinical prior"
    out.loc[out["error_type"].eq("false_negative") & high_prior, "failure_mode"] = "subtle/high-risk case missed despite clinical prior"
    keep = ["patient_id_hash", "case_id_hash", "center", "histology_stratum", "y_cin2", "y_cin3", "score", "pred_binary", "error_type", "failure_mode"]
    return out[keep].copy()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_pair_paths(out: Path, stem: str) -> list[Path]:
    return [(out / stem).with_suffix(".png"), (out / stem).with_suffix(".pdf")]


def model_label(name: str) -> str:
    return MODEL_DISPLAY.get(name, str(name).replace("_", " "))


def center_label(name: str) -> str:
    return CENTER_DISPLAY.get(name, str(name))


def fig_cin3_false_negatives(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention", "OCTOnly_ViT"])].copy()
    sub["model"] = sub["model_name"].map(model_label)
    sub["center_display"] = sub["test_center"].map(center_label)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if sns:
        sns.barplot(data=sub, x="center_display", y="cin3_false_negatives_test", hue="model", ax=ax, palette=PALETTE)
    ax.set_title("CIN3+ False Negatives by Held-out Center")
    ax.set_xlabel("Held-out center")
    ax.set_ylabel("False negatives")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="", fontsize=9)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_cin3_tradeoff(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sub = df.copy()
    sub["model"] = sub["model_name"].map(model_label)
    if sns:
        sns.scatterplot(data=sub, x="cin3_specificity_test", y="cin3_sensitivity_test", hue="model", size="referral_rate_test", sizes=(40, 240), alpha=0.8, ax=ax, palette=PALETTE)
    ax.axhline(0.95, color="#b57979", linestyle="--", linewidth=2)
    ax.set_title("CIN3+ Sensitivity-Specificity Trade-off")
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_locked_thresholds(df: pd.DataFrame, stem: Path) -> None:
    plot = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention"])].copy()
    plot["model"] = plot["model_name"].map(model_label)
    plot["fold_display"] = plot["fold_id"].str.replace("loco_", "", regex=False).map(center_label)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if sns:
        sns.lineplot(data=plot, x="fold_display", y="threshold_cin3_safety_val", hue="model", marker="o", linewidth=2.5, ax=ax, palette=PALETTE)
    ax.set_title("Validation-Locked Safety Thresholds")
    ax.set_xlabel("Held-out fold")
    ax.set_ylabel("CIN3+ safety threshold")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="", fontsize=9)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_auc_ci_forest(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention", "OCTOnly_ViT"])].copy()
    sub["label"] = sub["model_name"].map(model_label) + " / " + sub["center"].map(center_label)
    sub = sub.sort_values("auroc")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(sub))))
    y = np.arange(len(sub))
    ax.errorbar(sub["auroc"], y, xerr=[sub["auroc"] - sub["auroc_ci_low"], sub["auroc_ci_high"] - sub["auroc"]], fmt="o", color="#8b98b3", ecolor="#b57979", elinewidth=1.5, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.set_xlabel("AUROC with 95% patient bootstrap CI")
    ax.set_title("Center-wise AUROC Confidence Intervals")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_sens_spec_ci(df: pd.DataFrame, stem: Path) -> None:
    sub = df.copy()
    sub["model"] = sub["model_name"].map(model_label)
    sub = sub.sort_values("auroc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sub))
    ax.errorbar(x - 0.08, sub["sensitivity"], yerr=[sub["sensitivity"] - sub["sensitivity_ci_low"], sub["sensitivity_ci_high"] - sub["sensitivity"]], fmt="o", label="Sensitivity", color="#8b98b3")
    ax.errorbar(x + 0.08, sub["specificity"], yerr=[sub["specificity"] - sub["specificity_ci_low"], sub["specificity_ci_high"] - sub["specificity"]], fmt="s", label="Specificity", color="#b57979")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"], rotation=35, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Pooled Sensitivity and Specificity CIs")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, stem)


def fig_paired_auc_diff(df: pd.DataFrame, stem: Path) -> None:
    sub = df.copy().sort_values("auc_difference")
    sub["baseline"] = sub["baseline_model"].map(model_label)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(sub))))
    y = np.arange(len(sub))
    ax.axvline(0, color="#333333", linewidth=1)
    diff = sub["auc_difference"].to_numpy(dtype=float)
    left = np.maximum(diff - sub["ci_low"].to_numpy(dtype=float), 0)
    right = np.maximum(sub["ci_high"].to_numpy(dtype=float) - diff, 0)
    ax.errorbar(sub["auc_difference"], y, xerr=[left, right], fmt="o", color="#dbb98c", ecolor="#8b98b3", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["baseline"])
    ax.set_xlabel("AUROC difference vs HyDRA")
    ax.set_title("Paired Patient-Level AUROC Difference")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_same_backbone_auc(df: pd.DataFrame, stem: Path) -> None:
    plot = df.sort_values("auroc", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if sns:
        sns.barplot(data=plot, x="variant", y="auroc", ax=ax, color="#8b98b3")
    ax.set_title("Available Same-Backbone/Fusion Control AUROC")
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_same_backbone_safety(df: pd.DataFrame, stem: Path) -> None:
    rows = []
    for variant, g in df.groupby("variant"):
        th = float(g["threshold_cin3_safety_val"].median()) if g["threshold_cin3_safety_val"].notna().any() else 0.5
        m = binary_metrics(g["y_cin3"], g["score"], th)
        rows.append({"variant": variant, "cin3_sensitivity": m["sensitivity"], "cin3_false_negatives": m["fn"]})
    plot = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(plot["variant"], plot["cin3_sensitivity"], marker="o", color="#8b98b3", linewidth=2.5)
    ax2.bar(plot["variant"], plot["cin3_false_negatives"], alpha=0.35, color="#b57979")
    ax1.set_ylabel("CIN3+ sensitivity")
    ax2.set_ylabel("CIN3+ false negatives")
    ax1.set_title("CIN3+ Safety by Fusion Variant")
    ax1.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_same_backbone_calibration(df: pd.DataFrame, stem: Path) -> None:
    rows = []
    for variant, g in df.groupby("variant"):
        rows.append({"variant": variant, "ece": expected_calibration_error(g["y_cin2"], g["score"]), "brier": safe_brier(g["y_cin2"], g["score"])})
    plot = pd.DataFrame(rows).sort_values("ece")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(plot["ece"], plot["brier"], s=120, color="#dbb98c", edgecolor="#333333")
    for _, r in plot.iterrows():
        ax.text(r["ece"], r["brier"], str(r["variant"]).replace("same_backbone_", ""), fontsize=8)
    ax.set_xlabel("ECE")
    ax.set_ylabel("Brier score")
    ax.set_title("Calibration by Fusion Variant")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_modality_auc(df: pd.DataFrame, stem: Path) -> None:
    plot = df.sort_values("auroc", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    if sns:
        sns.barplot(data=plot, x="variant", y="auroc", ax=ax, color="#abb8cc")
    ax.set_title("Modality Contribution")
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_removal_auc_npv(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(df) and {"setting", "auc", "npv"}.issubset(df.columns):
        plot = df.groupby("setting")[["auc", "npv"]].mean(numeric_only=True).reset_index().melt("setting")
        if sns:
            sns.pointplot(data=plot, x="setting", y="value", hue="variable", ax=ax, palette=["#8b98b3", "#b57979"])
        ax.tick_params(axis="x", rotation=25)
    ax.set_title("Complete Modality Removal")
    ax.set_xlabel("Removal condition")
    ax.set_ylabel("Metric")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_dropout_placeholder(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.55, "Random 10/30/50% modality-dropout\npatient-level predictions unavailable", ha="center", va="center", fontsize=15, color="#30335f", fontweight="bold")
    ax.text(0.5, 0.28, "Rows are recorded as NOT_EXECUTABLE in random_dropout_stress_table.csv", ha="center", va="center", fontsize=10, color="#555555")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_missing_cin3_fn(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(df):
        plot = df.groupby("setting").size().reset_index(name="available_rows")
        if sns:
            sns.barplot(data=plot, x="setting", y="available_rows", ax=ax, color="#dbb98c")
        ax.tick_params(axis="x", rotation=25)
    ax.set_title("Missing-Modality Stress Test Availability")
    ax.set_ylabel("Available metric rows")
    ax.set_xlabel("Condition")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_calibration_ece(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention"])].copy()
    sub["model"] = sub["model_name"].map(model_label)
    sub["center_display"] = sub["center"].map(center_label)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    if sns:
        sns.barplot(data=sub, x="center_display", y="ece", hue="calibration_variant", ax=ax, palette=PALETTE)
    ax.set_title("Calibration ECE by Center")
    ax.set_xlabel("Center")
    ax.set_ylabel("ECE")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="", fontsize=7)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_reliability_curves(df: pd.DataFrame, stem: Path) -> None:
    sub = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["calibration_variant"].isin(["threshold_lock_only", "logit_median_matching_target_unlabeled"]))].copy()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#b3b0b0")
    for i, (variant, g) in enumerate(sub.groupby("calibration_variant")):
        if len(np.unique(g["y_cin2"])) < 2:
            continue
        frac, mean = calibration_curve(g["y_cin2"].astype(int), np.clip(g["calibrated_score"], 0, 1), n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=variant, color=PALETTE[i])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title("Reliability Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_probability_density(df: pd.DataFrame, stem: Path) -> None:
    sub = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["calibration_variant"].isin(["threshold_lock_only", "logit_median_matching_target_unlabeled"]))].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if sns:
        sns.kdeplot(data=sub, x="calibrated_score", hue="calibration_variant", common_norm=False, fill=True, alpha=0.25, ax=ax, palette=PALETTE)
    ax.set_title("Probability Density After Calibration")
    ax.set_xlabel("Calibrated score")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_calibration_cin3_safety(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].eq(PRIMARY_MODEL)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if sns:
        sns.scatterplot(data=sub, x="referral_rate", y="cin3_false_negatives", hue="calibration_variant", s=100, ax=ax, palette=PALETTE)
    ax.set_title("Calibration Effect on CIN3+ Safety")
    ax.set_xlabel("Referral rate")
    ax.set_ylabel("CIN3+ false negatives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_center_confusion(labels: list[str], conf: np.ndarray, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf, cmap=LinearSegmentedColormap.from_list("muted", ["#f7f3ef", "#abb8cc", "#8b98b3"]))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([center_label(x) for x in labels], rotation=35, ha="right")
    ax.set_yticklabels([center_label(x) for x in labels])
    ax.set_title("Center Classifier Confusion")
    ax.set_xlabel("Predicted center")
    ax.set_ylabel("True center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_distance_heatmap(df: pd.DataFrame, stem: Path, metric_label: str) -> None:
    value_col = "mmd_distance" if metric_label == "MMD" else "coral_distance"
    rep = "feature_concat_proxy" if "feature_concat_proxy" in set(df["representation"]) else first_value(df["representation"])
    sub = df[df["representation"].eq(rep)]
    mat = sub.pivot(index="center_a", columns="center_b", values=value_col)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    if sns:
        sns.heatmap(mat, cmap=LinearSegmentedColormap.from_list("rb", ["#8b98b3", "#f7f3ef", "#b57979"]), ax=ax, square=True, cbar_kws={"label": metric_label})
    ax.set_xticklabels([center_label(x.get_text()) for x in ax.get_xticklabels()], rotation=35, ha="right")
    ax.set_yticklabels([center_label(x.get_text()) for x in ax.get_yticklabels()], rotation=0)
    ax.set_title(f"{metric_label} Center Distance")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_umap_proxy(rep: dict[str, object], stem: Path, title: str) -> None:
    X = reduce_features(rep["X"], 2)
    plot = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "center": pd.Series(rep["center"]).map(center_label), "label": rep["label"]})
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns:
        sns.scatterplot(data=plot, x="x", y="y", hue="center", style="label", s=35, alpha=0.75, ax=ax, palette=PALETTE)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_shift_scatter(df: pd.DataFrame, y_col: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if len(df):
        ax.scatter(df["mean_shift_distance"], df[y_col], s=120, color="#8b98b3", edgecolor="#333333")
        for _, r in df.iterrows():
            ax.text(r["mean_shift_distance"], r[y_col], center_label(r["center"]), fontsize=8)
    ax.set_title(f"Center Shift vs {y_col}")
    ax.set_xlabel("Mean shift distance")
    ax.set_ylabel(y_col)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_decision_curve(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention", "OCTOnly_ViT"])].copy()
    sub["model"] = sub["model_name"].map(model_label)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if sns:
        sns.lineplot(data=sub, x="threshold_probability", y="net_benefit", hue="model", ax=ax, palette=PALETTE, linewidth=2.4)
    if len(df):
        base = df.groupby("threshold_probability").agg(treat_all=("treat_all_net_benefit", "mean"), treat_none=("treat_none_net_benefit", "mean")).reset_index()
        ax.plot(base["threshold_probability"], base["treat_all"], color="#b3b0b0", linestyle="--", label="Treat all")
        ax.plot(base["threshold_probability"], base["treat_none"], color="#333333", linestyle=":", label="Treat none")
    ax.set_title("Decision Curve Analysis")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_referral_vs_missed(df: pd.DataFrame, stem: Path) -> None:
    sub = df.copy()
    sub["model"] = sub["model_name"].map(model_label)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if sns:
        sns.scatterplot(data=sub, x="referral_rate", y="missed_cin3_per_1000", hue="model", style="center", s=100, ax=ax, palette=PALETTE)
    ax.set_title("Referral Rate vs Missed CIN3+")
    ax.set_xlabel("Model-positive referral rate")
    ax.set_ylabel("Missed CIN3+ per 1,000")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_per1000(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention"])].copy()
    sub["model"] = sub["model_name"].map(model_label)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if sns:
        sns.barplot(data=sub, x="model", y="missed_cin2_per_1000", hue="center", ax=ax, palette=PALETTE)
    ax.set_title("Missed CIN2+ per 1,000 Patients")
    ax.set_xlabel("")
    ax.set_ylabel("Per 1,000")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, title="Center")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_locked_tradeoff(df: pd.DataFrame, stem: Path) -> None:
    sub = df[df["model_name"].isin([PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention"])].copy()
    sub["model"] = sub["model_name"].map(model_label)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if sns:
        sns.lineplot(data=sub, x="center", y="referral_rate", hue="model", marker="o", ax=ax, palette=PALETTE)
    ax.set_xticklabels([center_label(t.get_text()) for t in ax.get_xticklabels()], rotation=25, ha="right")
    ax.set_title("Locked-Threshold Referral Trade-off")
    ax.set_xlabel("Center")
    ax.set_ylabel("Referral rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_reliability_distribution(rel: pd.DataFrame, stem: Path) -> None:
    long = rel.melt(value_vars=["alpha_clinical", "alpha_colposcopy", "alpha_oct"], var_name="modality", value_name="weight")
    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.violinplot(data=long, x="modality", y="weight", ax=ax, palette=PALETTE, inner="quartile")
    ax.set_title("Reliability Weight Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_reliability_by_center(rel: pd.DataFrame, stem: Path) -> None:
    long = rel.melt(id_vars=["center"], value_vars=["alpha_clinical", "alpha_colposcopy", "alpha_oct"], var_name="modality", value_name="weight")
    long["center_display"] = long["center"].map(center_label)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if sns:
        sns.pointplot(data=long, x="center_display", y="weight", hue="modality", ax=ax, palette=PALETTE)
    ax.set_title("Reliability Weights by Center")
    ax.set_xlabel("Center")
    ax.set_ylabel("Mean weight")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_reliability_corruption(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(df) and {"corruption", "auc"}.issubset(df.columns):
        plot = df.groupby("corruption")["auc"].mean().reset_index()
        if sns:
            sns.barplot(data=plot, x="corruption", y="auc", ax=ax, color="#dbb98c")
        ax.tick_params(axis="x", rotation=25)
    ax.set_title("Corruption Response (Metric Only)")
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Corruption")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_quality_scatter(rel: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rel["logvar_oct"], rel["alpha_oct"], s=40, alpha=0.55, color="#8b98b3", edgecolor="none")
    ax.set_title("OCT Uncertainty Proxy vs OCT Weight")
    ax.set_xlabel("OCT uncertainty proxy")
    ax.set_ylabel("OCT reliability weight")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_entropy_error(rel: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.boxplot(data=rel, x="error_type", y="reliability_entropy", ax=ax, palette=PALETTE)
    ax.set_title("Reliability Entropy vs Error")
    ax.set_xlabel("")
    ax.set_ylabel("Entropy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_reliability_simplex(rel: pd.DataFrame, stem: Path) -> None:
    w = rel[["alpha_clinical", "alpha_colposcopy", "alpha_oct"]].to_numpy(dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum(axis=1, keepdims=True)
    x = 0.5 * (2 * w[:, 2] + w[:, 1])
    y = np.sqrt(3) / 2 * w[:, 1]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = np.where(rel["prediction_correct"], "#8b98b3", "#b57979")
    ax.scatter(x, y, s=24, c=colors, alpha=0.7, edgecolor="none")
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#333333")
    ax.text(-0.04, -0.04, "Clinical", fontsize=10)
    ax.text(1.0, -0.04, "OCT", fontsize=10)
    ax.text(0.42, np.sqrt(3) / 2 + 0.03, "Colpo", fontsize=10)
    ax.axis("off")
    ax.set_title("Reliability Simplex")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_coe_placeholder(summary: pd.DataFrame, stem: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=14, color="#30335f", fontweight="bold")
    ax.text(0.5, 0.32, "Required intervention logits are not available in saved outputs.", ha="center", va="center", fontsize=10, color="#555555")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_coe_delta_heatmap(coe: pd.DataFrame, stem: Path) -> None:
    mat = coe.groupby("center")[["original_z1_logit", "original_z2_logit", "original_z3_logit"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    if sns:
        sns.heatmap(mat, cmap=LinearSegmentedColormap.from_list("coe", ["#8b98b3", "#f7f3ef", "#b57979"]), center=0, ax=ax)
    ax.set_yticklabels([center_label(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    ax.set_title("CoE Stepwise Delta Proxy")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_stratified(strat: pd.DataFrame, stem: Path) -> None:
    sub = strat[strat["stratum_type"].isin(["center", "age"])].copy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if sns:
        sns.scatterplot(data=sub, x="sensitivity", y="npv", hue="stratum_type", size="n", sizes=(40, 220), ax=ax, palette=PALETTE)
    ax.set_title("Stratified Sensitivity and NPV")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("NPV")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_histology_score(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if sns:
        sns.violinplot(data=df, x="histology_stratum", y="score", ax=ax, palette=PALETTE, inner="quartile")
    ax.set_title("Score Distribution by Histology Grade")
    ax.set_xlabel("Histology")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_failure_counts(summary: pd.DataFrame, stem: Path) -> None:
    sub = summary[summary["error_type"].ne("not_error")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if len(sub) and sns:
        sns.barplot(data=sub, x="failure_mode", y="n", hue="error_type", ax=ax, palette=PALETTE)
    ax.set_title("Proxy Failure Mode Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Cases")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_false_negative_center_grade(failures: pd.DataFrame, stem: Path) -> None:
    fn = failures[failures["error_type"].eq("false_negative")]
    pivot = fn.pivot_table(index="center", columns="histology_stratum", values="score", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if sns and len(pivot):
        sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list("fn", ["#f7f3ef", "#dea3a2", "#b57979"]), annot=True, fmt=".0f", ax=ax)
    ax.set_yticklabels([center_label(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    ax.set_title("False Negatives by Center and Grade")
    fig.tight_layout()
    save_figure(fig, stem)


def fig_main_roc_dca_calibration(test: pd.DataFrame, thresholds: pd.DataFrame, stem: Path) -> None:
    sub_models = [PRIMARY_MODEL, "BioMedCLIP_Finetuned", "ColposcopyOCTText_CrossAttention", "OCTOnly_ViT"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for i, model in enumerate(sub_models):
        g = test[test["model_name"].eq(model)]
        if len(g) and len(np.unique(g["y_cin2"])) > 1:
            fpr, tpr, _ = roc_curve(g["y_cin2"], g["score"])
            axes[0].plot(fpr, tpr, label=model_label(model), color=PALETTE[i], linewidth=2.3)
            frac, mean = calibration_curve(g["y_cin2"].astype(int), np.clip(g["score"], 0, 1), n_bins=10, strategy="quantile")
            axes[2].plot(mean, frac, marker="o", color=PALETTE[i], label=model_label(model))
    axes[0].plot([0, 1], [0, 1], "--", color="#b3b0b0")
    axes[0].set_title("ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    dca_path = OUT / "08_clinical_utility/decision_curve_net_benefit.csv"
    if dca_path.exists():
        dca = read_csv(dca_path)
        dca = dca[dca["model_name"].isin(sub_models)]
        dca["model"] = dca["model_name"].map(model_label)
        if sns:
            sns.lineplot(data=dca, x="threshold_probability", y="net_benefit", hue="model", ax=axes[1], palette=PALETTE, legend=False)
    axes[1].set_title("Decision Curve")
    axes[1].set_xlabel("Threshold probability")
    axes[1].set_ylabel("Net benefit")
    axes[2].plot([0, 1], [0, 1], "--", color="#b3b0b0")
    axes[2].set_title("Calibration")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Observed")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, stem)


def fig_center_shift_combo(stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, fname, title in [
        (axes[0], OUT / "07_center_shift/mmd_pairwise_by_representation.csv", "MMD"),
        (axes[1], OUT / "07_center_shift/coral_pairwise_by_representation.csv", "CORAL"),
    ]:
        if fname.exists():
            df = read_csv(fname)
            value = "mmd_distance" if title == "MMD" else "coral_distance"
            rep = "feature_concat_proxy" if "feature_concat_proxy" in set(df["representation"]) else first_value(df["representation"])
            mat = df[df["representation"].eq(rep)].pivot(index="center_a", columns="center_b", values=value)
            if sns:
                sns.heatmap(mat, ax=ax, cmap=LinearSegmentedColormap.from_list("shift", ["#8b98b3", "#f7f3ef", "#b57979"]), cbar=False)
            ax.set_title(title)
            ax.set_xticklabels([center_label(t.get_text()) for t in ax.get_xticklabels()], rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels([center_label(t.get_text()) for t in ax.get_yticklabels()], fontsize=8)
    umap_src = OUT / "07_center_shift/shift_performance_correlation.csv"
    if umap_src.exists():
        corr = read_csv(umap_src)
        axes[2].scatter(corr["mean_shift_distance"], corr["auc_drop_vs_best"], s=120, color="#dbb98c", edgecolor="#333333")
        for _, r in corr.iterrows():
            axes[2].text(r["mean_shift_distance"], r["auc_drop_vs_best"], center_label(r["center"]), fontsize=8)
    axes[2].set_title("Shift vs AUROC Drop")
    axes[2].set_xlabel("Mean shift")
    axes[2].set_ylabel("AUROC drop")
    fig.tight_layout()
    save_figure(fig, stem)


def copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        if dst.suffix == ".csv":
            pd.DataFrame([{"status": "SOURCE_MISSING", "source": str_rel(src)}]).to_csv(dst, index=False, encoding="utf-8-sig")
        else:
            write_text(dst, f"Missing source: {str_rel(src)}\n")


def make_results_summary() -> str:
    return "\n".join(
        [
            "# Results Summary for Manuscript",
            "",
            "The supplementary experiment package was generated from locked patient-level LOCO predictions, validation predictions, and cached feature arrays.",
            "",
            "Use the final claim-lock before updating manuscript claims. The strongest defensible framing is a multicenter reliability and calibration-boundary analysis, with CoE and reliability weights presented as internal diagnostics unless perturbation/export gaps are closed.",
            "",
        ]
    )


def best_baseline_summary(pooled: pd.DataFrame) -> object:
    if len(pooled) == 0 or "auroc" not in pooled:
        return "not available"
    sub = pooled[~pooled["model_name"].eq(PRIMARY_MODEL)].copy()
    if len(sub) == 0:
        return "not available"
    row = sub.sort_values("auroc", ascending=False).head(1)
    return row.to_dict("records")


def environment_snapshot() -> str:
    lines = [
        f"created_at: {now()}",
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"hostname: {socket.gethostname()}",
        f"working_directory: {ROOT}",
    ]
    for pkg in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn", "scipy", "torch"]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            lines.append(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{pkg}: unavailable ({exc})")
    try:
        import torch

        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
    except Exception:
        lines.append("cuda_available: unknown")
    for cmd, label in [(["git", "rev-parse", "HEAD"], "git_hash"), (["git", "status", "--short"], "git_status_short")]:
        try:
            res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            lines.append(f"{label}: {res.stdout.strip()}")
        except Exception as exc:
            lines.append(f"{label}: unavailable ({exc})")
    return "\n".join(lines) + "\n"


def make_zip_package(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include_suffix = {".csv", ".md", ".txt", ".json", ".yaml", ".yml", ".png", ".pdf", ".py"}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in OUT.rglob("*"):
            if file == path or not file.is_file():
                continue
            if file.suffix.lower() in include_suffix:
                zf.write(file, arcname=str(file.relative_to(OUT)))
        for file in [
            ROOT / "scripts/if_supplementary/run_all_if_supplementary_experiments.py",
            ROOT / "scripts/if_supplementary/train_same_backbone_fusion_baselines.py",
            ROOT / "scripts/if_supplementary/evaluate_same_backbone_fusion_baselines.py",
            ROOT / "configs/if_supplementary_same_backbone_baselines.yaml",
        ]:
            if file.exists():
                zf.write(file, arcname=f"reproducibility/{file.relative_to(ROOT)}")


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    runner = SupplementaryRunner()
    runner.run()
    print(f"Supplementary package written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
