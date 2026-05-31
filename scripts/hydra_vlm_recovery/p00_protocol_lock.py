#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


OUT = C.OUT / "p00_protocol_lock"


def main() -> None:
    C.ensure_dirs()
    df = C.load_data_lock()
    required = ["case_id", "patient_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]
    missing = [c for c in required if c not in df.columns]

    folds = C.make_folds(df)
    C.write_json(OUT / "locked_folds.json", folds)
    C.write_csv(OUT / "locked_folds.csv", pd.DataFrame(folds))

    endpoint = {
        "primary_endpoint": "CIN2+",
        "primary_endpoint_column": "pathology_cin2plus",
        "safety_endpoint": "CIN3+",
        "safety_endpoint_column": "pathology_cin3plus",
        "patient_id_column": "patient_id",
        "case_id_column": "case_id",
        "centre_column": "center_name",
    }
    C.write_json(OUT / "endpoint_definition.json", endpoint)

    modality_rows = []
    modality_specs = {
        "colposcopy": ["colposcopy_available", "colposcopy_paths", "colposcopy_num_images"],
        "oct": ["oct_available", "oct_paths", "oct_num_bscans"],
        "clinical_prior": ["clinical_prior_available", "age", "hpv16_18_status", "tct_status_harmonized"],
        "vlm_cache": ["vlm_cache_available"],
    }
    for mod, cols in modality_specs.items():
        present = [c for c in cols if c in df.columns]
        missing_cols = [c for c in cols if c not in df.columns]
        availability_col = present[0] if present and present[0].endswith("_available") else None
        n_available = int(df[availability_col].fillna(False).astype(bool).sum()) if availability_col else int(len(df))
        modality_rows.append(
            {
                "modality": mod,
                "detected_columns": ";".join(present),
                "missing_columns": ";".join(missing_cols),
                "n_available": n_available,
                "availability_rate": n_available / len(df),
            }
        )
    C.write_csv(OUT / "modality_availability.csv", pd.DataFrame(modality_rows))

    leakage_rows = []
    blocked = False
    for f in folds:
        src = df[df["center_name"].isin(f["source_centres"])]
        tgt = df[df["center_name"].eq(f["target_centre"])]
        overlap = sorted(set(src["patient_id"].astype(str)).intersection(set(tgt["patient_id"].astype(str))))
        if overlap:
            blocked = True
        leakage_rows.append(
            {
                "fold_id": f["fold_id"],
                "held_out_centre": f["held_out_centre"],
                "n_source_patient_ids": src["patient_id"].astype(str).nunique(),
                "n_target_patient_ids": tgt["patient_id"].astype(str).nunique(),
                "n_cross_source_target_duplicate_patient_ids": len(overlap),
                "example_duplicates": ";".join(overlap[:10]),
                "blocks_downstream": bool(overlap),
            }
        )
    dup_all = int(df["patient_id"].astype(str).duplicated().sum())
    patient_audit = pd.DataFrame(leakage_rows)
    C.write_csv(OUT / "patient_id_integrity_audit.csv", patient_audit)

    rules = """# Leakage Control Rules

- Source-only training, feature adaptation, model selection, threshold selection, and ensemble selection must use source centres only.
- Held-out target-centre labels are prohibited for source-only training, feature adaptation, threshold tuning, hyperparameter selection, and model selection.
- Target-centre unlabelled data may be used only in explicitly marked transductive analyses.
- Fold-wise LoRA tuning, if available, must be trained on `D_src` only.
- In this recovery run, BioMedCLIP-LoRA is not assumed. Cached-feature adapters must be labelled `CACHED_FEATURE_ADAPTER`.
- Target labels may be stored in feature files only for downstream evaluation after predictions are locked.
"""
    C.write_text(OUT / "leakage_control_rules.md", rules)

    status = "PASS"
    notes = "Required IDs, centres, and endpoints are available; no cross source/target patient-ID overlap detected."
    if missing:
        status = "FAILED"
        notes = f"Missing required columns: {missing}"
    elif blocked:
        status = "FAILED"
        notes = "Cross source/target patient-ID overlap detected."
    elif any(r["missing_columns"] for r in modality_rows):
        status = "FAILED_PARTIAL"
        notes = "Core protocol is locked, but some non-critical modality descriptor columns are missing."

    report = [
        "# P00 Protocol Lock Report",
        "",
        f"Status: `{status}`",
        "",
        notes,
        "",
        f"Cohort rows: {len(df)}.",
        f"Centres: {df['center_name'].nunique()}.",
        f"Duplicate patient IDs within full table: {dup_all} (reported, not automatically leakage if within the same centre/fold role).",
        "",
        "## Folds",
        "",
        C.md_table(pd.DataFrame(folds)),
        "",
        "## Patient Leakage Audit",
        "",
        C.md_table(patient_audit),
    ]
    C.write_text(OUT / "protocol_lock_report.md", "\n".join(report) + "\n")
    C.status_json(OUT / "status.json", status, notes)
    C.file_manifest(OUT, OUT / "p00_file_manifest.csv")


if __name__ == "__main__":
    main()
