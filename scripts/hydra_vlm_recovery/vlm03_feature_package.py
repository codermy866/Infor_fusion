#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

OUT = C.OUT / "vlm03_feature_package"
VLM01 = C.OUT / "vlm01_foldwise_lora"


def save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32) if arr.dtype.kind == "f" else arr)


def scaled(src: np.ndarray, tgt: np.ndarray):
    if src.shape[1] == 0:
        return src, tgt, {"mean": [], "scale": []}
    scaler = StandardScaler()
    xs = scaler.fit_transform(np.nan_to_num(src)).astype(np.float32)
    xt = scaler.transform(np.nan_to_num(tgt)).astype(np.float32)
    return xs, xt, {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def main() -> None:
    C.ensure_dirs()
    rows = []
    blocking_errors = []
    for fold_dir in C.fold_dirs(VLM01):
        src = C.read_table(fold_dir / "source_features.parquet")
        tgt = C.read_table(fold_dir / "target_features.parquet")
        fold_name = fold_dir.name
        out = OUT / fold_name
        out.mkdir(parents=True, exist_ok=True)
        issues = []
        for name, df in [("source", src), ("target", tgt)]:
            if df["patient_id"].isna().any():
                issues.append(f"{name}: missing patient_id")
            if df["centre"].isna().any():
                issues.append(f"{name}: missing centre")
            if df["patient_id"].duplicated().any():
                issues.append(f"{name}: duplicate patient_id rows")
            for col in ["cin2_label", "cin3_label"]:
                if col not in df:
                    issues.append(f"{name}: missing {col}")
        groups = {
            "clinical": C.feature_matrix(src, ["clinical"], frozen=False),
            "colpo": C.feature_matrix(src, ["colpo"], frozen=False),
            "oct": C.feature_matrix(src, ["oct"], frozen=False),
            "text": C.feature_matrix(src, ["text"], frozen=False),
            "combined": C.feature_matrix(src, ["clinical", "text", "colpo", "oct"], frozen=False),
        }
        tgt_groups = {
            "clinical": C.feature_matrix(tgt, ["clinical"], frozen=False),
            "colpo": C.feature_matrix(tgt, ["colpo"], frozen=False),
            "oct": C.feature_matrix(tgt, ["oct"], frozen=False),
            "text": C.feature_matrix(tgt, ["text"], frozen=False),
            "combined": C.feature_matrix(tgt, ["clinical", "text", "colpo", "oct"], frozen=False),
        }
        scalers = {}
        for name in groups:
            xs, xt, params = scaled(groups[name], tgt_groups[name])
            save_array(out / f"X_source_{name}.npy", xs)
            save_array(out / f"X_target_{name}.npy", xt)
            scalers[name] = params
        for label in ["cin2", "cin3"]:
            np.save(out / f"y_source_{label}.npy", src[f"{label}_label"].astype(int).to_numpy())
            np.save(out / f"y_target_{label}.npy", tgt[f"{label}_label"].astype(int).to_numpy())
        C.write_csv(out / "patient_ids_source.csv", src[["patient_id", "case_id", "original_patient_id", "centre", "centre_label"]])
        C.write_csv(out / "patient_ids_target.csv", tgt[["patient_id", "case_id", "original_patient_id", "centre", "centre_label"]])
        C.write_json(out / "source_scaler.json", scalers)
        manifest = {
            "fold": fold_name,
            "source_rows": int(len(src)),
            "target_rows": int(len(tgt)),
            "feature_source_type": sorted(src["feature_source_type"].dropna().unique().tolist()),
            "dimensions": {k: int(v.shape[1]) for k, v in groups.items()},
            "validation_issues": issues,
            "target_labels_saved_for_evaluation_only": True,
        }
        C.write_json(out / "feature_package_manifest.json", manifest)
        rows.append(
            {
                "fold": fold_name,
                "status": "PASS" if not issues else "FAILED_PARTIAL",
                "source_rows": len(src),
                "target_rows": len(tgt),
                "clinical_dim": groups["clinical"].shape[1],
                "colpo_dim": groups["colpo"].shape[1],
                "oct_dim": groups["oct"].shape[1],
                "text_dim": groups["text"].shape[1],
                "combined_dim": groups["combined"].shape[1],
                "issues": ";".join(issues),
            }
        )
        blocking_errors.extend(issues)

    summary = pd.DataFrame(rows)
    C.write_csv(OUT / "vlm03_feature_package_summary.csv", summary)
    status = "PASS" if not blocking_errors else "FAILED_PARTIAL"
    report = [
        "# VLM03 Feature Package Audit Report",
        "",
        f"Status: `{status}`",
        "",
        "Model-ready arrays were generated for clinical, colposcopy, OCT, text, and combined cached-adapter features. Scalers were fitted on source folds only and applied to target folds. Target labels are stored only for evaluation.",
        "",
        C.md_table(summary),
    ]
    C.write_text(OUT / "vlm03_audit_report.md", "\n".join(report) + "\n")
    C.status_json(OUT / "status.json", status, "Feature package generated.", blocking_errors=blocking_errors[:20])
    C.file_manifest(OUT, OUT / "vlm03_file_manifest.csv")


if __name__ == "__main__":
    main()
