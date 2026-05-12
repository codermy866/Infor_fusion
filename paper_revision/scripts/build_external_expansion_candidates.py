#!/usr/bin/env python3
"""Audit and prepare legitimate external-set expansion candidates.

This script does not relabel internal data as external. It creates auditable
candidate files for supplementary external validation:

1. official_full_multimodal_external_148.csv
   The fixed full-multimodal external set already used in the paper package.
2. supplementary_oct_only_external_902.csv
   A non-overlapping 10-center OCT-only external candidate set. This can be
   reported as supplementary OCT-only external validation or used with a
   missing-modality protocol, but it should not be described as a full
   multimodal external cohort.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = EXP_ROOT / "paper_revision"
OUT_DIR = PAPER_DIR / "data_expansion"
TABLE_DIR = PAPER_DIR / "tables"

FIVE_C_ROOT = Path("/data2/hmy/5Center_datas/5centers_multi_leave_centers_out")
TEN_C_ROOT = Path("/data2/hmy/VLM_Caus_Rm_Mics/data/loc5out_10centers_oct")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def oct_ids(df: pd.DataFrame) -> set[str]:
    col = "OCT" if "OCT" in df.columns else "oct_id"
    return set(df[col].astype(str))


def normalize_official_external(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "case_id": df["ID"].astype(str),
            "oct_id": df["OCT"].astype(str),
            "center": df["OCT"].astype(str).str.split("_").str[0],
            "label": df["label"].astype(int),
            "age": df.get("AGE", ""),
            "hpv": df.get("HPV清洗", ""),
            "tct": df.get("TCT清洗", ""),
            "modality_profile": "full_multimodal",
            "source_csv": str(FIVE_C_ROOT / "external_test_labels.csv"),
            "recommended_use": "primary_official_external_full_multimodal",
        }
    )
    return out


def normalize_oct_only_external(df: pd.DataFrame, used_5c_ids: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["oct_id"] = out["oct_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["center"] = out["center_id"].astype(str)
    out["case_id"] = out["oct_id"]
    out["oct_path_count"] = out["oct_paths"].fillna("").astype(str).map(lambda x: 0 if not x else len(x.split(";")))
    out["overlap_with_5center_current_splits"] = out["oct_id"].isin(used_5c_ids)
    out["missing_oct_paths"] = out["oct_paths"].fillna("").astype(str).eq("")
    out["modality_profile"] = "oct_only"
    out["source_csv"] = str(TEN_C_ROOT / "external_test_labels_fixed.csv")
    out["recommended_use"] = out["overlap_with_5center_current_splits"].map(
        {
            True: "exclude_overlap",
            False: "supplementary_external_oct_only_or_missing_modality_external",
        }
    )
    cols = [
        "case_id",
        "oct_id",
        "center",
        "label",
        "oct_paths",
        "oct_path_count",
        "source",
        "modality_profile",
        "overlap_with_5center_current_splits",
        "missing_oct_paths",
        "source_csv",
        "recommended_use",
    ]
    return out[cols]


def summarize(name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for center, group in df.groupby("center", dropna=False):
        rows.append(
            {
                "candidate_set": name,
                "center": center,
                "n": int(len(group)),
                "positive": int(group["label"].sum()),
                "negative": int((group["label"] == 0).sum()),
                "positive_rate": round(float(group["label"].mean()), 4),
                "modality_profile": str(group["modality_profile"].iloc[0]),
            }
        )
    rows.append(
        {
            "candidate_set": name,
            "center": "TOTAL",
            "n": int(len(df)),
            "positive": int(df["label"].sum()),
            "negative": int((df["label"] == 0).sum()),
            "positive_rate": round(float(df["label"].mean()), 4),
            "modality_profile": str(df["modality_profile"].iloc[0]),
        }
    )
    return pd.DataFrame(rows)


def write_strategy_doc(official: pd.DataFrame, oct_only: pd.DataFrame, summary: pd.DataFrame) -> None:
    eligible_oct = oct_only[
        (~oct_only["overlap_with_5center_current_splits"]) & (~oct_only["missing_oct_paths"])
    ]
    lines = [
        "# External Dataset Expansion Strategy",
        "",
        "## Recommendation",
        "",
        "Use a two-tier external validation design rather than merging everything into one ambiguous external set.",
        "",
        "1. **External-A: fixed full-multimodal external set.** Keep the current 148 Jingzhou/Shiyan cases as the primary full-multimodal external test set.",
        "2. **External-B: supplementary large OCT-only external set.** Use the non-overlapping 10-center OCT-only cohort as supplementary external validation under an OCT-only or missing-modality protocol.",
        "3. **External-C: future full-multimodal expansion.** Add newly collected cases only if they include matched OCT, colposcopy, clinical fields, labels, and center/time metadata.",
        "",
        "## Current Audit Result",
        "",
        f"- External-A full multimodal cases: {len(official)}.",
        f"- External-B OCT-only candidate cases: {len(oct_only)}.",
        f"- Eligible non-overlapping OCT-only cases with OCT paths: {len(eligible_oct)}.",
        f"- OCT-only candidate positives: {int(eligible_oct['label'].sum())}; negatives: {int((eligible_oct['label'] == 0).sum())}.",
        "",
        "The OCT-only cohort should **not** be described as an expanded full-multimodal external set because colposcopy and clinical prior fields are not present in the candidate CSV. It can be used as a reviewer-facing supplementary external stress test for modality incompleteness and cross-center OCT generalization.",
        "",
        "## Clean Expansion Rules",
        "",
        "- Do not move internal train/validation cases into the external test set.",
        "- Do not select external cases based on model performance.",
        "- Freeze the expanded external CSV before running final model evaluation.",
        "- Report External-A and External-B separately.",
        "- Use patient/OCT ID overlap checks and path-existence checks before evaluation.",
        "- For new full-multimodal cases, require all matched modalities and a documented acquisition time after the training cohort when possible.",
        "",
        "## Candidate Center Summary",
        "",
        summary.to_csv(index=False),
    ]
    (OUT_DIR / "external_expansion_strategy.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    five_train = read_csv(FIVE_C_ROOT / "train_labels.csv")
    five_val = read_csv(FIVE_C_ROOT / "val_labels.csv")
    five_external = read_csv(FIVE_C_ROOT / "external_test_labels.csv")
    used_5c_ids = oct_ids(five_train) | oct_ids(five_val) | oct_ids(five_external)

    ten_external = read_csv(TEN_C_ROOT / "external_test_labels_fixed.csv")

    official = normalize_official_external(five_external)
    oct_only = normalize_oct_only_external(ten_external, used_5c_ids)
    eligible_oct_only = oct_only[
        (~oct_only["overlap_with_5center_current_splits"]) & (~oct_only["missing_oct_paths"])
    ].copy()

    official.to_csv(OUT_DIR / "official_full_multimodal_external_148.csv", index=False)
    oct_only.to_csv(OUT_DIR / "supplementary_oct_only_external_902_all_audit.csv", index=False)
    eligible_oct_only.to_csv(OUT_DIR / "supplementary_oct_only_external_902.csv", index=False)

    summary = pd.concat(
        [
            summarize("External-A_full_multimodal_official", official),
            summarize("External-B_oct_only_supplementary", eligible_oct_only),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT_DIR / "external_expansion_candidate_summary.csv", index=False)
    summary.to_csv(TABLE_DIR / "external_expansion_candidate_summary.csv", index=False)
    write_strategy_doc(official, oct_only, summary)

    print(f"Wrote expansion audit to {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
