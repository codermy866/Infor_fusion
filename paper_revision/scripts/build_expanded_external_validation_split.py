#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a larger patient-level external holdout from the locked 1897 cohort.

This split is intended for paper-facing sensitivity analyses where the original
20% patient-level external test set is too small for stable centre-wise tables.
It is still a patient-level multi-centre holdout, not strict centre-external
validation. Strict centre-external testing remains the LOCO protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.build_target_adapted_validation_splits import (
    DEFAULT_INPUT,
    EXPECTED_N,
    SEED,
    TARGET_ROOT,
    TABLE_DIR,
    group_table,
    prepare_locked_cohort,
    read_csv,
    stratified_group_split,
    subset_by_rows,
    summarize,
    write_split,
)
from paper_revision.scripts.clinical_variable_mapping import assert_no_report_columns


DEFAULT_SPLIT_NAME = "all_center_patient_holdout_60_10_30"


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be positive.")


def _patient_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    train_ids = set(train["patient_id"].astype(str))
    val_ids = set(val["patient_id"].astype(str))
    test_ids = set(test["patient_id"].astype(str))
    return {
        "train_val_overlap": len(train_ids & val_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "val_test_overlap": len(val_ids & test_ids),
    }


def build_expanded_split(
    df: pd.DataFrame,
    split_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    assert_no_report_columns(df.columns)

    groups = group_table(df)
    train_rows, val_rows, test_rows = stratified_group_split(
        groups,
        strata_cols=["center_name", "label"],
        ratios=[train_ratio, val_ratio, test_ratio],
        seed=seed,
    )
    train = subset_by_rows(df, train_rows, seed)
    val = subset_by_rows(df, val_rows, seed + 1)
    test = subset_by_rows(df, test_rows, seed + 2)

    overlap = _patient_overlap(train, val, test)
    if any(overlap.values()):
        raise ValueError(f"Patient overlap detected in expanded split: {overlap}")

    split_dir = TARGET_ROOT / split_name
    write_split(
        split_dir,
        train,
        val,
        test,
        f"""
# {split_name}

Expanded patient-level multi-centre holdout from the locked 1897 tri-modal
cohort. All five hospitals may appear in train, validation, and external_test.
This split increases the held-out test size for stable paper-facing estimates,
but it is not strict centre-external validation. LOCO splits remain the
supplementary cross-centre generalization tests.

No examination reports, diagnosis reports, generated reports, or VLM evidence
caches are used. Inputs remain colposcopy images, OCT images/B-scans, and
HPV/TCT/Age clinical text variables.
""",
    )

    summary_rows = [
        summarize(split_name, "train", train, "expanded_patient_level_multi_center_holdout"),
        summarize(split_name, "val", val, "expanded_patient_level_multi_center_holdout"),
        summarize(split_name, "external_test", test, "expanded_patient_level_multi_center_holdout"),
    ]
    split_summary = pd.DataFrame(summary_rows)

    center_rows: list[dict[str, object]] = []
    for subset_name, subset in [("train", train), ("val", val), ("external_test", test)]:
        for center_name, center_df in subset.groupby("center_name", dropna=False):
            center_rows.append(
                {
                    "split_name": split_name,
                    "subset": subset_name,
                    "center_name": center_name,
                    "center_idx": int(center_df["center_idx"].mode().iloc[0]),
                    "n_cases": int(len(center_df)),
                    "positive": int(center_df["label"].sum()),
                    "negative": int((center_df["label"] == 0).sum()),
                    "positive_rate": round(float(center_df["label"].mean()), 4),
                    "total_colposcopy_images": int(center_df["col_count"].sum()),
                    "total_oct_images": int(center_df["oct_count"].sum()),
                    "total_images": int(center_df["col_count"].sum() + center_df["oct_count"].sum()),
                }
            )
    center_summary = pd.DataFrame(center_rows)

    split_summary = split_summary.merge(
        pd.DataFrame([{"split_name": split_name, **overlap}]),
        on="split_name",
        how="left",
    )
    return train, val, test, split_summary, center_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split-name", default=DEFAULT_SPLIT_NAME)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED + 3030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    df = prepare_locked_cohort(read_csv(args.input), args.expected_n, args.allow_count_mismatch)

    train, val, test, split_summary, center_summary = build_expanded_split(
        df=df,
        split_name=args.split_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_summary.to_csv(TABLE_DIR / "expanded_external_split_summary.csv", index=False, encoding="utf-8-sig")
    center_summary.to_csv(TABLE_DIR / "expanded_external_center_summary.csv", index=False, encoding="utf-8-sig")

    print(split_summary.to_string(index=False))
    print(center_summary.to_string(index=False))
    print(f"Wrote expanded split to {TARGET_ROOT / args.split_name}")
    print(
        "External test size:",
        len(test),
        "cases; total images:",
        int(test["col_count"].sum() + test["oct_count"].sum()),
    )


if __name__ == "__main__":
    main()
