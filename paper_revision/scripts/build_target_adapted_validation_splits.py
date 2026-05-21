#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build locked 1897-cohort patient-level and LOCO validation splits.

All downstream no-report HyDRA-CoE experiments should use
final_1897_case_index.csv as the source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.clinical_variable_mapping import (
    assert_no_report_columns,
    clinical_info_from_row,
    normalize_age,
    normalize_hpv,
    normalize_tct,
)


DEFAULT_INPUT = EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "final_1897_case_index.csv"
TARGET_ROOT = EXP_ROOT / "paper_revision" / "splits" / "target_adapted_validation"
LOCO_ROOT = EXP_ROOT / "paper_revision" / "splits" / "loco"
SPLITS_ROOT = EXP_ROOT / "paper_revision" / "splits"
TABLE_DIR = EXP_ROOT / "paper_revision" / "tables"
EXPECTED_N = 1897
SEED = 2026

# Fixed five-hospital order used by the paper tables and the main model.
# Several hospitals have more than one raw center_code prefix; these prefixes
# must not be treated as separate centers.
CENTER_NAME_ORDER = [
    "武大人民医院",
    "恩施州中心医院",
    "襄阳市中心医院",
    "十堰市人民医院",
    "荆州市第一人民医院",
]
CENTER_NAME_TO_IDX = {name: idx for idx, name in enumerate(CENTER_NAME_ORDER)}
CENTER_NAME_TO_CANONICAL_CODE = {
    "武大人民医院": "C01_WUHAN_RENMIN",
    "恩施州中心医院": "C02_ENSHI",
    "襄阳市中心医院": "C03_XIANGYANG",
    "十堰市人民医院": "C04_SHIYAN",
    "荆州市第一人民医院": "C05_JINGZHOU",
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _mode_or_first(values: pd.Series):
    modes = values.dropna().mode()
    if len(modes):
        return modes.iloc[0]
    return values.iloc[0] if len(values) else ""


def prepare_locked_cohort(df: pd.DataFrame, expected_n: int, allow_count_mismatch: bool) -> pd.DataFrame:
    assert_no_report_columns(df.columns)
    if len(df) != expected_n:
        message = f"Locked cohort N mismatch: observed {len(df)}, expected {expected_n}."
        if allow_count_mismatch:
            print(f"WARNING: {message}")
        else:
            raise ValueError(message)

    required = {
        "case_id",
        "patient_id",
        "oct_id",
        "center_name",
        "center_code",
        "age",
        "hpv_raw",
        "hpv_clean",
        "tct_raw",
        "tct_clean",
        "label",
        "colposcopy_paths",
        "oct_paths",
        "col_count",
        "oct_count",
        "full_multimodal_complete",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"final_1897_case_index.csv missing required columns: {missing}")

    df = df.copy()
    df["col_count"] = _safe_numeric(df["col_count"]).astype(int)
    df["oct_count"] = _safe_numeric(df["oct_count"]).astype(int)
    no_images = df[(df["col_count"] <= 0) | (df["oct_count"] <= 0)]
    if len(no_images):
        raise ValueError(f"{len(no_images)} rows do not have both colposcopy and OCT/B-scan images.")

    df["label"] = _safe_numeric(df["label"]).astype(int).clip(0, 1)
    if "raw_center_code" not in df.columns:
        df["raw_center_code"] = df["center_code"]
    df["age"] = df.apply(lambda row: normalize_age(row.get("age")), axis=1)
    df["hpv_clean"] = df.apply(lambda row: normalize_hpv(row.get("hpv_clean", row.get("hpv_raw"))), axis=1)
    df["tct_clean"] = df.apply(lambda row: normalize_tct(row.get("tct_clean", row.get("tct_raw"))), axis=1)
    df["clinical_info"] = df.apply(clinical_info_from_row, axis=1)
    if df["clinical_info"].str.contains("label|pathology|CIN", case=False, regex=True).any():
        raise ValueError("Pathology label text leaked into clinical_info.")

    observed_centers = set(df["center_name"].astype(str).unique())
    unknown_centers = sorted(observed_centers - set(CENTER_NAME_TO_IDX))
    if unknown_centers:
        raise ValueError(
            "Unknown center_name values in locked cohort. "
            f"Expected the fixed five-hospital mapping, got: {unknown_centers}"
        )
    df["center_idx"] = df["center_name"].astype(str).map(CENTER_NAME_TO_IDX).astype(int)
    df["center_group_id"] = df["center_idx"]
    df["center_code"] = df["center_name"].astype(str).map(CENTER_NAME_TO_CANONICAL_CODE)

    # Compatibility aliases for legacy dataset loaders. They still carry only
    # no-report variables and the locked image paths.
    df["ID"] = df["patient_id"]
    df["OCT"] = df["oct_id"]
    df["AGE"] = df["age"]
    df["HPV清洗"] = df["hpv_clean"]
    df["TCT清洗"] = df["tct_clean"]
    df["col_paths"] = df["colposcopy_paths"]

    assert_no_report_columns(df.columns)
    return df


def write_center_mapping_audit(df: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for center_no, center_name in enumerate(CENTER_NAME_ORDER, start=1):
        subset = df[df["center_name"].astype(str).eq(center_name)]
        rows.append(
            {
                "paper_center_no": center_no,
                "center_idx": CENTER_NAME_TO_IDX[center_name],
                "center_name": center_name,
                "center_code": CENTER_NAME_TO_CANONICAL_CODE[center_name],
                "raw_center_codes": ";".join(sorted(subset["raw_center_code"].astype(str).unique())),
                "n_cases": int(len(subset)),
                "positive": int(subset["label"].sum()) if len(subset) else 0,
                "negative": int((subset["label"] == 0).sum()) if len(subset) else 0,
                "total_colposcopy_images": int(subset["col_count"].sum()) if len(subset) else 0,
                "total_oct_images": int(subset["oct_count"].sum()) if len(subset) else 0,
                "total_images": int(subset["col_count"].sum() + subset["oct_count"].sum()) if len(subset) else 0,
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "center_code_mapping_audit.csv", index=False, encoding="utf-8-sig")


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in df.groupby("patient_id", dropna=False):
        rows.append(
            {
                "patient_id": patient_id,
                "row_indices": list(group.index),
                "center_name": _mode_or_first(group["center_name"]),
                "center_code": _mode_or_first(group["center_code"]),
                "label": int(group["label"].max()),
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def stratified_group_split(
    groups: pd.DataFrame,
    strata_cols: Iterable[str],
    ratios: list[float],
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in ratios]
    for _, stratum in groups.groupby(list(strata_cols), dropna=False):
        indices = stratum.index.to_numpy()
        rng.shuffle(indices)
        n = len(indices)
        counts = [int(round(n * ratio)) for ratio in ratios[:-1]]
        if n >= len(ratios):
            counts = [max(1, c) for c in counts]
        used = sum(counts)
        if used >= n:
            overflow = used - (n - 1)
            for pos in range(len(counts) - 1, -1, -1):
                take = min(overflow, max(0, counts[pos] - 1))
                counts[pos] -= take
                overflow -= take
                if overflow <= 0:
                    break
        splits = np.split(indices, np.cumsum(counts))
        for bucket, split_indices in zip(buckets, splits):
            row_indices = groups.loc[split_indices, "row_indices"].tolist()
            for rows in row_indices:
                bucket.extend(rows)
    return buckets


def subset_by_rows(df: pd.DataFrame, rows: list[int], seed: int) -> pd.DataFrame:
    if not rows:
        return df.iloc[[]].copy()
    return df.loc[rows].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_split(split_dir: Path, train: pd.DataFrame, val: pd.DataFrame, external: pd.DataFrame, readme: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, subset in [
        ("train_labels.csv", train),
        ("val_labels.csv", val),
        ("external_test_labels.csv", external),
    ]:
        assert_no_report_columns(subset.columns)
        subset.to_csv(split_dir / subset_name, index=False, encoding="utf-8-sig")
    (split_dir / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")


def summarize(split_name: str, subset: str, df: pd.DataFrame, definition: str, held_out_center: str = "") -> dict[str, object]:
    assert_no_report_columns(df.columns)
    return {
        "split_name": split_name,
        "held_out_center": held_out_center,
        "subset": subset,
        "definition": definition,
        "n_cases": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()) if len(df) else 0,
        "positive": int(df["label"].sum()) if len(df) else 0,
        "negative": int((df["label"] == 0).sum()) if len(df) else 0,
        "positive_rate": round(float(df["label"].mean()), 4) if len(df) else np.nan,
        "centers": ",".join(sorted(df["center_name"].astype(str).unique())) if len(df) else "",
        "total_colposcopy_images": int(df["col_count"].sum()) if len(df) else 0,
        "total_oct_images": int(df["oct_count"].sum()) if len(df) else 0,
        "total_images": int(df["col_count"].sum() + df["oct_count"].sum()) if len(df) else 0,
    }


def build_all_center_patient_holdout(df: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    split_name = "all_center_patient_holdout_70_10_20"
    groups = group_table(df)
    train_rows, val_rows, test_rows = stratified_group_split(
        groups,
        strata_cols=["center_name", "label"],
        ratios=[0.70, 0.10, 0.20],
        seed=seed,
    )
    train = subset_by_rows(df, train_rows, seed)
    val = subset_by_rows(df, val_rows, seed + 1)
    test = subset_by_rows(df, test_rows, seed + 2)
    write_split(
        TARGET_ROOT / split_name,
        train,
        val,
        test,
        """
# all_center_patient_holdout_70_10_20

Patient-level multi-center holdout from the locked 1897 tri-modal cohort.
All centers may appear in train, validation, and external_test. This is not
strict center-external validation.
""",
    )
    return [
        summarize(split_name, "train", train, "patient_level_multi_center_holdout"),
        summarize(split_name, "val", val, "patient_level_multi_center_holdout"),
        summarize(split_name, "external_test", test, "patient_level_multi_center_holdout"),
    ]


def build_loco_splits(df: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, center_name in enumerate(sorted(df["center_name"].astype(str).unique())):
        held_out = df[df["center_name"].astype(str).eq(center_name)].copy().reset_index(drop=True)
        internal = df[~df["center_name"].astype(str).eq(center_name)].copy().reset_index(drop=True)
        internal_groups = group_table(internal)
        train_rows, val_rows = stratified_group_split(
            internal_groups,
            strata_cols=["center_name", "label"],
            ratios=[0.90, 0.10],
            seed=seed + 100 + offset,
        )
        train = subset_by_rows(internal, train_rows, seed + 1000 + offset)
        val = subset_by_rows(internal, val_rows, seed + 2000 + offset)
        split_dir = LOCO_ROOT / center_name
        write_split(
            split_dir,
            train,
            val,
            held_out,
            f"""
# LOCO: {center_name}

Leave-one-centre-out split. The held-out centre is external_test only and must
not be used for threshold selection or hyperparameter tuning.
""",
        )
        split_name = f"loco_{center_name}"
        rows.extend(
            [
                summarize(split_name, "train", train, "leave_one_center_out", held_out_center=center_name),
                summarize(split_name, "val", val, "leave_one_center_out", held_out_center=center_name),
                summarize(split_name, "external_test", held_out, "leave_one_center_out", held_out_center=center_name),
            ]
        )
    return rows


def write_split_policy() -> None:
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    policy = """
# Split Policy

All splits are generated from the locked final 1897-case tri-modal cohort.

all_center_patient_holdout_70_10_20 is patient-level multi-center holdout, not strict center-external validation.

LOCO splits are supplementary cross-center generalization tests.

For each LOCO split, the held-out centre is used only as external_test. It must
not be used for threshold selection, model selection, or hyperparameter tuning.

The current main experiment uses colposcopy images, OCT images/B-scans, and
HPV/TCT/Age clinical text variables. Examination reports are not used.

The five model centers follow the paper table order:
0 武大人民医院; 1 恩施州中心医院; 2 襄阳市中心医院; 3 十堰市人民医院; 4 荆州市第一人民医院.
Raw center_code prefixes are retained as provenance only and are not used as
separate model centers.
"""
    (SPLITS_ROOT / "SPLIT_POLICY.md").write_text(policy.strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    LOCO_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_locked_cohort(read_csv(args.input), args.expected_n, args.allow_count_mismatch)
    write_center_mapping_audit(df)
    all_center_rows = build_all_center_patient_holdout(df, args.seed)
    loco_rows = build_loco_splits(df, args.seed)
    write_split_policy()

    all_center_summary = pd.DataFrame(all_center_rows)
    loco_summary = pd.DataFrame(loco_rows)
    all_center_summary.to_csv(TABLE_DIR / "split_summary_1897.csv", index=False, encoding="utf-8-sig")
    loco_summary.to_csv(TABLE_DIR / "loco_split_summary.csv", index=False, encoding="utf-8-sig")

    print(all_center_summary.to_string(index=False))
    print(loco_summary.to_string(index=False))
    print(f"Wrote all-center split to {TARGET_ROOT / 'all_center_patient_holdout_70_10_20'}")
    print(f"Wrote LOCO splits to {LOCO_ROOT}")


if __name__ == "__main__":
    main()
