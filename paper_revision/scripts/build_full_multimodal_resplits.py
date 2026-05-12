#!/usr/bin/env python3
"""Re-split the original complete 5-center full-multimodal dataset.

The source dataset is `/data2/hmy_pri/VLM_Caus_Rm_Mics/data/5centers_multi`,
which contains matched OCT, colposcopy, and clinical labels. OCT-only external
centers are intentionally excluded from these full-multimodal splits.

Outputs include:
- recommended_enshi_external: a larger single-center full-multimodal external
  split using Enshi as held-out external center.
- official_jingzhou_shiyan_external: the prior Jingzhou+Shiyan center holdout
  rebuilt from the complete original dataset.
- case_stratified_holdout: a case-level stratified holdout, marked as not
  center-external.
- leave_one_center_out/*: five full-multimodal center-held-out folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SOURCE_ROOT = Path("/data2/hmy_pri/VLM_Caus_Rm_Mics/data/5centers_multi")
EXP_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit"
TABLE_DIR = EXP_ROOT / "paper_revision" / "tables"

SEED = 2026
VAL_RATIO = 0.20
CASE_EXTERNAL_RATIO = 0.25

CENTER_MAP = {
    "M22105": "Enshi",
    "M20105": "Wuda",
    "M20203": "Wuda",
    "M22102": "Xiangyang",
    "M0008": "Jingzhou",
    "M22101": "Jingzhou",
    "M22104": "Shiyan",
}

CENTER_CODES_BY_GROUP = {
    "Enshi": ["M22105"],
    "Wuda": ["M20105", "M20203"],
    "Xiangyang": ["M22102"],
    "Jingzhou": ["M0008", "M22101"],
    "Shiyan": ["M22104"],
}

CENTER_GROUP_ID = {
    "Enshi": 0,
    "Wuda": 1,
    "Xiangyang": 2,
    "Jingzhou": 3,
    "Shiyan": 4,
}


def split_paths(text: str) -> list[str]:
    if pd.isna(text) or str(text).strip() == "":
        return []
    return [p for p in str(text).split(";") if p]


def collect_paths(base: Path, patterns: Iterable[str]) -> list[str]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return [str(p) for p in sorted(files)]


def load_complete_cases() -> pd.DataFrame:
    rows = []
    for original_split in ["train", "test"]:
        labels = pd.read_csv(SOURCE_ROOT / f"{original_split}_labels.csv")
        for _, row in labels.iterrows():
            patient_id = str(row["ID"])
            oct_id = str(row["OCT"])
            center_code = oct_id.split("_")[0]
            center_name = CENTER_MAP.get(center_code, center_code)
            oct_dir = SOURCE_ROOT / original_split / "oct" / oct_id
            col_dir = SOURCE_ROOT / original_split / "col" / patient_id
            oct_paths = collect_paths(oct_dir, ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"])
            col_paths = collect_paths(col_dir, ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"])
            rows.append(
                {
                    "ID": patient_id,
                    "OCT": oct_id,
                    "AGE": row.get("AGE", ""),
                    "HPV清洗": row.get("HPV清洗", ""),
                    "TCT清洗": row.get("TCT清洗", ""),
                    "label": int(row["label"]),
                    "patient_id": patient_id,
                    "oct_id": oct_id,
                    "age": row.get("AGE", ""),
                    "hpv": row.get("HPV清洗", ""),
                    "tct": row.get("TCT清洗", ""),
                    "center_code": center_code,
                    "center_name": center_name,
                    "center_group_id": CENTER_GROUP_ID.get(center_name, -1),
                    "original_split": original_split,
                    "oct_paths": ";".join(oct_paths),
                    "col_paths": ";".join(col_paths),
                    "oct_count": len(oct_paths),
                    "col_count": len(col_paths),
                    "is_positive_patient": int(row["label"]) == 1,
                    "source_root": str(SOURCE_ROOT),
                }
            )
    df = pd.DataFrame(rows)
    df["full_multimodal_complete"] = (df["oct_count"] > 0) & (df["col_count"] > 0)
    return df


def stratified_train_val(df: pd.DataFrame, val_ratio: float = VAL_RATIO, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    group_cols = ["center_name", "label"]
    for _, group in df.groupby(group_cols, dropna=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        if len(idx) <= 3:
            n_val = 0
        else:
            n_val = max(1, int(round(len(idx) * val_ratio)))
            n_val = min(n_val, len(idx) - 1)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        if len(train_idx):
            train_parts.append(df.loc[train_idx])
        if len(val_idx):
            val_parts.append(df.loc[val_idx])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    val = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy()
    return train.sample(frac=1.0, random_state=seed).reset_index(drop=True), val.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def case_stratified_split(df: pd.DataFrame, external_ratio: float = CASE_EXTERNAL_RATIO, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    external_parts = []
    internal_parts = []
    for _, group in df.groupby(["center_name", "label"], dropna=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        if len(idx) <= 3:
            n_ext = 1
        else:
            n_ext = max(1, int(round(len(idx) * external_ratio)))
            n_ext = min(n_ext, len(idx) - 1)
        external_parts.append(df.loc[idx[:n_ext]])
        internal_parts.append(df.loc[idx[n_ext:]])
    external = pd.concat(external_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    internal = pd.concat(internal_parts, ignore_index=True).reset_index(drop=True)
    train, val = stratified_train_val(internal, seed=seed)
    return train, val, external


def write_split(split_dir: Path, train: pd.DataFrame, val: pd.DataFrame, external: pd.DataFrame, description: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train_labels.csv", index=False, encoding="utf-8-sig")
    val.to_csv(split_dir / "val_labels.csv", index=False, encoding="utf-8-sig")
    external.to_csv(split_dir / "external_test_labels.csv", index=False, encoding="utf-8-sig")
    (split_dir / "README.md").write_text(description + "\n", encoding="utf-8")


def summarize_subset(name: str, subset: str, df: pd.DataFrame) -> dict[str, object]:
    centers = ",".join(sorted(df["center_name"].unique())) if len(df) else ""
    codes = ",".join(sorted(df["center_code"].unique())) if len(df) else ""
    return {
        "split_name": name,
        "subset": subset,
        "n": int(len(df)),
        "positive": int(df["label"].sum()) if len(df) else 0,
        "negative": int((df["label"] == 0).sum()) if len(df) else 0,
        "positive_rate": round(float(df["label"].mean()), 4) if len(df) else np.nan,
        "centers": centers,
        "center_codes": codes,
        "min_oct_count": int(df["oct_count"].min()) if len(df) else 0,
        "median_oct_count": float(df["oct_count"].median()) if len(df) else 0,
        "min_col_count": int(df["col_count"].min()) if len(df) else 0,
        "median_col_count": float(df["col_count"].median()) if len(df) else 0,
    }


def build_outputs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df_all = load_complete_cases()
    df_all.to_csv(OUT_ROOT / "full_multimodal_all_cases_audit.csv", index=False, encoding="utf-8-sig")

    complete = df_all[df_all["full_multimodal_complete"]].copy().reset_index(drop=True)
    incomplete = df_all[~df_all["full_multimodal_complete"]].copy().reset_index(drop=True)
    incomplete.to_csv(OUT_ROOT / "excluded_incomplete_cases.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    center_summary = (
        complete.groupby(["center_name", "center_code", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    center_summary["total"] = center_summary.get(0, 0) + center_summary.get(1, 0)
    center_summary["positive_rate"] = center_summary.get(1, 0) / center_summary["total"]
    center_summary.to_csv(OUT_ROOT / "source_center_summary.csv", index=False, encoding="utf-8-sig")
    center_summary.to_csv(TABLE_DIR / "full_multimodal_resplit_source_center_summary.csv", index=False, encoding="utf-8-sig")

    split_specs: list[tuple[str, list[str], str]] = [
        (
            "recommended_enshi_external",
            ["Enshi"],
            "Recommended larger single-center full-multimodal external split. External center: Enshi (largest complete center). Internal train/val uses all other complete centers with stratified train/val split.",
        ),
        (
            "official_jingzhou_shiyan_external",
            ["Jingzhou", "Shiyan"],
            "Prior-style full-multimodal center-holdout split. External centers: Jingzhou and Shiyan. This remains the strict two-center external set but is only 148 cases.",
        ),
        (
            "expanded_xiangyang_external",
            ["Xiangyang"],
            "Alternative larger single-center full-multimodal external split. External center: Xiangyang. Use as sensitivity analysis, not as the only selected external result.",
        ),
    ]

    for split_name, external_centers, description in split_specs:
        external = complete[complete["center_name"].isin(external_centers)].copy().reset_index(drop=True)
        internal = complete[~complete["center_name"].isin(external_centers)].copy().reset_index(drop=True)
        train, val = stratified_train_val(internal, seed=SEED)
        write_split(OUT_ROOT / split_name, train, val, external, description)
        for subset_name, subset_df in [("train", train), ("val", val), ("external_test", external)]:
            summary_rows.append(summarize_subset(split_name, subset_name, subset_df))

    train, val, external = case_stratified_split(complete, seed=SEED)
    write_split(
        OUT_ROOT / "case_stratified_holdout",
        train,
        val,
        external,
        "Case-level stratified holdout from all complete centers. This increases external sample size, but it is not a center-external validation split because all centers appear in training.",
    )
    for subset_name, subset_df in [("train", train), ("val", val), ("external_test", external)]:
        summary_rows.append(summarize_subset("case_stratified_holdout", subset_name, subset_df))

    lco_root = OUT_ROOT / "leave_one_center_out"
    for center_name in sorted(CENTER_CODES_BY_GROUP):
        external = complete[complete["center_name"].eq(center_name)].copy().reset_index(drop=True)
        internal = complete[~complete["center_name"].eq(center_name)].copy().reset_index(drop=True)
        train, val = stratified_train_val(internal, seed=SEED)
        split_name = f"lco_{center_name.lower()}"
        write_split(
            lco_root / split_name,
            train,
            val,
            external,
            f"Leave-one-center-out full-multimodal fold. Held-out external center: {center_name}.",
        )
        for subset_name, subset_df in [("train", train), ("val", val), ("external_test", external)]:
            summary_rows.append(summarize_subset(split_name, subset_name, subset_df))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "resplit_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "full_multimodal_resplit_summary.csv", index=False, encoding="utf-8-sig")

    policy = f"""# Full-Multimodal Re-Split Policy

Source dataset: `{SOURCE_ROOT}`.

This re-split uses only complete 5-center multimodal cases with matched OCT and colposcopy paths. OCT-only external cohorts such as AnYang, HuaXi, ZhengDaSanFu, and liaoning are excluded from these full-multimodal splits.

Total source cases: {len(df_all)}.
Complete full-multimodal cases retained: {len(complete)}.
Incomplete cases excluded: {len(incomplete)}.

Recommended reporting strategy:

1. Use `recommended_enshi_external` when a single larger center-held-out external set is needed. It holds out Enshi as external test set with 404 complete cases.
2. Use `leave_one_center_out` as the strongest reviewer-facing protocol, because each complete center becomes external once and the total external evidence covers all 985 complete cases across folds.
3. Keep `official_jingzhou_shiyan_external` as a continuity check for the strict two-center external split.
4. Use `case_stratified_holdout` only as a case-level holdout analysis; do not describe it as center-external validation.
"""
    (OUT_ROOT / "README.md").write_text(policy, encoding="utf-8")

    print(f"Wrote full multimodal re-splits to: {OUT_ROOT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    build_outputs()
