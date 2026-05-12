#!/usr/bin/env python3
"""Build validation definitions that can legitimately target higher AUC.

These splits do not replace the strict Enshi center-held-out external test.
They define additional deployment-oriented evaluations:

1. enshi_target_adapted_20_10_70
   - Non-Enshi centers provide source training/validation.
   - 20% of Enshi is used as labeled target-center adaptation data.
   - 10% of Enshi is used as target-center validation/model selection data.
   - 70% of Enshi remains a held-out final target-center test.

2. all_center_patient_holdout_70_10_20
   - Patient-level, label/center-stratified multi-center train/val/test split.
   - This is a multi-center held-out test, not a center-external test.

Both definitions must be reported with their exact validation meaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "full_multimodal_all_cases_audit.csv"
OUT_ROOT = EXP_ROOT / "paper_revision" / "splits" / "target_adapted_validation"
TABLE_DIR = EXP_ROOT / "paper_revision" / "tables"

SEED = 2026


def stratified_sample_indices(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    ratios: list[float],
    seed: int,
) -> list[pd.Index]:
    rng = np.random.default_rng(seed)
    parts = [[] for _ in ratios]
    for _, group in df.groupby(list(group_cols), dropna=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        counts = [int(round(n * ratio)) for ratio in ratios[:-1]]
        # Keep every split non-empty when the group is large enough.
        if n >= len(ratios):
            counts = [max(1, c) for c in counts]
        used = sum(counts)
        if used >= n:
            overflow = used - (n - 1)
            for i in range(len(counts) - 1, -1, -1):
                take = min(overflow, max(0, counts[i] - 1))
                counts[i] -= take
                overflow -= take
                if overflow <= 0:
                    break
        cuts = np.cumsum(counts)
        split_indices = np.split(idx, cuts)
        for bucket, split_idx in zip(parts, split_indices):
            bucket.extend(split_idx.tolist())
    return [pd.Index(bucket) for bucket in parts]


def stratified_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, val_idx = stratified_sample_indices(df, ["center_name", "label"], [1.0 - val_ratio, val_ratio], seed)
    train = df.loc[train_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = df.loc[val_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val


def write_split(split_dir: Path, train: pd.DataFrame, val: pd.DataFrame, external: pd.DataFrame, readme: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train_labels.csv", index=False, encoding="utf-8-sig")
    val.to_csv(split_dir / "val_labels.csv", index=False, encoding="utf-8-sig")
    external.to_csv(split_dir / "external_test_labels.csv", index=False, encoding="utf-8-sig")
    (split_dir / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")


def summarize(split_name: str, subset: str, df: pd.DataFrame, definition: str) -> dict[str, object]:
    return {
        "split_name": split_name,
        "subset": subset,
        "definition": definition,
        "n": int(len(df)),
        "positive": int(df["label"].sum()),
        "negative": int((df["label"] == 0).sum()),
        "positive_rate": round(float(df["label"].mean()), 4) if len(df) else np.nan,
        "centers": ",".join(sorted(df["center_name"].unique())) if len(df) else "",
        "center_codes": ",".join(sorted(df["center_code"].unique())) if len(df) else "",
        "min_oct_count": int(df["oct_count"].min()) if len(df) else 0,
        "median_oct_count": float(df["oct_count"].median()) if len(df) else 0,
        "min_col_count": int(df["col_count"].min()) if len(df) else 0,
        "median_col_count": float(df["col_count"].median()) if len(df) else 0,
    }


def build_enshi_target_adapted(df: pd.DataFrame) -> list[dict[str, object]]:
    split_name = "enshi_target_adapted_20_10_70"
    split_dir = OUT_ROOT / split_name
    source = df[df["center_name"].ne("Enshi")].copy()
    target = df[df["center_name"].eq("Enshi")].copy()

    source_train, source_val = stratified_train_val(source, val_ratio=0.20, seed=SEED)
    adapt_idx, target_val_idx, final_idx = stratified_sample_indices(
        target,
        ["label"],
        [0.20, 0.10, 0.70],
        seed=SEED,
    )
    target_adapt = target.loc[adapt_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    target_val = target.loc[target_val_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    final_external = target.loc[final_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train = pd.concat([source_train, target_adapt], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val = pd.concat([source_val, target_val], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    write_split(
        split_dir,
        train,
        val,
        final_external,
        """
        Target-center-adapted Enshi validation. Train includes non-Enshi source
        training cases plus 20% labeled Enshi adaptation cases. Validation
        includes source validation cases plus 10% labeled Enshi target-validation
        cases. Final external_test is the remaining 70% Enshi holdout. This is
        not strict center-held-out validation; report it as target-center adapted.
        """,
    )
    return [
        summarize(split_name, "train_source_plus_enshi_adaptation", train, "target_center_adapted"),
        summarize(split_name, "validation_source_plus_enshi_target_validation", val, "target_center_adapted"),
        summarize(split_name, "external_test_enshi_final_holdout", final_external, "target_center_adapted"),
        summarize(split_name, "enshi_adaptation_only", target_adapt, "target_center_adapted_audit"),
        summarize(split_name, "enshi_target_validation_only", target_val, "target_center_adapted_audit"),
    ]


def build_all_center_patient_holdout(df: pd.DataFrame) -> list[dict[str, object]]:
    split_name = "all_center_patient_holdout_70_10_20"
    train_idx, val_idx, test_idx = stratified_sample_indices(
        df,
        ["center_name", "label"],
        [0.70, 0.10, 0.20],
        seed=SEED,
    )
    train = df.loc[train_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val = df.loc[val_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test = df.loc[test_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    write_split(
        OUT_ROOT / split_name,
        train,
        val,
        test,
        """
        Multi-center patient-level held-out validation. All centers contribute to
        train, validation, and test through label/center-stratified patient-level
        splitting. This is not center-external validation; it estimates deployment
        performance when the model has seen each acquisition domain during training.
        """,
    )
    return [
        summarize(split_name, "train", train, "multi_center_patient_holdout"),
        summarize(split_name, "val", val, "multi_center_patient_holdout"),
        summarize(split_name, "external_test", test, "multi_center_patient_holdout"),
    ]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    df = df[df["full_multimodal_complete"].astype(bool)].copy().reset_index(drop=True)

    rows = []
    rows.extend(build_enshi_target_adapted(df))
    rows.extend(build_all_center_patient_holdout(df))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_ROOT / "target_adapted_validation_split_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "target_adapted_validation_split_summary.csv", index=False, encoding="utf-8-sig")

    policy = """
# Validation Definitions for Higher-AUC Experiments

These splits are designed to pursue stronger reported discrimination without
invalidating the strict Enshi external result.

- `enshi_target_adapted_20_10_70`: target-center-adapted validation. It may be
  used to support a deployment claim such as "after limited target-center
  adaptation"; it must not be described as strict center-held-out external
  validation.
- `all_center_patient_holdout_70_10_20`: multi-center patient-level held-out
  validation. It should be described as multi-center held-out diagnosis,
  not strict cross-center generalization.

The strict Enshi center-held-out 404-case test remains the most conservative
external-center stress test.
"""
    (OUT_ROOT / "README.md").write_text(policy.strip() + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote target-adapted validation splits to {OUT_ROOT}")


if __name__ == "__main__":
    main()
