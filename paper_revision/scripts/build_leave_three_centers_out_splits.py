#!/usr/bin/env python3
"""Build deterministic two-source / three-external-center splits.

The resulting protocol is stricter than patient-level holdout: all patients
from three hospitals are held out for external testing, while the remaining
two hospitals are split into train/internal-validation subsets.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    EXP_ROOT
    / "paper_revision"
    / "splits"
    / "full_multimodal_resplit"
    / "full_multimodal_all_cases_audit.csv"
)
DEFAULT_OUTPUT = (
    EXP_ROOT
    / "paper_revision"
    / "splits"
    / "full_multimodal_resplit"
    / "leave_three_centers_out"
)


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def joined(values: Iterable[str]) -> str:
    return "+".join(sorted(str(item) for item in values))


def stratified_internal_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split internal-center rows by center and label with a fixed seed."""
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []

    for (_, _), group in df.groupby(["center_name", "label"], sort=True):
        indices = group.index.to_numpy()
        if len(indices) <= 1:
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_fraction))
        if len(shuffled) >= 5:
            val_count = max(1, val_count)
        val_count = min(max(0, val_count), len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count].tolist())

    val_index = pd.Index(sorted(val_indices))
    val_df = df.loc[val_index].copy()
    train_df = df.drop(index=val_index).copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def subset_summary(split_name: str, subset: str, df: pd.DataFrame) -> dict[str, object]:
    labels = df["label"].astype(int)
    centers = sorted(df["center_name"].dropna().astype(str).unique())
    codes = sorted(df["center_code"].dropna().astype(str).unique())
    positive = int(labels.sum())
    negative = int((labels == 0).sum())
    return {
        "split_name": split_name,
        "subset": subset,
        "n": int(len(df)),
        "positive": positive,
        "negative": negative,
        "positive_rate": round(float(labels.mean()), 4) if len(df) else np.nan,
        "centers": ",".join(centers),
        "center_codes": ",".join(codes),
        "single_class_centers": ",".join(
            center for center, group in df.groupby("center_name") if group["label"].nunique() < 2
        )
        or "-",
        "min_oct_count": int(df["oct_count"].min()) if "oct_count" in df and len(df) else np.nan,
        "median_oct_count": float(df["oct_count"].median()) if "oct_count" in df and len(df) else np.nan,
        "min_col_count": int(df["col_count"].min()) if "col_count" in df and len(df) else np.nan,
        "median_col_count": float(df["col_count"].median()) if "col_count" in df and len(df) else np.nan,
    }


def write_readme(
    path: Path,
    split_name: str,
    internal_centers: tuple[str, ...],
    external_centers: tuple[str, ...],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> None:
    text = f"""# {split_name}

Protocol: two-source-center training with three completely held-out external centers.

- Internal centers: {joined(internal_centers)}
- External centers: {joined(external_centers)}
- Train: n={len(train_df)}
- Internal validation: n={len(val_df)}
- External test: n={len(external_df)}

The internal train/validation split is deterministic and stratified within each
available `(center_name, label)` group. No external-center patient is used for
training, threshold selection, checkpoint selection, or hyperparameter tuning.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument(
        "--only",
        default="",
        help="Optional internal-center pair, e.g. Enshi+Wuda. By default all pairs are generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"center_name", "center_code", "label", "patient_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    centers = tuple(sorted(df["center_name"].dropna().astype(str).unique()))
    requested = {item.strip() for item in args.only.split("+") if item.strip()}

    summary_rows: list[dict[str, object]] = []
    combo_rows: list[dict[str, object]] = []

    for internal_centers in combinations(centers, 2):
        internal_set = set(internal_centers)
        if requested and internal_set != requested:
            continue
        external_centers = tuple(center for center in centers if center not in internal_set)
        split_name = "internal_" + "_".join(safe_name(center) for center in internal_centers)
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        internal_df = df[df["center_name"].isin(internal_centers)].copy()
        external_df = df[df["center_name"].isin(external_centers)].copy().reset_index(drop=True)
        train_df, val_df = stratified_internal_split(internal_df, args.val_fraction, args.seed)

        train_df.to_csv(split_dir / "train_labels.csv", index=False)
        val_df.to_csv(split_dir / "val_labels.csv", index=False)
        external_df.to_csv(split_dir / "external_test_labels.csv", index=False)
        write_readme(split_dir / "README.md", split_name, internal_centers, external_centers, train_df, val_df, external_df)

        for subset, subset_df in (
            ("train", train_df),
            ("internal_validation", val_df),
            ("external_test", external_df),
        ):
            summary_rows.append(subset_summary(split_name, subset, subset_df))

        internal_labels = internal_df["label"].astype(int)
        external_labels = external_df["label"].astype(int)
        combo_rows.append(
            {
                "split_name": split_name,
                "internal_centers": joined(internal_centers),
                "external_centers": joined(external_centers),
                "internal_n": int(len(internal_df)),
                "internal_positive": int(internal_labels.sum()),
                "internal_negative": int((internal_labels == 0).sum()),
                "internal_positive_rate": round(float(internal_labels.mean()), 4),
                "external_n": int(len(external_df)),
                "external_positive": int(external_labels.sum()),
                "external_negative": int((external_labels == 0).sum()),
                "external_positive_rate": round(float(external_labels.mean()), 4),
                "external_single_class_centers": ",".join(
                    center for center, group in external_df.groupby("center_name") if group["label"].nunique() < 2
                )
                or "-",
            }
        )

    summary = pd.DataFrame(summary_rows)
    combos = pd.DataFrame(combo_rows)
    if combos.empty:
        raise ValueError("No splits generated; check --only center names.")

    combos = combos.sort_values(
        ["external_single_class_centers", "external_n", "internal_n"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    combos.to_csv(output_dir / "leave_three_centers_out_combo_summary.csv", index=False)
    summary.to_csv(output_dir / "leave_three_centers_out_split_summary.csv", index=False)

    print(f"Wrote splits under {output_dir}")
    print(combos.to_string(index=False))


if __name__ == "__main__":
    main()
