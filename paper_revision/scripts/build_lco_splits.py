#!/usr/bin/env python3
"""Create leave-one-center-out split CSVs for additional cross-center validation.

The original external test split remains unchanged. These folds are meant as
supplementary robustness evidence, not as a replacement for the official held-out
external test.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data2/hmy_pri/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out")
OUT_ROOT = EXP_ROOT / "paper_revision" / "splits" / "leave_one_center_out"

CENTER_NAMES = {
    "M22105": "Enshi",
    "M20105": "Wuda",
    "M20203": "Wuda",
    "M22102": "Xiangyang",
    "M0008": "Jingzhou",
    "M22101": "Jingzhou",
    "M22104": "Shiyan",
}


def center_from_oct(oct_id: str) -> str:
    code = str(oct_id).split("_")[0]
    return CENTER_NAMES.get(code, code)


def stratified_val_split(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    val_indices = []
    for _, group in df.groupby("label", dropna=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction))) if len(indices) > 1 else 0
        val_indices.extend(indices[:n_val].tolist())
    val_set = set(val_indices)
    val_df = df.loc[sorted(val_set)].copy()
    train_df = df.drop(index=sorted(val_set)).copy()
    return train_df, val_df


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ["train_labels.csv", "val_labels.csv", "external_test_labels.csv"]:
        df = pd.read_csv(DATA_ROOT / name)
        df["source_split"] = name.replace("_labels.csv", "")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["center"] = all_df["OCT"].map(center_from_oct)

    summary_rows = []
    for center in sorted(all_df["center"].unique()):
        fold_dir = OUT_ROOT / center
        fold_dir.mkdir(parents=True, exist_ok=True)
        external_df = all_df[all_df["center"] == center].copy()
        development_df = all_df[all_df["center"] != center].copy()
        train_df, val_df = stratified_val_split(development_df, val_fraction=0.2, seed=42)

        keep_cols = ["ID", "OCT", "AGE", "HPV清洗", "TCT清洗", "label"]
        train_df[keep_cols].to_csv(fold_dir / "train_labels.csv", index=False)
        val_df[keep_cols].to_csv(fold_dir / "val_labels.csv", index=False)
        external_df[keep_cols].to_csv(fold_dir / "external_test_labels.csv", index=False)

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("external", external_df),
        ]:
            summary_rows.append(
                {
                    "held_out_center": center,
                    "split": split_name,
                    "n": int(len(split_df)),
                    "positive": int(split_df["label"].sum()),
                    "negative": int(len(split_df) - split_df["label"].sum()),
                    "positive_rate": float(split_df["label"].mean()) if len(split_df) else 0.0,
                    "binary_auc_valid": bool(split_df["label"].nunique() == 2),
                    "centers": ",".join(sorted(split_df["center"].unique())),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_ROOT / "lco_split_summary.csv", index=False)
    (OUT_ROOT / "README.md").write_text(
        "# Leave-One-Center-Out Splits\n\n"
        "These supplementary folds address the small official external test set by "
        "evaluating center holdout robustness across all five centers. The official "
        "148-subject external test set is not modified.\n\n"
        "Folds with a single external class should not be used for ROC-AUC; report "
        "thresholded metrics for those folds or exclude them from mean AUC summaries "
        "with an explicit note.\n",
        encoding="utf-8",
    )
    print(f"Wrote LCO splits to {OUT_ROOT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
