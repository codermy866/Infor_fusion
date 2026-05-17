#!/usr/bin/env python3
"""Summarize Stage-1 contrastive transfer prediction CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from metrics_utils import aggregate_metric_table, summarize_prediction_dataframe


RESULT_ROOT = EXP_ROOT / "paper_revision" / "results" / "stage1_contrastive_transfer"
PRED_DIR = RESULT_ROOT / "predictions"
TABLE_DIR = EXP_ROOT / "paper_revision" / "tables"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs under {PRED_DIR}")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if "modality_setting" in df.columns:
        df = df[df["modality_setting"].fillna("none").eq("none")]
    if "input_corruption" in df.columns:
        df = df[df["input_corruption"].fillna("none").eq("none")]

    run_metrics = summarize_prediction_dataframe(df)
    aggregate = aggregate_metric_table(run_metrics)
    external = aggregate[aggregate["split"] == "external_test"].copy()
    external = external.sort_values("auc_mean", ascending=False).reset_index(drop=True)
    external.insert(0, "external_auc_rank", range(1, len(external) + 1))

    run_metrics.to_csv(TABLE_DIR / "stage1_contrastive_transfer_run_level_metrics.csv", index=False)
    aggregate.to_csv(TABLE_DIR / "stage1_contrastive_transfer_summary_metrics.csv", index=False)
    external.to_csv(TABLE_DIR / "stage1_contrastive_transfer_external_ranked_metrics.csv", index=False)
    print(external[["external_auc_rank", "method", "runs", "n", "auc", "auprc", "f1", "ece", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
