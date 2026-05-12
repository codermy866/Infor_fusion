#!/usr/bin/env python3
"""Build center-wise calibration and missing-modality summary tables."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
TABLE_DIR = PAPER_DIR / "tables"
PRED_DIR = PAPER_DIR / "results" / "predictions"

sys.path.insert(0, str(SCRIPT_DIR))
from metrics_utils import binary_metrics, read_prediction_files, select_threshold


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = read_prediction_files(PRED_DIR)
    if df.empty:
        print("No prediction files available for center-wise calibration.")
        return

    rows = []
    for (method, run_id, seed), run_df in df.groupby(["method", "run_id", "seed"], dropna=False):
        source = run_df[run_df["split"] == "internal_validation"]
        threshold = select_threshold(source["y_true"], source["y_prob"]) if not source.empty else select_threshold(run_df["y_true"], run_df["y_prob"])
        external = run_df[run_df["split"] == "external_test"]
        if external.empty or "center" not in external:
            continue
        for center, center_df in external.groupby("center", dropna=False):
            if center_df["y_true"].nunique() < 2:
                metric = binary_metrics(center_df["y_true"], center_df["y_prob"], threshold=threshold)
                auc = float("nan")
                auprc = float("nan")
            else:
                metric = binary_metrics(center_df["y_true"], center_df["y_prob"], threshold=threshold)
                auc = metric.auc
                auprc = metric.auprc
            rows.append(
                {
                    "method": method,
                    "run_id": run_id,
                    "seed": seed,
                    "center": center,
                    "n": metric.n,
                    "positive": metric.positives,
                    "negative": metric.negatives,
                    "auc": auc,
                    "auprc": auprc,
                    "sensitivity": metric.sensitivity,
                    "specificity": metric.specificity,
                    "npv": metric.npv,
                    "ece": metric.ece,
                    "brier": metric.brier,
                    "threshold": threshold,
                }
            )

    centerwise = pd.DataFrame(rows)
    centerwise.to_csv(TABLE_DIR / "centerwise_calibration_metrics.csv", index=False)
    if not centerwise.empty:
        display = centerwise.copy()
        for col in ["auc", "auprc", "sensitivity", "specificity", "npv", "ece", "brier", "threshold"]:
            display[col] = display[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
        display[["method", "center", "n", "positive", "negative", "auc", "sensitivity", "specificity", "npv", "ece", "brier"]].to_csv(
            TABLE_DIR / "centerwise_calibration_metrics_formatted.csv",
            index=False,
        )

    missing = df[df.get("modality_setting", "none").astype(str) != "none"].copy()
    if not missing.empty:
        rows = []
        for (method, setting), group in missing.groupby(["method", "modality_setting"], dropna=False):
            metric = binary_metrics(group["y_true"], group["y_prob"])
            rows.append(
                {
                    "method": method,
                    "setting": setting,
                    "n": metric.n,
                    "auc": metric.auc,
                    "sensitivity": metric.sensitivity,
                    "specificity": metric.specificity,
                    "npv": metric.npv,
                    "ece": metric.ece,
                    "brier": metric.brier,
                }
            )
        missing_table = pd.DataFrame(rows)
        missing_table.to_csv(TABLE_DIR / "missing_modality_robustness_metrics.csv", index=False)
        display = missing_table.copy()
        for col in ["auc", "sensitivity", "specificity", "npv", "ece", "brier"]:
            display[col] = display[col].map(lambda x: f"{x:.3f}")
        display.to_csv(TABLE_DIR / "missing_modality_robustness_metrics_formatted.csv", index=False)

    print(f"Wrote center-wise calibration tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
