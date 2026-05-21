#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build locked-threshold metric tables from prediction CSVs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from metrics_utils import binary_metrics, read_prediction_files, select_threshold


def _clean_predictions(df: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "run_id", "seed", "split", "y_true", "y_prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction files missing required columns: {missing}")
    modality = df["modality_setting"].fillna("none").astype(str) if "modality_setting" in df else pd.Series("none", index=df.index)
    corruption = df["input_corruption"].fillna("none").astype(str) if "input_corruption" in df else pd.Series("none", index=df.index)
    return df[modality.eq("none") & corruption.eq("none")].copy()


def build_tables(pred_dir: Path, output_dir: Path, threshold_rule: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _clean_predictions(read_prediction_files(pred_dir))
    if df.empty:
        raise ValueError(f"No clean prediction CSVs found in {pred_dir}")

    threshold_rows = []
    internal_rows = []
    external_rows = []
    for (method, run_id, seed), run_df in df.groupby(["method", "run_id", "seed"], dropna=False):
        internal = run_df[run_df["split"].isin(["internal_validation", "val"])]
        external = run_df[run_df["split"].isin(["external_test", "external"])]
        if internal.empty:
            continue
        threshold = select_threshold(internal["y_true"], internal["y_prob"], threshold_rule)
        threshold_rows.append(
            {
                "method": method,
                "run_id": run_id,
                "seed": seed,
                "threshold_rule": threshold_rule,
                "locked_threshold": threshold,
                "threshold_source_split": "internal_validation",
                "external_reselected": False,
            }
        )
        internal_metric = binary_metrics(internal["y_true"], internal["y_prob"], threshold=threshold)
        internal_rows.append({"method": method, "run_id": run_id, "seed": seed, "split": "internal_validation", **asdict(internal_metric)})
        if not external.empty:
            external_metric = binary_metrics(external["y_true"], external["y_prob"], threshold=threshold)
            external_rows.append({"method": method, "run_id": run_id, "seed": seed, "split": "external_test", **asdict(external_metric)})

    threshold_table = pd.DataFrame(threshold_rows)
    internal_table = pd.DataFrame(internal_rows)
    external_table = pd.DataFrame(external_rows)
    formatted = external_table.copy()
    for col in ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier", "threshold"]:
        if col in formatted:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")

    paths = {
        "locked_thresholds": output_dir / "locked_thresholds_by_run.csv",
        "internal_metrics": output_dir / "internal_validation_metrics_locked_threshold.csv",
        "external_metrics": output_dir / "external_test_metrics_locked_threshold.csv",
        "formatted_main": output_dir / "formatted_locked_threshold_main_table.csv",
    }
    threshold_table.to_csv(paths["locked_thresholds"], index=False, encoding="utf-8-sig")
    internal_table.to_csv(paths["internal_metrics"], index=False, encoding="utf-8-sig")
    external_table.to_csv(paths["external_metrics"], index=False, encoding="utf-8-sig")
    formatted.to_csv(paths["formatted_main"], index=False, encoding="utf-8-sig")
    return paths


def run_toy_test() -> None:
    y_val = [1, 1, 1, 0, 0, 0]
    p_val = [0.96, 0.80, 0.70, 0.60, 0.40, 0.10]
    threshold = select_threshold(y_val, p_val, "max_specificity_at_sensitivity:0.95")
    y_ext = [1, 0, 0]
    p_ext = [0.75, 0.50, 0.20]
    metric = binary_metrics(y_ext, p_ext, threshold=threshold)
    assert threshold in set(p_val)
    assert metric.threshold == threshold
    print(f"toy_test_passed locked_threshold={threshold:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "tables")
    parser.add_argument("--threshold-rule", default="max_specificity_at_sensitivity:0.95")
    parser.add_argument("--toy-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.toy_test:
        run_toy_test()
    paths = build_tables(args.pred_dir, args.output_dir, args.threshold_rule)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
