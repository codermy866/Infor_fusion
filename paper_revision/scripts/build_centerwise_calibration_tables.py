#!/usr/bin/env python3
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_prediction_files(args.pred_dir)
    rows = []
    if not df.empty and "center" in df:
        clean = df[(df.get("modality_setting", "none").fillna("none") == "none") & (df.get("input_corruption", "none").fillna("none") == "none")]
        for (method, run_id, seed), run_df in clean.groupby(["method", "run_id", "seed"], dropna=False):
            internal = run_df[run_df["split"].eq("internal_validation")]
            threshold = select_threshold(internal["y_true"], internal["y_prob"], "max_specificity_at_sensitivity:0.95") if not internal.empty else 0.5
            external = run_df[run_df["split"].eq("external_test")]
            for center, group in external.groupby("center", dropna=False):
                metric = binary_metrics(group["y_true"], group["y_prob"], threshold=threshold)
                rows.append({"method": method, "run_id": run_id, "seed": seed, "center": center, **asdict(metric)})
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "centerwise_ece_brier_table.csv", index=False, encoding="utf-8-sig")
    summary = out.groupby("center", dropna=False)[["ece", "brier"]].agg(["mean", "std", "count"]).reset_index() if not out.empty else pd.DataFrame()
    summary.to_csv(args.output_dir / "centerwise_reliability_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote center-wise calibration rows: {len(out)}")


if __name__ == "__main__":
    main()
