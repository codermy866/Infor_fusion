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
    if df.empty:
        pd.DataFrame().to_csv(args.output_dir / "loco_metrics_by_center.csv", index=False)
        return
    modality = df["modality_setting"].fillna("none").astype(str) if "modality_setting" in df else pd.Series("none", index=df.index)
    corruption = df["input_corruption"].fillna("none").astype(str) if "input_corruption" in df else pd.Series("none", index=df.index)
    df = df[modality.eq("none") & corruption.eq("none")].copy()
    rows = []
    for (method, run_id, seed), run_df in df.groupby(["method", "run_id", "seed"], dropna=False):
        internal = run_df[run_df["split"].eq("internal_validation")]
        threshold = select_threshold(internal["y_true"], internal["y_prob"], "max_specificity_at_sensitivity:0.95") if not internal.empty else 0.5
        external = run_df[run_df["split"].eq("external_test")]
        if external.empty or "center" not in external:
            continue
        for center, group in external.groupby("center", dropna=False):
            metric = binary_metrics(group["y_true"], group["y_prob"], threshold=threshold)
            row = {"method": method, "run_id": run_id, "seed": seed, "held_out_center": center, **asdict(metric)}
            if group["y_true"].nunique() < 2:
                row["auc"] = float("nan")
                row["auc_reason"] = "single_class_held_out_center"
            else:
                row["auc_reason"] = ""
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "loco_metrics_by_center.csv", index=False, encoding="utf-8-sig")
    display = out.copy()
    for col in ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "ece", "brier", "threshold"]:
        if col in display:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    display.to_csv(args.output_dir / "loco_formatted_table.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} LOCO/center rows")


if __name__ == "__main__":
    main()
