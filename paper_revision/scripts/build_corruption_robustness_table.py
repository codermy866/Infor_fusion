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


def locked_thresholds(df: pd.DataFrame) -> dict[tuple[object, object, object], float]:
    thresholds: dict[tuple[object, object, object], float] = {}
    if df.empty:
        return thresholds
    modality = df["modality_setting"].fillna("none").astype(str) if "modality_setting" in df else pd.Series("none", index=df.index)
    corruption = df["input_corruption"].fillna("none").astype(str) if "input_corruption" in df else pd.Series("none", index=df.index)
    clean = df[modality.eq("none") & corruption.eq("none") & df["split"].eq("internal_validation")]
    for key, group in clean.groupby(["method", "run_id", "seed"], dropna=False):
        thresholds[key] = select_threshold(group["y_true"], group["y_prob"], "max_specificity_at_sensitivity:0.95")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_prediction_files(args.pred_dir)
    if df.empty or "input_corruption" not in df:
        pd.DataFrame().to_csv(args.output_dir / "input_corruption_robustness_metrics.csv", index=False)
        return
    thresholds = locked_thresholds(df)
    df = df[df["input_corruption"].fillna("none").astype(str).ne("none")].copy()
    rows = []
    for keys, group in df.groupby(["method", "run_id", "seed", "split", "input_corruption", "corruption_severity"], dropna=False):
        method, run_id, seed, split, corruption, severity = keys
        threshold = thresholds.get((method, run_id, seed), 0.5)
        metric = binary_metrics(group["y_true"], group["y_prob"], threshold=threshold)
        rows.append({
            "method": method,
            "run_id": run_id,
            "seed": seed,
            "split": split,
            "corruption": corruption,
            "severity": severity,
            "threshold_source_split": "internal_validation",
            **asdict(metric),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "input_corruption_robustness_metrics.csv", index=False, encoding="utf-8-sig")
    display = out.copy()
    for col in ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "ece", "brier"]:
        if col in display:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    display.to_csv(args.output_dir / "input_corruption_robustness_metrics_formatted.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} corruption rows")


if __name__ == "__main__":
    main()
