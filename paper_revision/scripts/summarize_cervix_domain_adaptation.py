#!/usr/bin/env python3
"""Summarize cervix-domain adaptation prediction CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from metrics_utils import aggregate_metric_table, binary_metrics, select_threshold, summarize_prediction_dataframe


RESULT_ROOT = EXP_ROOT / "paper_revision" / "results" / "cervix_domain_adaptation"
PRED_DIR = RESULT_ROOT / "predictions"
TABLE_DIR = EXP_ROOT / "paper_revision" / "tables"

METHOD_ORDER = [
    "CervixAdapt_StaticPrior",
    "CervixAdapt_VisualAdapterOnly",
    "CervixAdapt_BERTAdapterOnly",
    "CervixAdapt_VisualBERTAdapter",
    "CervixAdapt_BERTLastLayerFT",
    "CervixAdapt_VisualFullTextFT",
]


def read_predictions(prediction_dir: Path) -> pd.DataFrame:
    files = sorted(prediction_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs under {prediction_dir}")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if "modality_setting" in df.columns:
        df = df[df["modality_setting"].fillna("none").eq("none")]
    if "input_corruption" in df.columns:
        df = df[df["input_corruption"].fillna("none").eq("none")]
    return df


def latex_table(df: pd.DataFrame) -> str:
    cols = ["method", "runs", "n", "auc", "auprc", "sensitivity", "specificity", "npv", "f1", "ece", "brier"]
    display = df[cols].copy()
    display.columns = ["Method", "Runs", "n", "AUC", "AUPRC", "Sensitivity", "Specificity", "NPV", "F1", "ECE", "Brier"]
    return display.to_latex(index=False, escape=False)


def summarize_center_metrics(
    df: pd.DataFrame,
    threshold_rule: str = "youden",
    threshold_from_split: str = "internal_validation",
) -> pd.DataFrame:
    required = {"method", "run_id", "seed", "split", "center", "y_true", "y_prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction dataframe is missing columns for center metrics: {missing}")

    rows = []
    for (method, run_id, seed), run_df in df.groupby(["method", "run_id", "seed"], dropna=False):
        threshold_source = run_df[run_df["split"] == threshold_from_split]
        if threshold_source.empty:
            threshold_source = run_df
        threshold = select_threshold(threshold_source["y_true"], threshold_source["y_prob"], threshold_rule)

        external = run_df[run_df["split"] == "external_test"]
        for center, center_df in external.groupby("center", dropna=False):
            bundle = binary_metrics(center_df["y_true"], center_df["y_prob"], threshold=threshold)
            rows.append(
                {
                    "method": method,
                    "run_id": run_id,
                    "seed": seed,
                    "split": "external_test",
                    "center": center,
                    **bundle.__dict__,
                }
            )
    return pd.DataFrame(rows)


def aggregate_center_metric_table(center_metrics: pd.DataFrame) -> pd.DataFrame:
    if center_metrics.empty:
        return center_metrics
    metric_cols = ["auc", "auprc", "sensitivity", "specificity", "npv", "f1", "ece", "brier"]
    rows = []
    for (method, center), group in center_metrics.groupby(["method", "center"], dropna=False):
        row = {
            "method": method,
            "center": center,
            "runs": int(group.shape[0]),
            "n": int(group["n"].median()) if "n" in group else "",
            "positives": int(group["positives"].median()) if "positives" in group else "",
            "negatives": int(group["negatives"].median()) if "negatives" in group else "",
        }
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = "TBD" if pd.isna(mean) else f"{mean:.3f} +/- {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(RESULT_ROOT), help="Directory containing predictions/.")
    parser.add_argument("--prediction-dir", default="", help="Override prediction CSV directory.")
    parser.add_argument("--table-dir", default=str(TABLE_DIR), help="Directory for output tables.")
    parser.add_argument("--table-prefix", default="cervix_domain_adaptation", help="Output table filename prefix.")
    parser.add_argument("--threshold-rule", default="youden")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = EXP_ROOT / result_root
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else result_root / "predictions"
    if not prediction_dir.is_absolute():
        prediction_dir = EXP_ROOT / prediction_dir
    table_dir = Path(args.table_dir)
    if not table_dir.is_absolute():
        table_dir = EXP_ROOT / table_dir
    table_dir.mkdir(parents=True, exist_ok=True)

    df = read_predictions(prediction_dir)
    run_metrics = summarize_prediction_dataframe(df, threshold_rule=args.threshold_rule)
    run_metrics["method"] = pd.Categorical(run_metrics["method"], categories=METHOD_ORDER, ordered=True)
    run_metrics = run_metrics.sort_values(["split", "method", "seed"]).reset_index(drop=True)
    run_metrics["method"] = run_metrics["method"].astype(str)

    aggregate = aggregate_metric_table(run_metrics)
    aggregate["method"] = pd.Categorical(aggregate["method"], categories=METHOD_ORDER, ordered=True)
    aggregate = aggregate.sort_values(["split", "method"]).reset_index(drop=True)
    aggregate["method"] = aggregate["method"].astype(str)

    external = aggregate[aggregate["split"] == "external_test"].copy()
    external = external.sort_values("auc_mean", ascending=False).reset_index(drop=True)
    external.insert(0, "external_auc_rank", range(1, len(external) + 1))

    center_metrics = summarize_center_metrics(df, threshold_rule=args.threshold_rule)
    center_aggregate = aggregate_center_metric_table(center_metrics)
    if not center_aggregate.empty:
        center_aggregate["method"] = pd.Categorical(center_aggregate["method"], categories=METHOD_ORDER, ordered=True)
        center_aggregate = center_aggregate.sort_values(["method", "center"]).reset_index(drop=True)
        center_aggregate["method"] = center_aggregate["method"].astype(str)

    prefix = args.table_prefix
    run_metrics.to_csv(table_dir / f"{prefix}_run_level_metrics.csv", index=False)
    aggregate.to_csv(table_dir / f"{prefix}_summary_metrics.csv", index=False)
    external.to_csv(table_dir / f"{prefix}_external_ranked_metrics.csv", index=False)
    center_metrics.to_csv(table_dir / f"{prefix}_external_center_run_level_metrics.csv", index=False)
    center_aggregate.to_csv(table_dir / f"{prefix}_external_center_summary_metrics.csv", index=False)
    (table_dir / f"{prefix}_external_ranked_metrics.tex").write_text(latex_table(external), encoding="utf-8")

    print(f"Wrote {table_dir / f'{prefix}_run_level_metrics.csv'}")
    print(f"Wrote {table_dir / f'{prefix}_summary_metrics.csv'}")
    print(f"Wrote {table_dir / f'{prefix}_external_ranked_metrics.csv'}")
    print(f"Wrote {table_dir / f'{prefix}_external_center_summary_metrics.csv'}")
    print(external[["external_auc_rank", "method", "runs", "n", "auc", "auprc", "f1", "ece", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
