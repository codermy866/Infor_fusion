#!/usr/bin/env python3
"""Summarize cervix-domain adaptation prediction CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from metrics_utils import aggregate_metric_table, summarize_prediction_dataframe


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


def read_predictions() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs under {PRED_DIR}")
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


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = read_predictions()
    run_metrics = summarize_prediction_dataframe(df)
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

    run_metrics.to_csv(TABLE_DIR / "cervix_domain_adaptation_run_level_metrics.csv", index=False)
    aggregate.to_csv(TABLE_DIR / "cervix_domain_adaptation_summary_metrics.csv", index=False)
    external.to_csv(TABLE_DIR / "cervix_domain_adaptation_external_ranked_metrics.csv", index=False)
    (TABLE_DIR / "cervix_domain_adaptation_external_ranked_metrics.tex").write_text(latex_table(external), encoding="utf-8")

    print(f"Wrote {TABLE_DIR / 'cervix_domain_adaptation_run_level_metrics.csv'}")
    print(f"Wrote {TABLE_DIR / 'cervix_domain_adaptation_summary_metrics.csv'}")
    print(f"Wrote {TABLE_DIR / 'cervix_domain_adaptation_external_ranked_metrics.csv'}")
    print(external[["external_auc_rank", "method", "runs", "n", "auc", "auprc", "f1", "ece", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
