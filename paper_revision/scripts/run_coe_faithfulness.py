#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CoE faithfulness proxy summaries without report generation.")
    parser.add_argument("--pred-dir", type=Path, default=ROOT / "paper_revision" / "results" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_revision" / "results" / "coe_faithfulness")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.pred_dir.glob("*.csv"))
    frames = [pd.read_csv(path) for path in files] if files else []
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        pd.DataFrame().to_csv(args.output_dir / "coe_faithfulness_proxy_metrics.csv", index=False)
        return

    rows = []
    if "modality_setting" in df:
        for setting, group in df.groupby(df["modality_setting"].fillna("none")):
            rows.append({"experiment": "evidence_removal", "condition": setting, "n": len(group), "mean_y_prob": group["y_prob"].mean()})
    if {"coe_z0_norm", "coe_z1_norm", "coe_z2_norm", "coe_z3_norm"}.issubset(df.columns):
        for split, group in df.groupby("split", dropna=False):
            rows.append(
                {
                    "experiment": "posterior_trajectory",
                    "condition": split,
                    "n": len(group),
                    "mean_z0_norm": pd.to_numeric(group["coe_z0_norm"], errors="coerce").mean(),
                    "mean_z3_norm": pd.to_numeric(group["coe_z3_norm"], errors="coerce").mean(),
                }
            )
    wrong = df[(df["y_prob"].ge(0.5).astype(int) != df["y_true"].astype(int))].copy()
    wrong.to_csv(args.output_dir / "failure_case_mining_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(args.output_dir / "coe_faithfulness_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    note = (
        "These analyses are faithfulness proxies and are not equivalent to clinical validation of generated explanations.\n"
        "No report generation or report supervision is used.\n"
    )
    (args.output_dir / "README.md").write_text(note, encoding="utf-8")
    print(f"Wrote CoE faithfulness proxy outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
