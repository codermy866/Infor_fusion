#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "paper_revision" / "results" / "coe_faithfulness")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_revision" / "results" / "tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = args.input_dir / "coe_faithfulness_proxy_metrics.csv"
    failures = args.input_dir / "failure_case_mining_candidates.csv"
    if metrics.exists():
        pd.read_csv(metrics).to_csv(args.output_dir / "coe_faithfulness_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    if failures.exists():
        failure_df = pd.read_csv(failures)
        failure_df.head(100).to_csv(args.output_dir / "coe_failure_case_mining_top100.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote CoE faithfulness tables to {args.output_dir}")


if __name__ == "__main__":
    main()
