#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "paper_revision" / "tables" / "label_noise_stress_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_revision" / "results" / "tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        pd.DataFrame().to_csv(args.output_dir / "label_noise_metrics.csv", index=False)
        print(f"No label-noise metrics found: {args.input}")
        return
    df = pd.read_csv(args.input)
    df.to_csv(args.output_dir / "label_noise_metrics.csv", index=False, encoding="utf-8-sig")
    clean = df[df.get("noise_rate", 1).astype(float).eq(0.0)] if "noise_rate" in df else pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        if clean.empty or "seed" not in clean or "split" not in clean or "auc" not in clean:
            auc_base = float("nan")
        else:
            base = clean[
                clean["seed"].astype(str).eq(str(row.get("seed", "")))
                & clean["split"].astype(str).eq(str(row.get("split", "")))
            ]
            auc_base = float(base["auc"].iloc[0]) if not base.empty else float("nan")
        row_auc = pd.to_numeric(pd.Series([row.get("auc", float("nan"))]), errors="coerce").iloc[0]
        rows.append({**row.to_dict(), "auc_degradation_vs_clean": auc_base - row_auc})
    degradation = pd.DataFrame(rows)
    degradation.to_csv(args.output_dir / "label_noise_degradation_vs_clean.csv", index=False, encoding="utf-8-sig")
    formatted = degradation.copy()
    for col in ["auc", "auprc", "sensitivity", "specificity", "auc_degradation_vs_clean"]:
        if col in formatted:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    formatted.to_csv(args.output_dir / "formatted_label_noise_table.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote label-noise tables to {args.output_dir}")


if __name__ == "__main__":
    main()
