#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from metrics_utils import confusion_counts, decision_curve_points


def lookup_threshold(row: pd.Series, table: pd.DataFrame) -> float:
    match = table[
        table["method"].astype(str).eq(str(row["method"]))
        & table["run_id"].astype(str).eq(str(row["run_id"]))
        & table["seed"].astype(str).eq(str(row["seed"]))
    ]
    if match.empty:
        return 0.5
    return float(match["locked_threshold"].iloc[0])


def per_1000_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    n = max(len(y_true), 1)
    scale = 1000.0 / n
    predicted_positive = tp + fp
    return {
        "predicted_positive_per_1000": predicted_positive * scale,
        "predicted_negative_per_1000": (tn + fn) * scale,
        "true_positive_detected_per_1000": tp * scale,
        "false_positive_referral_per_1000": fp * scale,
        "false_negative_missed_per_1000": fn * scale,
        "low_yield_referral_per_1000": fp * scale,
        "number_needed_to_refer_for_one_positive": float(predicted_positive / tp) if tp else float("inf"),
        "net_benefit": (tp / n) - (fp / n) * (threshold / max(1.0 - threshold, 1e-12)),
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, seed: int, n_bootstrap: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    metrics = []
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    for _ in range(n_bootstrap):
        if len(pos) and len(neg):
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        else:
            idx = rng.choice(np.arange(len(y_true)), len(y_true), replace=True)
        metrics.append(per_1000_metrics(y_true[idx], y_prob[idx], threshold))
    out = {}
    for key in metrics[0]:
        vals = np.asarray([m[key] for m in metrics], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"{key}_ci_low"] = float(np.percentile(vals, 2.5)) if len(vals) else float("nan")
        out[f"{key}_ci_high"] = float(np.percentile(vals, 97.5)) if len(vals) else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", default=[str(EXP_ROOT / "paper_revision" / "results" / "predictions" / "*.csv")])
    parser.add_argument("--threshold-table", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "tables" / "locked_thresholds_by_run.csv")
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "paper_revision" / "results" / "clinical_decision")
    parser.add_argument("--split", default="external_test")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for pattern in args.predictions for path in glob.glob(pattern)]
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    thresholds = pd.read_csv(args.threshold_table) if args.threshold_table.exists() else pd.DataFrame(columns=["method", "run_id", "seed", "locked_threshold"])
    rows = []
    dca_rows = []
    for keys, group in df[df.get("split", "").eq(args.split)].groupby(["method", "run_id", "seed"], dropna=False):
        method, run_id, seed = keys
        ref = pd.Series({"method": method, "run_id": run_id, "seed": seed})
        threshold = lookup_threshold(ref, thresholds)
        y_true = group["y_true"].astype(int).to_numpy()
        y_prob = group["y_prob"].astype(float).to_numpy()
        metric = per_1000_metrics(y_true, y_prob, threshold)
        metric.update(bootstrap_ci(y_true, y_prob, threshold, args.seed, args.n_bootstrap))
        rows.append({"method": method, "run_id": run_id, "seed": seed, "split": args.split, "threshold": threshold, "n": len(group), **metric})
        dca = decision_curve_points(y_true, y_prob)
        dca["method"] = method
        dca["run_id"] = run_id
        dca["seed"] = seed
        dca_rows.append(dca)
    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "clinical_decision_per_1000_table.csv", index=False, encoding="utf-8-sig")
    pd.concat(dca_rows, ignore_index=True).to_csv(args.output_dir / "decision_curve_points.csv", index=False, encoding="utf-8-sig") if dca_rows else pd.DataFrame().to_csv(args.output_dir / "decision_curve_points.csv", index=False)
    formatted = out.copy()
    for col in formatted.columns:
        if col.endswith("_per_1000") or col in {"number_needed_to_refer_for_one_positive", "net_benefit", "threshold"}:
            formatted[col] = pd.to_numeric(formatted[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}")
    formatted.to_csv(args.output_dir / "formatted_clinical_utility_table.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote clinical decision quality outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
