#!/usr/bin/env python3
"""Recompute external-test metrics on the official Jingzhou+Shiyan (n=148) subset.

Several runs export `external_test` predictions for a wider all-center holdout
(n≈196). For Information Fusion reporting, discrimination and calibration on the
fixed official external set should use only centers {Jingzhou, Shiyan}.

Threshold selection remains on the full internal-validation split for each
(method, run_id, seed), matching `experiment_registry.json` threshold_policy.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
TABLE_DIR = PAPER_DIR / "tables"
PRED_DIR = PAPER_DIR / "results" / "predictions"

OFFICIAL_CENTERS = frozenset({"Jingzhou", "Shiyan"})

sys.path.insert(0, str(SCRIPT_DIR))
from metrics_utils import binary_metrics, select_threshold


def _is_full_modality(df: pd.DataFrame) -> pd.Series:
    if "modality_setting" not in df.columns:
        return pd.Series(True, index=df.index)
    s = df["modality_setting"].astype(str).str.lower()
    return s.isin(["none", "nan", "full", ""]) | df["modality_setting"].isna()


def collect_run_frames() -> dict[tuple[str, object, object], dict[str, pd.DataFrame]]:
    """Map (method, run_id, seed) -> {'internal_validation': df, 'external_test': df}."""
    buckets: dict[tuple[str, object, object], dict[str, pd.DataFrame]] = {}
    for path in sorted(PRED_DIR.glob("*.csv")):
        name = path.name
        if "internal_validation_full" not in name and "external_test_full" not in name:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df[_is_full_modality(df)].copy()
        method = str(df["method"].iloc[0])
        run_id = df["run_id"].iloc[0] if "run_id" in df.columns else ""
        seed = df["seed"].iloc[0] if "seed" in df.columns else ""
        split = str(df["split"].iloc[0])
        key = (method, run_id, seed)
        buckets.setdefault(key, {})[split] = df
    return buckets


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (method, run_id, seed), splits in collect_run_frames().items():
        internal = splits.get("internal_validation")
        external = splits.get("external_test")
        if internal is None or external is None:
            continue
        if "center" not in external.columns:
            continue
        thr = select_threshold(internal["y_true"], internal["y_prob"], "youden")
        ext_off = external[external["center"].astype(str).isin(OFFICIAL_CENTERS)].copy()
        if ext_off.empty:
            continue
        bundle = binary_metrics(ext_off["y_true"], ext_off["y_prob"], threshold=thr)
        rows.append(
            {
                "method": method,
                "run_id": run_id,
                "seed": seed,
                "n_official_external": bundle.n,
                "positives": bundle.positives,
                "negatives": bundle.negatives,
                "threshold_internal_youden": thr,
                **{k: getattr(bundle, k) for k in ("auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier")},
            }
        )

    out = pd.DataFrame(rows)
    out.sort_values(["method", "seed", "run_id"], inplace=True)
    out_path = TABLE_DIR / "main_performance_official_external_148_aligned.csv"
    out.to_csv(out_path, index=False)

    # Aggregate duplicate seeds for manuscript summary
    agg_rows = []
    metric_cols = ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier"]
    for method, group in out.groupby("method"):
        agg_rows.append(
            {
                "method": method,
                "runs": len(group),
                "n_official_external": int(group["n_official_external"].median()),
                **{f"{m}_mean": float(group[m].mean()) for m in metric_cols},
                **{f"{m}_std": float(group[m].std(ddof=1)) if len(group) > 1 else 0.0 for m in metric_cols},
            }
        )
    agg = pd.DataFrame(agg_rows).sort_values("auc_mean", ascending=False)
    agg.to_csv(TABLE_DIR / "main_performance_official_external_148_aligned_aggregate.csv", index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    print(f"Wrote {TABLE_DIR / 'main_performance_official_external_148_aligned_aggregate.csv'} ({len(agg)} methods)")


if __name__ == "__main__":
    main()
