#!/usr/bin/env python3
"""Summarize full 1897 component ablation (all modules, g1 control)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts.shared_lora_biocot.run_1897_improved_loco import (  # noqa: E402
    ABLATION_PRESETS,
    FULL_ABLATION_SUITE,
)

OUT_ROOT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897"
ABLATION_ROOT = OUT_ROOT / "ablations"
BASELINE_RAW = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv"
)
CONTROL = "g1"


def load_group_summary(group: str) -> dict | None:
    metrics_path = ABLATION_ROOT / group / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    auc = df["cin2plus_auc"].dropna()
    preset = ABLATION_PRESETS.get(group, {})
    return {
        "group": group,
        "description": preset.get("description", group),
        "removed_components": preset.get("removed_components", ""),
        "mean_cin2plus_auc": float(auc.mean()) if len(auc) else None,
        "std_cin2plus_auc": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
        "n_valid_folds": int(len(auc)),
        "fold_metrics_path": str(metrics_path),
    }


def per_center_pivot() -> pd.DataFrame:
    rows = []
    if not ABLATION_ROOT.exists():
        return pd.DataFrame()
    for group_dir in sorted(ABLATION_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        metrics_path = group_dir / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        for _, row in df.iterrows():
            rows.append(
                {
                    "group": group_dir.name,
                    "held_out_center": row["held_out_center"],
                    "cin2plus_auc": row.get("cin2plus_auc"),
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df
    return long_df.pivot_table(index="held_out_center", columns="group", values="cin2plus_auc", aggfunc="first").sort_index()


def attribute_vs_control(comparison: pd.DataFrame, control_group: str = CONTROL) -> pd.DataFrame:
    control = comparison[comparison["group"] == control_group]
    if control.empty:
        return pd.DataFrame()
    control_auc = float(control.iloc[0]["mean_cin2plus_auc"])
    rows = []
    for _, row in comparison.iterrows():
        group = row["group"]
        if group in {control_group, "baseline_formal_raw", "g0"}:
            continue
        auc = row.get("mean_cin2plus_auc")
        if pd.isna(auc):
            continue
        delta = float(auc) - control_auc
        if delta > 0.01:
            verdict = "HARMFUL (remove from production)"
        elif delta < -0.01:
            verdict = "USEFUL (keep)"
        else:
            verdict = "NEUTRAL (candidate for removal)"
        rows.append(
            {
                "group": group,
                "removed_components": row.get("removed_components", ""),
                "mean_cin2plus_auc": float(auc),
                "delta_vs_g1": delta,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_vs_g1", ascending=False)


def main() -> None:
    table_dir = OUT_ROOT / "tables"
    report_dir = OUT_ROOT / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    pending = []
    for group in FULL_ABLATION_SUITE:
        summary = load_group_summary(group)
        if summary:
            summaries.append(summary)
        else:
            pending.append(group)

    if BASELINE_RAW.exists():
        base = pd.read_csv(BASELINE_RAW)
        base_auc = base["cin2plus_auc"].dropna()
        summaries.insert(
            0,
            {
                "group": "baseline_formal_raw",
                "description": "Original formal LOCO (raw images, all modules)",
                "removed_components": "none",
                "mean_cin2plus_auc": float(base_auc.mean()) if len(base_auc) else None,
                "std_cin2plus_auc": float(base_auc.std(ddof=1)) if len(base_auc) > 1 else 0.0,
                "n_valid_folds": int(len(base_auc)),
                "fold_metrics_path": str(BASELINE_RAW),
            },
        )

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(table_dir / "Table_Improved1897_Ablation_Comparison.csv", index=False, encoding="utf-8-sig")

    pivot = per_center_pivot()
    if not pivot.empty:
        pivot.to_csv(table_dir / "Table_Improved1897_Ablation_PerCenter_Pivot.csv", encoding="utf-8-sig")

    attribution = attribute_vs_control(comparison, control_group=CONTROL)
    if not attribution.empty:
        attribution.to_csv(table_dir / "Table_Improved1897_Component_Attribution.csv", index=False, encoding="utf-8-sig")

    design_rows = []
    for g in FULL_ABLATION_SUITE:
        p = ABLATION_PRESETS.get(g, {})
        design_rows.append(f"| {g} | {p.get('removed_components', '')} | {p.get('description', '')} |")

    lines = [
        "# 1897 全组件消融实验",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**对照组**: `{CONTROL}`（已关 L_adv）",
        "",
        "## 实验矩阵（leave-one-out，统一 cached 1897 LOCO）",
        "",
        "| Group | 移除组件 | 说明 |",
        "|-------|---------|------|",
        *design_rows,
        "",
        f"**待完成**: {', '.join(pending) if pending else '无'}",
        "",
        "## 汇总",
        "",
        comparison.to_markdown(index=False) if len(comparison) else "_尚无结果_",
        "",
    ]
    if not attribution.empty:
        lines.extend(["## 相对 g1 的组件归因", "", attribution.to_markdown(index=False), ""])
    if not pivot.empty:
        lines.extend(["## 逐中心 AUC", "", pivot.to_markdown(), ""])

    report_path = report_dir / "Report_1897_Component_Ablation.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {table_dir / 'Table_Improved1897_Ablation_Comparison.csv'}")
    print(f"Wrote {report_path}")
    if pending:
        print(f"Pending groups ({len(pending)}): {', '.join(pending)}")


if __name__ == "__main__":
    main()
