#!/usr/bin/env python3
"""Build final paper-ready tables comparing baseline, ablation, and 100-ep production."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

OUT_ROOT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897"
FINAL_ROOT = OUT_ROOT / "final_production_100ep"
BASELINE_RAW = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv"
)
PRUNE_DECISIONS = OUT_ROOT / "tables/Table_Ablation_Prune_Decisions.csv"
PRUNED_JSON = OUT_ROOT / "tables/Table_Pruned_Production_Defaults.json"


def load_fold_summary(label: str, metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    auc = df["cin2plus_auc"].dropna()
    return {
        "protocol": label,
        "epochs": None,
        "mean_cin2plus_auc": float(auc.mean()) if len(auc) else None,
        "std_cin2plus_auc": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
        "n_valid_folds": int(len(auc)),
        "mean_cin2plus_sensitivity": float(df["cin2plus_sensitivity"].dropna().mean()) if "cin2plus_sensitivity" in df else None,
        "mean_cin2plus_specificity": float(df["cin2plus_specificity"].dropna().mean()) if "cin2plus_specificity" in df else None,
        "source": str(metrics_path),
    }


def per_center_table() -> pd.DataFrame:
    rows = []
    sources = [
        ("baseline_formal_raw_20ep", BASELINE_RAW),
        ("ablation_g1_20ep", OUT_ROOT / "ablations/g1/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"),
        ("production_pruned_100ep", FINAL_ROOT / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"),
    ]
    for protocol, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "protocol": protocol,
                    "held_out_center": r["held_out_center"],
                    "n_test": r.get("n_test"),
                    "cin2plus_auc": r.get("cin2plus_auc"),
                    "cin2plus_sensitivity": r.get("cin2plus_sensitivity"),
                    "cin2plus_specificity": r.get("cin2plus_specificity"),
                    "cin3plus_auc": r.get("cin3plus_auc"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    table_dir = OUT_ROOT / "tables"
    report_dir = OUT_ROOT / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for label, path, ep in [
        ("baseline_formal_raw", BASELINE_RAW, 20),
        ("ablation_control_g1", OUT_ROOT / "ablations/g1/tables/Table_Improved1897_LOCO_Fold_Metrics.csv", 20),
        ("production_pruned_final", FINAL_ROOT / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv", 100),
    ]:
        row = load_fold_summary(label, path)
        if row:
            row["epochs"] = ep
            summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    summary_path = table_dir / "Table_Final_1897_LOCO_Summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    center_df = per_center_table()
    center_path = table_dir / "Table_Final_1897_LOCO_PerCenter.csv"
    if len(center_df):
        center_df.to_csv(center_path, index=False, encoding="utf-8-sig")
        pivot = center_df.pivot_table(index="held_out_center", columns="protocol", values="cin2plus_auc", aggfunc="first")
        pivot.to_csv(table_dir / "Table_Final_1897_LOCO_PerCenter_Pivot.csv", encoding="utf-8-sig")

    kept_modules = []
    removed_modules = []
    if PRUNE_DECISIONS.exists():
        dec = pd.read_csv(PRUNE_DECISIONS)
        kept_modules = dec[dec["action"] == "keep"]["component"].tolist()
        removed_modules = dec[dec["action"] == "remove"]["component"].tolist()

    pruned_cfg = {}
    if PRUNED_JSON.exists():
        pruned_cfg = json.loads(PRUNED_JSON.read_text(encoding="utf-8"))

    prod = summary_df[summary_df["protocol"] == "production_pruned_final"]
    prod_auc = float(prod.iloc[0]["mean_cin2plus_auc"]) if len(prod) else None

    lines = [
        "# 1897 最终实验汇总（消融裁剪 + 100 epoch 生产运行）",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 主结果",
        "",
    ]
    if prod_auc is not None:
        lines.append(f"**Production (100 ep, pruned stack) CIN2+ LOCO AUC = {prod_auc:.4f}**")
        lines.append("")
    lines.extend(
        [
            "## 协议对比",
            "",
            summary_df.to_markdown(index=False) if len(summary_df) else "_No summary yet._",
            "",
            "## 保留模块",
            "",
            ", ".join(kept_modules) if kept_modules else "_See prune decisions._",
            "",
            "## 移除模块",
            "",
            ", ".join(removed_modules) if removed_modules else "_See prune decisions._",
            "",
            "## 生产配置",
            "",
            "```json",
            json.dumps(pruned_cfg, indent=2, ensure_ascii=False) if pruned_cfg else "{}",
            "```",
            "",
        ]
    )
    if len(center_df):
        lines.extend(["## 逐中心 AUC", "", center_df.pivot_table(index="held_out_center", columns="protocol", values="cin2plus_auc").to_markdown(), ""])

    report_path = report_dir / "Report_Final_1897_Experiment.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
