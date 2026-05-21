#!/usr/bin/env python3
"""Emit bilingual manuscript snippets from aligned official-external metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
TABLE_DIR = PAPER_DIR / "tables"
MAN_DIR = PAPER_DIR / "manuscript_sections"


def fmt3(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.3f}"


def main() -> None:
    MAN_DIR.mkdir(parents=True, exist_ok=True)
    agg_path = TABLE_DIR / "main_performance_official_external_148_aligned_aggregate.csv"
    if not agg_path.exists():
        raise SystemExit(f"Missing {agg_path}; run build_official_148_aligned_metrics.py first.")
    agg = pd.read_csv(agg_path)
    full148 = agg[agg["n_official_external"].eq(148)].sort_values("auc_mean", ascending=False)
    partial = agg[~agg["n_official_external"].eq(148)].sort_values("auc_mean", ascending=False)

    # Curated ordering for main text (only n=148 rows are comparable for headline ranking)
    preferred = [
        "HyDRA_Full_Pretrained",
        "Late_Fusion",
        "Concat_Fusion",
        "CrossAttention_Fusion",
        "Gated_Fusion",
        "HyDRA_Variational",
        "DirectLate_AllCenterPatientHoldout",
        "DirectLateAUCSelect_AllCenterPatientHoldout",
        "HyDRA_ELBO_StructuredPrior_AllCenter",
    ]
    rows = []
    for name in preferred:
        hit = full148[full148["method"].eq(name)]
        if not hit.empty:
            rows.append(hit.iloc[0])
    table_df = pd.DataFrame(rows) if rows else full148.head(12)
    top = full148.iloc[0] if not full148.empty else agg.sort_values("auc_mean", ascending=False).iloc[0]

    lines = [
        "# Paper-ready results (auto-generated)",
        "",
        "## Figure paths (this workspace)",
        "",
        "- Decision curve (official external, selected methods): `paper_revision/figures/decision_curve_external.png`",
        "- Missing-modality stress (selected methods): `paper_revision/figures/fig_missing_modality_robustness.png`",
        "- Input corruption stress (selected methods): `paper_revision/figures/fig_input_corruption_robustness.png`",
        "",
        "## English paragraph (Results, diagnostic performance — official 148 subset)",
        "",
    ]

    lines.append(
        "We report metrics on the fixed official external cohort (148 cases from Jingzhou and Shiyan). "
        "Operating thresholds were selected on the internal validation split using Youden's J rule and held fixed for external evaluation. "
        "Some exported `external_test` CSVs include additional centers; for headline discrimination numbers we therefore **re-evaluate** "
        "each run after filtering predictions to Jingzhou+Shiyan only. Methods whose exports omit most official-center cases receive "
        f"a reduced `n_official_external` and should be reported only as partial-coverage diagnostics. "
        f"Among runs with **full official coverage (n=148)**, the highest aligned external ROC-AUC was achieved by **{top['method']}** "
        f"(AUC={fmt3(top['auc_mean'])}, AUPRC={fmt3(top['auprc_mean'])}, sensitivity={fmt3(top['sensitivity_mean'])}, "
        f"specificity={fmt3(top['specificity_mean'])}, NPV={fmt3(top['npv_mean'])}, ECE={fmt3(top['ece_mean'])}, "
        f"Brier={fmt3(top['brier_mean'])}). "
        "Direct fusion baselines exported under the all-center holdout sometimes list only a subset of official-center subjects; "
        "re-exporting predictions for the strict 148-case cohort is recommended before claiming head-to-head parity. "
        "Missing-modality and input-corruption stress tests used the exported stress-test prediction files; "
        "training-time label-noise sweeps are reported from `label_noise_stress_metrics.csv` when that table is present."
    )
    lines.extend(["", "## 中文段落（结果—官方外部148子集）", ""])
    lines.append(
        "我们在固定的官方外部队列（荆州与十堰共148例）上报告指标；阈值在内部验证集上按Youden法则选取并固定用于外部测试。"
        "部分导出的 `external_test` 预测包含其他中心，因此正文主表对预测按荆州/十堰子集重算；若导出文件缺失大部分官方中心病例，"
        f"则 `n_official_external` 会小于148，仅可作为覆盖度诊断。**在完整覆盖148例的运行中**，对齐后外部ROC-AUC最高的是 **{top['method']}**"
        f"（AUC={fmt3(top['auc_mean'])}，AUPRC={fmt3(top['auprc_mean'])}，灵敏度={fmt3(top['sensitivity_mean'])}，"
        f"特异度={fmt3(top['specificity_mean'])}，NPV={fmt3(top['npv_mean'])}，ECE={fmt3(top['ece_mean'])}，"
        f"Brier={fmt3(top['brier_mean'])}）。"
        "全中心 holdout 导出的直接融合基线若官方子集覆盖不足，建议在严格148例队列上重新导出预测后再写头对头比较。"
        "缺失模态与输入损坏压力测试基于已导出的压力测试预测；训练期标签噪声压力测试在 `label_noise_stress_metrics.csv` 存在时按该表报告。"
    )

    lines.extend(["", "## Markdown table (official 148 subset, aligned)", ""])
    show = table_df[
        [
            "method",
            "runs",
            "n_official_external",
            "auc_mean",
            "auprc_mean",
            "sensitivity_mean",
            "specificity_mean",
            "npv_mean",
            "ece_mean",
            "brier_mean",
        ]
    ].copy()
    for c in show.columns:
        if c.endswith("_mean"):
            show[c] = show[c].map(fmt3)
    headers = list(show.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    if not partial.empty:
        lines.extend(
            [
                "",
                "## Partial official-center coverage (diagnostic only; n<148)",
                "",
                "These methods had fewer than 148 Jingzhou+Shiyan rows inside the exported `external_test_full` files; "
                "do not rank them against full-coverage runs without re-exporting predictions.",
                "",
            ]
        )
        pshow = partial.head(8)[["method", "runs", "n_official_external", "auc_mean"]].copy()
        pshow["auc_mean"] = pshow["auc_mean"].map(fmt3)
        ph = list(pshow.columns)
        lines.append("| " + " | ".join(ph) + " |")
        lines.append("| " + " | ".join(["---"] * len(ph)) + " |")
        for _, row in pshow.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in ph) + " |")

    (MAN_DIR / "PAPER_READY_RESULTS_zh_en.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Minimal LaTeX table
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Official external test (Jingzhou+Shiyan, $n{=}148$) after center-aligned filtering of exported predictions.}",
        r"\label{tab:official148_aligned}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Method & runs & AUC & AUPRC & Sens. & Spec. & NPV & ECE & Brier \\",
        r"\midrule",
    ]
    for _, r in table_df.head(10).iterrows():
        m = str(r["method"]).replace("_", "\\_")
        tex.append(
            f"{m} & {int(r['runs'])} & "
            f"{fmt3(r['auc_mean'])} & {fmt3(r['auprc_mean'])} & {fmt3(r['sensitivity_mean'])} & {fmt3(r['specificity_mean'])} & "
            f"{fmt3(r['npv_mean'])} & {fmt3(r['ece_mean'])} & {fmt3(r['brier_mean'])} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (MAN_DIR / "table_official148_aligned.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {MAN_DIR / 'PAPER_READY_RESULTS_zh_en.md'}")
    print(f"Wrote {MAN_DIR / 'table_official148_aligned.tex'}")


if __name__ == "__main__":
    main()
