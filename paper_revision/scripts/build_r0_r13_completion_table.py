#!/usr/bin/env python3
"""Build R0-R13 clean-rerun completion status table from on-disk artifacts."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORRECTED = ROOT / "paper_revision" / "results" / "real_50epoch_5center_corrected"


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def external_n_from_preds(pred_dir: Path, method_hint: str = "") -> str:
    if not pred_dir.exists():
        return "no_dir"
    files = list(pred_dir.glob("*external_test*full.csv"))
    if not files:
        return "no_csv"
    ns = []
    for f in files:
        try:
            df = pd.read_csv(f)
            ext = df[df["split"].astype(str).isin(["external_test", "external"])]
            ns.append(len(ext))
        except Exception:
            ns.append(-1)
    if not ns:
        return "no_csv"
    if all(n == 403 for n in ns if n >= 0):
        return f"403 ({len(ns)} files)"
    return f"mixed {sorted(set(ns))}"


def manifest_summary(path: Path) -> str:
    if not path.exists():
        return "missing"
    df = pd.read_csv(path)
    if df.empty:
        return "empty"
    if "status" not in df.columns:
        return f"{len(df)} rows"
    ok = df["status"].isin(["success", "skipped_existing"]).sum()
    fail = (df["status"] == "failed").sum()
    ext403 = 0
    if "external_n" in df.columns:
        ext403 = (pd.to_numeric(df["external_n"], errors="coerce") == 403).sum()
    return f"ok={ok} fail={fail} ext403={ext403}/{len(df)}"


def latest_train_epoch() -> str:
    logs = sorted(
        CORRECTED.rglob("train_seed*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    logs += sorted(
        (ROOT / "paper_revision" / "results").rglob("train_seed*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seen = set()
    for log in logs[:10]:
        if log in seen:
            continue
        seen.add(log)
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        import re

        hits = re.findall(r"Epoch (\d+)/50", text)
        if hits:
            return f"epoch {hits[-1]}/50 ({log.relative_to(ROOT)})"
    return "not started"


def pipeline_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "run_corrected403_clean_rerun"],
            capture_output=True,
            text=True,
        )
        return out.returncode == 0
    except Exception:
        return False


def build_rows() -> list[dict]:
    rows = [
        {
            "stage": "R0",
            "title": "Archive legacy mixed-n results",
            "status": "完成" if exists("paper_revision/results/archive_legacy_mixed_n/legacy_results_manifest.csv") else "未完成",
            "evidence": "legacy_results_manifest.csv + CLEAN_RERUN_SCOPE.md",
            "external_n_check": "N/A",
            "notes": "",
        },
        {
            "stage": "R1",
            "title": "Verify cohort/split/cache",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/cohort_cache_verification.json")
            and json.loads(
                (CORRECTED / "cohort_cache_verification.json").read_text(encoding="utf-8")
            ).get("passed")
            else "未完成/失败",
            "evidence": "cohort_cache_verification.md",
            "external_n_check": "403 expected",
            "notes": "",
        },
        {
            "stage": "R2",
            "title": "Stage-1 adapter injection check",
            "status": "完成"
            if exists("paper_revision/results/stage1_adapter_injection_check.json")
            else "未完成",
            "evidence": "stage1_adapter_injection_check.md",
            "external_n_check": "N/A",
            "notes": "",
        },
        {
            "stage": "R3",
            "title": "Full HyDRA-CoE 50-epoch × seeds",
            "status": "进行中" if pipeline_running() else "待确认",
            "evidence": manifest_summary(CORRECTED / "full_hydra_coe" / "full_model_run_manifest.csv"),
            "external_n_check": external_n_from_preds(CORRECTED / "full_hydra_coe" / "predictions"),
            "notes": latest_train_epoch() if pipeline_running() else "",
        },
        {
            "stage": "R4",
            "title": "Baselines under 403 protocol",
            "status": "待确认",
            "evidence": manifest_summary(CORRECTED / "baselines" / "baseline_run_manifest.csv"),
            "external_n_check": external_n_from_preds(CORRECTED / "baselines" / "predictions"),
            "notes": "",
        },
        {
            "stage": "R5",
            "title": "Requirement-level ablations",
            "status": "待确认",
            "evidence": manifest_summary(CORRECTED / "ablations" / "ablation_run_manifest.csv"),
            "external_n_check": external_n_from_preds(CORRECTED / "ablations" / "predictions"),
            "notes": "",
        },
        {
            "stage": "R6",
            "title": "Missing-modality & corruption robustness",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/robustness/missing_modality_robustness_metrics.csv")
            else "未完成",
            "evidence": "robustness/*.csv",
            "external_n_check": "403 required",
            "notes": "",
        },
        {
            "stage": "R7",
            "title": "Label-noise stress test",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/label_noise/label_noise_metrics.csv")
            else "未完成",
            "evidence": "label_noise/*.csv",
            "external_n_check": "403 required",
            "notes": "",
        },
        {
            "stage": "R8",
            "title": "LOCO & center-wise calibration",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/loco/loco_metrics_by_center.csv")
            else "未完成",
            "evidence": "loco/loco_metrics_by_center.csv",
            "external_n_check": "per held-out center",
            "notes": "",
        },
        {
            "stage": "R9",
            "title": "CoE faithfulness proxy",
            "status": "完成"
            if exists(
                "paper_revision/results/real_50epoch_5center_corrected/coe_faithfulness/coe_faithfulness_proxy_metrics.csv"
            )
            else "未完成",
            "evidence": "coe_faithfulness/*.csv",
            "external_n_check": "403",
            "notes": "proxy only",
        },
        {
            "stage": "R10",
            "title": "Clinical decision per 1000 women",
            "status": "完成"
            if exists(
                "paper_revision/results/real_50epoch_5center_corrected/clinical_decision/clinical_decision_per_1000_table.csv"
            )
            else "未完成",
            "evidence": "clinical_decision/*.csv",
            "external_n_check": "403",
            "notes": "",
        },
        {
            "stage": "R11",
            "title": "Paper-ready tables",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/paper_tables/Table1_main_performance.csv")
            else "未完成",
            "evidence": "paper_tables/Table*.csv",
            "external_n_check": "403 enforced",
            "notes": "",
        },
        {
            "stage": "R12",
            "title": "Real final pipeline manifest",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/final_pipeline_manifest.csv")
            else "未完成",
            "evidence": "final_pipeline_manifest.csv (dry_run=false)",
            "external_n_check": "N/A",
            "notes": "",
        },
        {
            "stage": "R13",
            "title": "Paper-facing naming cleanup",
            "status": "完成"
            if exists("paper_revision/results/real_50epoch_5center_corrected/PAPER_READY_RESULT_SCOPE.md")
            else "未完成",
            "evidence": "name_cleanup_audit.csv",
            "external_n_check": "N/A",
            "notes": "",
        },
    ]

    # Auto-upgrade R3-R5 status from manifests
    for row in rows:
        if row["stage"] == "R3":
            m = CORRECTED / "full_hydra_coe" / "full_model_run_manifest.csv"
            if m.exists():
                df = pd.read_csv(m)
                need = 3
                ok = (
                    (df["status"].isin(["success", "skipped_existing"]))
                    & (pd.to_numeric(df.get("external_n", 0), errors="coerce") == 403)
                ).sum()
                row["status"] = "完成" if ok >= need else ("部分完成" if ok > 0 else "失败")
        if row["stage"] in {"R4", "R5"}:
            sub = "baselines" if row["stage"] == "R4" else "ablations"
            m = CORRECTED / sub / f"{sub[:-1]}_run_manifest.csv" if sub == "baselines" else CORRECTED / "ablations" / "ablation_run_manifest.csv"
            if m.exists():
                df = pd.read_csv(m)
                ok = (
                    (df["status"].isin(["success", "skipped_existing"]))
                    & (pd.to_numeric(df.get("external_n", 0), errors="coerce") == 403)
                ).sum()
                row["status"] = "完成" if ok == len(df) and len(df) > 0 else ("部分完成" if ok > 0 else "失败")

    if pipeline_running():
        for row in rows:
            if row["status"] == "待确认" and row["stage"] in {"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}:
                row["status"] = "等待中"

    return rows


def main() -> None:
    rows = build_rows()
    out_dir = CORRECTED / "status"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    csv_path = out_dir / "R0_R13_completion_table.csv"
    md_path = out_dir / "R0_R13_completion_table.md"

    df = pd.DataFrame(rows)
    df["updated_at"] = ts
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    running = pipeline_running()
    done = sum(1 for r in rows if r["status"] == "完成")
    lines = [
        "# HyDRA-CoE Corrected403 Clean Rerun — R0–R13 完成表",
        "",
        f"更新时间: {ts}",
        f"流水线运行中: **{'是' if running else '否'}**",
        f"已完成阶段: **{done}/14**",
        "",
        "| 阶段 | 任务 | 状态 | 外部测试 n | 证据 | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['title']} | {r['status']} | {r['external_n_check']} | {r['evidence']} | {r['notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
