#!/usr/bin/env python3
"""Audit formal LOCO splits: prevalence shift, leakage, hard-center val simulation."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts.shared_lora_biocot.run_formal_loco import (  # noqa: E402
    _choose_hard_validation_center,
    build_loco_splits,
    read_csv,
)

DEFAULT_SPLIT_ROOT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/formal_loco/splits/loco"
DEFAULT_MANIFEST = EXP_ROOT / "outputs/publishable_v2/splits/split_manifest_v2.csv"
DEFAULT_DATA_LOCK = EXP_ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv"
DEFAULT_COHORT = EXP_ROOT / "paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv"
DEFAULT_OUT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897/diagnostics/loco_split"
DEFAULT_BASELINE_METRICS = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv"
)
DEFAULT_G2_METRICS = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/improved_1897/ablations/g2/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
)

ISSUE_ALL_POSITIVE = "all_positive_external"
ISSUE_EXTREME_IMBALANCE = "extreme_imbalance"
ISSUE_VAL_EXTERNAL_GAP = "val_external_prevalence_gap"
ISSUE_SMALL_EXTERNAL = "small_external_n"


def _fold_dirs(split_root: Path) -> list[Path]:
    return sorted(p for p in split_root.glob("loco_*") if p.is_dir())


def _held_out_name(fold_dir: Path) -> str:
    return fold_dir.name.replace("loco_", "", 1)


def audit_patient_leakage(split_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_dir in _fold_dirs(split_root):
        train = read_csv(fold_dir / "train_labels.csv")
        val = read_csv(fold_dir / "val_labels.csv")
        ext = read_csv(fold_dir / "external_test_labels.csv")
        pid = "patient_id"
        train_ext = len(set(train[pid].astype(str)) & set(ext[pid].astype(str)))
        val_ext = len(set(val[pid].astype(str)) & set(ext[pid].astype(str)))
        rows.append(
            {
                "held_out_center": _held_out_name(fold_dir),
                "train_ext_patient_overlap": train_ext,
                "val_ext_patient_overlap": val_ext,
                "leakage_detected": bool(train_ext or val_ext),
            }
        )
    return pd.DataFrame(rows)


def audit_prevalence_shift(split_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_dir in _fold_dirs(split_root):
        train = read_csv(fold_dir / "train_labels.csv")
        val = read_csv(fold_dir / "val_labels.csv")
        ext = read_csv(fold_dir / "external_test_labels.csv")
        held = _held_out_name(fold_dir)
        tr = float(train["label"].mean())
        va = float(val["label"].mean())
        te = float(ext["label"].mean())
        gap = abs(va - te)
        flags: list[str] = []
        if te >= 0.99:
            flags.append(ISSUE_ALL_POSITIVE)
        if te <= 0.08 or te >= 0.92:
            flags.append(ISSUE_EXTREME_IMBALANCE)
        if gap > 0.10:
            flags.append(ISSUE_VAL_EXTERNAL_GAP)
        if len(ext) < 120:
            flags.append(ISSUE_SMALL_EXTERNAL)
        rows.append(
            {
                "held_out_center": held,
                "n_train": len(train),
                "n_val": len(val),
                "n_external": len(ext),
                "train_pos_rate": tr,
                "val_pos_rate": va,
                "external_pos_rate": te,
                "val_external_gap_pp": gap,
                "external_positives": int(ext["label"].sum()),
                "issues": ";".join(flags) if flags else "none",
            }
        )
    return pd.DataFrame(rows)


def cohort_center_summary(cohort_path: Path) -> pd.DataFrame:
    df = read_csv(cohort_path)
    label_col = "label" if "label" in df.columns else "pathology_cin2plus"
    out = (
        df.groupby("center_name")
        .agg(n=(label_col, "size"), positives=(label_col, "sum"), pos_rate=(label_col, "mean"))
        .reset_index()
        .sort_values("pos_rate")
    )
    return out


def simulate_hard_center_manifest(
    manifest_path: Path,
    data_lock_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_csv(manifest_path)
    data_lock = read_csv(data_lock_path)
    keep_cols = [
        "case_id",
        "patient_id",
        "center_id",
        "center_name",
        "age",
        "hpv_status_harmonized",
        "tct_status_harmonized",
        "pathology_cin2plus",
        "pathology_cin3plus",
        "oct_num_bscans",
        "colposcopy_num_images",
        "colposcopy_paths",
        "oct_paths",
    ]
    merged = manifest.merge(
        data_lock[keep_cols],
        on=["case_id", "patient_id", "center_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"],
        how="left",
        validate="many_to_one",
    )
    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for fold_id, fold_df in merged.groupby("fold_id", sort=True):
        held_out = sorted(fold_df.loc[fold_df["split_role"].eq("test"), "center_name"].unique())
        held_name = held_out[0] if len(held_out) == 1 else ";".join(held_out)
        test_df = fold_df[fold_df["split_role"].eq("test")].copy()
        source_pool = fold_df[~fold_df["split_role"].eq("test")].copy()
        manifest_val = fold_df[fold_df["split_role"].eq("validation")].copy()
        hard_val_center, candidates = _choose_hard_validation_center(source_pool)
        candidates = candidates.copy()
        candidates.insert(0, "fold_id", fold_id)
        candidates.insert(1, "held_out_center", held_name)
        candidate_rows.extend(candidates.to_dict("records"))
        hard_val = source_pool[source_pool["center_name"].astype(str).eq(hard_val_center)]
        fold_rows.append(
            {
                "fold_id": fold_id,
                "held_out_center": held_name,
                "manifest_val_pos_rate": float(manifest_val["pathology_cin2plus"].mean()),
                "hard_val_center": hard_val_center,
                "hard_val_pos_rate": float(hard_val["pathology_cin2plus"].mean()),
                "external_pos_rate": float(test_df["pathology_cin2plus"].mean()),
                "manifest_val_external_gap_pp": abs(
                    float(manifest_val["pathology_cin2plus"].mean()) - float(test_df["pathology_cin2plus"].mean())
                ),
                "hard_val_external_gap_pp": abs(
                    float(hard_val["pathology_cin2plus"].mean()) - float(test_df["pathology_cin2plus"].mean())
                ),
                "gap_reduction_pp": abs(
                    float(manifest_val["pathology_cin2plus"].mean()) - float(test_df["pathology_cin2plus"].mean())
                )
                - abs(float(hard_val["pathology_cin2plus"].mean()) - float(test_df["pathology_cin2plus"].mean())),
                "hard_val_n": len(hard_val),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(candidate_rows)


def merge_auc_metrics(
    prevalence: pd.DataFrame,
    baseline_metrics: Path | None,
    g2_metrics: Path | None,
) -> pd.DataFrame:
    out = prevalence.copy()
    for label, path in [("baseline_auc", baseline_metrics), ("g2_auc", g2_metrics)]:
        if path is None or not path.exists():
            out[label] = np.nan
            continue
        metrics = read_csv(path)
        center_col = "held_out_center" if "held_out_center" in metrics.columns else "center"
        auc_col = "cin2plus_auc" if "cin2plus_auc" in metrics.columns else "auc"
        mini = metrics[[center_col, auc_col]].rename(columns={center_col: "held_out_center", auc_col: label})
        out = out.merge(mini, on="held_out_center", how="left")
    return out


def recommend_hard_center(
    prevalence: pd.DataFrame,
    hard_sim: pd.DataFrame,
    leakage: pd.DataFrame,
) -> tuple[bool, list[str], pd.DataFrame]:
    evaluable = prevalence[prevalence["external_pos_rate"] < 0.99].copy()
    manifest_gap = float(evaluable["val_external_gap_pp"].mean()) if len(evaluable) else np.nan
    hard_gap = float(evaluable.merge(hard_sim[["held_out_center", "hard_val_external_gap_pp"]], on="held_out_center")["hard_val_external_gap_pp"].mean()) if len(evaluable) else np.nan
    gap_reduction = manifest_gap - hard_gap if pd.notna(manifest_gap) and pd.notna(hard_gap) else np.nan
    rel_reduction = gap_reduction / manifest_gap if manifest_gap and manifest_gap > 0 else np.nan

    n_gap_issues = int((prevalence["issues"].str.contains(ISSUE_VAL_EXTERNAL_GAP, na=False)).sum())
    n_all_pos = int((prevalence["issues"].str.contains(ISSUE_ALL_POSITIVE, na=False)).sum())
    n_leak = int(leakage["leakage_detected"].sum())

    reasons: list[str] = []
    if n_leak:
        reasons.append(f"检测到 patient 泄漏 {n_leak} 折（需立即修复划分）")
    if n_all_pos:
        reasons.append(f"{n_all_pos} 折 external 全阳（武大），macro AUC 不应纳入该折")
    if n_gap_issues >= 3:
        reasons.append(f"{n_gap_issues}/5 折 manifest val 与 external 阳性率差 >10pp（平均 {manifest_gap:.1%}）")

    recommend = False
    if pd.notna(rel_reduction) and rel_reduction >= 0.25 and manifest_gap >= 0.08:
        recommend = True
        reasons.append(
            f"hard-center val 可将 evaluable 折 val↔external 阳性率差从 {manifest_gap:.1%} 降到 {hard_gap:.1%} "
            f"（相对缩小 {rel_reduction:.0%}）"
        )
    elif manifest_gap >= 0.12:
        recommend = True
        reasons.append(
            f"manifest val 与 external 平均差 {manifest_gap:.1%} 过大，建议至少做 hard-center 对比实验"
        )
    else:
        reasons.append("manifest val 与 external 差距尚可，hard-center 为可选 sensitivity analysis")

    score_rows = [
        {"criterion": "manifest_val_external_gap_mean", "value": manifest_gap, "threshold": 0.10, "pass": manifest_gap < 0.10 if pd.notna(manifest_gap) else False},
        {"criterion": "hard_center_gap_mean", "value": hard_gap, "threshold": manifest_gap, "pass": hard_gap < manifest_gap if pd.notna(hard_gap) and pd.notna(manifest_gap) else False},
        {"criterion": "relative_gap_reduction", "value": rel_reduction, "threshold": 0.25, "pass": bool(recommend)},
        {"criterion": "patient_leakage_folds", "value": n_leak, "threshold": 0, "pass": n_leak == 0},
        {"criterion": "all_positive_external_folds", "value": n_all_pos, "threshold": 0, "pass": n_all_pos == 0},
    ]
    return recommend, reasons, pd.DataFrame(score_rows)


def write_report(
    out_dir: Path,
    prevalence: pd.DataFrame,
    leakage: pd.DataFrame,
    cohort: pd.DataFrame,
    hard_sim: pd.DataFrame,
    hard_candidates: pd.DataFrame,
    merged_auc: pd.DataFrame,
    recommend: bool,
    reasons: list[str],
    scorecard: pd.DataFrame,
    validation_policy: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(out_dir / "Table_LOCO_Prevalence_Shift.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(out_dir / "Table_LOCO_Patient_Leakage.csv", index=False, encoding="utf-8-sig")
    cohort.to_csv(out_dir / "Table_LOCO_Cohort_Center_Prevalence.csv", index=False, encoding="utf-8-sig")
    hard_sim.to_csv(out_dir / "Table_LOCO_HardCenter_Val_Simulation.csv", index=False, encoding="utf-8-sig")
    hard_candidates.to_csv(out_dir / "Table_LOCO_HardCenter_Candidates.csv", index=False, encoding="utf-8-sig")
    merged_auc.to_csv(out_dir / "Table_LOCO_Prevalence_vs_AUC.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(out_dir / "Table_LOCO_HardCenter_Recommendation_Scorecard.csv", index=False, encoding="utf-8-sig")

    evaluable = merged_auc[merged_auc["external_pos_rate"] < 0.99].copy()
    macro_baseline = evaluable["baseline_auc"].dropna()
    macro_g2 = evaluable["g2_auc"].dropna()
    exclude_xy = evaluable[~evaluable["held_out_center"].eq("襄阳市中心医院")]
    macro_g2_3center = exclude_xy["g2_auc"].dropna()

    rec_text = "**建议切换 hard-center val**" if recommend else "**可保留 manifest val，建议做 hard-center 对比**"
    lines = [
        "# LOCO Split Diagnostic Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Split root: `{validation_policy}` on-disk CSVs",
        "",
        "## Recommendation",
        "",
        rec_text,
        "",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## External test prevalence (current splits)",
            "",
            prevalence.to_markdown(index=False),
            "",
            "## Val vs external gap: manifest vs simulated hard-center",
            "",
            hard_sim[
                [
                    "held_out_center",
                    "manifest_val_pos_rate",
                    "hard_val_center",
                    "hard_val_pos_rate",
                    "external_pos_rate",
                    "manifest_val_external_gap_pp",
                    "hard_val_external_gap_pp",
                    "gap_reduction_pp",
                ]
            ].to_markdown(index=False),
            "",
            "## AUC context (if metrics available)",
            "",
            f"- Evaluable folds baseline mean AUC: {macro_baseline.mean():.4f} (n={len(macro_baseline)})" if len(macro_baseline) else "- Baseline AUC: n/a",
            f"- Evaluable folds g2 mean AUC: {macro_g2.mean():.4f} (n={len(macro_g2)})" if len(macro_g2) else "- g2 AUC: n/a",
            f"- g2 mean AUC excluding 襄阳: {macro_g2_3center.mean():.4f} (n={len(macro_g2_3center)})" if len(macro_g2_3center) else "",
            "",
            "## Next steps",
            "",
            "1. 武大 external 仅报 sensitivity/specificity，不纳入 macro AUC。",
            "2. 运行 hard-center 对比：",
            "   `python scripts/shared_lora_biocot/run_1897_hardcenter_val_comparison.py --gpu 1`",
            "3. 若 hard-center 更优，100-epoch production 改用 `splits/loco_hard_center`。",
            "",
        ]
    )
    (out_dir / "Report_LOCO_Split_Diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnosis(args: argparse.Namespace) -> tuple[bool, Path]:
    split_root = Path(args.split_root)
    out_dir = Path(args.out_dir)
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    prevalence = audit_prevalence_shift(split_root)
    leakage = audit_patient_leakage(split_root)
    cohort = cohort_center_summary(Path(args.cohort_csv))
    hard_sim, hard_candidates = simulate_hard_center_manifest(Path(args.manifest), Path(args.data_lock))
    merged_auc = merge_auc_metrics(
        prevalence,
        Path(args.baseline_metrics) if args.baseline_metrics else None,
        Path(args.g2_metrics) if args.g2_metrics else None,
    )
    recommend, reasons, scorecard = recommend_hard_center(prevalence, hard_sim, leakage)
    write_report(
        out_dir,
        prevalence,
        leakage,
        cohort,
        hard_sim,
        hard_candidates,
        merged_auc,
        recommend,
        reasons,
        scorecard,
        args.validation_policy,
    )
    print(f"Wrote LOCO diagnostics to {out_dir}")
    print(f"Recommend hard-center val: {recommend}")
    for reason in reasons:
        print(f"  - {reason}")
    return recommend, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose formal LOCO split quality and hard-center val need.")
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--data-lock", default=str(DEFAULT_DATA_LOCK))
    parser.add_argument("--cohort-csv", default=str(DEFAULT_COHORT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_BASELINE_METRICS))
    parser.add_argument("--g2-metrics", default=str(DEFAULT_G2_METRICS))
    parser.add_argument("--validation-policy", default="manifest", help="Label for report only unless building splits.")
    parser.add_argument(
        "--build-hard-center-splits",
        action="store_true",
        help="Also materialize hard-center CSV splits under --hard-center-split-root.",
    )
    parser.add_argument(
        "--hard-center-split-root",
        default=str(EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897/splits/loco_hard_center"),
    )
    args = parser.parse_args()

    recommend, out_dir = run_diagnosis(args)

    if args.build_hard_center_splits:
        hard_root = Path(args.hard_center_split_root)
        hard_root.mkdir(parents=True, exist_ok=True)
        print(f"Building hard-center splits under {hard_root} ...")
        build_loco_splits(
            Path(args.manifest),
            Path(args.data_lock),
            hard_root,
            validation_policy="hard-center",
        )
        print(f"Hard-center splits ready: {hard_root}")

    if recommend:
        sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
