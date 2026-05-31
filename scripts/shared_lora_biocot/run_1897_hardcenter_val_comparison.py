#!/usr/bin/env python3
"""Compare manifest val vs hard-center val LOCO using the g2 (best ablation) stack."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts.shared_lora_biocot.diagnose_loco_splits import run_diagnosis  # noqa: E402
from scripts.shared_lora_biocot.run_formal_loco import (  # noqa: E402
    DEFAULT_PYTHON,
    build_loco_splits,
    read_csv,
)
from scripts.shared_lora_biocot.run_1897_improved_loco import (  # noqa: E402
    CACHE_1897,
    FORMAL_PRETRAIN_CKPT,
    FORMAL_SPLIT_ROOT,
    OUT_ROOT_DEFAULT,
    group_is_complete,
    run_ablation_group,
)

COMP_ROOT_DEFAULT = OUT_ROOT_DEFAULT / "hardcenter_val_comparison"
MANIFEST_SPLIT_ROOT = FORMAL_SPLIT_ROOT
HARD_SPLIT_ROOT_DEFAULT = OUT_ROOT_DEFAULT / "splits/loco_hard_center"
DIAG_OUT_DEFAULT = OUT_ROOT_DEFAULT / "diagnostics/loco_split"


def ensure_hard_center_splits(manifest: Path, data_lock: Path, hard_root: Path) -> None:
    marker = hard_root / "formal_loco_split_manifest.csv"
    if marker.exists():
        print(f"Hard-center splits already exist: {hard_root}")
        return
    hard_root.mkdir(parents=True, exist_ok=True)
    print(f"Building hard-center splits under {hard_root} ...")
    build_loco_splits(manifest, data_lock, hard_root, validation_policy="hard-center")


COMP_ROOT_DEFAULT = OUT_ROOT_DEFAULT / "hardcenter_val_comparison"
MANIFEST_SPLIT_ROOT = FORMAL_SPLIT_ROOT
HARD_SPLIT_ROOT_DEFAULT = OUT_ROOT_DEFAULT / "splits/loco_hard_center"
DIAG_OUT_DEFAULT = OUT_ROOT_DEFAULT / "diagnostics/loco_split"
GROUP_NAME = "g2"


def resolve_group_metrics(arm_parent: Path) -> Path:
    """Return fold metrics CSV for an arm (supports flat or ablations/g2 layout)."""
    candidates = [
        arm_parent / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv",
        arm_parent / "ablations" / GROUP_NAME / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_or_run_arm(
    arm: str,
    split_root: Path,
    arm_parent: Path,
    expert_checkpoint: Path,
    python: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    env: dict[str, str],
    skip_train: bool,
    skip_eval: bool,
    reuse_manifest_g2: bool,
    fold_filter: list[str] | None = None,
) -> Path:
    group_root = arm_parent / "ablations" / GROUP_NAME
    metrics_path = group_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"

    if arm == "manifest" and reuse_manifest_g2:
        src = OUT_ROOT_DEFAULT / "ablations/g2/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
        if src.exists() and group_is_complete(OUT_ROOT_DEFAULT / "ablations/g2"):
            print(f"[{arm}] Reusing completed g2 ablation metrics from {src}")
            return src

    if metrics_path.exists() and group_is_complete(group_root):
        print(f"[{arm}] Already complete: {metrics_path}")
        return metrics_path

    arm_parent.mkdir(parents=True, exist_ok=True)
    run_ablation_group(
        group=GROUP_NAME,
        out_root=arm_parent,
        split_root=split_root,
        expert_checkpoint=expert_checkpoint,
        python=python,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        env=env,
        skip_train=skip_train,
        skip_eval=skip_eval,
        fold_filter=fold_filter,
    )
    metrics_path = resolve_group_metrics(arm_parent)
    if not metrics_path.exists():
        raise FileNotFoundError(f"[{arm}] Expected metrics at {metrics_path}")
    return metrics_path


def build_comparison_table(
    manifest_metrics: Path,
    hard_metrics: Path,
    prevalence_csv: Path,
    out_dir: Path,
) -> pd.DataFrame:
    m = read_csv(manifest_metrics)
    h = read_csv(hard_metrics)
    prev = read_csv(prevalence_csv) if prevalence_csv.exists() else pd.DataFrame()
    merged = m[["held_out_center", "n_test", "cin2plus_auc", "cin2plus_sensitivity", "cin2plus_specificity"]].rename(
        columns={
            "cin2plus_auc": "manifest_auc",
            "cin2plus_sensitivity": "manifest_sensitivity",
            "cin2plus_specificity": "manifest_specificity",
        }
    )
    merged = merged.merge(
        h[["held_out_center", "cin2plus_auc", "cin2plus_sensitivity", "cin2plus_specificity"]].rename(
            columns={
                "cin2plus_auc": "hardcenter_auc",
                "cin2plus_sensitivity": "hardcenter_sensitivity",
                "cin2plus_specificity": "hardcenter_specificity",
            }
        ),
        on="held_out_center",
        how="outer",
    )
    merged["delta_auc_hard_minus_manifest"] = merged["hardcenter_auc"] - merged["manifest_auc"]
    if not prev.empty and "external_pos_rate" in prev.columns:
        merged = merged.merge(
            prev[["held_out_center", "external_pos_rate", "val_external_gap_pp"]],
            on="held_out_center",
            how="left",
        )

    evaluable = merged[merged["external_pos_rate"].fillna(1.0) < 0.99].copy()
    summary = {
        "held_out_center": ["__macro_evaluable__"],
        "manifest_auc": [evaluable["manifest_auc"].dropna().mean()],
        "hardcenter_auc": [evaluable["hardcenter_auc"].dropna().mean()],
        "delta_auc_hard_minus_manifest": [
            evaluable["hardcenter_auc"].dropna().mean() - evaluable["manifest_auc"].dropna().mean()
        ],
    }
    out = pd.concat([merged, pd.DataFrame(summary)], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "Table_HardCenter_Val_Comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Hard-Center vs Manifest Validation Comparison (g2 stack)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Per-fold external CIN2+ AUC",
        "",
        merged.to_markdown(index=False),
        "",
        "## Summary",
        "",
        f"- Evaluable folds manifest mean AUC: **{evaluable['manifest_auc'].dropna().mean():.4f}**",
        f"- Evaluable folds hard-center mean AUC: **{evaluable['hardcenter_auc'].dropna().mean():.4f}**",
        f"- Delta (hard - manifest): **{summary['delta_auc_hard_minus_manifest'][0]:+.4f}**",
        "",
        "Interpretation: positive delta suggests hard-center val selection aligns better with external generalization.",
        "",
    ]
    (out_dir / "Report_HardCenter_Val_Comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(COMP_ROOT_DEFAULT))
    parser.add_argument("--manifest-split-root", default=str(MANIFEST_SPLIT_ROOT))
    parser.add_argument("--hard-split-root", default=str(HARD_SPLIT_ROOT_DEFAULT))
    parser.add_argument("--manifest", default=str(EXP_ROOT / "outputs/publishable_v2/splits/split_manifest_v2.csv"))
    parser.add_argument("--data-lock", default=str(EXP_ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv"))
    parser.add_argument("--diagnostics-out", default=str(DIAG_OUT_DEFAULT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-diagnose", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-manifest-arm", action="store_true", help="Only run hard-center arm (reuse manifest g2).")
    parser.add_argument("--reuse-manifest-g2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fold", action="append", default=None, help="Optional fold subset, e.g. loco_襄阳市中心医院")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training; build comparison table from existing arm metrics.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    manifest_split_root = Path(args.manifest_split_root)
    hard_split_root = Path(args.hard_split_root)
    expert = FORMAL_PRETRAIN_CKPT
    if not expert.exists():
        raise FileNotFoundError(f"Expert checkpoint missing: {expert}")
    if not CACHE_1897.exists():
        raise FileNotFoundError(f"1897 cache missing: {CACHE_1897}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not args.skip_diagnose:
        diag_args = argparse.Namespace(
            split_root=str(manifest_split_root),
            manifest=args.manifest,
            data_lock=args.data_lock,
            cohort_csv=str(EXP_ROOT / "paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv"),
            out_dir=args.diagnostics_out,
            baseline_metrics=str(
                EXP_ROOT
                / "outputs/publishable_v2/shared_lora_biocot/formal_loco/tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv"
            ),
            g2_metrics=str(OUT_ROOT_DEFAULT / "ablations/g2/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"),
            validation_policy="manifest",
            build_hard_center_splits=False,
            hard_center_split_root=str(hard_split_root),
        )
        run_diagnosis(diag_args)

    ensure_hard_center_splits(Path(args.manifest), Path(args.data_lock), hard_split_root)

    manifest_arm_root = out_root / "g2_manifest_val"
    hard_arm_parent = out_root / "g2_hardcenter_val"

    if not args.compare_only:
        if not args.skip_manifest_arm:
            load_or_run_arm(
                "manifest",
                manifest_split_root,
                manifest_arm_root,
                expert,
                Path(args.python),
                args.seed,
                args.epochs,
                args.batch_size,
                env,
                args.skip_train,
                args.skip_eval,
                args.reuse_manifest_g2,
                args.fold,
            )

        load_or_run_arm(
            "hard-center",
            hard_split_root,
            hard_arm_parent,
            expert,
            Path(args.python),
            args.seed,
            args.epochs,
            args.batch_size,
            env,
            args.skip_train,
            args.skip_eval,
            reuse_manifest_g2=False,
            fold_filter=args.fold,
        )

    manifest_metrics = resolve_group_metrics(manifest_arm_root)
    if not manifest_metrics.exists() and args.reuse_manifest_g2:
        manifest_metrics = OUT_ROOT_DEFAULT / "ablations/g2/tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
    hard_metrics = resolve_group_metrics(hard_arm_parent)
    if not hard_metrics.exists():
        raise FileNotFoundError(f"Hard-center arm metrics missing under {hard_arm_parent}")

    comparison = build_comparison_table(
        manifest_metrics,
        hard_metrics,
        Path(args.diagnostics_out) / "Table_LOCO_Prevalence_Shift.csv",
        out_root / "tables",
    )
    print(comparison.to_string(index=False))
    print(f"\nComparison written to {out_root / 'tables'}")


if __name__ == "__main__":
    main()
