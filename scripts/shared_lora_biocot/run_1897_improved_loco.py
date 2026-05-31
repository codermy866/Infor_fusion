#!/usr/bin/env python3
"""Run improved Shared-LoRA LOCO experiments on the locked n=1897 cohort.

Implements G3-a loss ablations (g0–g4), G4-a LoRA rank, cached features, and G5-b
CIN3+ threshold selection. Reuses formal LOCO splits and the Phase-0 expert checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
    DEFAULT_PYTHON,
    aggregate_results,
    binary_metrics,
    newest_checkpoint,
    read_csv,
    run_command,
    safe_name,
    target_free_logit_median_match,
    threshold_for_sensitivity,
    threshold_max_f1,
    write_config,
)

OUT_ROOT_DEFAULT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897"
FORMAL_SPLIT_ROOT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/formal_loco/splits/loco"
FORMAL_PRETRAIN_CKPT = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/pretrain_oct_text_expert/checkpoints/best_model_v3_20260526_215640.pth"
)
CACHE_1897 = EXP_ROOT / "paper_revision/cache/patch_features_final_1897.pt"
CASE_MANIFEST = EXP_ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv"
BASELINE_METRICS = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv"
)

# Validated control: g1 (no adversarial). All leave-one-out ablations start here.
G1_CONTROL: dict[str, Any] = {
    "lambda_cls": 2.0,
    "lambda_ot": 0.5,
    "lambda_align": 0.5,
    "lambda_adv": 0.0,
    "lambda_consist": 0.2,
    "lambda_colpo_bridge_ot": 0.2,
    "lambda_colpo_bridge_align": 0.05,
    "lambda_coe": 0.05,
    "lambda_reliability_kl": 0.01,
    "lambda_posterior_smooth": 0.01,
    "lambda_modality_likelihood": 0.05,
    "use_adversarial": False,
    "use_coe_readout": True,
    "use_coe_supervision": True,
    "use_ot": True,
    "use_dual": True,
    "use_cross_attn": True,
    "use_colpo_lora_bridge": True,
    "use_posterior_refinement": True,
    "use_variational_reliability": True,
    "use_modality_likelihood": True,
    "use_center_aware_reliability": True,
    "shared_lora_rank": 8,
    "shared_lora_alpha": 16.0,
}


def _abl(name: str, description: str, removed: str, **overrides: Any) -> dict[str, Any]:
    preset = dict(G1_CONTROL)
    preset.update(overrides)
    preset["description"] = description
    preset["removed_components"] = removed
    return preset


ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    "g1": _abl("g1", "Control: g1 (no L_adv, full HyDRA stack)", "none"),
    "g0": _abl(
        "g0",
        "Historical: full losses incl. L_adv (cached)",
        "none (includes L_adv)",
        lambda_adv=0.5,
        use_adversarial=True,
    ),
    "g2": _abl(
        "g2",
        "Minimal: cls + OT only",
        "L_align,L_bridge,L_coe,L_consist,posterior,variational,modality_likelihood,cross_attn",
        lambda_align=0.0,
        lambda_colpo_bridge_ot=0.0,
        lambda_colpo_bridge_align=0.0,
        lambda_coe=0.0,
        lambda_consist=0.0,
        lambda_reliability_kl=0.0,
        lambda_posterior_smooth=0.0,
        lambda_modality_likelihood=0.0,
        use_coe_readout=False,
        use_coe_supervision=False,
        use_dual=False,
        use_cross_attn=False,
        use_colpo_lora_bridge=False,
        use_posterior_refinement=False,
        use_variational_reliability=False,
        use_modality_likelihood=False,
        use_center_aware_reliability=False,
    ),
    "g3": _abl(
        "g3",
        "Remove CoE supervision + readout",
        "L_coe,CoE_readout",
        lambda_coe=0.0,
        use_coe_readout=False,
        use_coe_supervision=False,
    ),
    "g4": _abl(
        "g4",
        "Rebalanced cls vs auxiliary (no adv)",
        "L_adv (rebalanced lambdas)",
        lambda_cls=4.0,
        lambda_ot=0.2,
        lambda_align=0.2,
        lambda_colpo_bridge_ot=0.1,
        lambda_colpo_bridge_align=0.02,
    ),
    "no_align": _abl("no_align", "Remove L_align", "L_align", lambda_align=0.0),
    "no_ot": _abl("no_ot", "Remove L_ot", "L_ot", use_ot=False, lambda_ot=0.0),
    "no_bridge": _abl(
        "no_bridge",
        "Remove bridge OT/align losses",
        "L_colpo_bridge_ot,L_colpo_bridge_align",
        lambda_colpo_bridge_ot=0.0,
        lambda_colpo_bridge_align=0.0,
    ),
    "no_coe": _abl(
        "no_coe",
        "Remove CoE (alias g3)",
        "L_coe,CoE_readout",
        lambda_coe=0.0,
        use_coe_readout=False,
        use_coe_supervision=False,
    ),
    "no_consist": _abl(
        "no_consist",
        "Remove counterfactual consistency (L_consist)",
        "L_consist,dual_head",
        use_dual=False,
        lambda_consist=0.0,
    ),
    "no_posterior": _abl(
        "no_posterior",
        "Remove posterior refinement stack",
        "posterior_refinement,L_posterior_smooth",
        use_posterior_refinement=False,
        lambda_posterior_smooth=0.0,
    ),
    "no_variational": _abl(
        "no_variational",
        "Remove variational reliability",
        "variational_reliability,L_reliability_kl",
        use_variational_reliability=False,
        lambda_reliability_kl=0.0,
    ),
    "no_modality_likelihood": _abl(
        "no_modality_likelihood",
        "Remove modality likelihood head",
        "modality_likelihood,L_modality_likelihood",
        use_modality_likelihood=False,
        lambda_modality_likelihood=0.0,
    ),
    "no_cross_attn": _abl("no_cross_attn", "Remove final cross-attention fusion", "cross_attn", use_cross_attn=False),
    "no_bridge_module": _abl(
        "no_bridge_module",
        "Remove Shared-LoRA bridge module",
        "SharedLoRA_bridge",
        use_colpo_lora_bridge=False,
        lambda_colpo_bridge_ot=0.0,
        lambda_colpo_bridge_align=0.0,
    ),
    "no_center_reliability": _abl(
        "no_center_reliability",
        "Remove center-aware reliability",
        "center_aware_reliability",
        use_center_aware_reliability=False,
    ),
    "best": _abl(
        "best",
        "Combined g1+g4 + LoRA rank 32",
        "L_adv (rebalanced + rank32)",
        lambda_cls=4.0,
        lambda_ot=0.2,
        lambda_align=0.2,
        lambda_colpo_bridge_ot=0.1,
        lambda_colpo_bridge_align=0.02,
        shared_lora_rank=32,
        shared_lora_alpha=64.0,
    ),
}

# Full leave-one-out suite for reviewer-facing module ablation (g1 = control).
FULL_ABLATION_SUITE = [
    "g1",
    "g0",
    "g2",
    "g3",
    "no_align",
    "no_ot",
    "no_bridge",
    "no_coe",
    "no_consist",
    "no_posterior",
    "no_variational",
    "no_modality_likelihood",
    "no_cross_attn",
    "no_bridge_module",
    "no_center_reliability",
]

DEFAULT_ABLATION_SUITE = FULL_ABLATION_SUITE


def group_is_complete(group_root: Path) -> bool:
    metrics_path = group_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
    if not metrics_path.exists():
        return False
    df = pd.read_csv(metrics_path)
    return len(df) >= 5 and df["cin2plus_auc"].notna().sum() >= 4


def all_ablation_groups_complete(ablation_root: Path | None = None) -> bool:
    root = ablation_root or (OUT_ROOT_DEFAULT / "ablations")
    for group in FULL_ABLATION_SUITE:
        if not group_is_complete(root / group):
            return False
    return True


def pending_ablation_groups(ablation_root: Path | None = None) -> list[str]:
    root = ablation_root or (OUT_ROOT_DEFAULT / "ablations")
    return [g for g in FULL_ABLATION_SUITE if not group_is_complete(root / g)]


def threshold_sensitivity_constrained(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target: float = 0.90,
    fallback: float = 0.85,
) -> float:
    """G5-b: maximize specificity subject to sensitivity >= target."""
    from sklearn.metrics import roc_curve

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
    mask = tpr >= target
    if mask.sum() == 0:
        mask = tpr >= fallback
    if mask.sum() == 0:
        return threshold_for_sensitivity(y_true_arr, y_prob_arr, target=fallback)
    feasible_fpr = fpr[mask]
    feasible_thr = thresholds[mask]
    return float(feasible_thr[int(np.argmin(feasible_fpr))])


def append_improved_config_fields(
    config_path: Path,
    feature_cache_path: Path,
    case_manifest: Path,
    preset: dict[str, Any],
) -> None:
    text = config_path.read_text(encoding="utf-8")
    lines = [
        "",
        "    use_cached_patch_features: bool = True",
        f"    feature_cache_path: str = r\"{feature_cache_path}\"",
        "    expected_aligned_n: int = 1897",
        "    vit_pretrained: bool = True",
        "    pass_raw_oct_to_model: bool = False",
        "    pass_raw_colpo_to_model: bool = False",
        f"    case_manifest_csv: str = r\"{case_manifest}\"",
    ]
    for key, value in preset.items():
        if key in {"description", "removed_components"}:
            continue
        if isinstance(value, bool):
            lines.append(f"    {key}: bool = {value}")
        elif isinstance(value, int):
            lines.append(f"    {key}: int = {value}")
        elif isinstance(value, float):
            lines.append(f"    {key}: float = {value}")
        else:
            lines.append(f"    {key} = {value!r}")
    extra = "\n".join(lines) + "\n"
    if "use_cached_patch_features" not in text:
        text = text.rstrip() + extra
        config_path.write_text(text, encoding="utf-8")


def evaluate_split(
    config: Path,
    checkpoint: Path,
    csv_path: Path,
    split_name: str,
    run_id: str,
    seed: int,
    pred_root: Path,
    log_path: Path,
    python: Path,
    batch_size: int,
    env: dict[str, str],
) -> None:
    run_command(
        [
            str(python),
            "paper_revision/scripts/evaluate_checkpoint_predictions.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--csv",
            str(csv_path),
            "--split",
            split_name,
            "--method",
            "SharedLoRA_BioCOT",
            "--run_id",
            run_id,
            "--seed",
            str(seed),
            "--output-dir",
            str(pred_root),
            "--batch-size",
            str(batch_size),
        ],
        log_path,
        env,
    )


def run_training_job(
    config_path: Path,
    train_csv: Path,
    val_csv: Path,
    log_path: Path,
    python: Path,
    seed: int,
    env: dict[str, str],
    skip_train: bool,
) -> Path:
    if not skip_train:
        run_command(
            [
                str(python),
                "training/train_bio_cot_v3.2.py",
                "--config",
                str(config_path),
                "--train-csv",
                str(train_csv),
                "--val-csv",
                str(val_csv),
                "--seed",
                str(seed),
            ],
            log_path,
            env,
        )
    text = config_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("checkpoint_dir:"):
            raw = line.split("=", 1)[1].strip()
            if raw.startswith("r"):
                raw = raw[1:].strip().strip('"').strip("'")
            else:
                raw = raw.strip().strip('"').strip("'")
            return newest_checkpoint(Path(raw))
    raise RuntimeError(f"Could not resolve checkpoint_dir from {config_path}")


def aggregate_results_g5b(out_root: Path, fold_rows: list[dict[str, object]]) -> None:
    """Like aggregate_results but uses G5-b CIN3+ threshold (sens>=0.90, max spec)."""
    pred_dir = out_root / "predictions"
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    report_dir = out_root / "reports"
    for path in [table_dir, fig_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    patient_predictions = []
    for fold in fold_rows:
        fold_id = str(fold["fold_id"])
        run_id = safe_name(fold_id)
        val_path = pred_dir / f"SharedLoRA_BioCOT_run{run_id}_seed{fold['seed']}_val_full.csv"
        test_path = pred_dir / f"SharedLoRA_BioCOT_run{run_id}_seed{fold['seed']}_external_test_full.csv"
        if not val_path.exists() or not test_path.exists():
            continue
        val = read_csv(val_path)
        test = read_csv(test_path)
        test_split = read_csv(Path(fold["test_csv"]))
        labels = test_split[["ID", "pathology_cin3plus", "center_name"]].rename(columns={"ID": "case_id"})
        test = test.merge(labels, on="case_id", how="left", suffixes=("", "_split"))
        val_split = read_csv(Path(fold["val_csv"]))
        val_labels = val_split[["ID", "pathology_cin3plus"]].rename(columns={"ID": "case_id"})
        val = val.merge(val_labels, on="case_id", how="left")

        thr_cin2 = threshold_max_f1(val["y_true"].to_numpy(), val["y_prob"].to_numpy())
        thr_cin3 = threshold_sensitivity_constrained(
            val["pathology_cin3plus"].fillna(0).astype(int).to_numpy(),
            val["y_prob"].to_numpy(),
            target=0.90,
        )
        test["y_prob_target_free_calibrated"], target_free_shift = target_free_logit_median_match(
            val["y_prob"].to_numpy(),
            test["y_prob"].to_numpy(),
        )
        cin2 = binary_metrics(test["y_true"], test["y_prob"], thr_cin2)
        cin3 = binary_metrics(test["pathology_cin3plus"].fillna(0).astype(int), test["y_prob"], thr_cin3)
        cin2_tfc = binary_metrics(test["y_true"], test["y_prob_target_free_calibrated"], thr_cin2)
        cin3_tfc = binary_metrics(
            test["pathology_cin3plus"].fillna(0).astype(int),
            test["y_prob_target_free_calibrated"],
            thr_cin3,
        )
        row = {
            "fold_id": fold_id,
            "held_out_center": fold["held_out_center"],
            "n_test": len(test),
            "cin2plus_prevalence": float(test["y_true"].mean()),
            "cin3plus_prevalence": float(test["pathology_cin3plus"].fillna(0).mean()),
            "target_free_logit_shift": target_free_shift,
            **{f"cin2plus_{k}": v for k, v in cin2.items()},
            **{f"cin3plus_{k}": v for k, v in cin3.items()},
            **{f"cin2plus_tfc_{k}": v for k, v in cin2_tfc.items()},
            **{f"cin3plus_tfc_{k}": v for k, v in cin3_tfc.items()},
            "checkpoint": fold["checkpoint"],
            "train_log": fold["train_log"],
            "val_predictions": str(val_path.relative_to(EXP_ROOT)),
            "test_predictions": str(test_path.relative_to(EXP_ROOT)),
        }
        rows.append(row)
        test["fold_id"] = fold_id
        test["held_out_center"] = fold["held_out_center"]
        test["cin2_threshold_from_val"] = thr_cin2
        test["cin3_safety_threshold_from_val"] = thr_cin3
        patient_predictions.append(test)

    if not rows:
        raise RuntimeError("No fold predictions found to aggregate.")

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(table_dir / "Table_Improved1897_LOCO_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    if patient_predictions:
        pd.concat(patient_predictions, ignore_index=True).to_csv(
            pred_dir / "Improved1897_LOCO_All_Patient_Predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_rows = []
    metric_cols = [
        c
        for c in fold_metrics.columns
        if c.startswith(("cin2plus_", "cin3plus_")) and pd.api.types.is_numeric_dtype(fold_metrics[c])
    ]
    for metric in metric_cols:
        values = fold_metrics[metric].dropna()
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "Table_Improved1897_LOCO_Aggregate_Metrics.csv", index=False, encoding="utf-8-sig")

    cin2_auc = fold_metrics["cin2plus_auc"].dropna()
    report_lines = [
        "# Improved 1897 LOCO Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Mean CIN2+ AUC (valid folds): **{cin2_auc.mean():.4f}** ± {cin2_auc.std(ddof=1):.4f}",
        "",
        "## Fold Metrics",
        "",
        fold_metrics.to_markdown(index=False),
        "",
        "## Aggregate Metrics",
        "",
        summary.to_markdown(index=False),
    ]
    (report_dir / "Report_Improved1897_LOCO.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_ablation_group(
    group: str,
    out_root: Path,
    split_root: Path,
    expert_checkpoint: Path,
    python: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    env: dict[str, str],
    skip_train: bool,
    skip_eval: bool,
    fold_filter: list[str] | None,
) -> dict[str, Any]:
    if group not in ABLATION_PRESETS:
        raise ValueError(f"Unknown group {group}; choose from {list(ABLATION_PRESETS)}")

    preset = ABLATION_PRESETS[group]
    group_root = out_root / "ablations" / group
    config_root = group_root / "configs"
    logs_root = group_root / "logs"
    pred_root = group_root / "predictions"
    runs_root = group_root / "runs"
    for path in [config_root, logs_root, pred_root, runs_root, group_root / "tables", group_root / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    (group_root / "preset.json").write_text(
        json.dumps({"group": group, **preset}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state_rows: list[dict[str, object]] = []
    for fold_dir in sorted(split_root.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("loco_"):
            continue
        fold_id = fold_dir.name
        if fold_filter and fold_id not in fold_filter:
            continue
        held_out = fold_id.replace("loco_", "")
        run_out = runs_root / fold_id
        config_path = config_root / f"{fold_id}_config.py"
        write_config(
            config_path,
            fold_dir,
            run_out,
            epochs,
            batch_size,
            0,
            8,
            f"1897 improved {group} {fold_id}",
            pretrain_without_colpo=False,
            load_expert_checkpoint=expert_checkpoint,
            colpo_pretrained=True,
        )
        append_improved_config_fields(config_path, CACHE_1897, CASE_MANIFEST, preset)

        train_log = logs_root / f"{fold_id}_train.log"
        print(f"[{group}] Training {fold_id} ...")
        checkpoint = run_training_job(
            config_path,
            fold_dir / "train_labels.csv",
            fold_dir / "val_labels.csv",
            train_log,
            python,
            seed,
            env,
            skip_train,
        )
        if not skip_eval:
            for split_name, csv_path in [
                ("val", fold_dir / "val_labels.csv"),
                ("external_test", fold_dir / "external_test_labels.csv"),
            ]:
                evaluate_split(
                    config_path,
                    checkpoint,
                    csv_path,
                    split_name,
                    fold_id,
                    seed,
                    pred_root,
                    logs_root / f"{fold_id}_eval_{split_name}.log",
                    python,
                    batch_size,
                    env,
                )
        state_rows.append(
            {
                "fold_id": fold_id,
                "held_out_center": held_out,
                "seed": seed,
                "config": str(config_path),
                "train_csv": str(fold_dir / "train_labels.csv"),
                "val_csv": str(fold_dir / "val_labels.csv"),
                "test_csv": str(fold_dir / "external_test_labels.csv"),
                "checkpoint": str(checkpoint),
                "train_log": str(train_log),
            }
        )

    (group_root / "loco_run_state.json").write_text(
        json.dumps(state_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    aggregate_results_g5b(group_root, state_rows)

    fold_df = read_csv(group_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv")
    auc_vals = fold_df["cin2plus_auc"].dropna()
    summary = {
        "group": group,
        "description": preset["description"],
        "mean_cin2plus_auc": float(auc_vals.mean()) if len(auc_vals) else None,
        "std_cin2plus_auc": float(auc_vals.std(ddof=1)) if len(auc_vals) > 1 else 0.0,
        "n_valid_folds": int(len(auc_vals)),
        "fold_metrics_path": str(group_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"),
    }
    print(f"[{group}] mean CIN2+ AUC = {summary['mean_cin2plus_auc']:.4f} (n={summary['n_valid_folds']})")
    return summary


def build_ablation_comparison(out_root: Path, summaries: list[dict[str, Any]]) -> None:
    table_dir = out_root / "tables"
    report_dir = out_root / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary in summaries:
        rows.append(summary)
    if BASELINE_METRICS.exists():
        base = read_csv(BASELINE_METRICS)
        base_auc = base["cin2plus_auc"].dropna()
        rows.insert(
            0,
            {
                "group": "baseline_formal_raw",
                "description": "Original formal LOCO (raw images, default losses)",
                "removed_components": "none (raw colpo path)",
                "mean_cin2plus_auc": float(base_auc.mean()) if len(base_auc) else None,
                "std_cin2plus_auc": float(base_auc.std(ddof=1)) if len(base_auc) > 1 else 0.0,
                "n_valid_folds": int(len(base_auc)),
                "fold_metrics_path": str(BASELINE_METRICS),
                "status": "reference",
            },
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(table_dir / "Table_Improved1897_Ablation_Comparison.csv", index=False, encoding="utf-8-sig")

    best_row = comparison.loc[comparison["mean_cin2plus_auc"].idxmax()]
    lines = [
        "# 1897 Improved LOCO Ablation Comparison",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Best group by mean CIN2+ AUC: **{best_row['group']}** ({best_row['mean_cin2plus_auc']:.4f})",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Diagnostic summary (DX-01)",
        "",
        "Baseline per-fold AUC: 十堰 0.767, 恩施 0.690, 武大 N/A, 荆州 0.547, 襄阳 0.549.",
        "Predictions concentrated ~0.35–0.55 → weak discrimination + 2 weak folds.",
        "Interventions: G3 loss ablation (disable adv), G4 rebalanced cls, cached features, G5-b threshold.",
    ]
    (report_dir / "Report_Improved1897_Ablation_Comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        choices=list(ABLATION_PRESETS),
        help="Ablation group(s) to run sequentially. Default: full component ablation suite.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip groups that already have 5-fold metrics under ablations/<group>/.",
    )
    parser.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    parser.add_argument("--split-root", default=str(FORMAL_SPLIT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expert-checkpoint", default=str(FORMAL_PRETRAIN_CKPT))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fold", action="append", default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = EXP_ROOT / out_root
    split_root = Path(args.split_root)
    expert_checkpoint = Path(args.expert_checkpoint)
    if not expert_checkpoint.exists():
        raise FileNotFoundError(f"Expert checkpoint not found: {expert_checkpoint}")
    if not CACHE_1897.exists():
        raise FileNotFoundError(f"1897 feature cache not found: {CACHE_1897}")
    if not split_root.exists():
        raise FileNotFoundError(f"LOCO splits not found: {split_root}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    groups = args.group if args.group else DEFAULT_ABLATION_SUITE
    summaries = []
    for group in groups:
        group_root = out_root / "ablations" / group
        if args.skip_complete and group_is_complete(group_root):
            print(f"[{group}] already complete, skipping.")
            preset = ABLATION_PRESETS[group]
            fold_df = read_csv(group_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv")
            auc_vals = fold_df["cin2plus_auc"].dropna()
            summaries.append(
                {
                    "group": group,
                    "description": preset["description"],
                    "removed_components": preset.get("removed_components", ""),
                    "mean_cin2plus_auc": float(auc_vals.mean()) if len(auc_vals) else None,
                    "std_cin2plus_auc": float(auc_vals.std(ddof=1)) if len(auc_vals) > 1 else 0.0,
                    "n_valid_folds": int(len(auc_vals)),
                    "fold_metrics_path": str(group_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"),
                    "status": "skipped_complete",
                }
            )
            continue
        summary = run_ablation_group(
            group=group,
            out_root=out_root,
            split_root=split_root,
            expert_checkpoint=expert_checkpoint,
            python=Path(args.python),
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            env=env,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            fold_filter=args.fold,
        )
        summary["removed_components"] = ABLATION_PRESETS[group].get("removed_components", "")
        summary["status"] = "completed"
        summaries.append(summary)

    build_ablation_comparison(out_root, summaries)
    print(f"Comparison written to {out_root / 'tables/Table_Improved1897_Ablation_Comparison.csv'}")


if __name__ == "__main__":
    main()
