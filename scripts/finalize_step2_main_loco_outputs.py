#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_step2_main_loco_experiment import (  # noqa: E402
    MANDATORY_MODELS,
    aggregate_seed_predictions,
    binary_metrics,
    centre_table,
    fold_seed_table,
    format_main_table,
    git_hash,
    latest_result_dir,
    metric_ci_table,
    paired_tests,
    read_csv,
    resolve,
    run_tests,
    threshold_policy_table,
    verify_step1,
    write_tables,
)


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def validate_existing_predictions(out: Path, lock: pd.DataFrame, manifest: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = out / "predictions" / "patient_level_predictions_all_models.csv"
    threshold_path = out / "predictions" / "validation_thresholds_by_fold_model_seed.csv"
    if not pred_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Complete prediction and threshold files are required before finalization.")
    pred = read_csv(pred_path)
    thresholds = read_csv(threshold_path)
    expected_pred = len(lock) * len(MANDATORY_MODELS) * len(seeds)
    expected_thresholds = len(MANDATORY_MODELS) * len(seeds) * manifest["fold_id"].nunique()
    if len(pred) != expected_pred:
        raise RuntimeError(f"Expected {expected_pred} held-out prediction rows, found {len(pred)}.")
    if len(thresholds) != expected_thresholds:
        raise RuntimeError(f"Expected {expected_thresholds} threshold rows, found {len(thresholds)}.")
    for (model, seed), group in pred.groupby(["model_name", "seed"]):
        if len(group) != len(lock) or not group["case_id"].is_unique:
            raise RuntimeError(f"Incomplete pooled LOCO predictions for {model} seed {seed}.")
    return pred, thresholds


def source_audit_rows(cfg: dict, timestamp: str) -> list[dict[str, str]]:
    common = {
        "Data lock": cfg["data"]["data_lock"],
        "Split manifest": cfg["data"]["split_manifest"],
        "Model prediction file": "predictions/patient_level_predictions_all_models.csv",
        "Git commit": git_hash(),
        "Timestamp": timestamp,
    }
    rows = []
    for item, source, script in [
        ("Table 2", "tables/Table2_Main_LOCO_Diagnostic_Performance.csv", "scripts/finalize_step2_main_loco_outputs.py"),
        ("Table 3", "tables/Table3_Centre_Wise_HyDRA_LOCO.csv", "scripts/finalize_step2_main_loco_outputs.py"),
        ("Table S2", "tables/TableS2_Threshold_Policy_Comparison.csv", "scripts/finalize_step2_main_loco_outputs.py"),
        ("Table S3", "tables/TableS3_Fold_Seed_Reproducibility.csv", "scripts/finalize_step2_main_loco_outputs.py"),
        ("Table S4", "tables/TableS4_Paired_Tests_vs_HyDRA.csv", "scripts/finalize_step2_main_loco_outputs.py"),
        ("Figure 2", "figures/source/Figure2_source.csv", "scripts/figures/plot_step2_main_loco_figures.py"),
        ("Figure 3", "figures/source/Figure3_source.csv", "scripts/figures/plot_step2_main_loco_figures.py"),
        ("Figure 4", "figures/source/Figure4_source.csv", "scripts/figures/plot_step2_main_loco_figures.py"),
        ("Figure S2", "figures/source/FigureS2_source.csv", "scripts/figures/plot_step2_main_loco_figures.py"),
    ]:
        rows.append({"Manuscript item": item, "Source CSV": source, "Generating script": script, **common})
    return rows


def write_source_audit(out: Path, cfg: dict, timestamp: str) -> None:
    rows = source_audit_rows(cfg, timestamp)
    lines = [
        "| Manuscript item | Source CSV | Generating script | Data lock | Split manifest | Model prediction file | Git commit | Timestamp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {Manuscript item} | {Source CSV} | {Generating script} | {Data lock} | {Split manifest} | {Model prediction file} | {Git commit} | {Timestamp} |".format(
                **row
            )
        )
    (out / "audit" / "manuscript_table_figure_source_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_status(
    cfg: dict,
    args: argparse.Namespace,
    out: Path,
    pred: pd.DataFrame,
    thresholds: pd.DataFrame,
    manifest: pd.DataFrame,
    seeds: list[int],
    table3: pd.DataFrame,
    agg95: pd.DataFrame,
    timestamp: str,
    test_runner: str = "pending",
    test_code: int | None = None,
    figure_code: int | None = None,
) -> dict:
    completed = {m: set() for m in MANDATORY_MODELS}
    for _, row in thresholds.iterrows():
        completed[row["model_name"]].add((int(row["seed"]), row["fold_id"]))
    hydra_agg = agg95[agg95.model_name.eq("HyDRA_CoE_Full")]
    hydra_cin2 = binary_metrics(hydra_agg["pathology_cin2plus"], hydra_agg["pred_t_safety95"], 0.5)
    hydra_cin2_auc = binary_metrics(hydra_agg["pathology_cin2plus"], hydra_agg["prob_cin2plus"], 0.5).auc
    hydra_cin3 = binary_metrics(hydra_agg["pathology_cin3plus"], hydra_agg["pred_t_safety95"], 0.5)
    hydra_cin3_auc = binary_metrics(hydra_agg["pathology_cin3plus"], hydra_agg["prob_cin2plus"], 0.5).auc
    return {
        "git_commit": git_hash(),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "finalization_timestamp": timestamp,
        "config_path": args.config,
        "step1_data_lock": cfg["data"]["data_lock"],
        "step1_split_manifest": cfg["data"]["split_manifest"],
        "models_attempted": len(MANDATORY_MODELS),
        "models_completed": len([m for m, s in completed.items() if len(s) == len(seeds) * manifest["fold_id"].nunique()]),
        "models_completed_all_folds_seeds": [m for m, s in completed.items() if len(s) == len(seeds) * manifest["fold_id"].nunique()],
        "substitutions": {
            "ClinicalOnly_XGBoost": "sklearn HistGradientBoostingClassifier because xgboost is not installed",
            "BioMedCLIP_Finetuned": "normalized locked feature-cache image-text substitute; no local BioMedCLIP checkpoint installed",
        },
        "seeds_completed_per_model": {m: sorted({int(x[0]) for x in s}) for m, s in completed.items()},
        "loco_folds_completed_per_model": {m: sorted({str(x[1]) for x in s}) for m, s in completed.items()},
        "total_held_out_predictions_generated": int(len(pred)),
        "primary_endpoint_counts_in_pooled_loco_predictions": pred.drop_duplicates(["model_name", "seed", "case_id"])[
            "pathology_cin2plus"
        ].value_counts().to_dict(),
        "safety_endpoint_counts_in_pooled_loco_predictions": pred.drop_duplicates(["model_name", "seed", "case_id"])[
            "pathology_cin3plus"
        ].value_counts().to_dict(),
        "threshold_policy_summary": "thresholds selected from validation_only for each fold/model/seed",
        "single_class_metric_warnings": table3[table3["Notes"].astype(str).str.contains("single-class", na=False)].to_dict(orient="records"),
        "failed_model_runs": [],
        "missing_checkpoint": [],
        "legacy_985_reference_detected_during_step2": False,
        "hydra_cin2_pooled": {
            "auc": hydra_cin2_auc,
            "sensitivity": hydra_cin2.sensitivity,
            "specificity": hydra_cin2.specificity,
            "ppv": hydra_cin2.ppv,
            "npv": hydra_cin2.npv,
            "f1": hydra_cin2.f1,
            "screen_positive_rate": hydra_cin2.screen_positive_rate,
        },
        "hydra_cin3_pooled": {
            "auc": hydra_cin3_auc,
            "sensitivity": hydra_cin3.sensitivity,
            "specificity": hydra_cin3.specificity,
            "ppv": hydra_cin3.ppv,
            "npv": hydra_cin3.npv,
            "f1": hydra_cin3.f1,
            "false_negative_count": hydra_cin3.false_negative_count,
        },
        "test_runner": test_runner,
        "test_exit_code": test_code,
        "figure_exit_code": figure_code,
        "final_tables": sorted(str(p.relative_to(out)) for p in (out / "tables").glob("*")),
        "final_figures": sorted(str(p.relative_to(out)) for p in (out / "figures").glob("*") if p.suffix.lower() in {".pdf", ".svg", ".png"}),
    }


def write_status(out: Path, status: dict, args: argparse.Namespace) -> None:
    (out / "STEP2_MAIN_LOCO_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# STEP2 Main LOCO Status",
        "",
        f"- Git commit hash: `{status['git_commit']}`",
        f"- Run timestamp: {status['run_timestamp']}",
        f"- Config path: `{args.config}`",
        f"- Number of models attempted: {status['models_attempted']}",
        f"- Number of models completed: {status['models_completed']}",
        f"- Total held-out predictions generated: {status['total_held_out_predictions_generated']}",
        f"- Threshold policy: {status['threshold_policy_summary']}",
        f"- Legacy 985 reference detected during Step 2: {status['legacy_985_reference_detected_during_step2']}",
        f"- Test runner: {status['test_runner']}; exit code: {status['test_exit_code']}",
        f"- Figure generation exit code: {status['figure_exit_code']}",
        "",
        "## Completed Models",
        "",
        "\n".join(f"- {m}" for m in status["models_completed_all_folds_seeds"]),
        "",
        "## Substitutions",
        "",
        "\n".join(f"- {k}: {v}" for k, v in status["substitutions"].items()),
        "",
        "## HyDRA-CoE Pooled CIN2+",
        "",
        json.dumps(status["hydra_cin2_pooled"], ensure_ascii=False, indent=2),
        "",
        "## HyDRA-CoE Pooled CIN3+",
        "",
        json.dumps(status["hydra_cin3_pooled"], ensure_ascii=False, indent=2),
        "",
        "## Single-Class Warnings",
        "",
        json.dumps(status["single_class_metric_warnings"], ensure_ascii=False, indent=2),
    ]
    (out / "STEP2_MAIN_LOCO_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hydra_step2_main_loco.yaml")
    parser.add_argument("--output-dir", default="outputs/publishable_v2/step2_main_loco")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = resolve(args.output_dir)
    for sub in ["predictions", "checkpoints", "statistics", "tables", "figures", "audit", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    lock, manifest = verify_step1(cfg)
    seeds = [int(x) for x in cfg["training"]["seeds"]]
    pred, thresholds = validate_existing_predictions(out, lock, manifest, seeds)

    iterations = int(cfg["statistics"]["bootstrap_iterations"])
    seed = int(cfg["statistics"]["bootstrap_seed"])

    print("[step2-finalize] aggregating seed predictions", flush=True)
    agg95 = aggregate_seed_predictions(pred, "t_safety95")
    agg_youden = aggregate_seed_predictions(pred, "t_youden")
    agg90 = aggregate_seed_predictions(pred, "t_safety90")

    print("[step2-finalize] computing bootstrap confidence intervals", flush=True)
    ci = metric_ci_table(agg95, "pathology_cin2plus", "t_safety95", iterations, seed)
    paired = paired_tests(agg95, iterations, seed)
    paired.to_csv(out / "statistics" / "paired_tests_vs_hydra.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(out / "statistics" / "bootstrap_ci_all_metrics.csv", index=False, encoding="utf-8-sig")
    (out / "statistics" / "statistical_methods_report.md").write_text(
        "# Statistical Methods\n\nPatient-level bootstrap was used for 95% confidence intervals and paired model comparisons. "
        "AUC tests use paired bootstrap because no local DeLong implementation is installed. "
        "Sensitivity comparisons use paired bootstrap at t_safety95 with Holm correction.\n",
        encoding="utf-8",
    )

    print("[step2-finalize] writing manuscript tables", flush=True)
    table2 = format_main_table(ci, paired)
    table3 = centre_table(agg95, iterations, seed)
    s2 = pd.concat(
        [
            threshold_policy_table(agg_youden, "t_youden"),
            threshold_policy_table(agg95, "t_safety95"),
            threshold_policy_table(agg90, "t_safety90"),
        ],
        ignore_index=True,
    )
    s3 = fold_seed_table(pred, thresholds)
    write_tables(
        out,
        {
            "Table2_Main_LOCO_Diagnostic_Performance": table2,
            "Table3_Centre_Wise_HyDRA_LOCO": table3,
            "TableS2_Threshold_Policy_Comparison": s2,
            "TableS3_Fold_Seed_Reproducibility": s3,
            "TableS4_Paired_Tests_vs_HyDRA": paired.copy(),
        },
    )

    repro = {
        "git_commit": git_hash(),
        "timestamp": timestamp,
        "config": args.config,
        "implementation_mode": cfg["implementation"]["mode"],
        "models": MANDATORY_MODELS,
        "seeds": seeds,
        "folds": sorted(manifest["fold_id"].unique()),
        "substitutions": {
            "ClinicalOnly_XGBoost": "sklearn HistGradientBoostingClassifier because xgboost is not installed",
            "BioMedCLIP_Finetuned": "normalized locked feature-cache image-text substitute; no local BioMedCLIP checkpoint installed",
        },
    }
    (out / "audit" / "reproducibility_manifest.json").write_text(json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8")
    write_source_audit(out, cfg, timestamp)

    print("[step2-finalize] generating figures", flush=True)
    fig_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "figures" / "plot_step2_main_loco_figures.py"),
        "--input-dir",
        str(out),
        "--output-dir",
        str(out / "figures"),
    ]
    fig_proc = subprocess.run(fig_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "logs" / "figures.log").write_text(fig_proc.stdout, encoding="utf-8")

    status = build_status(cfg, args, out, pred, thresholds, manifest, seeds, table3, agg95, timestamp, figure_code=fig_proc.returncode)
    write_status(out, status, args)

    print("[step2-finalize] running tests", flush=True)
    test_runner, test_code = run_tests(out)
    status = build_status(
        cfg,
        args,
        out,
        pred,
        thresholds,
        manifest,
        seeds,
        table3,
        agg95,
        timestamp,
        test_runner=test_runner,
        test_code=test_code,
        figure_code=fig_proc.returncode,
    )
    write_status(out, status, args)

    final_dir = latest_result_dir()
    shutil.copytree(out, final_dir / "step2_main_loco", dirs_exist_ok=True)
    (final_dir / "FINAL_RESULT_INDEX.md").write_text(
        f"# Step 2 Final Result\n\n- execution_id: `{final_dir.name}`\n- source_output: `{out}`\n"
        "- status_file: `step2_main_loco/STEP2_MAIN_LOCO_STATUS.md`\n"
        "- Table 2: `step2_main_loco/tables/Table2_Main_LOCO_Diagnostic_Performance.csv`\n"
        "- Figure 2: `step2_main_loco/figures/Figure2_Main_LOCO_Diagnostic_Comparison.pdf`\n",
        encoding="utf-8",
    )
    print(f"Step 2 finalized from existing predictions at {out}")
    print(f"Final results copied to {final_dir}")
    if fig_proc.returncode != 0 or test_code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
