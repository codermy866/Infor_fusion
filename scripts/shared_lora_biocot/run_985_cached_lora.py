#!/usr/bin/env python3
"""Run cached-feature Shared-LoRA experiments on the balanced 985-case cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from data.cached_patch_dataset import case_key
from scripts.shared_lora_biocot.build_985_splits import build_985_splits
from scripts.shared_lora_biocot.run_formal_loco import (
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


OUT_ROOT_DEFAULT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/cached_985"
SOURCE_ROOT_DEFAULT = Path("/data2/hmy/5Center_datas/5centers_multi_leave_centers_out")
CACHE_1897 = EXP_ROOT / "paper_revision/cache/patch_features_final_1897.pt"
FORMAL_PRETRAIN_CKPT = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/pretrain_oct_text_expert/checkpoints/best_model_v3_20260526_215640.pth"
)
FORMAL_LOCO_METRICS = (
    EXP_ROOT
    / "outputs/publishable_v2/shared_lora_biocot/formal_loco/tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv"
)


def append_cached_config_fields(config_path: Path, feature_cache_path: Path, case_manifest: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    extra = f"""
    use_cached_patch_features: bool = True
    feature_cache_path: str = r"{feature_cache_path}"
    expected_aligned_n: int = 985
    vit_pretrained: bool = True
    pass_raw_oct_to_model: bool = False
    pass_raw_colpo_to_model: bool = False
    num_workers: int = 0
    case_manifest_csv: str = r"{case_manifest}"
"""
    if "use_cached_patch_features" not in text:
        text = text.rstrip() + "\n" + extra
        config_path.write_text(text, encoding="utf-8")


def build_985_feature_cache(
    manifest_csv: Path,
    output_cache: Path,
    source_cache: Path,
    python: Path,
    gpu: str,
) -> None:
    manifest = read_csv(manifest_csv)
    keys = [case_key(row) for _, row in manifest.iterrows()]
    payload = torch.load(source_cache, map_location="cpu", weights_only=False)
    source_features = payload.get("features", payload)
    features = {key: source_features[key] for key in keys if key in source_features}
    missing_keys = [key for key in keys if key not in features]
    metadata = {
        "source_case_index": str(manifest_csv),
        "expected_aligned_n": 985,
        "subset_from": str(source_cache),
        "missing_after_subset": missing_keys,
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "metadata": metadata, "config": "cached_985_subset"}, output_cache)
    print(f"Subset cache: {len(features)}/{len(keys)} cases -> {output_cache}")
    if not missing_keys:
        return

    print(f"Precomputing {len(missing_keys)} missing cases from images...")
    precompute_config = output_cache.parent / "precompute_985_config.py"
    official_root = manifest_csv.parent / "official"
    precompute_config.write_text(
        f'''#!/usr/bin/env python3
from dataclasses import dataclass
from configs.shared_lora_loco_template import SharedLoRALOCOConfig

@dataclass
class Precompute985Config(SharedLoRALOCOConfig):
    data_root: str = r"{official_root}"
    feature_cache_path: str = r"{output_cache}"
    oct_frames: int = 8
    colposcopy_images: int = 3
    vit_batch_size: int = 8
    vit_pretrained: bool = True
    expected_aligned_n: int = 985
''',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    run_command(
        [
            str(python),
            "paper_revision/scripts/precompute_patch_feature_cache.py",
            "--config",
            str(precompute_config),
            "--output",
            str(output_cache),
            "--batch-size",
            "4",
        ],
        output_cache.parent / "precompute_missing.log",
        env,
    )
    payload = torch.load(output_cache, map_location="cpu", weights_only=False)
    final_features = payload.get("features", payload)
    still_missing = [key for key in keys if key not in final_features]
    if still_missing:
        raise RuntimeError(f"Feature cache still missing {len(still_missing)} cases: {still_missing[:3]}")


@dataclass
class ExperimentState:
    run_id: str
    held_out_center: str
    seed: int
    config: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    checkpoint: Path
    train_log: Path


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


def metrics_from_predictions(
    val_csv: Path,
    test_csv: Path,
    val_pred: Path,
    test_pred: Path,
    held_out_center: str,
) -> dict[str, object]:
    val = read_csv(val_pred)
    test = read_csv(test_pred)
    val_labels = read_csv(val_csv)
    test_labels = read_csv(test_csv)
    val = val.merge(
        val_labels[["ID", "pathology_cin3plus"]].rename(columns={"ID": "case_id"}),
        on="case_id",
        how="left",
    )
    test = test.merge(
        test_labels[["ID", "pathology_cin3plus", "center_name"]].rename(columns={"ID": "case_id"}),
        on="case_id",
        how="left",
    )
    thr_cin2 = threshold_max_f1(val["y_true"].to_numpy(), val["y_prob"].to_numpy())
    thr_cin3 = threshold_for_sensitivity(
        val["pathology_cin3plus"].fillna(0).astype(int).to_numpy(),
        val["y_prob"].to_numpy(),
        target=0.95,
    )
    test["y_prob_target_free_calibrated"], shift = target_free_logit_median_match(
        val["y_prob"].to_numpy(),
        test["y_prob"].to_numpy(),
    )
    cin2 = binary_metrics(test["y_true"], test["y_prob"], thr_cin2)
    cin3 = binary_metrics(test["pathology_cin3plus"].fillna(0).astype(int), test["y_prob"], thr_cin3)
    return {
        "held_out_center": held_out_center,
        "n_test": len(test),
        "cin2plus_prevalence": float(test["y_true"].mean()),
        "cin3plus_prevalence": float(test["pathology_cin3plus"].fillna(0).mean()),
        "target_free_logit_shift": shift,
        **{f"cin2plus_{k}": v for k, v in cin2.items()},
        **{f"cin3plus_{k}": v for k, v in cin3.items()},
        "val_predictions": str(val_pred),
        "test_predictions": str(test_pred),
    }


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


def write_official_report(
    out_root: Path,
    row: dict[str, object],
    protocol: str,
) -> None:
    report_dir = out_root / "reports"
    table_dir = out_root / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame([row])
    fold_df.to_csv(table_dir / f"Table_985_{protocol}_Metrics.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# 985 Cached Shared-LoRA {protocol} Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        fold_df.to_markdown(index=False),
    ]
    (report_dir / f"Report_985_{protocol}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison_table(out_root: Path) -> Path:
    rows = []
    loco_985 = out_root / "tables/Table_SharedLoRA_985_LOCO_Fold_Metrics.csv"
    if loco_985.exists():
        df985 = read_csv(loco_985)
        for _, row in df985.iterrows():
            rows.append(
                {
                    "protocol": "985_LOCO_cached",
                    "held_out_center": row["held_out_center"],
                    "n_test": row["n_test"],
                    "cin2plus_auc": row.get("cin2plus_auc"),
                    "cin2plus_sensitivity": row.get("cin2plus_sensitivity"),
                    "cin2plus_specificity": row.get("cin2plus_specificity"),
                    "cin3plus_auc": row.get("cin3plus_auc"),
                }
            )
    official_985 = out_root / "tables/Table_985_official_external_Metrics.csv"
    if official_985.exists():
        df = read_csv(official_985)
        if len(df):
            r = df.iloc[0]
            rows.append(
                {
                    "protocol": "985_official_external_cached",
                    "held_out_center": "十堰+荆州 (official external)",
                    "n_test": r.get("n_test"),
                    "cin2plus_auc": r.get("cin2plus_auc"),
                    "cin2plus_sensitivity": r.get("cin2plus_sensitivity"),
                    "cin2plus_specificity": r.get("cin2plus_specificity"),
                    "cin3plus_auc": r.get("cin3plus_auc"),
                }
            )
    if FORMAL_LOCO_METRICS.exists():
        df1897 = read_csv(FORMAL_LOCO_METRICS)
        for _, row in df1897.iterrows():
            rows.append(
                {
                    "protocol": "1897_LOCO_formal",
                    "held_out_center": row["held_out_center"],
                    "n_test": row["n_test"],
                    "cin2plus_auc": row.get("cin2plus_auc"),
                    "cin2plus_sensitivity": row.get("cin2plus_sensitivity"),
                    "cin2plus_specificity": row.get("cin2plus_specificity"),
                    "cin3plus_auc": row.get("cin3plus_auc"),
                }
            )
    comparison = pd.DataFrame(rows)
    out_path = out_root / "tables/Table_985_vs_1897_Comparison.csv"
    comparison.to_csv(out_path, index=False, encoding="utf-8-sig")

    pivot_rows = []
    for center in sorted(set(comparison["held_out_center"].astype(str))):
        sub985 = comparison[
            (comparison["protocol"].astype(str).str.startswith("985"))
            & (comparison["held_out_center"].astype(str).eq(center))
        ]
        sub1897 = comparison[
            (comparison["protocol"].astype(str).eq("1897_LOCO_formal"))
            & (comparison["held_out_center"].astype(str).eq(center))
        ]
        auc985 = sub985["cin2plus_auc"].dropna()
        auc1897 = sub1897["cin2plus_auc"].dropna()
        if len(sub985) or len(sub1897):
            pivot_rows.append(
                {
                    "held_out_center": center,
                    "985_loco_auc": float(auc985.iloc[0]) if len(auc985) else None,
                    "1897_loco_auc": float(auc1897.iloc[0]) if len(auc1897) else None,
                    "985_n_test": int(sub985["n_test"].iloc[0]) if len(sub985) else None,
                    "1897_n_test": int(sub1897["n_test"].iloc[0]) if len(sub1897) else None,
                }
            )
    pivot = pd.DataFrame(pivot_rows)
    pivot_path = out_root / "tables/Table_985_vs_1897_LOCO_Pivot.csv"
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")

    report = [
        "# 985 vs 1897 Comparison",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Side-by-side LOCO pivot",
        "",
        pivot.to_markdown(index=False) if len(pivot) else "_No overlapping centres._",
        "",
        "## Full comparison table",
        "",
        comparison.to_markdown(index=False),
    ]
    (out_root / "reports/Report_985_vs_1897_Comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(SOURCE_ROOT_DEFAULT))
    parser.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-loco", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--expert-checkpoint", default=str(FORMAL_PRETRAIN_CKPT))
    parser.add_argument("--phase", choices=["all", "splits", "cache", "official", "loco", "compare"], default="all")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = EXP_ROOT / out_root
    split_root = out_root / "splits"
    config_root = out_root / "configs"
    logs_root = out_root / "logs"
    pred_root = out_root / "predictions"
    cache_path = out_root / "cache/patch_features_985.pt"
    for path in [config_root, logs_root, pred_root, out_root / "cache", out_root / "tables", out_root / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifest_csv = split_root / "case_manifest_985.csv"
    if args.phase in {"all", "splits"} and not args.skip_splits:
        build_985_splits(Path(args.source_root), split_root, seed=args.seed)

    if args.phase in {"all", "cache"} and not args.skip_cache:
        build_985_feature_cache(manifest_csv, cache_path, CACHE_1897, Path(args.python), args.gpu)

    expert_checkpoint = Path(args.expert_checkpoint)
    if not expert_checkpoint.exists():
        raise FileNotFoundError(f"Expert checkpoint not found: {expert_checkpoint}")

    if args.phase in {"all", "official"} and not args.skip_official:
        official_root = split_root / "official"
        run_id = "official_external_148"
        run_out = out_root / "runs" / run_id
        config_path = config_root / f"{run_id}_config.py"
        write_config(
            config_path,
            official_root,
            run_out,
            args.epochs,
            args.batch_size,
            0,
            8,
            "985 official train/val with external 148",
            pretrain_without_colpo=False,
            load_expert_checkpoint=expert_checkpoint,
            colpo_pretrained=True,
        )
        append_cached_config_fields(config_path, cache_path, manifest_csv)
        train_log = logs_root / f"{run_id}_train.log"
        checkpoint = run_training_job(
            config_path,
            official_root / "train_labels.csv",
            official_root / "val_labels.csv",
            train_log,
            Path(args.python),
            args.seed,
            env,
            args.skip_train,
        )
        if not args.skip_eval:
            for split_name, csv_path in [
                ("val", official_root / "val_labels.csv"),
                ("external_test", official_root / "external_test_labels.csv"),
            ]:
                evaluate_split(
                    config_path,
                    checkpoint,
                    csv_path,
                    split_name,
                    run_id,
                    args.seed,
                    pred_root,
                    logs_root / f"{run_id}_eval_{split_name}.log",
                    Path(args.python),
                    args.batch_size,
                    env,
                )
            row = metrics_from_predictions(
                official_root / "val_labels.csv",
                official_root / "external_test_labels.csv",
                pred_root / f"SharedLoRA_BioCOT_985_run{run_id}_seed{args.seed}_val_full.csv",
                pred_root / f"SharedLoRA_BioCOT_985_run{run_id}_seed{args.seed}_external_test_full.csv",
                held_out_center="十堰+荆州 (official external 148)",
            )
            row["checkpoint"] = str(checkpoint)
            write_official_report(out_root, row, "official_external")

    if args.phase in {"all", "loco"} and not args.skip_loco:
        loco_root = split_root / "loco"
        state_rows = []
        for fold_dir in sorted(loco_root.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith("loco_"):
                continue
            fold_id = fold_dir.name
            held_out = fold_id.replace("loco_", "")
            run_out = out_root / "runs" / fold_id
            config_path = config_root / f"{fold_id}_config.py"
            write_config(
                config_path,
                fold_dir,
                run_out,
                args.epochs,
                args.batch_size,
                0,
                8,
                f"985 LOCO {fold_id}",
                pretrain_without_colpo=False,
                load_expert_checkpoint=expert_checkpoint,
                colpo_pretrained=True,
            )
            append_cached_config_fields(config_path, cache_path, manifest_csv)
            train_log = logs_root / f"{fold_id}_train.log"
            checkpoint = run_training_job(
                config_path,
                fold_dir / "train_labels.csv",
                fold_dir / "val_labels.csv",
                train_log,
                Path(args.python),
                args.seed,
                env,
                args.skip_train,
            )
            if not args.skip_eval:
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
                        args.seed,
                        pred_root,
                        logs_root / f"{fold_id}_eval_{split_name}.log",
                        Path(args.python),
                        args.batch_size,
                        env,
                    )
            state_rows.append(
                {
                    "fold_id": fold_id,
                    "held_out_center": held_out,
                    "seed": args.seed,
                    "config": str(config_path),
                    "train_csv": str(fold_dir / "train_labels.csv"),
                    "val_csv": str(fold_dir / "val_labels.csv"),
                    "test_csv": str(fold_dir / "external_test_labels.csv"),
                    "checkpoint": str(checkpoint),
                    "train_log": str(train_log),
                }
            )
        (out_root / "loco_run_state.json").write_text(json.dumps(state_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate_results(out_root, state_rows)
        loco_metrics = read_csv(out_root / "tables/Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv")
        loco_metrics.to_csv(out_root / "tables/Table_SharedLoRA_985_LOCO_Fold_Metrics.csv", index=False, encoding="utf-8-sig")

    if args.phase in {"all", "compare"}:
        build_comparison_table(out_root)
        print(f"Comparison written under {out_root / 'tables'}")


if __name__ == "__main__":
    main()
