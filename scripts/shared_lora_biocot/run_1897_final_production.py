#!/usr/bin/env python3
"""Run 100-epoch formal LOCO with ablation-pruned production config."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts.shared_lora_biocot.run_1897_improved_loco import (  # noqa: E402
    CACHE_1897,
    CASE_MANIFEST,
    DEFAULT_PYTHON,
    FORMAL_PRETRAIN_CKPT,
    FORMAL_SPLIT_ROOT,
    OUT_ROOT_DEFAULT,
    aggregate_results_g5b,
    append_improved_config_fields,
    evaluate_split,
    run_training_job,
    write_config,
)

PRUNED_JSON = OUT_ROOT_DEFAULT / "tables/Table_Pruned_Production_Defaults.json"
FINAL_ROOT_DEFAULT = OUT_ROOT_DEFAULT / "final_production_100ep"


def load_production_preset() -> dict[str, Any]:
    if not PRUNED_JSON.exists():
        raise FileNotFoundError(
            f"Missing {PRUNED_JSON}. Run apply_ablation_pruning.py after ablation completes."
        )
    preset = json.loads(PRUNED_JSON.read_text(encoding="utf-8"))
    preset["description"] = "Ablation-pruned production stack (100 epochs)"
    preset["removed_components"] = "see Table_Ablation_Prune_Decisions.csv"
    return preset


def run_final_production(
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
) -> None:
    preset = load_production_preset()
    for path in [out_root / "configs", out_root / "logs", out_root / "predictions", out_root / "runs", out_root / "tables", out_root / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    (out_root / "production_preset.json").write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")

    state_rows: list[dict[str, object]] = []
    for fold_dir in sorted(split_root.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("loco_"):
            continue
        fold_id = fold_dir.name
        held_out = fold_id.replace("loco_", "")
        run_out = out_root / "runs" / fold_id
        config_path = out_root / "configs" / f"{fold_id}_config.py"
        write_config(
            config_path,
            fold_dir,
            run_out,
            epochs,
            batch_size,
            0,
            8,
            f"1897 production 100ep {fold_id}",
            pretrain_without_colpo=False,
            load_expert_checkpoint=expert_checkpoint,
            colpo_pretrained=True,
        )
        append_improved_config_fields(config_path, CACHE_1897, CASE_MANIFEST, preset)

        train_log = out_root / "logs" / f"{fold_id}_train.log"
        print(f"[production] Training {fold_id} for {epochs} epochs ...")
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
                    out_root / "predictions",
                    out_root / "logs" / f"{fold_id}_eval_{split_name}.log",
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

    (out_root / "final_run_state.json").write_text(json.dumps(state_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_results_g5b(out_root, state_rows)

    fold_df = __import__("pandas").read_csv(out_root / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv")
    agg_df = __import__("pandas").read_csv(out_root / "tables/Table_Improved1897_LOCO_Aggregate_Metrics.csv")
    auc_row = agg_df[agg_df["metric"] == "cin2plus_auc"]
    mean_auc = float(auc_row["mean"].iloc[0]) if len(auc_row) else float("nan")
    std_auc = float(auc_row["std"].iloc[0]) if len(auc_row) else float("nan")

    report = [
        "# 1897 Production LOCO (100 epochs, pruned stack)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**Mean CIN2+ AUC**: {mean_auc:.4f} ± {std_auc:.4f}",
        "",
        f"Preset: `{PRUNED_JSON.relative_to(EXP_ROOT)}`",
        "",
        "## Fold metrics",
        "",
        fold_df.to_markdown(index=False),
    ]
    (out_root / "reports/Report_Final_Production_100ep.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[production] Mean CIN2+ AUC = {mean_auc:.4f} ± {std_auc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(FINAL_ROOT_DEFAULT))
    parser.add_argument("--split-root", default=str(FORMAL_SPLIT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expert-checkpoint", default=str(FORMAL_PRETRAIN_CKPT))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    expert_checkpoint = Path(args.expert_checkpoint)
    if not expert_checkpoint.exists():
        raise FileNotFoundError(expert_checkpoint)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_final_production(
        out_root=out_root,
        split_root=Path(args.split_root),
        expert_checkpoint=expert_checkpoint,
        python=Path(args.python),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        env=env,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
