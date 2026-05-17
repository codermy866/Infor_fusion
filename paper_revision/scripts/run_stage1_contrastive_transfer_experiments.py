#!/usr/bin/env python3
"""Run Stage-1 contrastive adapter pretraining followed by Stage-2 HyDRA transfer."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


EXP_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DEFAULT = "/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
RESULT_ROOT = EXP_ROOT / "paper_revision" / "results" / "stage1_contrastive_transfer"
PRETRAIN_DIR = RESULT_ROOT / "stage1"
PRED_DIR = RESULT_ROOT / "predictions"
CONFIG_DIR = RESULT_ROOT / "generated_configs"
LOG_DIR = RESULT_ROOT / "run_logs"
STATUS_PATH = RESULT_ROOT / "stage1_transfer_status.jsonl"


def append_status(payload: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_command(command: List[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write(" ".join(command) + "\n")
        log_f.flush()
        proc = subprocess.Popen(command, cwd=str(EXP_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def newest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("best_model_v3_*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return candidates[0]


def prediction_exists(run_id: str, seed: int, split: str) -> bool:
    return (PRED_DIR / f"CervixAdapt_Stage1Contrastive_run{run_id}_seed{seed}_{split}_full.csv").exists()


def write_stage2_config(seed: int, pretrain_path: Path) -> Path:
    out_base = RESULT_ROOT / "CervixAdapt_Stage1Contrastive" / f"seed{seed}"
    config_path = CONFIG_DIR / f"CervixAdapt_Stage1Contrastive_seed{seed}.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generated Stage-2 config for Stage-1 contrastive transfer, seed={seed}."""

from dataclasses import dataclass
from pathlib import Path

from paper_revision.configs.cervix_domain_adaptation_configs import CervixAdaptStage1ContrastiveConfig


@dataclass
class Stage1TransferConfig(CervixAdaptStage1ContrastiveConfig):
    experiment_name: str = "CervixAdapt_Stage1Contrastive"
    load_domain_pretrain_path: str = "{pretrain_path}"
    output_dir: str = "{out_base / "results"}"
    checkpoint_dir: str = "{out_base / "checkpoints"}"
    log_dir: str = "{out_base / "logs"}"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
'''
    config_path.write_text(content, encoding="utf-8")
    return config_path


def run_job(seed: int, gpu: int, pretrain_epochs: int, finetune_epochs: int, run_id: str, python_bin: str, skip_existing: bool) -> dict[str, object]:
    started = datetime.now().isoformat(timespec="seconds")
    log_path = LOG_DIR / f"CervixAdapt_Stage1Contrastive_{run_id}_seed{seed}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HOME": env.get("HF_HOME", "/data2/hmy_pri/.cache/huggingface"),
            "TORCH_HOME": env.get("TORCH_HOME", "/data2/hmy_pri/.cache/torch"),
            "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": env.get("TRANSFORMERS_OFFLINE", "1"),
        }
    )

    try:
        if skip_existing and prediction_exists(run_id, seed, "internal_validation") and prediction_exists(run_id, seed, "external_test"):
            status = {
                "method": "CervixAdapt_Stage1Contrastive",
                "seed": seed,
                "gpu": gpu,
                "status": "skipped_existing",
                "started": started,
                "finished": datetime.now().isoformat(timespec="seconds"),
            }
            append_status(status)
            return status

        seed_stage1_dir = PRETRAIN_DIR / f"seed{seed}"
        pretrain_path = seed_stage1_dir / f"stage1_contrastive_seed{seed}.pt"
        if not pretrain_path.exists() or not skip_existing:
            pretrain_cmd = [
                python_bin,
                "paper_revision/scripts/pretrain_cervix_contrastive_adapters.py",
                "--seed",
                str(seed),
                "--epochs",
                str(pretrain_epochs),
                "--gpu",
                "0",
                "--output-dir",
                str(seed_stage1_dir),
            ]
            run_command(pretrain_cmd, env, log_path)

        config_path = write_stage2_config(seed, pretrain_path)
        train_cmd = [
            python_bin,
            "training/train_bio_cot_v3.2.py",
            "--config",
            str(config_path),
            "--epochs",
            str(finetune_epochs),
            "--seed",
            str(seed),
        ]
        run_command(train_cmd, env, log_path)
        checkpoint = newest_checkpoint(RESULT_ROOT / "CervixAdapt_Stage1Contrastive" / f"seed{seed}" / "checkpoints")

        for split in ("internal_validation", "external_test"):
            eval_cmd = [
                python_bin,
                "paper_revision/scripts/evaluate_checkpoint_predictions.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                split,
                "--method",
                "CervixAdapt_Stage1Contrastive",
                "--run_id",
                run_id,
                "--seed",
                str(seed),
                "--output-dir",
                str(PRED_DIR),
            ]
            run_command(eval_cmd, env, log_path)

        status = {
            "method": "CervixAdapt_Stage1Contrastive",
            "seed": seed,
            "gpu": gpu,
            "status": "completed",
            "checkpoint": str(checkpoint),
            "stage1_checkpoint": str(pretrain_path),
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        append_status(status)
        return status
    except Exception as exc:
        status = {
            "method": "CervixAdapt_Stage1Contrastive",
            "seed": seed,
            "gpu": gpu,
            "status": "failed",
            "error": repr(exc),
            "log": str(log_path),
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        append_status(status)
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--run-id", default="stage1_transfer")
    parser.add_argument("--python", default=PYTHON_DEFAULT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    workers = args.max_workers if args.max_workers > 0 else len(gpus)
    workers = max(1, min(workers, len(seeds)))
    failures = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futs = []
        for idx, seed in enumerate(seeds):
            gpu = gpus[idx % len(gpus)]
            futs.append(executor.submit(run_job, seed, gpu, args.pretrain_epochs, args.finetune_epochs, args.run_id, args.python, args.skip_existing))
        for future in futures.as_completed(futs):
            status = future.result()
            print(json.dumps(status, ensure_ascii=False))
            if status.get("status") == "failed":
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
