#!/usr/bin/env python3
"""Run cervix-domain visual-language adaptation experiments."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


EXP_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DEFAULT = "/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
RESULT_ROOT = EXP_ROOT / "paper_revision" / "results" / "cervix_domain_adaptation"
PRED_DIR = RESULT_ROOT / "predictions"
CONFIG_DIR = RESULT_ROOT / "generated_configs"
LOG_DIR = RESULT_ROOT / "run_logs"
STATUS_PATH = RESULT_ROOT / "domain_adaptation_status.jsonl"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    class_name: str


CONFIG_MODULE = "paper_revision.configs.cervix_domain_adaptation_configs"
METHODS: List[MethodSpec] = [
    MethodSpec("CervixAdapt_StaticPrior", "CervixAdaptStaticPriorConfig"),
    MethodSpec("CervixAdapt_VisualAdapterOnly", "CervixAdaptVisualAdapterOnlyConfig"),
    MethodSpec("CervixAdapt_BERTAdapterOnly", "CervixAdaptBERTAdapterOnlyConfig"),
    MethodSpec("CervixAdapt_VisualBERTAdapter", "CervixAdaptVisualBERTAdapterConfig"),
    MethodSpec("CervixAdapt_BERTLastLayerFT", "CervixAdaptBERTLastLayerFTConfig"),
    MethodSpec("CervixAdapt_VisualFullTextFT", "CervixAdaptVisualFullTextFTConfig"),
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_config(spec: MethodSpec, seed: int) -> Path:
    safe = safe_name(spec.method)
    out_base = RESULT_ROOT / safe / f"seed{seed}"
    config_path = CONFIG_DIR / f"{safe}_seed{seed}.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generated config for {spec.method}, seed={seed}."""

from dataclasses import dataclass
from pathlib import Path

from {CONFIG_MODULE} import {spec.class_name}


@dataclass
class DomainAdaptationConfig({spec.class_name}):
    experiment_name: str = "{spec.method}"
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


def newest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("best_model_v3_*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return candidates[0]


def run_command(command: List[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n" + "=" * 80 + "\n")
        log_f.write(" ".join(command) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(EXP_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def prediction_exists(method: str, run_id: str, seed: int, split: str) -> bool:
    return (PRED_DIR / f"{method}_run{run_id}_seed{seed}_{split}_full.csv").exists()


def append_status(payload: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_job(
    spec: MethodSpec,
    seed: int,
    gpu: int,
    epochs: int,
    run_id: str,
    python_bin: str,
    skip_existing: bool,
) -> dict[str, object]:
    started = datetime.now().isoformat(timespec="seconds")
    safe = safe_name(spec.method)
    log_path = LOG_DIR / f"{safe}_{run_id}_seed{seed}.log"
    config_path = write_config(spec, seed)
    checkpoint_dir = RESULT_ROOT / safe / f"seed{seed}" / "checkpoints"
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
        if skip_existing and prediction_exists(spec.method, run_id, seed, "internal_validation") and prediction_exists(
            spec.method, run_id, seed, "external_test"
        ):
            checkpoint = newest_checkpoint(checkpoint_dir)
            status = {
                "method": spec.method,
                "seed": seed,
                "gpu": gpu,
                "status": "skipped_existing",
                "checkpoint": str(checkpoint),
                "started": started,
                "finished": datetime.now().isoformat(timespec="seconds"),
            }
            append_status(status)
            return status

        train_cmd = [
            python_bin,
            "training/train_bio_cot_v3.2.py",
            "--config",
            str(config_path),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
        ]
        run_command(train_cmd, env, log_path)
        checkpoint = newest_checkpoint(checkpoint_dir)

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
                spec.method,
                "--run_id",
                run_id,
                "--seed",
                str(seed),
                "--output-dir",
                str(PRED_DIR),
            ]
            run_command(eval_cmd, env, log_path)

        status = {
            "method": spec.method,
            "seed": seed,
            "gpu": gpu,
            "status": "completed",
            "checkpoint": str(checkpoint),
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        append_status(status)
        return status
    except Exception as exc:
        status = {
            "method": spec.method,
            "seed": seed,
            "gpu": gpu,
            "status": "failed",
            "error": repr(exc),
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "log": str(log_path),
        }
        append_status(status)
        return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids assigned round-robin.")
    parser.add_argument("--run-id", default="domain_adapt")
    parser.add_argument("--python", default=PYTHON_DEFAULT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--methods", default="", help="Optional comma-separated method subset.")
    parser.add_argument("--max-workers", type=int, default=0, help="Default: number of provided GPUs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    requested = {item.strip() for item in args.methods.split(",") if item.strip()}
    methods = [spec for spec in METHODS if not requested or spec.method in requested]
    if not methods:
        raise ValueError(f"No matching methods for {sorted(requested)}")

    jobs = []
    for spec in methods:
        for seed in seeds:
            gpu = gpus[len(jobs) % len(gpus)]
            jobs.append((spec, seed, gpu))

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    workers = args.max_workers if args.max_workers > 0 else len(gpus)
    workers = max(1, min(workers, len(jobs)))
    print(f"Running {len(jobs)} jobs with {workers} workers; predictions -> {PRED_DIR}")

    failures = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(run_job, spec, seed, gpu, args.epochs, args.run_id, args.python, args.skip_existing): (spec, seed, gpu)
            for spec, seed, gpu in jobs
        }
        for future in futures.as_completed(future_to_job):
            status = future.result()
            print(json.dumps(status, ensure_ascii=False))
            if status.get("status") == "failed":
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
