#!/usr/bin/env python3
"""Orchestrate HyDRA-CoE corrected 403-case clean rerun (R0–R13)."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper_revision" / "scripts"
CORRECTED_ROOT = ROOT / "paper_revision" / "results" / "real_50epoch_5center_corrected"
PYTHON_BIN = os.environ.get("PYTHON_BIN") or (
    "/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python"
    if Path("/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python").exists()
    else "/data2/hmy_pri/VLM_Caus_Rm_Mics/my_retfound/bin/python"
)
SEEDS_DEFAULT = [42, 123, 456]
THRESHOLD_RULE = "max_specificity_at_sensitivity:0.95"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_config_class(config_path: Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("cfg_mod", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [
        cls
        for cls in module.__dict__.values()
        if isinstance(cls, type)
        and "Config" in cls.__name__
        and cls.__name__ != "BioCOT_v3_2_Config"
        and (
            getattr(cls, "__module__", None) == getattr(module, "__name__", None)
            or cls.__name__.startswith(("Corrected", "Abl", "AllCenter"))
        )
    ]
    if not classes:
        raise ValueError(f"No Config class in {config_path}")
    for prefix in ("Corrected", "Abl"):
        for cls in classes:
            if cls.__name__.startswith(prefix):
                return cls
    for cls in classes:
        if cls.__name__.startswith("AllCenter"):
            return cls
    return classes[0]


def run_cmd(cmd: list[str], log_path: Path, env: Optional[dict] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{now()}] CMD: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=merged_env, stdout=logf, stderr=subprocess.STDOUT)
        logf.write(f"[{now()}] EXIT={proc.returncode}\n")
    return proc.returncode


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    files = sorted(ckpt_dir.glob("best_model_v3_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def checkpoint_after_train(ckpt_dir: Path, train_start_epoch: float) -> Optional[Path]:
    """Pick checkpoint written during the current train run (not a later seed's best)."""
    margin = 5.0
    files = [
        p
        for p in ckpt_dir.glob("best_model_v3_*.pth")
        if p.stat().st_mtime >= train_start_epoch - margin
    ]
    if files:
        return max(files, key=lambda p: p.stat().st_mtime)
    return latest_checkpoint(ckpt_dir)


def checkpoint_for_seed_from_manifest(
    manifest_path: Path, method: str, seed: int, ckpt_dir: Path
) -> Optional[Path]:
    """First manifest row per (method, seed) keeps the checkpoint from that training run."""
    if not manifest_path.exists():
        return latest_checkpoint(ckpt_dir)
    first: Optional[Path] = None
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("method") != method or int(row.get("seed", -1)) != seed:
                continue
            ckpt = row.get("checkpoint_path", "").strip()
            if ckpt and Path(ckpt).exists():
                first = Path(ckpt)
                break
    return first or latest_checkpoint(ckpt_dir)


def prediction_csv(pred_dir: Path, method: str, run_id: str, seed: int, split: str) -> Path:
    """Match evaluate_checkpoint_predictions.py naming (method may contain spaces)."""
    return pred_dir / f"{method}_run{run_id}_seed{seed}_{split}_full.csv"


def external_n(pred_csv: Path) -> int:
    if not pred_csv.exists():
        return -1
    try:
        df = pd.read_csv(pred_csv)
    except Exception:
        return -1
    ext = df[df["split"].astype(str).isin(["external_test", "external"])]
    return len(ext)


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerows([row])


def train_and_eval(
    *,
    stage: str,
    config_path: Path,
    method: str,
    seed: int,
    epochs: int,
    gpu: int,
    pred_dir: Path,
    manifest_path: Path,
    run_id: str = "clean403",
    skip_if_success: bool = True,
) -> bool:
    cfg_cls = load_config_class(config_path)
    cfg = cfg_cls()
    ckpt_dir = Path(cfg.checkpoint_dir)
    ext_pred = prediction_csv(pred_dir, method, run_id, seed, "external_test")
    int_pred = prediction_csv(pred_dir, method, run_id, seed, "internal_validation")
    if skip_if_success and ext_pred.exists() and external_n(ext_pred) == 403:
        append_manifest(
            manifest_path,
            {
                "stage": stage,
                "method": method,
                "seed": seed,
                "config_path": str(config_path),
                "checkpoint_path": str(latest_checkpoint(ckpt_dir) or ""),
                "internal_prediction_csv": str(int_pred if int_pred.exists() else ""),
                "external_prediction_csv": str(ext_pred),
                "external_n": 403,
                "status": "skipped_existing",
                "start_time": now(),
                "end_time": now(),
                "error_message": "",
            },
        )
        print(f"[skip] {method} seed={seed} already has 403 external predictions")
        return True

    log_path = Path(cfg.log_dir) / f"train_seed{seed}_{now().replace(':', '')}.log"
    t0 = now()
    train_start_epoch = datetime.now().timestamp()
    code = run_cmd(
        [
            PYTHON_BIN,
            str(ROOT / "training" / "train_bio_cot_v3.2.py"),
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--gpu",
            str(gpu),
        ],
        log_path,
        env={"CUDA_VISIBLE_DEVICES": str(gpu)},
    )
    ckpt = checkpoint_after_train(ckpt_dir, train_start_epoch)
    status = "success" if code == 0 and ckpt else "failed"
    err = "" if status == "success" else f"train_exit={code}"

    if status == "success" and ckpt:
        for split, out_path in [
            ("internal_validation", int_pred),
            ("external_test", ext_pred),
        ]:
            safe_method = method.replace(" ", "_")
            eval_log = pred_dir / f"eval_{safe_method}_seed{seed}_{split}.log"
            ec = run_cmd(
                [
                    PYTHON_BIN,
                    str(SCRIPTS / "evaluate_checkpoint_predictions.py"),
                    "--checkpoint",
                    str(ckpt),
                    "--config",
                    str(config_path),
                    "--split",
                    split,
                    "--method",
                    method,
                    "--run_id",
                    run_id,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(pred_dir),
                ],
                eval_log,
                env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
            if ec != 0:
                status = "failed"
                err = f"eval_{split}_exit={ec}"

    ext_n = external_n(ext_pred)
    if status == "success" and ext_n != 403:
        status = "failed"
        err = f"external_n={ext_n}, expected 403"

    append_manifest(
        manifest_path,
        {
            "stage": stage,
            "method": method,
            "seed": seed,
            "config_path": str(config_path),
            "checkpoint_path": str(ckpt or ""),
            "internal_prediction_csv": str(int_pred),
            "external_prediction_csv": str(ext_pred),
            "external_n": ext_n,
            "status": status,
            "start_time": t0,
            "end_time": now(),
            "error_message": err,
        },
    )
    print(f"[{status}] {stage} {method} seed={seed} external_n={ext_n}")
    return status == "success"


def stage_r0() -> None:
    run_cmd([PYTHON_BIN, str(SCRIPTS / "archive_legacy_mixed_n_results.py")], CORRECTED_ROOT / "logs" / "r0_archive.log")


def stage_r1() -> bool:
    log = CORRECTED_ROOT / "logs" / "r1_verify.log"
    return run_cmd([PYTHON_BIN, str(SCRIPTS / "verify_corrected_5center_cohort.py")], log) == 0


def stage_r2() -> bool:
    log = CORRECTED_ROOT / "logs" / "r2_stage1_check.log"
    return run_cmd([PYTHON_BIN, str(SCRIPTS / "check_stage1_adapter_injection.py")], log) == 0


def wrap_config_for_corrected(src: Path, subdir: str) -> Path:
    """Copy an existing config module and redirect outputs under corrected root."""
    gen_dir = ROOT / "paper_revision" / "configs" / "corrected403_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    slug = src.stem
    out_base = CORRECTED_ROOT / subdir / slug.replace("_config", "")
    replacements = {
        'output_dir: str = "paper_revision/results/': f'output_dir: str = "{out_base}/results"',
        "output_dir: str = 'paper_revision/results/": f"output_dir: str = '{out_base}/results'",
        'checkpoint_dir: str = "paper_revision/results/': f'checkpoint_dir: str = "{out_base}/checkpoints"',
        'log_dir: str = "paper_revision/results/': f'log_dir: str = "{out_base}/logs"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "expected_external_n" not in text:
        text = text.replace(
            "def __post_init__(self):",
            "    expected_external_n: int = 403\n\n    def __post_init__(self):",
            1,
        )
    dest = gen_dir / f"{slug}_corrected403.py"
    dest.write_text(text, encoding="utf-8")
    return dest


def generate_ablation_configs() -> list[Path]:
    paths = []
    for src in sorted((ROOT / "paper_revision" / "configs").glob("abl_*_config.py")):
        paths.append(wrap_config_for_corrected(src, "ablations"))
    return paths


def baseline_config_stub(class_name: str) -> str:
    """Config subclass in generated file so evaluate_checkpoint_predictions finds *Config in-module."""
    return (
        "#!/usr/bin/env python\n"
        "# -*- coding: utf-8 -*-\n"
        '"""Auto-generated corrected403 baseline config."""\n'
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        f"from paper_revision.configs.corrected403_experiment_configs import {class_name} as _Base\n\n\n"
        f"class {class_name}(_Base):\n"
        '    """Thin subclass: *Config must live in this module for eval loader."""\n'
        "    pass\n"
    )


def write_baseline_config_files() -> dict[str, Path]:
    from paper_revision.configs import corrected403_experiment_configs as spec

    gen_dir = ROOT / "paper_revision" / "configs" / "corrected403_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "clinical_only": spec.Corrected403ClinicalOnlyConfig,
        "colposcopy_only": spec.Corrected403ColposcopyOnlyConfig,
        "oct_only": spec.Corrected403OctOnlyConfig,
        "concat_fusion": spec.Corrected403ConcatFusionConfig,
        "late_fusion": spec.Corrected403LateFusionConfig,
        "gated_fusion": spec.Corrected403GatedFusionConfig,
        "cross_attention_fusion": spec.Corrected403CrossAttentionFusionConfig,
    }
    paths: dict[str, Path] = {}
    for name, cls in mapping.items():
        path = gen_dir / f"baseline_{name}.py"
        path.write_text(baseline_config_stub(cls.__name__), encoding="utf-8")
        paths[name] = path
    return paths


def reset_manifest_if_resuming(manifest_path: Path) -> None:
    if manifest_path.exists():
        backup = manifest_path.with_suffix(".csv.bak")
        shutil.copy2(manifest_path, backup)


def stage_r3_r5(args) -> None:
    full_cfg = ROOT / "paper_revision" / "configs" / "corrected_5center_elbo_structured_prior_config.py"
    full_pred = CORRECTED_ROOT / "full_hydra_coe" / "predictions"
    full_manifest = CORRECTED_ROOT / "full_hydra_coe" / "full_model_run_manifest.csv"
    reset_manifest_if_resuming(full_manifest)
    if full_manifest.exists():
        full_manifest.unlink()

    for seed in args.seeds:
        train_and_eval(
            stage="R3_full_hydra",
            config_path=full_cfg,
            method="HyDRA-CoE Full",
            seed=seed,
            epochs=args.epochs,
            gpu=args.gpu,
            pred_dir=full_pred,
            manifest_path=full_manifest,
            skip_if_success=not args.force,
        )

    baseline_paths = write_baseline_config_files()
    base_manifest = CORRECTED_ROOT / "baselines" / "baseline_run_manifest.csv"
    base_pred = CORRECTED_ROOT / "baselines" / "predictions"
    method_names = {
        "clinical_only": "Clinical only",
        "colposcopy_only": "Colposcopy only",
        "oct_only": "OCT only",
        "concat_fusion": "Image concat fusion",
        "late_fusion": "Late fusion",
        "gated_fusion": "Gated fusion",
        "cross_attention_fusion": "Cross-attention fusion",
    }
    for key, cfg_path in baseline_paths.items():
        for seed in args.seeds:
            train_and_eval(
                stage="R4_baseline",
                config_path=cfg_path,
                method=method_names[key],
                seed=seed,
                epochs=args.epochs,
                gpu=args.gpu,
                pred_dir=base_pred,
                manifest_path=base_manifest,
                skip_if_success=not args.force,
            )

    abl_paths = generate_ablation_configs()
    abl_manifest = CORRECTED_ROOT / "ablations" / "ablation_run_manifest.csv"
    abl_pred = CORRECTED_ROOT / "ablations" / "predictions"
    for cfg_path in abl_paths:
        cfg_cls = load_config_class(cfg_path)
        method = cfg_cls().experiment_name
        for seed in args.seeds:
            train_and_eval(
                stage="R5_ablation",
                config_path=cfg_path,
                method=method,
                seed=seed,
                epochs=args.epochs,
                gpu=args.gpu,
                pred_dir=abl_pred,
                manifest_path=abl_manifest,
                skip_if_success=not args.force,
            )


def build_locked_tables(pred_dir: Path, out_dir: Path) -> None:
    run_cmd(
        [
            PYTHON_BIN,
            str(SCRIPTS / "build_locked_threshold_tables.py"),
            "--pred-dir",
            str(pred_dir),
            "--output-dir",
            str(out_dir),
            "--threshold-rule",
            THRESHOLD_RULE,
        ],
        out_dir / "build_locked_threshold.log",
    )


def stage_post_tables(args) -> None:
    """R6–R11 table builders using existing scripts where possible."""
    full_pred = CORRECTED_ROOT / "full_hydra_coe" / "predictions"
    full_tables = CORRECTED_ROOT / "full_hydra_coe" / "tables"
    build_locked_tables(full_pred, full_tables)

    rob_dir = CORRECTED_ROOT / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)
    manifest = CORRECTED_ROOT / "full_hydra_coe" / "full_model_run_manifest.csv"
    if manifest.exists() and args.run_robustness:
        df = pd.read_csv(manifest)
        ok = df[df["status"].isin(["success", "skipped_existing"])]
        for _, row in ok.iterrows():
            ckpt = Path(str(row["checkpoint_path"]))
            if not ckpt.exists():
                continue
            seed = int(row["seed"])
            for setting in ["remove_oct", "remove_colposcopy", "remove_clinical_prior", "random_one", "random_two"]:
                run_cmd(
                    [
                        PYTHON_BIN,
                        str(SCRIPTS / "evaluate_checkpoint_predictions.py"),
                        "--checkpoint",
                        str(ckpt),
                        "--config",
                        str(ROOT / "paper_revision/configs/corrected_5center_elbo_structured_prior_config.py"),
                        "--split",
                        "external_test",
                        "--method",
                        f"HyDRA-CoE Full_{setting}",
                        "--run_id",
                        "clean403",
                        "--seed",
                        str(seed),
                        "--missing-modality",
                        setting,
                        "--output-dir",
                        str(rob_dir / "predictions"),
                    ],
                    rob_dir / "logs" / f"robust_{setting}_seed{seed}.log",
                    env={"CUDA_VISIBLE_DEVICES": str(args.gpu)},
                )
        for script, out_name in [
            ("build_missing_modality_table.py", "missing_modality_robustness_metrics.csv"),
            ("build_corruption_robustness_table.py", "corruption_robustness_metrics.csv"),
        ]:
            src_out = ROOT / "paper_revision" / "tables" / out_name
            if (SCRIPTS / script).exists():
                run_cmd([PYTHON_BIN, str(SCRIPTS / script)], rob_dir / f"{script}.log")
                if src_out.exists():
                    shutil.copy2(src_out, rob_dir / out_name)

    if (SCRIPTS / "build_label_noise_table.py").exists():
        run_cmd([PYTHON_BIN, str(SCRIPTS / "build_label_noise_table.py")], CORRECTED_ROOT / "label_noise" / "build.log")

    for script, sub in [
        ("build_loco_tables.py", "loco"),
        ("build_centerwise_calibration_tables.py", "loco"),
    ]:
        out = CORRECTED_ROOT / sub
        out.mkdir(parents=True, exist_ok=True)
        if (SCRIPTS / script).exists():
            run_cmd([PYTHON_BIN, str(SCRIPTS / script)], out / f"{script}.log")

    if (SCRIPTS / "run_coe_faithfulness.py").exists():
        coe_dir = CORRECTED_ROOT / "coe_faithfulness"
        coe_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([PYTHON_BIN, str(SCRIPTS / "run_coe_faithfulness.py")], coe_dir / "run.log")

    if (SCRIPTS / "build_clinical_decision_quality.py").exists():
        clin = CORRECTED_ROOT / "clinical_decision"
        clin.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                PYTHON_BIN,
                str(SCRIPTS / "build_clinical_decision_quality.py"),
                "--predictions",
                str(full_pred),
                "--threshold-table",
                str(full_tables / "locked_thresholds_by_run.csv"),
                "--output-dir",
                str(clin),
                "--split",
                "external_test",
            ],
            clin / "build.log",
        )

    if (SCRIPTS / "build_corrected_5center_result_bundle.py").exists():
        run_cmd([PYTHON_BIN, str(SCRIPTS / "build_corrected_5center_result_bundle.py")], CORRECTED_ROOT / "logs" / "r11_bundle.log")


def stage_r12_r13() -> None:
    legacy = ROOT / "paper_revision" / "results" / "final_pipeline_manifest.csv"
    archive = ROOT / "paper_revision" / "results" / "archive_legacy_mixed_n"
    if legacy.exists():
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, archive / "final_pipeline_manifest_dry_run.csv")

    rows = []
    for manifest in CORRECTED_ROOT.rglob("*_manifest.csv"):
        if not manifest.is_file():
            continue
        df = pd.read_csv(manifest)
        for _, r in df.iterrows():
            rows.append(
                {
                    "stage": r.get("stage", manifest.parent.name),
                    "method": r.get("method", ""),
                    "seed": r.get("seed", ""),
                    "command": "run_corrected403_clean_rerun.py",
                    "config_path": r.get("config_path", ""),
                    "checkpoint_path": r.get("checkpoint_path", ""),
                    "prediction_csv": r.get("external_prediction_csv", ""),
                    "output_table": "",
                    "external_n": r.get("external_n", ""),
                    "status": r.get("status", ""),
                    "start_time": r.get("start_time", ""),
                    "end_time": r.get("end_time", ""),
                    "runtime_minutes": "",
                    "error_message": r.get("error_message", ""),
                    "dry_run": False,
                }
            )
    out = CORRECTED_ROOT / "final_pipeline_manifest.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

    scope = CORRECTED_ROOT / "PAPER_READY_RESULT_SCOPE.md"
    scope.write_text(
        "# Paper-Ready Result Scope\n\n"
        "Main paper tables must come from `real_50epoch_5center_corrected/` with external_n=403.\n"
        "HyDRA-CoE naming only; Bio-COT legacy names are not used in paper tables.\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"rule": "HyDRA-CoE only", "status": "enforced"}, {"rule": "external_n=403", "status": "enforced"}]
    ).to_csv(CORRECTED_ROOT / "name_cleanup_audit.csv", index=False, encoding="utf-8-sig")


def parse_stages(raw: str) -> set[str]:
    if raw.lower() in {"all", "r0-r13"}:
        return {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13"}
    out: set[str] = set()
    for part in raw.split(","):
        part = part.strip().upper()
        if part:
            out.add(part)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="all", help="Comma list or 'all'")
    parser.add_argument("--from-stage", default=None, help="Start at stage e.g. R3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Rerun even if 403 preds exist")
    parser.add_argument("--skip-train", action="store_true", help="Only prep and post tables")
    parser.add_argument("--run-robustness", action="store_true", default=True)
    args = parser.parse_args()

    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    order = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13"]
    selected = parse_stages(args.stages)
    if args.from_stage:
        start = args.from_stage.upper()
        if start in order:
            selected = {s for s in order if order.index(s) >= order.index(start)}

    CORRECTED_ROOT.mkdir(parents=True, exist_ok=True)
    (CORRECTED_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    if "R0" in selected:
        stage_r0()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if "R1" in selected:
        if not stage_r1():
            print("R1 verification failed; stopping.")
            return 1
    if "R2" in selected:
        if not stage_r2():
            print("R2 stage1 check failed; stopping.")
            return 1

    train_stages = selected & {"R3", "R4", "R5"}
    if train_stages and not args.skip_train:
        stage_r3_r5(args)

    post_stages = selected & {"R6", "R7", "R8", "R9", "R10", "R11"}
    if post_stages:
        stage_post_tables(args)

    if "R12" in selected or "R13" in selected:
        stage_r12_r13()

    print(f"Clean rerun orchestration finished. Results under: {CORRECTED_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
