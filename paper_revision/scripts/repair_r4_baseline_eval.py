#!/usr/bin/env python3
"""Re-run eval for R4 baselines that trained but failed evaluate_checkpoint_predictions."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_revision.scripts.run_corrected403_clean_rerun import (  # noqa: E402
    CORRECTED_ROOT,
    PYTHON_BIN,
    SCRIPTS,
    append_manifest,
    checkpoint_for_seed_from_manifest,
    external_n,
    now,
    prediction_csv,
    run_cmd,
    write_baseline_config_files,
)

METHOD_BY_KEY = {
    "clinical_only": "Clinical only",
    "colposcopy_only": "Colposcopy only",
    "oct_only": "OCT only",
    "concat_fusion": "Image concat fusion",
    "late_fusion": "Late fusion",
    "gated_fusion": "Gated fusion",
    "cross_attention_fusion": "Cross-attention fusion",
}


def repair_one(
    *,
    baseline_key: str,
    seed: int,
    gpu: int,
    run_id: str = "clean403",
    force: bool = False,
) -> bool:
    write_baseline_config_files()
    cfg_path = ROOT / "paper_revision" / "configs" / "corrected403_generated" / f"baseline_{baseline_key}.py"
    method = METHOD_BY_KEY[baseline_key]
    from paper_revision.scripts.run_corrected403_clean_rerun import load_config_class

    cfg = load_config_class(cfg_path)()
    ckpt_dir = Path(cfg.checkpoint_dir)
    manifest = CORRECTED_ROOT / "baselines" / "baseline_run_manifest.csv"
    ckpt = checkpoint_for_seed_from_manifest(manifest, method, seed, ckpt_dir)
    if not ckpt:
        print(f"[skip] no checkpoint for {method} seed={seed} in {ckpt_dir}")
        return False

    pred_dir = CORRECTED_ROOT / "baselines" / "predictions"
    int_pred = prediction_csv(pred_dir, method, run_id, seed, "internal_validation")
    ext_pred = prediction_csv(pred_dir, method, run_id, seed, "external_test")

    if ext_pred.exists() and external_n(ext_pred) == 403 and not force:
        print(f"[skip] {method} seed={seed} already has 403 external predictions (use --force to redo)")
        return True

    t0 = now()
    status = "success"
    err = ""
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
                str(cfg_path),
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
            break

    ext_n = external_n(ext_pred)
    if status == "success" and ext_n != 403:
        status = "failed"
        err = f"external_n={ext_n}, expected 403"

    append_manifest(
        manifest,
        {
            "stage": "R4_baseline",
            "method": method,
            "seed": seed,
            "config_path": str(cfg_path),
            "checkpoint_path": str(ckpt),
            "internal_prediction_csv": str(int_pred),
            "external_prediction_csv": str(ext_pred),
            "external_n": ext_n,
            "status": status,
            "start_time": t0,
            "end_time": now(),
            "error_message": err,
        },
    )
    print(f"[{status}] repair {method} seed={seed} external_n={ext_n} err={err}")
    return status == "success"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="clinical_only", choices=list(METHOD_BY_KEY))
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run eval even if 403 preds exist")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ok = 0
    for seed in seeds:
        if repair_one(
            baseline_key=args.baseline,
            seed=seed,
            gpu=args.gpu,
            force=args.force,
        ):
            ok += 1
    print(f"repair done: {ok}/{len(seeds)} succeeded")


if __name__ == "__main__":
    main()
