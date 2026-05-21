#!/usr/bin/env python3
"""R1: Verify corrected 1897 cohort, 403 external test, and feature cache."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_revision" / "results" / "real_50epoch_5center_corrected"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from paper_revision.configs.corrected_5center_elbo_structured_prior_config import (
        CorrectedFiveCenterELBOStructuredPriorConfig,
    )

    cfg = CorrectedFiveCenterELBOStructuredPriorConfig()
    checks: list[dict] = []
    ok = True

    def record(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        if not passed:
            ok = False
        checks.append({"check": name, "passed": passed, "detail": detail})

    index_path = ROOT / "paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv"
    train = ROOT / "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/train_labels.csv"
    val = ROOT / "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/val_labels.csv"
    test = ROOT / "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/external_test_labels.csv"
    cache_path = Path(cfg.feature_cache_path)

    import pandas as pd

    index_df = pd.read_csv(index_path)
    train_n = len(pd.read_csv(train))
    val_n = len(pd.read_csv(val))
    test_n = len(pd.read_csv(test))

    record("final_1897_case_index_exists", index_path.exists(), str(index_path))
    record("final_n_1897", len(index_df) == 1897, f"n={len(index_df)}")
    record("train_1317", train_n == 1317, f"train={train_n}")
    record("val_177", val_n == 177, f"val={val_n}")
    record("external_403", test_n == 403, f"external={test_n}")
    record("split_sum_1897", train_n + val_n + test_n == 1897, f"{train_n}+{val_n}+{test_n}")
    record("no_vlm", cfg.use_vlm_retriever is False, f"use_vlm_retriever={cfg.use_vlm_retriever}")
    record("no_report_mode", cfg.no_report_mode is True, f"no_report_mode={cfg.no_report_mode}")
    record("feature_cache_exists", cache_path.exists(), str(cache_path))

    if cache_path.exists():
        sys.path.insert(0, str(ROOT))
        from data.cached_patch_dataset import case_key

        payload = torch.load(cache_path, map_location="cpu")
        feature_map = payload.get("features", payload) if isinstance(payload, dict) else {}
        keys = set(feature_map.keys())
        split_frames = [pd.read_csv(p) for p in (train, val, test)]
        split_df = pd.concat(split_frames, ignore_index=True)
        required_keys = {case_key(row) for _, row in split_df.iterrows()}
        missing = sorted(required_keys - keys)[:5]
        record(
            "cache_covers_split_cases",
            len(missing) == 0,
            f"missing_keys_sample={missing}, required={len(required_keys)}, cached={len(keys)}",
        )

    report_cols = [c for c in index_df.columns if "report" in c.lower()]
    record("no_report_columns_in_index", len(report_cols) == 0, f"cols={report_cols}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "passed": ok,
        "checks": checks,
        "config": {
            "data_root": cfg.data_root,
            "feature_cache_path": cfg.feature_cache_path,
            "expected_external_n": getattr(cfg, "expected_external_n", 403),
        },
    }
    (OUT_DIR / "cohort_cache_verification.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# Cohort and Cache Verification", "", f"**Overall:** {'PASS' if ok else 'FAIL'}", ""]
    for row in checks:
        md.append(f"- [{'x' if row['passed'] else ' '}] {row['check']}: {row['detail']}")
    (OUT_DIR / "cohort_cache_verification.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((OUT_DIR / "cohort_cache_verification.md").read_text(encoding="utf-8"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
