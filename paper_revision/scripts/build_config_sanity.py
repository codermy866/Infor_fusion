#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write sanity checks for the locked HyDRA-CoE no-report configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.clinical_variable_mapping import assert_no_report_columns


DEFAULT_CONFIGS = [
    EXP_ROOT / "paper_revision" / "configs" / "all_center_patient_holdout_config.py",
    EXP_ROOT / "paper_revision" / "configs" / "all_center_elbo_structured_prior_config.py",
]
DEFAULT_OUTPUT = EXP_ROOT / "paper_revision" / "results" / "config_sanity" / "hydra_coe_config_sanity.json"
FINAL_INDEX = EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "final_1897_case_index.csv"


def load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    classes = [
        cls
        for cls in module.__dict__.values()
        if isinstance(cls, type)
        and "Config" in cls.__name__
        and getattr(cls, "__module__", None) == module.__name__
    ]
    if not classes:
        raise ValueError(f"No *Config class found in {path}")
    return classes[0]()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    assert_no_report_columns(df.columns)
    return df


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def inspect_cache(cache_path: Path, expected_n: int, warnings: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(cache_path),
        "exists": cache_path.exists(),
        "is_symlink": cache_path.is_symlink(),
        "mtime": iso_mtime(cache_path),
        "target": str(cache_path.resolve()) if cache_path.exists() else None,
        "feature_count": None,
        "metadata": None,
    }
    if not cache_path.exists():
        warnings.append(f"feature_cache_missing: {cache_path}")
        return info
    if FINAL_INDEX.exists() and cache_path.stat().st_mtime < FINAL_INDEX.stat().st_mtime:
        warnings.append("feature_cache_older_than_final_1897_case_index")
    try:
        import torch  # type: ignore
    except Exception as exc:
        warnings.append(f"feature_cache_metadata_not_loaded_torch_unavailable: {exc}")
        return info

    payload = torch.load(cache_path, map_location="cpu")
    features = payload.get("features", payload) if isinstance(payload, dict) else payload
    info["feature_count"] = len(features) if hasattr(features, "__len__") else None
    if isinstance(payload, dict):
        info["metadata"] = payload.get("metadata")
    if info["feature_count"] != expected_n:
        warnings.append(f"feature_cache_case_count_mismatch: {info['feature_count']} != {expected_n}")
    metadata = info.get("metadata") or {}
    source = str(metadata.get("source_case_index", "")) if isinstance(metadata, dict) else ""
    if not source:
        warnings.append("feature_cache_missing_source_case_index_metadata")
    elif Path(source).name != FINAL_INDEX.name:
        warnings.append(f"feature_cache_source_mismatch: {source}")
    return info


def summarize_config(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    warnings: list[str] = []
    data_root = Path(cfg.data_root)
    split_paths = {
        "train": data_root / "train_labels.csv",
        "val": data_root / "val_labels.csv",
        "external_test": data_root / "external_test_labels.csv",
    }
    splits: dict[str, Any] = {}
    total_n = 0
    total_col = 0
    total_oct = 0
    for split, path in split_paths.items():
        if not path.exists():
            warnings.append(f"missing_split_csv: {path}")
            splits[split] = {"path": str(path), "exists": False}
            continue
        df = read_csv(path)
        missing_images = int(((df["col_count"].astype(float) <= 0) | (df["oct_count"].astype(float) <= 0)).sum())
        if missing_images:
            warnings.append(f"{split}_missing_image_modalities: {missing_images}")
        splits[split] = {
            "path": str(path),
            "exists": True,
            "n": int(len(df)),
            "positive": int(df["label"].astype(int).sum()),
            "negative": int((df["label"].astype(int) == 0).sum()),
            "centers": sorted(df["center_name"].astype(str).unique().tolist()),
            "colposcopy_images": int(df["col_count"].sum()),
            "oct_images": int(df["oct_count"].sum()),
        }
        total_n += len(df)
        total_col += int(df["col_count"].sum())
        total_oct += int(df["oct_count"].sum())

    expected_n = int(getattr(cfg, "expected_aligned_n", 1897))
    if total_n != expected_n:
        warnings.append(f"split_total_n_mismatch: {total_n} != {expected_n}")
    if getattr(cfg, "clinical_feature_dim", None) != 14:
        warnings.append(f"clinical_feature_dim_not_14: {getattr(cfg, 'clinical_feature_dim', None)}")
    if getattr(cfg, "use_vlm_retriever", None):
        warnings.append("use_vlm_retriever_should_be_false")
    if getattr(cfg, "vlm_json_path", None) is not None:
        warnings.append("vlm_json_path_should_be_none")
    if not getattr(cfg, "no_report_mode", False):
        warnings.append("no_report_mode_should_be_true")

    expected_volume = int(getattr(cfg, "expected_image_volume_approx", 130000))
    total_images = total_col + total_oct
    if abs(total_images - expected_volume) > 15000:
        warnings.append(f"image_volume_not_approximately_expected: {total_images} vs {expected_volume}")

    return {
        "config_path": str(config_path),
        "experiment_name": getattr(cfg, "experiment_name", ""),
        "raw_registry_n": getattr(cfg, "raw_registry_n", None),
        "expected_aligned_n": expected_n,
        "input_modalities": getattr(cfg, "input_modalities", None),
        "no_report_mode": getattr(cfg, "no_report_mode", None),
        "use_vlm_retriever": getattr(cfg, "use_vlm_retriever", None),
        "vlm_json_path": getattr(cfg, "vlm_json_path", None),
        "clinical_feature_dim": getattr(cfg, "clinical_feature_dim", None),
        "data_root": str(data_root),
        "split_total_n": int(total_n),
        "total_colposcopy_images": int(total_col),
        "total_oct_images": int(total_oct),
        "total_images": int(total_images),
        "splits": splits,
        "feature_cache": inspect_cache(Path(cfg.feature_cache_path), expected_n, warnings),
        "warnings": warnings,
        "passed": not warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.config or DEFAULT_CONFIGS
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_case_index": str(FINAL_INDEX),
        "configs": [summarize_config(path) for path in configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote config sanity report to {args.output}")


if __name__ == "__main__":
    main()
