#!/usr/bin/env python3
"""Rebuild lightweight NPZ feature caches from the locked 1897 patch cache.

This avoids repeated image traversal through sshfs-backed OCT directories. The
source cache must have been generated from
paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv and
contains no report fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.clinical_variable_mapping import (
    assert_no_report_columns,
    clinical_features_from_row,
)


PAPER_DIR = EXP_ROOT / "paper_revision"
DEFAULT_SPLIT_DIR = PAPER_DIR / "splits" / "target_adapted_validation" / "all_center_patient_holdout_70_10_20"
DEFAULT_CACHE = PAPER_DIR / "cache" / "patch_features_final_1897.pt"
DEFAULT_OUTPUT_DIR = PAPER_DIR / "results" / "feature_cache"


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def case_key(row: pd.Series) -> str:
    patient_id = str(row.get("patient_id", row.get("ID", "")))
    oct_id = str(row.get("oct_id", row.get("OCT", "")))
    return f"{patient_id}||{oct_id}"


def pooled_feature(value: Any) -> np.ndarray:
    tensor = value.detach().float().cpu() if torch.is_tensor(value) else torch.as_tensor(value).float()
    if tensor.ndim == 1:
        return tensor.numpy().astype(np.float32)
    return tensor.reshape(-1, tensor.shape[-1]).mean(dim=0).numpy().astype(np.float32)


def build_one(split_name: str, csv_path: Path, features: dict[str, dict[str, torch.Tensor]], out_dir: Path) -> dict[str, Any]:
    df = read_csv(csv_path)
    assert_no_report_columns(df.columns)
    rows = []
    missing = []
    for _, row in df.iterrows():
        key = case_key(row)
        item = features.get(key)
        if item is None:
            missing.append(key)
            continue
        rows.append(
            {
                "case_id": str(row.get("ID", row.get("patient_id", ""))),
                "oct_id": str(row.get("OCT", row.get("oct_id", ""))),
                "center": str(row.get("center_name", "")),
                "center_idx": int(row.get("center_group_id", row.get("center_idx", 0))),
                "label": int(row["label"]),
                "oct": pooled_feature(item["oct"]),
                "col": pooled_feature(item["colpo"]),
                "clinical": np.asarray(clinical_features_from_row(row), dtype=np.float32),
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} cases from {csv_path} are missing in patch cache. Examples: {preview}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}_vit_patient_features.npz"
    np.savez_compressed(
        out_path,
        oct=np.stack([row["oct"] for row in rows]).astype(np.float32),
        col=np.stack([row["col"] for row in rows]).astype(np.float32),
        clinical=np.stack([row["clinical"] for row in rows]).astype(np.float32),
        y=np.asarray([row["label"] for row in rows], dtype=np.int64),
        center_idx=np.asarray([row["center_idx"] for row in rows], dtype=np.int64),
        case_id=np.asarray([row["case_id"] for row in rows], dtype=object),
        center=np.asarray([row["center"] for row in rows], dtype=object),
        oct_id=np.asarray([row["oct_id"] for row in rows], dtype=object),
    )
    return {
        "split": split_name,
        "csv_path": str(csv_path),
        "output_path": str(out_path),
        "n": len(rows),
        "positive": int(sum(row["label"] for row in rows)),
        "negative": int(len(rows) - sum(row["label"] for row in rows)),
        "centers": sorted({row["center"] for row in rows}),
        "clinical_feature_dim": int(rows[0]["clinical"].shape[0]) if rows else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.feature_cache, map_location="cpu")
    features = payload.get("features", payload)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if metadata.get("no_report_mode") is not True:
        raise ValueError(f"Feature cache does not declare no_report_mode=True: {args.feature_cache}")
    if len(features) != 1897:
        raise ValueError(f"Feature cache has {len(features)} cases; expected 1897.")

    summaries = []
    for split_name, file_name in [
        ("train", "train_labels.csv"),
        ("internal_validation", "val_labels.csv"),
        ("external_test", "external_test_labels.csv"),
    ]:
        summaries.append(build_one(split_name, args.split_dir / file_name, features, args.output_dir))

    summary_path = args.output_dir / "locked_1897_feature_cache_summary.json"
    summary_path.write_text(json.dumps({"source_cache": str(args.feature_cache), "splits": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_cache": str(args.feature_cache), "splits": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
