#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset that serves cached patient-level ViT patch features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def case_key(row: pd.Series | Dict[str, Any]) -> str:
    """Stable key shared by feature precomputation, training, and evaluation."""
    getter = row.get if hasattr(row, "get") else row.__getitem__
    patient_id = str(getter("patient_id", getter("ID", "")))
    oct_id = str(getter("oct_id", getter("OCT", "")))
    return f"{patient_id}||{oct_id}"


def center_from_row(row: pd.Series) -> int:
    if "center_group_id" in row and not pd.isna(row["center_group_id"]):
        return int(row["center_group_id"])
    if "center_idx" in row and not pd.isna(row["center_idx"]):
        return int(row["center_idx"])
    return 0


def parse_hpv(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1.0 if float(value) > 0 else 0.0
    value_str = str(value).lower()
    return 1.0 if any(token in value_str for token in ["16", "18", "positive", "阳性", "高危", "1"]) else 0.0


def parse_clinical_features(row: pd.Series) -> torch.Tensor:
    age_value = row["age"] if "age" in row else row.get("AGE", 50.0)
    age = float(age_value) if not pd.isna(age_value) else 50.0
    hpv_value = row["hpv"] if "hpv" in row else row.get("HPV清洗", 0)
    hpv = parse_hpv(hpv_value)
    tct_value = row["tct"] if "tct" in row else row.get("TCT清洗", "")
    tct_str = str(tct_value).upper() if not pd.isna(tct_value) else ""
    tct_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
    if "ASC-US" in tct_str:
        tct_onehot[0] = 1.0
    elif "ASC-H" in tct_str:
        tct_onehot[1] = 1.0
    elif "LSIL" in tct_str:
        tct_onehot[2] = 1.0
    elif "HSIL" in tct_str:
        tct_onehot[3] = 1.0
    elif "SCC" in tct_str or "癌" in tct_str:
        tct_onehot[4] = 1.0
    return torch.tensor([age / 100.0, hpv, *tct_onehot], dtype=torch.float32)


def first_name(paths_value: Any) -> str:
    if pd.isna(paths_value):
        return "unknown.jpg"
    paths = [item.strip() for item in str(paths_value).replace(",", ";").split(";") if item.strip()]
    if not paths:
        return "unknown.jpg"
    return Path(paths[0]).name


class CachedPatchFeatureDataset(Dataset):
    """Read split CSV metadata and cached patient-level patch features."""

    def __init__(self, csv_path: str | Path, feature_cache_path: str | Path):
        self.csv_path = Path(csv_path)
        try:
            self.df = pd.read_csv(self.csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_path, encoding="gbk")
        self.df["center_idx"] = self.df.apply(center_from_row, axis=1)

        cache_path = Path(feature_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached patch feature file not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        self.features: Dict[str, Dict[str, torch.Tensor]] = payload.get("features", payload)
        self.missing_keys = [case_key(row) for _, row in self.df.iterrows() if case_key(row) not in self.features]
        if self.missing_keys:
            preview = ", ".join(self.missing_keys[:3])
            raise KeyError(f"{len(self.missing_keys)} cases missing from feature cache. Examples: {preview}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        key = case_key(row)
        feats = self.features[key]
        age_value = row["age"] if "age" in row else row.get("AGE", 50)
        hpv_value = row["hpv"] if "hpv" in row else row.get("HPV清洗", 0)
        tct_value = row["tct"] if "tct" in row else row.get("TCT清洗", "NILM")
        clinical_info = (
            f"HPV: {'positive' if parse_hpv(hpv_value) > 0 else 'negative'}, "
            f"TCT: {str(tct_value).upper() if not pd.isna(tct_value) else 'NILM'}, "
            f"Age: {int(float(age_value)) if not pd.isna(age_value) else 50}"
        )
        return {
            "oct_patch_features": feats["oct"].float(),
            "colpo_patch_features": feats["colpo"].float(),
            "clinical_features": parse_clinical_features(row),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "center_idx": torch.tensor(int(row["center_idx"]), dtype=torch.long),
            "image_names": first_name(row.get("oct_paths", "")),
            "clinical_info": clinical_info,
            "oct_id": str(row.get("oct_id", row.get("OCT", ""))),
            "case_key": key,
        }
