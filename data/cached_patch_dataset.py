#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset that serves cached patient-level ViT patch features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from paper_revision.scripts.clinical_variable_mapping import (
    assert_no_report_columns,
    clinical_features_from_row,
    clinical_info_from_row,
    normalize_hpv,
)


REPORT_LIKE_COLUMNS = {
    "report",
    "report_text",
    "clinical_report",
    "diagnosis_report",
    "generated_report",
    "exam_report",
    "examination_report",
    "检查报告",
    "诊断报告",
}


def drop_report_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove report-like columns; not used in current experiments."""
    report_cols = [
        col
        for col in df.columns
        if str(col).strip().lower() in REPORT_LIKE_COLUMNS
        or "report" in str(col).strip().lower()
    ]
    if report_cols:
        return df.drop(columns=report_cols)
    return df


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
    """Legacy scalar wrapper around strict normalized HPV parsing.

    A lone "1" is no longer treated as HPV-positive. Only explicit positive
    patterns or recognized high-risk genotypes produce 1.0.
    """
    normalized = normalize_hpv(value)
    return 1.0 if normalized in {"hrhpv_positive", "hpv16_18_positive", "other_hrhpv_positive"} else 0.0


def parse_clinical_features(row: pd.Series) -> torch.Tensor:
    """Build 14-D clinical features from age, HPV, and TCT only."""
    return torch.tensor(clinical_features_from_row(row), dtype=torch.float32)


def first_name(paths_value: Any) -> str:
    if pd.isna(paths_value):
        return "unknown.jpg"
    paths = [item.strip() for item in str(paths_value).replace(",", ";").split(";") if item.strip()]
    if not paths:
        return "unknown.jpg"
    return Path(paths[0]).name


class CachedPatchFeatureDataset(Dataset):
    """Read split CSV metadata and cached patient-level patch features."""

    def __init__(
        self,
        csv_path: str | Path,
        feature_cache_path: str | Path,
        expected_clinical_feature_dim: int | None = None,
        expected_aligned_n: int | None = None,
        expected_case_index_csv: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        try:
            self.df = pd.read_csv(self.csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_path, encoding="gbk")
        assert_no_report_columns(self.df.columns)
        self.df = drop_report_like_columns(self.df)
        self.df["center_idx"] = self.df.apply(center_from_row, axis=1)

        cache_path = Path(feature_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached patch feature file not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        self.features: Dict[str, Dict[str, torch.Tensor]] = payload.get("features", payload)
        self._warn_if_cache_metadata_mismatch(
            payload=payload,
            cache_path=cache_path,
            expected_aligned_n=expected_aligned_n,
            expected_case_index_csv=expected_case_index_csv,
        )
        self.missing_keys = [case_key(row) for _, row in self.df.iterrows() if case_key(row) not in self.features]
        if self.missing_keys:
            preview = ", ".join(self.missing_keys[:3])
            raise KeyError(f"{len(self.missing_keys)} cases missing from feature cache. Examples: {preview}")
        if expected_clinical_feature_dim is not None:
            observed_dim = int(parse_clinical_features(self.df.iloc[0]).numel()) if len(self.df) else expected_clinical_feature_dim
            if observed_dim != int(expected_clinical_feature_dim):
                raise ValueError(
                    f"clinical_features dim mismatch: observed {observed_dim}, "
                    f"expected {expected_clinical_feature_dim}"
                )

    @staticmethod
    def _warn_if_cache_metadata_mismatch(
        payload: Any,
        cache_path: Path,
        expected_aligned_n: int | None,
        expected_case_index_csv: str | Path | None,
    ) -> None:
        if not isinstance(payload, dict) or "features" not in payload:
            warnings.warn(
                f"Feature cache {cache_path} has no cohort metadata; verify it was generated from final_1897_case_index.csv.",
                RuntimeWarning,
            )
            return
        features = payload.get("features", {})
        metadata = payload.get("metadata", {})
        if expected_aligned_n is not None and len(features) != int(expected_aligned_n):
            warnings.warn(
                f"Feature cache {cache_path} has {len(features)} cases; expected {expected_aligned_n}.",
                RuntimeWarning,
            )
        if expected_case_index_csv is not None:
            expected_name = Path(expected_case_index_csv).name
            source = str(metadata.get("source_case_index", metadata.get("case_index_csv", "")))
            if source and Path(source).name != expected_name:
                warnings.warn(
                    f"Feature cache {cache_path} metadata source {source} does not match {expected_name}.",
                    RuntimeWarning,
                )
            elif not source:
                warnings.warn(
                    f"Feature cache {cache_path} has no source_case_index metadata; verify cohort alignment manually.",
                    RuntimeWarning,
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        key = case_key(row)
        feats = self.features[key]
        # Current experiments do not use examination reports; clinical_info is
        # exactly HPV/TCT/Age and never contains pathology labels.
        clinical_info = clinical_info_from_row(row)
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
