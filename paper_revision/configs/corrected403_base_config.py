#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared base fields for corrected 403-case external-test clean reruns."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORRECTED_ROOT = ROOT / "paper_revision" / "results" / "real_50epoch_5center_corrected"
STAGE1_ADAPTER = (
    ROOT
    / "paper_revision"
    / "results"
    / "stage1_clinical_semantic_adapter"
    / "checkpoints"
    / "stage1_clinical_semantic_adapter_best.pth"
)


@dataclass
class Corrected403Mixin:
    """Mixin applied to corrected five-centre experiment configs."""

    expected_aligned_n: int = 1897
    expected_external_n: int = 403
    raw_registry_n: int = 3010
    no_report_mode: bool = True
    use_vlm_retriever: bool = False
    vlm_json_path: str = None
    data_root: str = str(
        ROOT
        / "paper_revision"
        / "splits"
        / "target_adapted_validation"
        / "all_center_patient_holdout_70_10_20"
    )
    feature_cache_path: str = str(ROOT / "paper_revision" / "cache" / "patch_features_final_1897.pt")
    use_cached_patch_features: bool = True
    num_epochs: int = 50
    load_clinical_semantic_adapter_path: str = str(STAGE1_ADAPTER)
    freeze_clinical_semantic_adapter_at_start: bool = True
    unfreeze_clinical_semantic_adapter_epoch: int = 5
    adapter_lr_multiplier: float = 0.1
    display_method_name: str = "HyDRA-CoE"
    paper_method_name: str = "HyDRA-CoE"
