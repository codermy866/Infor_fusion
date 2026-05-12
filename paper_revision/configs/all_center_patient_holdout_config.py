#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HyDRA config for multi-center patient-level held-out validation."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.full_pretrained_config import FullPretrainedHyDRAConfig


@dataclass
class AllCenterPatientHoldoutHyDRAConfig(FullPretrainedHyDRAConfig):
    experiment_name: str = "HyDRA_AllCenterPatientHoldout_70_10_20"
    experiment_description: str = "Multi-center patient-level held-out validation; not strict center-external validation."

    data_root: str = str(ROOT / "paper_revision" / "splits" / "target_adapted_validation" / "all_center_patient_holdout_70_10_20")
    num_centers: int = 5

    use_variational_reliability: bool = True
    use_adaptive_gating: bool = False
    fusion_strategy: str = "variational"
    lambda_reliability_kl: float = 0.01

    # Formal revision training strategy for the primary performance-oriented
    # analysis: multi-center patient-level holdout, selected by discrimination
    # with a calibration penalty.
    checkpoint_metric: str = "auc_minus_ece"
    ece_penalty: float = 0.25
    calibration_bins: int = 10
    center_balanced_sampling: bool = True
    train_modality_dropout_prob: float = 0.12
    train_two_modality_dropout_prob: float = 0.03
    train_image_corruption_prob: float = 0.15
    use_cached_patch_features: bool = True
    feature_cache_path: str = str(ROOT / "paper_revision" / "cache" / "patch_features_all_center_patient_holdout.pt")
    train_feature_noise_prob: float = 0.20
    train_feature_noise_std: float = 0.015
    oct_speckle_std: float = 0.05
    learning_rate: float = 0.0002
    weight_decay: float = 0.05
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True

    output_dir: str = "paper_revision/results/all_center_patient_holdout/results"
    checkpoint_dir: str = "paper_revision/results/all_center_patient_holdout/checkpoints"
    log_dir: str = "paper_revision/results/all_center_patient_holdout/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
