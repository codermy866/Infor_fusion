#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HyDRA config for labeled target-center-adapted Enshi validation."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.full_pretrained_config import FullPretrainedHyDRAConfig


@dataclass
class EnshiTargetAdaptedHyDRAConfig(FullPretrainedHyDRAConfig):
    experiment_name: str = "HyDRA_EnshiTargetAdapted_20_10_70"
    experiment_description: str = "Target-center-adapted Enshi validation with 20% labeled target adaptation, 10% target validation, and 70% held-out Enshi test."

    data_root: str = str(ROOT / "paper_revision" / "splits" / "target_adapted_validation" / "enshi_target_adapted_20_10_70")
    num_centers: int = 5

    use_variational_reliability: bool = True
    fusion_strategy: str = "variational"
    lambda_reliability_kl: float = 0.01

    # Formal revision training strategy: target labels are used only in the
    # predefined adaptation/validation partitions, while the 70% Enshi holdout
    # remains untouched for final reporting.
    checkpoint_metric: str = "auc_minus_ece"
    ece_penalty: float = 0.25
    calibration_bins: int = 10
    center_balanced_sampling: bool = True
    train_modality_dropout_prob: float = 0.15
    train_two_modality_dropout_prob: float = 0.05
    train_image_corruption_prob: float = 0.20
    use_cached_patch_features: bool = True
    feature_cache_path: str = str(ROOT / "paper_revision" / "cache" / "patch_features_enshi_target_adapted.pt")
    train_feature_noise_prob: float = 0.20
    train_feature_noise_std: float = 0.015
    oct_speckle_std: float = 0.05
    learning_rate: float = 0.00015
    weight_decay: float = 0.05
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True

    output_dir: str = "paper_revision/results/enshi_target_adapted/results"
    checkpoint_dir: str = "paper_revision/results/enshi_target_adapted/checkpoints"
    log_dir: str = "paper_revision/results/enshi_target_adapted/logs"

    def __post_init__(self):
        super().__post_init__()
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
