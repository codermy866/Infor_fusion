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
    expected_aligned_n: int = 1897
    raw_registry_n: int = 3010
    expected_image_volume_approx: int = 130000
    input_modalities: list = None
    no_report_mode: bool = True
    use_vlm_retriever: bool = False
    vlm_json_path: str = None
    clinical_feature_dim: int = 14
    primary_endpoint: str = "CIN2+"
    locked_sensitivity_target: float = 0.95
    threshold_rule: str = "max_specificity_at_sensitivity:0.95"
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
    feature_cache_path: str = str(ROOT / "paper_revision" / "cache" / "patch_features_final_1897.pt")
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
        if self.input_modalities is None:
            self.input_modalities = ["colposcopy", "oct", "clinical_text_hpv_tct_age"]
        super().__post_init__()
        self.use_vlm_retriever = False
        self.vlm_json_path = None
        for dir_name in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
