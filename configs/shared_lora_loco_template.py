#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path

from config import BioCOT_v3_2_Config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/shared_lora_biocot/loco"


@dataclass
class SharedLoRALOCOConfig(BioCOT_v3_2_Config):
    """Template for downstream LOCO Shared-LoRA adaptation.

    Use the training script with four source centres as train CSV and one
    source-only inner centre as validation CSV. The held-out centre must be
    evaluated only after checkpoint selection.
    """

    data_root: str = str(ROOT / "paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20")
    output_dir: str = str(OUT)
    checkpoint_dir: str = str(OUT / "checkpoints")
    log_dir: str = str(OUT / "logs")

    batch_size: int = 4
    num_epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    pin_memory: bool = True
    oct_frames: int = 8
    colposcopy_images: int = 3
    vit_batch_size: int = 8

    use_cached_patch_features: bool = False
    use_hierarchical: bool = False
    vit_pretrained: bool = False
    use_visual_notes: bool = False
    use_noise_aware: bool = False
    use_clinical_evolver: bool = False
    use_adaptive_gating: bool = False
    use_variational_reliability: bool = True
    use_posterior_refinement: bool = True
    use_asccp_prior: bool = False
    use_modality_likelihood: bool = True
    use_coe_readout: bool = True
    use_coe_supervision: bool = True
    use_ot: bool = True
    use_dual: bool = True
    use_cross_attn: bool = True

    pass_raw_colpo_to_model: bool = True
    enable_colpo_encoder: bool = True
    colpo_encoder_name: str = "vit_base_patch16_224"
    colpo_encoder_pretrained: bool = True
    train_colpo_encoder: bool = False
    freeze_expert_base_for_lora: bool = True
    freeze_colpo_encoder_for_lora: bool = True
    use_colpo_lora_bridge: bool = True
    shared_lora_rank: int = 8
    shared_lora_alpha: float = 16.0
    shared_lora_dropout: float = 0.05
    colpo_bridge_ot_weight: float = 1.0
    lambda_colpo_bridge_ot: float = 0.2
    lambda_colpo_bridge_align: float = 0.05
