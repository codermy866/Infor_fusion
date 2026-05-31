#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    pass_raw_oct_to_model: bool = False
    vit_model_name: str = "vit_base_patch16_224"
    vit_checkpoint_path: Optional[str] = None
    vit_pretrained: bool = False
    raw_oct_encoder_batch_size: int = 8
    use_visual_notes: bool = False
    use_noise_aware: bool = False
    use_clinical_evolver: bool = False
    use_adaptive_gating: bool = False
    use_variational_reliability: bool = False
    use_posterior_refinement: bool = False
    use_asccp_prior: bool = False
    use_modality_likelihood: bool = False
    use_coe_readout: bool = False
    use_coe_supervision: bool = False
    use_ot: bool = True
    use_dual: bool = False
    use_adversarial: bool = False
    use_cross_attn: bool = False

    pass_raw_colpo_to_model: bool = True
    enable_colpo_encoder: bool = True
    colpo_encoder_name: str = "vit_base_patch16_224"
    colpo_encoder_checkpoint_path: Optional[str] = None
    colpo_encoder_pretrained: bool = True
    train_colpo_encoder: bool = False
    freeze_expert_base_for_lora: bool = True
    freeze_colpo_encoder_for_lora: bool = True
    use_colpo_lora_bridge: bool = False
    shared_lora_rank: int = 8
    shared_lora_alpha: float = 16.0
    shared_lora_dropout: float = 0.05
    colpo_bridge_ot_weight: float = 1.0
    lambda_colpo_bridge_ot: float = 0.0
    lambda_colpo_bridge_align: float = 0.0
    use_oct_encoder_lora: bool = False
    use_colpo_encoder_lora: bool = False
    use_fusion_layer_lora: bool = False
    encoder_lora_rank: int = 8
    encoder_lora_alpha: float = 16.0
    encoder_lora_dropout: float = 0.05
    encoder_lora_targets: tuple[str, ...] = ("attn.qkv", "attn.proj")

    # Post-ablation validated defaults (1897 LOCO g1, May 2026)
    lambda_adv: float = 0.0
