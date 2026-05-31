#!/usr/bin/env python3
"""Auto-generated production config after full 1897 LOCO ablation."""
from dataclasses import dataclass
from configs.shared_lora_loco_template import SharedLoRALOCOConfig


@dataclass
class AblationPrunedLOCOConfig(SharedLoRALOCOConfig):
    """Best stack: g2 (ablation AUC=0.6809)."""

    lambda_adv: float = 0.0
    lambda_align: float = 0.0
    lambda_cls: float = 2.0
    lambda_coe: float = 0.0
    lambda_colpo_bridge_align: float = 0.0
    lambda_colpo_bridge_ot: float = 0.0
    lambda_consist: float = 0.0
    lambda_modality_likelihood: float = 0.0
    lambda_ot: float = 0.5
    lambda_posterior_smooth: float = 0.0
    lambda_reliability_kl: float = 0.0
    shared_lora_alpha: float = 16.0
    shared_lora_rank: int = 8
    use_adversarial: bool = False
    use_center_aware_reliability: bool = False
    use_coe_readout: bool = False
    use_coe_supervision: bool = False
    use_colpo_lora_bridge: bool = False
    use_cross_attn: bool = False
    use_dual: bool = False
    use_modality_likelihood: bool = False
    use_ot: bool = True
    use_posterior_refinement: bool = False
    use_variational_reliability: bool = False
