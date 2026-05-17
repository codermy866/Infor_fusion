#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cervix-domain visual-language adaptation experiments.

These configs keep the official cached patient-level ViT patch features fixed
unless a feature-space visual adapter is explicitly enabled. This makes the
screening runs cheap and avoids changing the frozen external-evaluation split.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_revision.configs.coe_lambda_002_config import CoELambda002Config


@dataclass
class CervixAdaptStaticPriorConfig(CoELambda002Config):
    experiment_name: str = "CervixAdapt_StaticPrior"
    experiment_description: str = "Cached frozen ViT features with structured prior and low-weight CoE."

    use_vlm_retriever: bool = False
    use_visual_domain_adapter: bool = False
    freeze_visual_encoder: bool = True
    train_text_encoder: bool = False
    text_encoder_trainable_layers: int = 0

    output_dir: str = "paper_revision/results/cervix_adapt_static_prior/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_static_prior/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_static_prior/logs"


@dataclass
class CervixAdaptVisualAdapterOnlyConfig(CervixAdaptStaticPriorConfig):
    experiment_name: str = "CervixAdapt_VisualAdapterOnly"
    experiment_description: str = "Cached frozen ViT features plus modality-specific residual feature adapters."

    use_vlm_retriever: bool = False
    use_visual_domain_adapter: bool = True
    visual_adapter_bottleneck: int = 192
    visual_adapter_dropout: float = 0.10
    train_text_encoder: bool = False
    text_encoder_trainable_layers: int = 0

    output_dir: str = "paper_revision/results/cervix_adapt_visual_adapter/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_visual_adapter/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_visual_adapter/logs"


@dataclass
class CervixAdaptBERTAdapterOnlyConfig(CervixAdaptStaticPriorConfig):
    experiment_name: str = "CervixAdapt_BERTAdapterOnly"
    experiment_description: str = "Frozen PubMedBERT/VLM semantic anchor with trainable text adapters."

    use_vlm_retriever: bool = True
    use_visual_domain_adapter: bool = False
    train_text_encoder: bool = False
    text_encoder_trainable_layers: int = 0
    text_model_name: str = str(ROOT / "paper_revision" / "cache" / "pubmedbert_safetensors")

    output_dir: str = "paper_revision/results/cervix_adapt_bert_adapter/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_bert_adapter/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_bert_adapter/logs"


@dataclass
class CervixAdaptVisualBERTAdapterConfig(CervixAdaptBERTAdapterOnlyConfig):
    experiment_name: str = "CervixAdapt_VisualBERTAdapter"
    experiment_description: str = "Residual visual feature adapters plus frozen PubMedBERT/VLM semantic adapter."

    use_visual_domain_adapter: bool = True
    visual_adapter_bottleneck: int = 192
    visual_adapter_dropout: float = 0.10

    output_dir: str = "paper_revision/results/cervix_adapt_visual_bert_adapter/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_visual_bert_adapter/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_visual_bert_adapter/logs"


@dataclass
class CervixAdaptBERTLastLayerFTConfig(CervixAdaptBERTAdapterOnlyConfig):
    experiment_name: str = "CervixAdapt_BERTLastLayerFT"
    experiment_description: str = "PubMedBERT/VLM semantic anchor with the last BERT layer fine-tuned."

    train_text_encoder: bool = True
    text_encoder_trainable_layers: int = 1
    learning_rate: float = 0.00005

    output_dir: str = "paper_revision/results/cervix_adapt_bert_last_layer_ft/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_bert_last_layer_ft/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_bert_last_layer_ft/logs"


@dataclass
class CervixAdaptVisualFullTextFTConfig(CervixAdaptVisualBERTAdapterConfig):
    experiment_name: str = "CervixAdapt_VisualFullTextFT"
    experiment_description: str = "Residual visual adapters plus full PubMedBERT fine-tuning at a conservative LR."

    train_text_encoder: bool = True
    text_encoder_trainable_layers: int = -1
    learning_rate: float = 0.00002

    output_dir: str = "paper_revision/results/cervix_adapt_visual_full_text_ft/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_visual_full_text_ft/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_visual_full_text_ft/logs"


@dataclass
class CervixAdaptStage1ContrastiveConfig(CervixAdaptVisualAdapterOnlyConfig):
    experiment_name: str = "CervixAdapt_Stage1Contrastive"
    experiment_description: str = (
        "Stage-2 HyDRA initialized from Stage-1 cross-modal contrastive visual/clinical adapters."
    )

    load_domain_pretrain_path: str = ""

    output_dir: str = "paper_revision/results/cervix_adapt_stage1_contrastive/results"
    checkpoint_dir: str = "paper_revision/results/cervix_adapt_stage1_contrastive/checkpoints"
    log_dir: str = "paper_revision/results/cervix_adapt_stage1_contrastive/logs"
