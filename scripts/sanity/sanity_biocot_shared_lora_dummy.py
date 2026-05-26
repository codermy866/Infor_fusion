#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT))

from models.bio_cot_v3_2 import BioCOT_v3_2


OUT = ROOT / "outputs/publishable_v2/shared_lora_biocot/audit"
OUT.mkdir(parents=True, exist_ok=True)


def build_model() -> BioCOT_v3_2:
    model = BioCOT_v3_2(
        embed_dim=64,
        input_dim=64,
        hidden_dim=64,
        num_classes=2,
        num_centers=5,
        use_hierarchical=False,
        vit_pretrained=False,
        use_visual_notes=False,
        use_ot=True,
        use_dual=True,
        use_cross_attn=True,
        use_adaptive_gating=False,
        use_text_adapter=True,
        use_variational_reliability=True,
        use_center_aware_reliability=True,
        direct_fusion_only=False,
        use_posterior_refinement=True,
        use_asccp_prior=False,
        use_modality_likelihood=True,
        use_coe_readout=True,
        use_coe_supervision=True,
        use_visual_domain_adapter=False,
        enable_colpo_encoder=False,
        train_colpo_encoder=False,
        use_colpo_lora_bridge=True,
        shared_lora_rank=4,
        shared_lora_alpha=8.0,
        shared_lora_dropout=0.0,
        colpo_bridge_ot_weight=1.0,
        dropout_rate=0.1,
    )
    model.freeze_expert_base()
    return model


def run_case(model: BioCOT_v3_2, f_colpo: torch.Tensor | None, tag: str) -> dict[str, object]:
    batch_size = 4
    f_oct = torch.randn(batch_size, 6, 64)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    center_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    clinical_features = torch.randn(batch_size, 14)
    image_names = [f"dummy_{i}.png" for i in range(batch_size)]
    clinical_info = [f"age {40 + i}; hpv test dummy" for i in range(batch_size)]

    model.train()
    output = model(
        f_oct=f_oct,
        f_colpo=f_colpo,
        image_names=image_names,
        clinical_info=clinical_info,
        center_labels=center_labels,
        labels=labels,
        clinical_features=clinical_features,
        return_loss_components=True,
    )
    loss = F.cross_entropy(output["logits"], labels)
    grad_terms = {}
    for name, value in output.get("loss_components", {}).items():
        if torch.is_tensor(value) and value.dim() == 0 and value.requires_grad:
            loss = loss + 0.01 * value
            grad_terms[name] = float(value.detach().cpu())
    model.zero_grad(set_to_none=True)
    loss.backward()
    trainable_with_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all()
    ]
    return {
        "tag": tag,
        "logits_shape": list(output["logits"].shape),
        "modality_mask": {k: v.detach().cpu().view(-1).tolist() for k, v in output["modality_mask"].items()},
        "loss": float(loss.detach().cpu()),
        "grad_terms": grad_terms,
        "has_colpo_lora_aligned": "colpo_lora_aligned" in output,
        "trainable_tensors_with_grad": trainable_with_grad,
    }


def main() -> None:
    torch.manual_seed(2026)
    model = build_model()
    summary = model.trainable_parameter_summary()
    missing_colpo = run_case(model, None, "missing_colpo")
    dummy_colpo = run_case(model, torch.randn(4, 6, 64), "dummy_colpo_features")
    payload = {
        "status": "PASS",
        "summary": summary,
        "cases": [missing_colpo, dummy_colpo],
    }
    (OUT / "biocot_shared_lora_dummy_sanity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
