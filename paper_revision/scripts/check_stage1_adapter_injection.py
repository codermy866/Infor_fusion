#!/usr/bin/env python3
"""R2: Verify Stage-1 clinical semantic adapter injection into Stage-2 model."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "paper_revision" / "results" / "stage1_adapter_injection_check.json"
OUT_MD = ROOT / "paper_revision" / "results" / "stage1_adapter_injection_check.md"

ALLOWED_PREFIXES = (
    "note_projector",
    "text_adapter",
    "clinical_feature_projector",
    "align_proj_img",
    "align_proj_text",
    "shared_align_proj",
)
FORBIDDEN_PREFIXES = (
    "classifier",
    "variational_reliability",
    "posterior_refiner",
    "asccp_prior",
    "coe_readout",
    "center_discriminator",
)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from paper_revision.configs.corrected_5center_elbo_structured_prior_config import (
        CorrectedFiveCenterELBOStructuredPriorConfig,
    )
    from models.bio_cot_v3_2 import create_hydra_coe

    cfg = CorrectedFiveCenterELBOStructuredPriorConfig()
    adapter_path = Path(cfg.load_clinical_semantic_adapter_path)
    payload = torch.load(adapter_path, map_location="cpu") if adapter_path.exists() else {}
    ckpt_keys = list(payload.keys()) if isinstance(payload, dict) else []

    model = create_hydra_coe(cfg)
    loaded_adapter_keys = []
    skipped_non_adapter_keys = []
    for key in ckpt_keys:
        if any(key.startswith(p) or key == p for p in ALLOWED_PREFIXES):
            loaded_adapter_keys.append(key)
        elif any(key.startswith(p) for p in FORBIDDEN_PREFIXES):
            skipped_non_adapter_keys.append(key)
        else:
            skipped_non_adapter_keys.append(key)

    adapter_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if any(name.startswith(pfx) for pfx in ALLOWED_PREFIXES)
    )
    trainable_epoch_1 = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and any(name.startswith(pfx) for pfx in ALLOWED_PREFIXES)
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "loaded_stage1_adapter": adapter_path.exists(),
        "adapter_path": str(adapter_path),
        "checkpoint_keys": ckpt_keys,
        "loaded_adapter_keys": loaded_adapter_keys,
        "skipped_non_adapter_keys": skipped_non_adapter_keys,
        "adapter_parameter_count": adapter_params,
        "adapter_trainable_epoch_1": trainable_epoch_1,
        "freeze_at_start": getattr(cfg, "freeze_clinical_semantic_adapter_at_start", None),
        "unfreeze_epoch": getattr(cfg, "unfreeze_clinical_semantic_adapter_epoch", None),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stage-1 Adapter Injection Check",
        "",
        f"- loaded_stage1_adapter: {report['loaded_stage1_adapter']}",
        f"- adapter_path: {adapter_path}",
        f"- loaded_adapter_keys: {loaded_adapter_keys}",
        f"- skipped_non_adapter_keys: {skipped_non_adapter_keys}",
        f"- adapter_parameter_count: {adapter_params}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0 if report["loaded_stage1_adapter"] and loaded_adapter_keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
