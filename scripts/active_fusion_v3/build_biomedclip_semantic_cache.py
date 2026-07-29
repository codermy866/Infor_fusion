#!/usr/bin/env python3
"""Build paired-concept semantic anchors without reading any outcome field."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "third_party/biomedclip_runtime"))
sys.path.insert(0, str(EXP_ROOT))

from scripts.active_fusion_v2.extract_qwen3vl_colposcopy import sha256_file  # noqa: E402
from scripts.active_fusion_v3.extract_biomedclip_concepts import (  # noqa: E402
    GENERIC_PAIRS,
    MEDICAL_PAIRS,
    encode_prompt_pairs,
    load_local_model,
    prompt_hash,
    save_payload,
)


def semantic_anchor(
    scores: torch.Tensor,
    text_pairs: torch.Tensor,
) -> torch.Tensor:
    """Expected text embedding under each paired concept probability."""
    positive = text_pairs[:, 0].cpu().float()
    negative = text_pairs[:, 1].cpu().float()
    expected = (
        scores.unsqueeze(-1) * positive
        + (1.0 - scores).unsqueeze(-1) * negative
    )
    return F.normalize(expected.mean(dim=0), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/"
            "biomedclip_paired_colposcopy_concepts.pt"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/biomedclip_model"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/"
            "biomedclip_paired_semantic_anchors.pt"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    if len(source["features"]) != 1897 or source.get("failures"):
        raise ValueError("paired concept cache must be complete and failure-free")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer, _ = load_local_model(args.model_dir, device)
    medical_text = encode_prompt_pairs(
        model, tokenizer, MEDICAL_PAIRS, device
    )
    generic_text = encode_prompt_pairs(
        model, tokenizer, GENERIC_PAIRS, device
    )
    features = {}
    for key, value in source["features"].items():
        medical_score = value["medical"].float()
        generic_score = value["generic"].float()
        medical_anchor = semantic_anchor(medical_score, medical_text)
        generic_anchor = semantic_anchor(generic_score, generic_text)
        features[key] = {
            "medical": torch.cat([medical_score, medical_anchor]),
            "generic": torch.cat([generic_score, generic_anchor]),
            "medical_concepts": medical_score,
            "generic_concepts": generic_score,
        }
    payload = {
        "schema": "biomedclip_paired_semantic_anchors_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "model_weight_sha256": sha256_file(
            args.model_dir / "open_clip_pytorch_model.bin"
        ),
        "medical_prompt_sha256": prompt_hash(MEDICAL_PAIRS),
        "generic_prompt_sha256": prompt_hash(GENERIC_PAIRS),
        "semantic_dimension": 519,
        "construction": (
            "seven paired probabilities concatenated with the normalized mean "
            "of their probability-weighted positive/negative text embeddings"
        ),
        "outcome_fields_used": [],
        "features": features,
        "failures": {},
    }
    save_payload(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "n": len(features),
                "dimension": payload["semantic_dimension"],
                "output": str(args.output),
                "sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
