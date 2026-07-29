#!/usr/bin/env python3
"""Extract label-free paired concepts with the official BiomedCLIP model.

The encoder sees only a deterministic colposcopy montage. Centre, pathology,
CIN2+, CIN3+ and patient outcome fields are projected out before inference.
Each medical concept is paired with a visible negative description. A matched
generic prompt bank provides a semantic-specificity control at the same
dimensionality and numerical scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


EXP_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = EXP_ROOT / "third_party/biomedclip_runtime"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(EXP_ROOT))

import open_clip  # noqa: E402
from open_clip.factory import _MODEL_CONFIGS  # noqa: E402

from scripts.active_fusion_v2.extract_qwen3vl_colposcopy import (  # noqa: E402
    case_key,
    make_montage,
    select_views,
    sha256_file,
    sha256_text,
)


MEDICAL_KEYS = (
    "transformation_zone_visibility",
    "acetowhite_extent",
    "vascular_abnormality",
    "surface_irregularity",
    "tissue_boundary_visibility",
    "image_quality",
    "assessment_uncertainty",
)
MEDICAL_PAIRS = (
    (
        "a colposcopy image with the cervical transformation zone clearly visible",
        "a colposcopy image with the cervical transformation zone obscured",
    ),
    (
        "a colposcopy image with extensive bright acetowhite epithelium after acetic acid",
        "a colposcopy image with little or no visible acetowhite epithelium",
    ),
    (
        "a colposcopy image with irregular coarse punctation or mosaic vascular patterns",
        "a colposcopy image with regular fine vascular patterns",
    ),
    (
        "a colposcopy image with an irregular raised or heterogeneous epithelial surface",
        "a colposcopy image with a smooth uniform epithelial surface",
    ),
    (
        "a colposcopy image with a clearly demarcated abnormal tissue boundary",
        "a colposcopy image without a clearly demarcated tissue boundary",
    ),
    (
        "a sharp well illuminated high quality colposcopy image",
        "a blurred poorly illuminated low quality colposcopy image",
    ),
    (
        "an obscured ambiguous colposcopy image that is difficult to assess visually",
        "a clear interpretable colposcopy image that is easy to assess visually",
    ),
)

GENERIC_KEYS = (
    "subject_visibility",
    "white_region_extent",
    "line_pattern_complexity",
    "surface_texture",
    "boundary_visibility",
    "image_quality",
    "assessment_uncertainty",
)
GENERIC_PAIRS = (
    (
        "an image with the main subject clearly visible",
        "an image with the main subject obscured",
    ),
    (
        "an image with extensive bright white regions",
        "an image with little or no bright white region",
    ),
    (
        "an image with complex irregular line patterns",
        "an image with simple regular line patterns",
    ),
    (
        "an image with an uneven heterogeneous textured surface",
        "an image with a smooth uniform surface",
    ),
    (
        "an image with a clearly demarcated object boundary",
        "an image without a clearly demarcated object boundary",
    ),
    (
        "a sharp well illuminated high quality image",
        "a blurred poorly illuminated low quality image",
    ),
    (
        "an obscured ambiguous image that is difficult to assess visually",
        "a clear interpretable image that is easy to assess visually",
    ),
)


def prompt_hash(pairs: Sequence[tuple[str, str]]) -> str:
    return sha256_text(json.dumps(pairs, ensure_ascii=False, sort_keys=False))


def save_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_local_model(model_dir: Path, device: torch.device):
    config = json.loads((model_dir / "open_clip_config.json").read_text())
    model_cfg = config["model_cfg"]
    model_cfg["text_cfg"]["hf_model_name"] = str(model_dir.resolve())
    model_cfg["text_cfg"]["hf_tokenizer_name"] = str(model_dir.resolve())
    model_name = "biomedclip_local"
    _MODEL_CONFIGS[model_name] = model_cfg
    preprocess_kwargs = {
        f"image_{key}": value
        for key, value in config["preprocess_cfg"].items()
    }
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=str((model_dir / "open_clip_pytorch_model.bin").resolve()),
        **preprocess_kwargs,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.to(device).eval(), preprocess, tokenizer, config


@torch.inference_mode()
def encode_prompt_pairs(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    device: torch.device,
    context_length: int = 256,
) -> torch.Tensor:
    flat = [text for pair in pairs for text in pair]
    tokens = tokenizer(flat, context_length=context_length).to(device)
    features = model.encode_text(tokens, normalize=True)
    return features.reshape(len(pairs), 2, -1)


def paired_probabilities(
    image_features: torch.Tensor,
    text_pairs: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits = (
        logit_scale.float()
        * torch.einsum("bd,cpd->bcp", image_features.float(), text_pairs.float())
    )
    return logits.softmax(dim=-1)[..., 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        choices=("biomedclip", "openai_clip"),
        default="biomedclip",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/publishable_v2/data_lock/data_lock_n1897.csv"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/biomedclip_model"
        ),
    )
    parser.add_argument(
        "--openai-weight",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/"
            "openai_clip_vit_b16_model/ViT-B-16.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/"
            "biomedclip_paired_colposcopy_concepts.pt"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preprocess-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    allowed = {
        "case_id",
        "patient_id",
        "exam_id_or_oct_id",
        "colposcopy_paths",
        "colposcopy_available",
    }
    rows = [{key: row.get(key, "") for key in allowed} for row in source_rows]
    if args.limit > 0:
        rows = rows[: args.limit]
    if len({case_key(row) for row in rows}) != len(rows):
        raise ValueError("case keys are not unique")

    features: dict[str, dict[str, torch.Tensor]] = {}
    failures: dict[str, str] = {}
    if args.output.exists():
        previous = torch.load(args.output, map_location="cpu", weights_only=False)
        features = previous.get("features", {})
        failures = previous.get("failures", {})
    pending = [row for row in rows if case_key(row) not in features]
    print(
        f"cohort={len(rows)} existing={len(features)} pending={len(pending)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder == "biomedclip":
        model, preprocess, tokenizer, model_config = load_local_model(
            args.model_dir, device
        )
        context_length = 256
        weight_path = args.model_dir / "open_clip_pytorch_model.bin"
        config_path = args.model_dir / "open_clip_config.json"
        schema = "biomedclip_paired_colposcopy_concepts_v1"
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained=str(args.openai_weight.resolve()),
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        model_config = {
            "model_name": "ViT-B-16",
            "pretraining": "OpenAI CLIP",
        }
        context_length = 77
        weight_path = args.openai_weight
        config_path = None
        schema = "openai_clip_paired_colposcopy_concepts_v1"
    medical_text = encode_prompt_pairs(
        model, tokenizer, MEDICAL_PAIRS, device, context_length
    )
    generic_text = encode_prompt_pairs(
        model, tokenizer, GENERIC_PAIRS, device, context_length
    )
    logit_scale = model.logit_scale.exp().detach()
    print(
        f"device={device} logit_scale={float(logit_scale):.6f}",
        flush=True,
    )

    def prepare(row):
        paths = select_views(row["colposcopy_paths"])
        if not paths:
            return row, None
        return row, preprocess(make_montage(paths))

    processed_batches = 0
    with ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
        for start in range(0, len(pending), args.batch_size):
            batch_rows = pending[start : start + args.batch_size]
            images = []
            valid = []
            for row, image in executor.map(prepare, batch_rows):
                if image is None:
                    failures[case_key(row)] = "no_existing_colposcopy_image"
                    continue
                images.append(image)
                valid.append(row)
            if valid:
                image_tensor = torch.stack(images).to(device)
                with torch.inference_mode():
                    image_features = F.normalize(
                        model.encode_image(image_tensor), dim=-1
                    )
                    medical = paired_probabilities(
                        image_features, medical_text, logit_scale
                    )
                    generic = paired_probabilities(
                        image_features, generic_text, logit_scale
                    )
                for index, row in enumerate(valid):
                    key = case_key(row)
                    features[key] = {
                        "medical": medical[index].cpu().to(torch.float32),
                        "generic": generic[index].cpu().to(torch.float32),
                        "image_embedding": image_features[index]
                        .cpu()
                        .to(torch.float16),
                    }
                    failures.pop(key, None)
            processed_batches += 1
            if (
                processed_batches % args.checkpoint_every != 0
                and start + args.batch_size < len(pending)
            ):
                continue
            payload = {
                "schema": schema,
                "created_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "input": str(args.input.resolve()),
                "input_sha256": sha256_file(args.input),
                "encoder": args.encoder,
                "model_dir": str(weight_path.parent.resolve()),
                "model_weight_sha256": sha256_file(weight_path),
                "model_config_sha256": (
                    sha256_file(config_path) if config_path else None
                ),
                "model_config": model_config,
                "medical_keys": MEDICAL_KEYS,
                "generic_keys": GENERIC_KEYS,
                "medical_pairs": MEDICAL_PAIRS,
                "generic_pairs": GENERIC_PAIRS,
                "medical_prompt_sha256": prompt_hash(MEDICAL_PAIRS),
                "generic_prompt_sha256": prompt_hash(GENERIC_PAIRS),
                "label_fields_excluded": [
                    "center_name",
                    "pathology_*",
                    "CIN2+",
                    "CIN3+",
                    "outcome",
                ],
                "logit_scale": float(logit_scale),
                "features": features,
                "failures": failures,
            }
            save_payload(args.output, payload)
            print(
                f"saved={len(features)} failures={len(failures)} "
                f"output={args.output}",
                flush=True,
            )
    if len(features) != len(rows):
        raise RuntimeError(
            f"incomplete cache: {len(features)}/{len(rows)}; "
            f"failures={len(failures)}"
        )


if __name__ == "__main__":
    main()
