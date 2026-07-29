#!/usr/bin/env python3
"""Extract label-free, prompt-conditioned Qwen3-VL colposcopy embeddings.

The model sees only a deterministic montage and a fixed prompt. Outcome,
pathology and centre fields are never inserted into the conversation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MEDICAL_PROMPT = (
    "Encode label-free evidence from these tiled colposcopy views. Attend to "
    "transformation-zone visibility, acetowhite appearance, vascular pattern, "
    "surface contour, lesion margins, and image quality. Represent only directly "
    "visible morphology. Do not infer diagnosis, pathology grade, biopsy result, "
    "or patient outcome."
)
GENERIC_PROMPT = (
    "Encode the visible content, composition, texture, colors, and image quality "
    "of these tiled images. Do not infer diagnosis, pathology, or patient outcome."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_key(row: dict[str, str]) -> str:
    return f"{row['patient_id']}||{row['exam_id_or_oct_id']}"


def select_views(value: str, maximum: int = 4) -> list[Path]:
    candidates = [Path(item.strip()) for item in value.split(";") if item.strip()]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return []
    if len(existing) <= maximum:
        return existing
    indices = [round(index * (len(existing) - 1) / (maximum - 1)) for index in range(maximum)]
    return [existing[index] for index in indices]


def make_montage(paths: list[Path], tile: int = 224) -> Image.Image:
    canvas = Image.new("RGB", (tile * 2, tile * 2), color=(0, 0, 0))
    for index in range(4):
        path = paths[min(index, len(paths) - 1)]
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (tile, tile))
        canvas.paste(image, ((index % 2) * tile, (index // 2) * tile))
    return canvas


def save_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/publishable_v2/data_lock/data_lock_n1897.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/data2/like/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/active_fusion_v2_20260729/shared/qwen3vl_colposcopy_embeddings.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed = {
        "case_id", "patient_id", "exam_id_or_oct_id",
        "colposcopy_paths", "colposcopy_available",
    }
    projected = [{key: row.get(key, "") for key in allowed} for row in rows]
    if len({case_key(row) for row in projected}) != len(projected):
        raise ValueError("case keys are not unique")

    existing: dict[str, dict[str, torch.Tensor]] = {}
    if args.output.exists():
        previous = torch.load(args.output, map_location="cpu", weights_only=False)
        existing = previous.get("features", {})
    pending = [row for row in projected if case_key(row) not in existing]
    print(f"cohort={len(projected)} existing={len(existing)} pending={len(pending)}", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=64 * 28 * 28,
        max_pixels=128 * 28 * 28,
    )
    failures: dict[str, str] = {}
    processed_batches = 0
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        conversations = []
        valid: list[tuple[dict[str, str], Image.Image]] = []
        for row in batch_rows:
            paths = select_views(row["colposcopy_paths"])
            if not paths:
                failures[case_key(row)] = "no_existing_colposcopy_image"
                continue
            montage = make_montage(paths)
            valid.append((row, montage))
            for prompt in (MEDICAL_PROMPT, GENERIC_PROMPT):
                conversations.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": montage},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                )
        if valid:
            inputs = processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            with torch.inference_mode():
                output = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            positions = torch.arange(
                inputs.attention_mask.shape[1],
                device=inputs.attention_mask.device,
            ).unsqueeze(0)
            last_positions = (positions * inputs.attention_mask).max(dim=1).values
            embedding = output.hidden_states[-1][
                torch.arange(len(conversations), device=last_positions.device),
                last_positions,
            ]
            embedding = F.normalize(embedding.float(), dim=-1).half().cpu()
            for index, (row, _) in enumerate(valid):
                existing[case_key(row)] = {
                    "medical": embedding[index * 2],
                    "generic": embedding[index * 2 + 1],
                }
        processed_batches += 1
        if processed_batches % args.checkpoint_every == 0 or start + args.batch_size >= len(pending):
            payload = {
                "schema": "qwen3vl_colposcopy_prompt_embeddings_v1",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model_path": str(args.model),
                "model_family": "Qwen3-VL-8B-Instruct",
                "input_sha256": sha256_file(args.input),
                "label_free": True,
                "outcome_fields_passed_to_model": False,
                "modalities": ["colposcopy"],
                "oct_vlm_status": "not_extracted_raw_OCT_paths_unavailable",
                "medical_prompt": MEDICAL_PROMPT,
                "medical_prompt_sha256": sha256_text(MEDICAL_PROMPT),
                "generic_prompt": GENERIC_PROMPT,
                "generic_prompt_sha256": sha256_text(GENERIC_PROMPT),
                "feature_dim": 4096,
                "normalization": "L2",
                "features": existing,
                "failures": failures,
            }
            save_payload(args.output, payload)
            print(
                f"saved={len(existing)} failures={len(failures)} "
                f"progress={min(start + args.batch_size, len(pending))}/{len(pending)}",
                flush=True,
            )
    if len(existing) + len(failures) != len(projected):
        raise RuntimeError("extraction accounting mismatch")


if __name__ == "__main__":
    main()
