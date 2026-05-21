#!/usr/bin/env python3
"""Precompute patient-level ViT patch features for fast formal reruns."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from data.cached_patch_dataset import case_key
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from training.extract_vit_patches import extract_patch_features_with_vit


def precompute_collate(batch):
    """Collate only the tensors needed for visual feature precomputation.

    The no-report clinical fields may contain normalized strings; default
    PyTorch collation can fail on mixed nested string/scalar dictionaries.
    """
    return {
        "oct_images": torch.stack([item["oct_images"] for item in batch], dim=0),
        "colposcopy_images": torch.stack([item["colposcopy_images"] for item in batch], dim=0),
        "label": torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "center_idx": torch.as_tensor([int(item["center_idx"]) for item in batch], dtype=torch.long),
    }


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("cache_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    classes = [
        cls for cls in module.__dict__.values()
        if isinstance(cls, type)
        and "Config" in cls.__name__
        and getattr(cls, "__module__", None) == module.__name__
    ]
    if not classes:
        raise ValueError(f"No config class found in {config_path}")
    return classes[0]()


def patch_features(images: torch.Tensor, device: torch.device, vit_batch_size: int, pretrained: bool) -> torch.Tensor:
    if len(images.shape) == 5:
        batch, frames = images.shape[:2]
        flat = images.view(batch * frames, *images.shape[2:])
        feats = extract_patch_features_with_vit(flat, device, batch_size=vit_batch_size, pretrained=pretrained)
        return feats.view(batch, frames, 196, 768).mean(dim=1)
    return extract_patch_features_with_vit(images, device, batch_size=vit_batch_size, pretrained=pretrained)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(args.output or getattr(cfg, "feature_cache_path"))
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split_csvs = [
        Path(cfg.data_root) / "train_labels.csv",
        Path(cfg.data_root) / "val_labels.csv",
        Path(cfg.data_root) / "external_test_labels.csv",
    ]
    expected_keys = set()
    for csv_path in split_csvs:
        split_df = FiveCentersMultimodalDatasetV3_2.read_csv(csv_path) if hasattr(FiveCentersMultimodalDatasetV3_2, "read_csv") else None
        if split_df is None:
            import pandas as pd

            try:
                split_df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                split_df = pd.read_csv(csv_path, encoding="gbk")
        expected_keys.update(case_key(row) for _, row in split_df.iterrows())
    metadata = {
        "config": str(args.config),
        "source_case_index": str(
            EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "final_1897_case_index.csv"
        ),
        "expected_aligned_n": getattr(cfg, "expected_aligned_n", None),
        "clinical_feature_dim": getattr(cfg, "clinical_feature_dim", None),
        "no_report_mode": getattr(cfg, "no_report_mode", True),
        "use_vlm_retriever": getattr(cfg, "use_vlm_retriever", False),
    }
    features: Dict[str, Dict[str, torch.Tensor]] = {}
    if output.exists():
        payload = torch.load(output, map_location="cpu")
        features.update(payload.get("features", payload))
        stale_keys = sorted(set(features) - expected_keys)
        if stale_keys:
            features = {key: value for key, value in features.items() if key in expected_keys}
            print(f"Pruned {len(stale_keys)} stale cached cases not present in the current locked split.")
        print(f"Loaded existing cache with {len(features)} cases: {output}")

    for csv_path in split_csvs:
        dataset = FiveCentersMultimodalDatasetV3_2(
            csv_path=str(csv_path),
            transform=transform,
            oct_num_frames=cfg.oct_frames,
            max_col_images=cfg.colposcopy_images,
            balance_negative_frames=False,
            data_root=str(cfg.data_root),
        )
        all_keys = [case_key(row) for _, row in dataset.df.iterrows()]
        missing_mask = [key not in features for key in all_keys]
        skipped = len(all_keys) - int(sum(missing_mask))
        dataset.df = dataset.df.loc[missing_mask].reset_index(drop=True)
        print(
            f"Precomputing {csv_path.name}: {len(dataset)} missing / {len(all_keys)} cases "
            f"({skipped} already cached)"
        )
        if len(dataset) == 0:
            continue
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=precompute_collate,
        )
        cursor = 0
        for batch in tqdm(loader, desc=csv_path.name):
            batch_size = int(batch["label"].shape[0])
            batch_df = dataset.df.iloc[cursor : cursor + batch_size]
            cursor += batch_size
            keys = [case_key(row) for _, row in batch_df.iterrows()]
            if all(key in features for key in keys):
                continue
            oct_images = batch["oct_images"].to(device, non_blocking=True)
            colpo_images = batch["colposcopy_images"].to(device, non_blocking=True)
            with torch.no_grad():
                oct_feats = patch_features(
                    oct_images,
                    device,
                    cfg.vit_batch_size,
                    pretrained=getattr(cfg, "vit_pretrained", False),
                ).detach().cpu().half()
                colpo_feats = patch_features(
                    colpo_images,
                    device,
                    cfg.vit_batch_size,
                    pretrained=getattr(cfg, "vit_pretrained", False),
                ).detach().cpu().half()
            for i, key in enumerate(keys):
                features[key] = {"oct": oct_feats[i], "colpo": colpo_feats[i]}
            if len(features) % 100 < batch_size:
                torch.save({"features": features, "metadata": metadata, "config": str(args.config)}, output)
                print(f"  cached {len(features)} cases -> {output}")

    torch.save({"features": features, "metadata": metadata, "config": str(args.config)}, output)
    print(f"Saved feature cache: {output} ({len(features)} cases)")


if __name__ == "__main__":
    main()
