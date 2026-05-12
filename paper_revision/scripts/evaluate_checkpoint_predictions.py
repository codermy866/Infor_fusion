#!/usr/bin/env python3
"""Evaluate a saved checkpoint and export auditable prediction CSVs.

This script intentionally writes only raw prediction rows. Metrics can be
aggregated later by the evaluation utilities in this directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from config import BioCOT_v3_2_Config
from data.cached_patch_dataset import CachedPatchFeatureDataset
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from training.extract_vit_patches import extract_patch_features_with_vit


CENTER_NAMES = {
    "M22105": "Enshi",
    "M20105": "Wuda",
    "M20203": "Wuda",
    "M22102": "Xiangyang",
    "M0008": "Jingzhou",
    "M22101": "Jingzhou",
    "M22104": "Shiyan",
}


def center_from_oct(oct_id: str) -> str:
    code = str(oct_id).split("_")[0]
    return CENTER_NAMES.get(code, code)


def load_config(config_path: Optional[str]) -> BioCOT_v3_2_Config:
    if not config_path:
        return BioCOT_v3_2_Config()
    spec = importlib.util.spec_from_file_location("eval_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config_classes = [
        cls
        for cls in module.__dict__.values()
        if isinstance(cls, type)
        and "Config" in cls.__name__
        and getattr(cls, "__module__", None) == module.__name__
    ]
    if not config_classes:
        raise ValueError(f"No *Config class found in {config_path}")
    return config_classes[0]()


def split_csv(config: BioCOT_v3_2_Config, split: str) -> Path:
    root = Path(config.data_root)
    mapping = {
        "train": root / "train_labels.csv",
        "internal_validation": root / "val_labels.csv",
        "val": root / "val_labels.csv",
        "external_test": root / "external_test_labels.csv",
        "external": root / "external_test_labels.csv",
    }
    if split not in mapping:
        raise ValueError(f"Unknown split: {split}")
    return mapping[split]


def clinical_features_from_batch(batch: Dict[str, object], batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if "clinical_features" in batch and batch["clinical_features"] is not None:
        return batch["clinical_features"].to(device, non_blocking=True)
    if "clinical_data" not in batch:
        return None
    clinical_data = batch["clinical_data"]
    if not isinstance(clinical_data, dict):
        return None
    hpv = clinical_data.get("hpv", torch.zeros(batch_size, device=device))
    age = clinical_data.get("age", torch.zeros(batch_size, device=device))
    tct = clinical_data.get("tct", torch.zeros(batch_size, device=device))
    if not isinstance(hpv, torch.Tensor):
        hpv = torch.as_tensor(hpv, device=device)
    else:
        hpv = hpv.to(device)
    if not isinstance(age, torch.Tensor):
        age = torch.as_tensor(age, device=device)
    else:
        age = age.to(device)
    if not isinstance(tct, torch.Tensor):
        tct = torch.zeros(batch_size, device=device)
    else:
        tct = tct.to(device)
    tct = torch.clamp(tct.long(), min=0, max=4)
    tct_onehot = torch.nn.functional.one_hot(tct, num_classes=5).float()
    return torch.cat([hpv.float().unsqueeze(1), age.float().unsqueeze(1), tct_onehot], dim=1)


def patch_features(
    images: torch.Tensor,
    device: torch.device,
    vit_batch_size: int,
    pretrained: bool = False,
) -> torch.Tensor:
    if len(images.shape) == 5:
        batch, frames = images.shape[:2]
        flat = images.view(batch * frames, *images.shape[2:])
        feats = extract_patch_features_with_vit(flat, device, batch_size=vit_batch_size, pretrained=pretrained)
        return feats.view(batch, frames, 196, 768).mean(dim=1)
    return extract_patch_features_with_vit(images, device, batch_size=vit_batch_size, pretrained=pretrained)


def normalize_string_list(values: object) -> List[str]:
    if isinstance(values, list):
        return [str(v) for v in values]
    if isinstance(values, tuple):
        return [str(v) for v in values]
    return [str(values)]


def choose_missing_modalities(setting: str, batch_size: int, seed: int) -> List[Sequence[str]]:
    rng = np.random.default_rng(seed)
    modalities = ["oct", "colposcopy", "clinical"]
    if setting == "none":
        return [[] for _ in range(batch_size)]
    if setting == "remove_oct":
        return [["oct"] for _ in range(batch_size)]
    if setting == "remove_colposcopy":
        return [["colposcopy"] for _ in range(batch_size)]
    if setting == "remove_clinical_prior":
        return [["clinical"] for _ in range(batch_size)]
    if setting == "random_one":
        return [[str(rng.choice(modalities))] for _ in range(batch_size)]
    if setting == "random_two":
        return [list(rng.choice(modalities, size=2, replace=False)) for _ in range(batch_size)]
    raise ValueError(f"Unknown missing-modality setting: {setting}")


def case_value(case, preferred: str, fallback: str, default: str = "") -> str:
    if preferred in case:
        value = case[preferred]
        if value is not None and not (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return str(value)
    if fallback in case:
        return str(case[fallback])
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="external_test", choices=["train", "val", "internal_validation", "external", "external_test"])
    parser.add_argument("--csv", default=None, help="Optional explicit CSV path for custom folds.")
    parser.add_argument("--data-root", default=None, help="Optional image data root override.")
    parser.add_argument("--method", default="HyDRA_Full")
    parser.add_argument("--run_id", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-modality",
        default="none",
        choices=["none", "remove_oct", "remove_colposcopy", "remove_clinical_prior", "random_one", "random_two"],
    )
    parser.add_argument("--output-dir", default=str(EXP_ROOT / "paper_revision" / "results" / "predictions"))
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config.data_root = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    csv_path = Path(args.csv) if args.csv else split_csv(config, args.split)
    if getattr(config, "use_cached_patch_features", False):
        dataset = CachedPatchFeatureDataset(csv_path, getattr(config, "feature_cache_path"))
    else:
        dataset = FiveCentersMultimodalDatasetV3_2(
            csv_path=str(csv_path),
            transform=transform,
            oct_num_frames=config.oct_frames,
            max_col_images=config.colposcopy_images,
            balance_negative_frames=False,
            data_root=str(config.data_root),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model = create_bio_cot_v3_2(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.missing_modality if args.missing_modality != "none" else "full"
    out_path = output_dir / f"{args.method}_run{args.run_id}_seed{args.seed}_{args.split}_{suffix}.csv"

    rows: List[Dict[str, object]] = []
    cursor = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            labels = batch["label"].to(device, non_blocking=True)
            centers = batch["center_idx"].to(device, non_blocking=True)
            image_names = normalize_string_list(batch["image_names"])
            clinical_info = normalize_string_list(batch.get("clinical_info", ""))
            batch_size = labels.shape[0]

            if "oct_patch_features" in batch and "colpo_patch_features" in batch:
                oct_feats = batch["oct_patch_features"].to(device, non_blocking=True).float()
                col_feats = batch["colpo_patch_features"].to(device, non_blocking=True).float()
            else:
                oct_images = batch["oct_images"].to(device, non_blocking=True)
                colposcopy_images = batch["colposcopy_images"].to(device, non_blocking=True)
                oct_feats = patch_features(
                    oct_images,
                    device,
                    config.vit_batch_size,
                    pretrained=getattr(config, "vit_pretrained", False),
                )
                col_feats = patch_features(
                    colposcopy_images,
                    device,
                    config.vit_batch_size,
                    pretrained=getattr(config, "vit_pretrained", False),
                )
            clinical_features = clinical_features_from_batch(batch, batch_size, device)

            missing = choose_missing_modalities(args.missing_modality, batch_size, args.seed + batch_idx)
            if any("oct" in item for item in missing):
                oct_feats = oct_feats.clone()
                for i, item in enumerate(missing):
                    if "oct" in item:
                        oct_feats[i].zero_()
            if any("colposcopy" in item for item in missing):
                col_feats = col_feats.clone()
                for i, item in enumerate(missing):
                    if "colposcopy" in item:
                        col_feats[i].zero_()
            if any("clinical" in item for item in missing):
                clinical_info = ["" if "clinical" in item else clinical_info[i] for i, item in enumerate(missing)]
                if clinical_features is not None:
                    clinical_features = clinical_features.clone()
                    for i, item in enumerate(missing):
                        if "clinical" in item:
                            clinical_features[i].zero_()

            outputs = model(
                f_oct=oct_feats,
                f_colpo=col_feats,
                image_names=image_names,
                clinical_info=clinical_info,
                center_labels=centers,
                clinical_features=clinical_features,
                return_loss_components=True,
                current_beta=0.1,
            )
            probs = torch.softmax(outputs["pred"], dim=1)[:, 1].detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            reliability_weights = outputs.get("fusion_weights", {})
            reliability_np = {}
            for key in ["oct", "colpo", "clinical_prior"]:
                value = reliability_weights.get(key) if isinstance(reliability_weights, dict) else None
                if value is not None:
                    reliability_np[key] = value.detach().cpu().view(-1).numpy()
            batch_df = dataset.df.iloc[cursor : cursor + batch_size]
            cursor += batch_size

            for i, (_, case) in enumerate(batch_df.iterrows()):
                case_id = case_value(case, "ID", "patient_id", f"case_{cursor - batch_size + i}")
                oct_id = case_value(case, "OCT", "oct_id", "")
                rows.append(
                    {
                        "method": args.method,
                        "run_id": args.run_id,
                        "seed": args.seed,
                        "split": "external_test" if args.split in {"external", "external_test"} else "internal_validation",
                        "case_id": case_id,
                        "center": center_from_oct(oct_id),
                        "y_true": int(labels_np[i]),
                        "y_prob": float(probs[i]),
                        "modality_setting": args.missing_modality,
                        "reliability_oct": float(reliability_np["oct"][i]) if "oct" in reliability_np else "",
                        "reliability_colposcopy": float(reliability_np["colpo"][i]) if "colpo" in reliability_np else "",
                        "reliability_clinical_prior": float(reliability_np["clinical_prior"][i]) if "clinical_prior" in reliability_np else "",
                    }
                )

    if not rows:
        raise RuntimeError(f"No predictions were produced for split={args.split}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
