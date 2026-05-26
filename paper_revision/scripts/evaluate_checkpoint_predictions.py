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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from config import BioCOT_v3_2_Config
from data.cached_patch_dataset import CachedPatchFeatureDataset, case_key
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from paper_revision.scripts.clinical_variable_mapping import assert_no_report_columns, clinical_features_from_row
from training.extract_vit_patches import extract_patch_features_with_vit


CENTER_NAMES = {
    "M22105": "Enshi",
    "M20105": "Wuda",
    "M20203": "Wuda",
    "M22102": "Xiangyang",
    "M0008": "Jingzhou",
    "M22101": "Shiyan",
    "M22104": "Shiyan",
}
_MISSING_INTERPRETABILITY_WARNED: set[str] = set()


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


def validate_evaluation_scope(config: BioCOT_v3_2_Config) -> None:
    """Check locked no-report all-center split before evaluation."""
    root = Path(config.data_root)
    split_paths = [
        root / "train_labels.csv",
        root / "val_labels.csv",
        root / "external_test_labels.csv",
    ]
    if not all(path.exists() for path in split_paths):
        return
    dfs = []
    for path in split_paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        assert_no_report_columns(df.columns)
        if {"col_count", "oct_count"}.issubset(df.columns):
            if getattr(config, "pretrain_without_colpo", False):
                missing = df[df["oct_count"].astype(float) <= 0]
                if len(missing):
                    raise ValueError(f"{path} contains {len(missing)} rows without OCT images.")
            else:
                missing = df[(df["col_count"].astype(float) <= 0) | (df["oct_count"].astype(float) <= 0)]
                if len(missing):
                    raise ValueError(f"{path} contains {len(missing)} rows without both colposcopy and OCT images.")
        dfs.append(df)
    expected_n = getattr(config, "expected_aligned_n", None)
    if expected_n is not None and "all_center_patient_holdout_70_10_20" in str(root):
        observed = sum(len(df) for df in dfs)
        if observed != int(expected_n):
            raise ValueError(f"All-center split N mismatch: train+val+external={observed}, expected {expected_n}.")
    if getattr(config, "clinical_feature_dim", 14) != 14:
        raise ValueError("Current no-report HPV/TCT/Age mapping expects clinical_feature_dim=14.")
    if getattr(config, "use_cached_patch_features", False):
        cache_path = Path(getattr(config, "feature_cache_path"))
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            features = payload.get("features", payload) if isinstance(payload, dict) else payload
            needed = [case_key(row) for df in dfs for _, row in df.iterrows()]
            missing_keys = [key for key in needed if key not in features]
            if missing_keys:
                preview = ", ".join(missing_keys[:3])
                raise KeyError(f"{len(missing_keys)} split cases missing from feature cache. Examples: {preview}")


def _check_clinical_feature_dim(features: torch.Tensor, expected_dim: int | None) -> torch.Tensor:
    if expected_dim is not None and features.shape[-1] != int(expected_dim):
        raise ValueError(f"clinical_features dim mismatch: got {features.shape[-1]}, expected {expected_dim}")
    return features


def _batch_value(values, index: int, default):
    if isinstance(values, torch.Tensor):
        return values[index].item()
    if isinstance(values, (list, tuple)):
        return values[index]
    return default if values is None else values


def clinical_features_from_batch(
    batch: Dict[str, object],
    batch_size: int,
    device: torch.device,
    expected_dim: int | None = None,
) -> Optional[torch.Tensor]:
    """Build evaluation clinical features from HPV, age, and TCT only."""
    if "clinical_features" in batch and batch["clinical_features"] is not None:
        features = batch["clinical_features"].to(device, non_blocking=True).float()
        return _check_clinical_feature_dim(features, expected_dim)
    if "clinical_data" not in batch:
        return None
    clinical_data = batch["clinical_data"]
    if not isinstance(clinical_data, dict):
        return None
    hpv_values = clinical_data.get("hpv")
    age_values = clinical_data.get("age")
    tct_values = clinical_data.get("tct")
    rows = [
        {
            "hpv": _batch_value(hpv_values, i, None),
            "age": _batch_value(age_values, i, None),
            "tct": _batch_value(tct_values, i, None),
        }
        for i in range(batch_size)
    ]
    features = torch.tensor([clinical_features_from_row(row) for row in rows], dtype=torch.float32, device=device)
    return _check_clinical_feature_dim(features, expected_dim)


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


def _warn_missing_once(field: str) -> None:
    if field not in _MISSING_INTERPRETABILITY_WARNED:
        print(f"WARNING: interpretability field missing: {field}; exporting empty string.")
        _MISSING_INTERPRETABILITY_WARNED.add(field)


def _scalar_from_tensor(value, sample_index: int, reducer: str = "item"):
    if value is None:
        return ""
    try:
        tensor = value.detach().cpu()
        if tensor.dim() == 0:
            return float(tensor.item())
        sample = tensor[sample_index]
        if reducer == "norm":
            return float(sample.float().norm().item())
        if reducer == "argmax":
            return int(sample.argmax().item())
        if reducer == "max":
            return float(sample.max().item())
        if sample.numel() == 1:
            return float(sample.view(-1)[0].item())
        return float(sample.float().mean().item())
    except Exception:
        return ""


def extract_interpretability_fields(outputs: Dict[str, object], sample_index: int) -> Dict[str, object]:
    """Extract scalar interpretability fields without exporting embeddings."""
    fields: Dict[str, object] = {
        "reliability_oct": "",
        "reliability_colposcopy": "",
        "reliability_clinical_text": "",
        "precision_oct": "",
        "precision_colposcopy": "",
        "precision_clinical_text": "",
        "guideline_assignment": "",
        "guideline_assignment_prob": "",
        "guideline_entropy": "",
        "coe_z0_norm": "",
        "coe_z1_norm": "",
        "coe_z2_norm": "",
        "coe_z3_norm": "",
        "coe_delta_clinical": "",
        "coe_delta_colposcopy": "",
        "coe_delta_oct": "",
        "z_causal_norm": "",
        "z_noise_norm": "",
    }
    reliability = outputs.get("reliability", {})
    weights = outputs.get("fusion_weights", {})
    precision = reliability.get("precision", {}) if isinstance(reliability, dict) else {}
    weights = reliability.get("weights", weights) if isinstance(reliability, dict) else weights
    if isinstance(weights, dict):
        fields["reliability_oct"] = _scalar_from_tensor(weights.get("oct"), sample_index)
        fields["reliability_colposcopy"] = _scalar_from_tensor(weights.get("colpo"), sample_index)
        fields["reliability_clinical_text"] = _scalar_from_tensor(weights.get("clinical_prior"), sample_index)
    if isinstance(precision, dict):
        fields["precision_oct"] = _scalar_from_tensor(precision.get("oct"), sample_index)
        fields["precision_colposcopy"] = _scalar_from_tensor(precision.get("colpo"), sample_index)
        fields["precision_clinical_text"] = _scalar_from_tensor(precision.get("clinical_prior"), sample_index)

    asccp = outputs.get("asccp_prior", {})
    if isinstance(asccp, dict):
        assignment = asccp.get("assignment")
        fields["guideline_assignment"] = _scalar_from_tensor(assignment, sample_index, reducer="argmax")
        fields["guideline_assignment_prob"] = _scalar_from_tensor(assignment, sample_index, reducer="max")
        fields["guideline_entropy"] = _scalar_from_tensor(asccp.get("entropy"), sample_index)

    trajectory = outputs.get("posterior_trajectory", {})
    if isinstance(trajectory, dict):
        for key in ["z0", "z1", "z2", "z3"]:
            fields[f"coe_{key}_norm"] = _scalar_from_tensor(trajectory.get(key), sample_index, reducer="norm")
        try:
            z0 = trajectory.get("z0").detach().cpu()[sample_index]
            z1 = trajectory.get("z1").detach().cpu()[sample_index]
            z2 = trajectory.get("z2").detach().cpu()[sample_index]
            z3 = trajectory.get("z3").detach().cpu()[sample_index]
            fields["coe_delta_clinical"] = float((z1 - z0).float().norm().item())
            fields["coe_delta_colposcopy"] = float((z2 - z1).float().norm().item())
            fields["coe_delta_oct"] = float((z3 - z2).float().norm().item())
        except Exception:
            pass

    fields["z_causal_norm"] = _scalar_from_tensor(outputs.get("z_causal"), sample_index, reducer="norm")
    fields["z_noise_norm"] = _scalar_from_tensor(outputs.get("z_noise"), sample_index, reducer="norm")
    for field, value in fields.items():
        if value == "":
            _warn_missing_once(field)
    return fields


def choose_missing_modalities(setting: str, batch_size: int, seed: int) -> List[Sequence[str]]:
    rng = np.random.default_rng(seed)
    modalities = ["oct", "colposcopy", "clinical"]
    if setting == "none":
        return [[] for _ in range(batch_size)]
    if setting == "remove_oct":
        return [["oct"] for _ in range(batch_size)]
    if setting == "remove_colposcopy":
        return [["colposcopy"] for _ in range(batch_size)]
    if setting in {"remove_clinical_prior", "remove_clinical_text"}:
        return [["clinical"] for _ in range(batch_size)]
    if setting == "random_one":
        return [[str(rng.choice(modalities))] for _ in range(batch_size)]
    if setting == "random_two":
        return [list(rng.choice(modalities, size=2, replace=False)) for _ in range(batch_size)]
    raise ValueError(f"Unknown missing-modality setting: {setting}")


def torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def token_mask(features: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    """Mask contiguous patient-level patch tokens for feature-cache corruption tests."""
    if features.ndim < 3 or ratio <= 0:
        return features
    out = features.clone()
    batch_size, token_count = out.shape[:2]
    width = max(1, min(token_count, int(round(token_count * ratio))))
    generator = torch_generator(out.device, seed)
    for row in range(batch_size):
        high = max(1, token_count - width + 1)
        start = int(torch.randint(high, (1,), generator=generator, device=out.device).item())
        out[row, start : start + width] = 0.0
    return out


def stripe_tokens(features: torch.Tensor, severity: float) -> torch.Tensor:
    if features.ndim < 3:
        return features
    out = features.clone()
    token_count = out.shape[1]
    grid = int(round(token_count ** 0.5))
    mask = torch.ones(token_count, device=out.device, dtype=out.dtype)
    if grid * grid == token_count:
        stripe_width = max(1, int(round(severity)))
        stride = max(3, int(round(7 - severity)))
        grid_mask = mask.view(grid, grid)
        grid_mask[:, ::stride] = 0.0
        if stripe_width > 1:
            for offset in range(1, stripe_width):
                grid_mask[:, offset::stride] = 0.0
    else:
        stride = max(3, int(round(8 - severity)))
        mask[::stride] = 0.0
    return out * mask.view(1, token_count, 1)


def smooth_patch_features(features: torch.Tensor, severity: float) -> torch.Tensor:
    if features.ndim < 3:
        return features
    blend = min(0.75, 0.18 * severity)
    patient_mean = features.mean(dim=1, keepdim=True)
    return (1.0 - blend) * features + blend * patient_mean


def scale_shift_features(features: torch.Tensor, scale: float, shift_strength: float) -> torch.Tensor:
    std = features.std(dim=tuple(range(1, features.ndim)), keepdim=True).clamp_min(1e-6)
    return features * scale + shift_strength * std


def apply_input_corruption(
    oct_feats: torch.Tensor,
    col_feats: torch.Tensor | None,
    corruption: str,
    severity: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Apply deterministic feature-space analogues of image corruption tests.

    Formal runs use cached patient-level ViT patch features for speed and
    auditability. These transformations mimic the expected downstream effects
    of modality-specific image corruptions on the extracted patch-token space.
    """
    if corruption == "none" or col_feats is None:
        return oct_feats, col_feats
    severity = max(0.1, float(severity))
    generator = torch_generator(oct_feats.device, seed)

    if corruption == "colpo_blur":
        return oct_feats, smooth_patch_features(col_feats, severity)
    if corruption == "colpo_brightness":
        return oct_feats, scale_shift_features(col_feats, scale=1.0 + 0.12 * severity, shift_strength=0.08 * severity)
    if corruption == "colpo_occlusion":
        return oct_feats, token_mask(col_feats, ratio=min(0.45, 0.10 * severity), seed=seed)
    if corruption == "oct_speckle":
        noise = torch.randn(oct_feats.shape, generator=generator, device=oct_feats.device, dtype=oct_feats.dtype)
        return oct_feats * (1.0 + noise * (0.05 * severity)), col_feats
    if corruption == "oct_stripe":
        return stripe_tokens(oct_feats, severity), col_feats
    if corruption == "oct_intensity":
        return scale_shift_features(oct_feats, scale=1.0 - 0.08 * severity, shift_strength=-0.06 * severity), col_feats
    raise ValueError(f"Unknown input corruption setting: {corruption}")


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
        choices=["none", "remove_oct", "remove_colposcopy", "remove_clinical_text", "remove_clinical_prior", "random_one", "random_two"],
    )
    parser.add_argument(
        "--input-corruption",
        default="none",
        choices=[
            "none",
            "colpo_blur",
            "colpo_brightness",
            "colpo_occlusion",
            "oct_speckle",
            "oct_stripe",
            "oct_intensity",
        ],
    )
    parser.add_argument("--corruption-severity", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(EXP_ROOT / "paper_revision" / "results" / "predictions"))
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root is not None:
        config.data_root = args.data_root
    validate_evaluation_scope(config)
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
        dataset = CachedPatchFeatureDataset(
            csv_path,
            getattr(config, "feature_cache_path"),
            expected_clinical_feature_dim=getattr(config, "clinical_feature_dim", None),
            expected_aligned_n=getattr(config, "expected_aligned_n", None),
            expected_case_index_csv=EXP_ROOT
            / "paper_revision"
            / "splits"
            / "full_multimodal_resplit"
            / "final_1897_case_index.csv",
        )
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
    suffix_parts = []
    if args.missing_modality != "none":
        suffix_parts.append(args.missing_modality)
    if args.input_corruption != "none":
        suffix_parts.append(args.input_corruption)
        suffix_parts.append(f"sev{str(args.corruption_severity).replace('.', 'p')}")
    suffix = "_".join(suffix_parts) if suffix_parts else "full"
    out_path = output_dir / f"{args.method}_run{args.run_id}_seed{args.seed}_{args.split}_{suffix}.csv"

    rows: List[Dict[str, object]] = []
    cursor = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            labels = batch["label"].to(device, non_blocking=True)
            centers = batch["center_idx"].to(device, non_blocking=True)
            image_names = normalize_string_list(batch["image_names"])
            # Current experiments do not use examination reports; clinical_info
            # is restricted to non-report clinical variables from the dataset.
            clinical_info = normalize_string_list(batch.get("clinical_info", ""))
            batch_size = labels.shape[0]

            pretrain_without_colpo = bool(getattr(config, "pretrain_without_colpo", False))
            pass_raw_colpo_to_model = bool(getattr(config, "pass_raw_colpo_to_model", False))

            if "oct_patch_features" in batch and "colpo_patch_features" in batch:
                oct_feats = batch["oct_patch_features"].to(device, non_blocking=True).float()
                col_feats = None if pretrain_without_colpo else batch["colpo_patch_features"].to(device, non_blocking=True).float()
            else:
                oct_images = batch["oct_images"].to(device, non_blocking=True)
                colposcopy_images = batch["colposcopy_images"].to(device, non_blocking=True)
                oct_feats = patch_features(
                    oct_images,
                    device,
                    config.vit_batch_size,
                    pretrained=getattr(config, "vit_pretrained", False),
                )
                if pretrain_without_colpo:
                    col_feats = None
                elif pass_raw_colpo_to_model:
                    col_feats = colposcopy_images.float()
                else:
                    col_feats = patch_features(
                        colposcopy_images,
                        device,
                        config.vit_batch_size,
                        pretrained=getattr(config, "vit_pretrained", False),
                    )
            clinical_features = clinical_features_from_batch(
                batch,
                batch_size,
                device,
                expected_dim=getattr(config, "clinical_feature_dim", None),
            )

            oct_feats, col_feats = apply_input_corruption(
                oct_feats,
                col_feats,
                args.input_corruption,
                args.corruption_severity,
                args.seed + 1009 * (batch_idx + 1),
            )
            missing = choose_missing_modalities(args.missing_modality, batch_size, args.seed + batch_idx)
            if any("oct" in item for item in missing):
                oct_feats = oct_feats.clone()
                for i, item in enumerate(missing):
                    if "oct" in item:
                        oct_feats[i].zero_()
            if any("colposcopy" in item for item in missing):
                if col_feats is not None:
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
                labels=labels,
                return_loss_components=True,
                current_beta=0.1,
            )
            probs = torch.softmax(outputs["pred"], dim=1)[:, 1].detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            batch_df = dataset.df.iloc[cursor : cursor + batch_size]
            cursor += batch_size

            for i, (_, case) in enumerate(batch_df.iterrows()):
                case_id = case_value(case, "ID", "patient_id", f"case_{cursor - batch_size + i}")
                oct_id = case_value(case, "OCT", "oct_id", "")
                interp = extract_interpretability_fields(outputs, i)
                row = {
                    "method": args.method,
                    "run_id": args.run_id,
                    "seed": args.seed,
                    "split": "external_test" if args.split in {"external", "external_test"} else "internal_validation",
                    "case_id": case_id,
                    "center": center_from_oct(oct_id),
                    "y_true": int(labels_np[i]),
                    "y_prob": float(probs[i]),
                    "modality_setting": args.missing_modality,
                    "input_corruption": args.input_corruption,
                    "corruption_severity": float(args.corruption_severity) if args.input_corruption != "none" else "",
                    **interp,
                }
                row["reliability_clinical_prior"] = row["reliability_clinical_text"]
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No predictions were produced for split={args.split}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
