#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 clinical semantic adapter pretraining without reports.

Dry-run mode performs all no-report/cohort checks without importing torch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.clinical_variable_mapping import assert_no_report_columns, clinical_features_from_row


ADAPTER_MODULES = [
    "note_projector",
    "text_adapter",
    "clinical_feature_projector",
    "align_proj_img",
    "align_proj_text",
    "shared_align_proj",
]


def load_config(config_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("stage1_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    classes = [
        cls
        for cls in module.__dict__.values()
        if isinstance(cls, type) and "Config" in cls.__name__ and getattr(cls, "__module__", None) == module.__name__
    ]
    if not classes:
        raise ValueError(f"No *Config class found in {config_path}")
    return classes[0]()


def read_split_csvs(data_root: Path) -> pd.DataFrame:
    frames = []
    for name in ["train_labels.csv", "val_labels.csv"]:
        path = data_root / name
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        assert_no_report_columns(df.columns)
        df["source_split"] = name.replace("_labels.csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dry_run(cfg: Any, config_path: Path) -> None:
    df = read_split_csvs(Path(cfg.data_root))
    feature_dim = len(clinical_features_from_row(df.iloc[0])) if len(df) else getattr(cfg, "clinical_feature_dim", None)
    cache_path = Path(cfg.feature_cache_path)
    report = {
        "config": str(config_path),
        "dataset_n_train_plus_val": int(len(df)),
        "clinical_features_dimension": int(feature_dim),
        "expected_clinical_feature_dim": getattr(cfg, "clinical_feature_dim", None),
        "no_report_field_check": "pass",
        "uses_pathology_labels_as_target": False,
        "feature_cache_path": str(cache_path),
        "feature_cache_exists": cache_path.exists(),
        "feature_cache_mtime": datetime.fromtimestamp(cache_path.stat().st_mtime).isoformat(timespec="seconds")
        if cache_path.exists()
        else None,
        "trainable_modules": ADAPTER_MODULES,
        "frozen_modules": ["classifier", "reliability_head", "posterior_refiner", "asccp_prior", "coe_readout"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(cfg: Any, config_path: Path) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch is required for non-dry-run Stage-1 training") from exc

    from data.cached_patch_dataset import CachedPatchFeatureDataset
    from models.bio_cot_v3_2 import create_hydra_coe

    train_csv = Path(cfg.data_root) / "train_labels.csv"
    val_csv = Path(cfg.data_root) / "val_labels.csv"
    train_ds = CachedPatchFeatureDataset(
        train_csv,
        cfg.feature_cache_path,
        expected_clinical_feature_dim=getattr(cfg, "clinical_feature_dim", 14),
        expected_aligned_n=getattr(cfg, "expected_aligned_n", 1897),
    )
    val_ds = CachedPatchFeatureDataset(
        val_csv,
        cfg.feature_cache_path,
        expected_clinical_feature_dim=getattr(cfg, "clinical_feature_dim", 14),
        expected_aligned_n=getattr(cfg, "expected_aligned_n", 1897),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.stage1_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.stage1_batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_hydra_coe(cfg).to(device)
    model.train()

    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(module) for module in ADAPTER_MODULES)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.stage1_learning_rate)

    best_val = float("inf")
    best_payload = None
    for epoch in range(1, cfg.stage1_epochs + 1):
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            oct_feat = batch["oct_patch_features"].to(device).float().mean(dim=1)
            col_feat = batch["colpo_patch_features"].to(device).float().mean(dim=1)
            cli_feat = batch["clinical_features"].to(device).float()
            visual = model.align_proj_img((oct_feat + col_feat) / 2.0)
            clinical = model.clinical_feature_projector(cli_feat)
            clinical = model.align_proj_text(clinical)
            logits = F.normalize(visual, dim=-1) @ F.normalize(clinical, dim=-1).t()
            labels = torch.arange(logits.shape[0], device=device)
            align_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            recon_loss = torch.zeros((), device=device)
            loss = cfg.lambda_align * align_loss + cfg.lambda_recon * recon_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1

        val_loss = 0.0
        val_steps = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                oct_feat = batch["oct_patch_features"].to(device).float().mean(dim=1)
                col_feat = batch["colpo_patch_features"].to(device).float().mean(dim=1)
                cli_feat = batch["clinical_features"].to(device).float()
                visual = model.align_proj_img((oct_feat + col_feat) / 2.0)
                clinical = model.align_proj_text(model.clinical_feature_projector(cli_feat))
                logits = F.normalize(visual, dim=-1) @ F.normalize(clinical, dim=-1).t()
                labels = torch.arange(logits.shape[0], device=device)
                val_loss += float((0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))).cpu())
                val_steps += 1
        model.train()
        val_metric = val_loss / max(val_steps, 1)
        print(f"epoch={epoch} train_loss={total_loss / max(steps, 1):.4f} val_align_loss={val_metric:.4f}")
        if val_metric < best_val:
            best_val = val_metric
            best_payload = {
                module: getattr(model, module).state_dict()
                for module in ADAPTER_MODULES
                if getattr(model, module, None) is not None
            }
            best_payload["config"] = str(config_path)
            best_payload["validation_alignment_metrics"] = {"val_align_loss": best_val}

    out = Path(cfg.checkpoint_dir) / "stage1_clinical_semantic_adapter_best.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, out)
    print(f"Saved Stage-1 adapter checkpoint: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        dry_run(cfg, args.config)
    else:
        train(cfg, args.config)


if __name__ == "__main__":
    main()
