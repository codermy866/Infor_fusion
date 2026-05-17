#!/usr/bin/env python3
"""Stage-1 feature-space contrastive pretraining for cervix adapters.

This is a conservative first implementation: it uses cached patient-level
features from the training split, so no external-test labels are touched. The
saved adapter checkpoint can initialize Stage-2 HyDRA runs.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from data.cached_patch_dataset import case_key, center_from_row, parse_clinical_features
from models.bio_cot_v3_2 import ResidualFeatureAdapter


DEFAULT_SPLIT_DIR = EXP_ROOT / "paper_revision" / "splits" / "target_adapted_validation" / "all_center_patient_holdout_70_10_20"
DEFAULT_CACHE = EXP_ROOT / "paper_revision" / "cache" / "patch_features_all_center_patient_holdout.pt"
DEFAULT_OUT = EXP_ROOT / "paper_revision" / "results" / "stage1_contrastive_pretrain"


class CachedContrastiveDataset(Dataset):
    def __init__(self, csv_paths: Iterable[Path], feature_cache_path: Path):
        frames = []
        for path in csv_paths:
            df = pd.read_csv(path)
            df["source_csv"] = str(path)
            frames.append(df)
        self.df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["patient_id", "oct_id"], keep="first")
        self.df["center_idx"] = self.df.apply(center_from_row, axis=1)

        payload = torch.load(feature_cache_path, map_location="cpu")
        self.features = payload.get("features", payload)
        self.df["case_key"] = self.df.apply(case_key, axis=1)
        self.df = self.df[self.df["case_key"].isin(self.features)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError("No cases matched the feature cache")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        feats = self.features[row["case_key"]]
        return {
            "oct": feats["oct"].float(),
            "colpo": feats["colpo"].float(),
            "clinical": parse_clinical_features(row),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "center": torch.tensor(int(row["center_idx"]), dtype=torch.long),
        }


class ClinicalProjector(nn.Module):
    def __init__(self, dim: int = 768, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.net(clinical)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = torch.matmul(a, b.t()) / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def center_invariant_nce(features: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(features, dim=-1)
    logits = torch.matmul(z, z.t()) / temperature
    logits = logits - torch.eye(z.shape[0], device=z.device) * 1e9
    positive = labels[:, None].eq(labels[None, :]) & centers[:, None].ne(centers[None, :])
    positive.fill_diagonal_(False)
    valid = positive.any(dim=1)
    if not valid.any():
        return features.new_tensor(0.0)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1).clamp_min(1))
    return per_anchor[valid].mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csvs", default=str(DEFAULT_SPLIT_DIR / "train_labels.csv"))
    parser.add_argument("--feature-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-img-clinical", type=float, default=0.5)
    parser.add_argument("--lambda-center", type=float, default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    csv_paths = [Path(item.strip()) for item in args.csvs.split(",") if item.strip()]
    dataset = CachedContrastiveDataset(csv_paths, Path(args.feature_cache))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    adapters = nn.ModuleDict(
        {
            "oct": ResidualFeatureAdapter(dim=768, bottleneck_dim=192, dropout=0.10),
            "colpo": ResidualFeatureAdapter(dim=768, bottleneck_dim=192, dropout=0.10),
        }
    ).to(device)
    clinical_projector = ClinicalProjector(dim=768, dropout=0.15).to(device)
    params = list(adapters.parameters()) + list(clinical_projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history_path = output_dir / f"stage1_contrastive_history_seed{args.seed}.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "loss_col_oct", "loss_img_cli", "loss_center"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            adapters.train()
            clinical_projector.train()
            totals = {"loss": 0.0, "loss_col_oct": 0.0, "loss_img_cli": 0.0, "loss_center": 0.0}
            n = 0
            for batch in loader:
                oct_feat = batch["oct"].to(device).float()
                colpo_feat = batch["colpo"].to(device).float()
                clinical = batch["clinical"].to(device).float()
                labels = batch["label"].to(device)
                centers = batch["center"].to(device)

                oct_vec = adapters["oct"](oct_feat).mean(dim=1)
                col_vec = adapters["colpo"](colpo_feat).mean(dim=1)
                img_vec = 0.5 * (oct_vec + col_vec)
                cli_vec = clinical_projector(clinical)

                loss_col_oct = info_nce(col_vec, oct_vec, args.temperature)
                loss_img_cli = info_nce(img_vec, cli_vec, args.temperature)
                loss_center = center_invariant_nce(img_vec, labels, centers, args.temperature)
                loss = loss_col_oct + args.lambda_img_clinical * loss_img_cli + args.lambda_center * loss_center

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                bs = int(labels.shape[0])
                n += bs
                totals["loss"] += float(loss.item()) * bs
                totals["loss_col_oct"] += float(loss_col_oct.item()) * bs
                totals["loss_img_cli"] += float(loss_img_cli.item()) * bs
                totals["loss_center"] += float(loss_center.item()) * bs

            row = {key: value / max(n, 1) for key, value in totals.items()}
            row["epoch"] = epoch
            writer.writerow(row)
            handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"loss={row['loss']:.4f} col_oct={row['loss_col_oct']:.4f} "
                    f"img_cli={row['loss_img_cli']:.4f} center={row['loss_center']:.4f}",
                    flush=True,
                )

    ckpt_path = output_dir / f"stage1_contrastive_seed{args.seed}.pt"
    torch.save(
        {
            "visual_domain_adapters": adapters.state_dict(),
            "clinical_feature_projector": clinical_projector.net.state_dict(),
            "seed": args.seed,
            "csvs": [str(path) for path in csv_paths],
            "n_cases": len(dataset),
            "objective": {
                "loss": "L_col_oct + lambda_img_clinical * L_img_cli + lambda_center * L_center",
                "lambda_img_clinical": args.lambda_img_clinical,
                "lambda_center": args.lambda_center,
                "temperature": args.temperature,
            },
        },
        ckpt_path,
    )
    print(f"Wrote {ckpt_path}")
    print(f"Wrote {history_path}")


if __name__ == "__main__":
    main()
