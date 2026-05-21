#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / "paper_revision" / "splits" / "target_adapted_validation" / "all_center_patient_holdout_70_10_20"
OUT_ROOT = ROOT / "paper_revision" / "splits" / "label_noise"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--noise-rate", type=float, action="append", default=[0.05, 0.10, 0.20])
    parser.add_argument("--seed", type=int, action="append", default=[42, 123, 456])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.split_root / "train_labels.csv", encoding="utf-8-sig")
    val = pd.read_csv(args.split_root / "val_labels.csv", encoding="utf-8-sig")
    test = pd.read_csv(args.split_root / "external_test_labels.csv", encoding="utf-8-sig")
    for rate in args.noise_rate:
        for seed in args.seed:
            rng = np.random.default_rng(seed)
            noisy = train.copy()
            n_flip = int(round(rate * len(noisy)))
            idx = rng.choice(noisy.index.to_numpy(), size=n_flip, replace=False) if n_flip else []
            flipped = noisy.loc[idx, ["case_id", "patient_id", "oct_id", "label"]].copy()
            flipped["original_label"] = flipped["label"].astype(int)
            noisy.loc[idx, "label"] = 1 - noisy.loc[idx, "label"].astype(int)
            flipped["flipped_label"] = noisy.loc[idx, "label"].astype(int).values
            flipped["noise_rate"] = rate
            flipped["seed"] = seed
            out_dir = args.output_root / f"noise_{rate:.2f}".replace(".", "p") / f"seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            noisy.to_csv(out_dir / "train_labels.csv", index=False, encoding="utf-8-sig")
            val.to_csv(out_dir / "val_labels.csv", index=False, encoding="utf-8-sig")
            test.to_csv(out_dir / "external_test_labels.csv", index=False, encoding="utf-8-sig")
            flipped.to_csv(out_dir / "flipped_cases.csv", index=False, encoding="utf-8-sig")
            print(f"{out_dir}: flipped {len(flipped)} training labels")


if __name__ == "__main__":
    main()
