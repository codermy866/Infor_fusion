#!/usr/bin/env python3
"""Run training-label noise stress tests on cached patient-level features.

This complements the full HyDRA runs with a fast, auditable stress test:
only the training labels are randomly flipped; internal validation and external
test labels remain untouched. The operating threshold is selected on the clean
internal validation split and then locked for external testing.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
RESULT_DIR = PAPER_DIR / "results"
PRED_DIR = RESULT_DIR / "label_noise_predictions"
TABLE_DIR = PAPER_DIR / "tables"

sys.path.insert(0, str(SCRIPT_DIR))

from metrics_utils import aggregate_metric_table, binary_metrics, select_threshold
from run_feature_space_experiments import (
    FeatureFusionNet,
    MethodSpec,
    batch_iter,
    load_features,
    make_tensors,
)


PRIMARY_METHOD = "HyDRA_FeatureVariational"
METRIC_COLS = ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "ece", "brier"]


def flip_training_labels(labels: np.ndarray, noise_rate: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).copy()
    if noise_rate <= 0:
        return y, np.zeros_like(y, dtype=bool)
    rng = np.random.default_rng(seed)
    n_flip = int(round(float(noise_rate) * len(y)))
    n_flip = max(0, min(n_flip, len(y)))
    mask = np.zeros(len(y), dtype=bool)
    if n_flip == 0:
        return y, mask
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    mask[flip_idx] = True
    return y, mask


def predict(model: FeatureFusionNet, data: Dict[str, np.ndarray], device: torch.device, seed: int) -> Tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    probs: List[np.ndarray] = []
    weights_all: List[np.ndarray] = []
    with torch.no_grad():
        for idx in batch_iter(data, 256, False, seed):
            oct_x = torch.as_tensor(data["oct"][idx], dtype=torch.float32, device=device)
            col_x = torch.as_tensor(data["col"][idx], dtype=torch.float32, device=device)
            cli_x = torch.as_tensor(data["clinical"][idx], dtype=torch.float32, device=device)
            center = torch.as_tensor(data["center_idx"][idx], dtype=torch.long, device=device)
            logits, extra = model(oct_x, col_x, cli_x, center)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            weights = extra.get("weights")
            if weights is not None:
                weights_all.append(weights.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(weights_all) if weights_all else None


def train_one(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    device: torch.device,
    seed: int,
    epochs: int,
) -> Tuple[FeatureFusionNet, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    spec = MethodSpec(
        PRIMARY_METHOD,
        "variational",
        variational=True,
        center_aware=True,
        refinement_steps=2,
        memory=True,
        prior="both",
    )
    model = FeatureFusionNet(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train["y"], minlength=2)
    class_weight = torch.as_tensor((len(train["y"]) / np.maximum(counts, 1)) / 2.0, dtype=torch.float32, device=device)
    best_score = -np.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for idx in batch_iter(train, 64, True, seed + epoch):
            oct_x = torch.as_tensor(train["oct"][idx], dtype=torch.float32, device=device)
            col_x = torch.as_tensor(train["col"][idx], dtype=torch.float32, device=device)
            cli_x = torch.as_tensor(train["clinical"][idx], dtype=torch.float32, device=device)
            center = torch.as_tensor(train["center_idx"][idx], dtype=torch.long, device=device)
            y = torch.as_tensor(train["y"][idx], dtype=torch.long, device=device)
            logits, extra = model(oct_x, col_x, cli_x, center, y)
            loss = F.cross_entropy(logits, y, weight=class_weight) + extra["aux_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_prob, _ = predict(model, val, device, seed)
        metric = binary_metrics(val["y"], val_prob)
        score = metric.auc if not np.isnan(metric.auc) else metric.auprc
        if score > best_score:
            best_score = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_score)


def write_predictions(
    noise_rate: float,
    seed: int,
    split: str,
    data: Dict[str, np.ndarray],
    meta: Dict[str, np.ndarray],
    probs: np.ndarray,
    weights: np.ndarray | None,
    threshold: float,
) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rate_tag = f"{noise_rate:.2f}".replace(".", "p")
    out = PRED_DIR / f"label_noise_{rate_tag}_{PRIMARY_METHOD}_seed{seed}_{split}_full.csv"
    pred = (probs >= threshold).astype(int)
    fieldnames = [
        "experiment",
        "method",
        "run_id",
        "seed",
        "noise_rate",
        "split",
        "case_id",
        "center",
        "y_true",
        "y_prob",
        "y_pred",
        "threshold",
        "reliability_oct",
        "reliability_colposcopy",
        "reliability_clinical_prior",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, prob in enumerate(probs):
            row = {
                "experiment": "label_noise_stress",
                "method": PRIMARY_METHOD,
                "run_id": f"label_noise_{rate_tag}",
                "seed": seed,
                "noise_rate": float(noise_rate),
                "split": split,
                "case_id": str(meta["case_id"][i]),
                "center": str(meta["center"][i]),
                "y_true": int(data["y"][i]),
                "y_prob": float(prob),
                "y_pred": int(pred[i]),
                "threshold": float(threshold),
                "reliability_oct": "",
                "reliability_colposcopy": "",
                "reliability_clinical_prior": "",
            }
            if weights is not None:
                row["reliability_oct"] = float(weights[i, 0])
                row["reliability_colposcopy"] = float(weights[i, 1])
                row["reliability_clinical_prior"] = float(weights[i, 2])
            writer.writerow(row)
    return out


def metric_row(
    noise_rate: float,
    seed: int,
    split: str,
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float,
    best_internal_auc: float,
    n_flipped: int,
) -> Dict[str, object]:
    metric = binary_metrics(y_true, y_prob, threshold=threshold)
    return {
        "experiment": "label_noise_stress",
        "method": PRIMARY_METHOD,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "split": split,
        "flipped_train_labels": int(n_flipped),
        "best_internal_auc": float(best_internal_auc),
        **asdict(metric),
    }


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in METRIC_COLS + ["threshold", "best_internal_auc"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    return display


def to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training-label noise stress test using cached patient-level multimodal features.}",
        r"\label{tab:label_noise_stress}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Noise rate & Runs & AUC & AUPRC & Sens. & Spec. & PPV & NPV & F1 & ECE & Brier \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{float(row['noise_rate']):.2f} & {int(row['runs'])} & {row['auc']} & {row['auprc']} & "
            f"{row['sensitivity']} & {row['specificity']} & {row['ppv']} & {row['npv']} & "
            f"{row['f1']} & {row['ece']} & {row['brier']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def summarize(metrics: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_DIR / "label_noise_stress_run_level_metrics.csv", index=False)
    external = metrics[metrics["split"].eq("external_test")].copy()
    agg = aggregate_metric_table(external.rename(columns={"noise_rate": "noise_rate_group"}))

    rows = []
    for noise_rate, group in external.groupby("noise_rate"):
        row = {
            "noise_rate": float(noise_rate),
            "runs": int(group.shape[0]),
            "n": int(group["n"].median()),
            "flipped_train_labels_mean": float(group["flipped_train_labels"].mean()),
        }
        for metric in METRIC_COLS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = "NA" if pd.isna(mean) else f"{mean:.3f} +/- {std:.3f}"
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("noise_rate")
    summary.to_csv(TABLE_DIR / "label_noise_stress_metrics.csv", index=False)
    summary.to_csv(TABLE_DIR / "label_noise_stress_metrics_formatted.csv", index=False)
    (TABLE_DIR / "label_noise_stress_metrics.tex").write_text(to_latex(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw = load_features("train")
    val_raw = load_features("internal_validation")
    ext_raw = load_features("external_test")
    base_train, stats = make_tensors(train_raw)
    val, _ = make_tensors(val_raw, stats)
    external, _ = make_tensors(ext_raw, stats)
    meta = {
        "internal_validation": {"case_id": val_raw["case_id"], "center": val_raw["center"]},
        "external_test": {"case_id": ext_raw["case_id"], "center": ext_raw["center"]},
    }

    rows: List[Dict[str, object]] = []
    for noise_rate in args.noise_rates:
        for seed in args.seeds:
            train = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base_train.items()}
            noisy_labels, flipped = flip_training_labels(base_train["y"], noise_rate, seed + 2026)
            train["y"] = noisy_labels
            model, best_internal_auc = train_one(train, val, device, seed, args.epochs)

            val_prob, val_weights = predict(model, val, device, seed)
            threshold = select_threshold(val["y"], val_prob)
            ext_prob, ext_weights = predict(model, external, device, seed)

            rows.append(metric_row(noise_rate, seed, "internal_validation", val["y"], val_prob, threshold, best_internal_auc, int(flipped.sum())))
            rows.append(metric_row(noise_rate, seed, "external_test", external["y"], ext_prob, threshold, best_internal_auc, int(flipped.sum())))
            write_predictions(noise_rate, seed, "internal_validation", val, meta["internal_validation"], val_prob, val_weights, threshold)
            write_predictions(noise_rate, seed, "external_test", external, meta["external_test"], ext_prob, ext_weights, threshold)
            print(
                f"noise={noise_rate:.2f} seed={seed}: flipped={int(flipped.sum())}, "
                f"best_internal_auc={best_internal_auc:.4f}, external_auc={binary_metrics(external['y'], ext_prob, threshold=threshold).auc:.4f}"
            )

    summarize(pd.DataFrame(rows))
    print(f"Wrote label-noise predictions to {PRED_DIR}")
    print(f"Wrote label-noise tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
