#!/usr/bin/env python3
"""Build a clean paper-facing bundle for the corrected five-centre experiment."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
OUT_ROOT = PAPER_DIR / "results" / "real_50epoch_5center_corrected"
TABLE_OUT = OUT_ROOT / "tables"
FIG_OUT = OUT_ROOT / "figures"
DOC_OUT = OUT_ROOT / "docs"

sys.path.insert(0, str(SCRIPT_DIR))
from metrics_utils import binary_metrics, read_prediction_files, summarize_prediction_dataframe


THRESHOLD_RULE = "max_specificity_at_sensitivity:0.95"
CENTER_LABEL_EN = {
    "武大人民医院": "Wuhan",
    "恩施州中心医院": "Enshi",
    "襄阳市中心医院": "Xiangyang",
    "十堰市人民医院": "Shiyan",
    "荆州市第一人民医院": "Jingzhou",
}


def center_labels(df: pd.DataFrame) -> list[str]:
    return [CENTER_LABEL_EN.get(str(name), str(name)) for name in df["center_name"]]


def ensure_dirs() -> None:
    for path in [TABLE_OUT, FIG_OUT, DOC_OUT]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def copy_key_tables() -> None:
    sources = [
        PAPER_DIR / "tables" / "center_code_mapping_audit.csv",
        PAPER_DIR / "tables" / "centerwise_aligned_case_summary.csv",
        PAPER_DIR / "tables" / "image_volume_summary.csv",
        PAPER_DIR / "tables" / "cohort_flow_3010_to_1897.csv",
        PAPER_DIR / "tables" / "split_summary_1897.csv",
        PAPER_DIR / "tables" / "loco_split_summary.csv",
        PAPER_DIR / "results" / "config_sanity" / "hydra_coe_config_sanity.json",
        PAPER_DIR / "results" / "feature_cache" / "locked_1897_feature_cache_summary.json",
    ]
    for source in sources:
        if source.exists():
            shutil.copy2(source, TABLE_OUT / source.name)


def build_feature_metrics() -> pd.DataFrame:
    pred_dir = PAPER_DIR / "results" / "predictions"
    files = sorted(pred_dir.glob("*_seed2026_*_full.csv"))
    if not files:
        raise FileNotFoundError(f"No seed2026 feature prediction CSVs found under {pred_dir}")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df = df[df["run_id"].astype(str).eq("feature")].copy()
    df = df[df.get("modality_setting", "none").fillna("none").astype(str).isin(["none", "full"])]
    metrics = summarize_prediction_dataframe(df, threshold_rule=THRESHOLD_RULE)
    metrics.to_csv(TABLE_OUT / "feature_space_metrics_seed2026_locked_threshold.csv", index=False, encoding="utf-8-sig")
    metrics[metrics["split"].eq("external_test")].to_csv(
        TABLE_OUT / "feature_space_external_metrics_seed2026_locked_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics


def checkpoint_metrics() -> pd.DataFrame:
    path = TABLE_OUT / "external_test_metrics_locked_threshold.csv"
    if path.exists():
        return read_csv(path)
    pred_dir = OUT_ROOT / "predictions"
    df = read_prediction_files(pred_dir)
    metrics = summarize_prediction_dataframe(df, threshold_rule=THRESHOLD_RULE)
    metrics.to_csv(TABLE_OUT / "checkpoint_metrics_locked_threshold.csv", index=False, encoding="utf-8-sig")
    return metrics[metrics["split"].eq("external_test")].copy()


def plot_center_cases(centerwise: pd.DataFrame) -> None:
    df = centerwise.sort_values("paper_center_no")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x, df["negative_cases"], label="Negative", color="#4C78A8")
    ax.bar(x, df["positive_cases"], bottom=df["negative_cases"], label="Positive", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(center_labels(df), rotation=25, ha="right")
    ax.set_ylabel("Patients")
    ax.set_title("Corrected five-centre aligned cohort")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "centerwise_case_distribution.png", dpi=300)
    plt.close(fig)


def plot_center_images(centerwise: pd.DataFrame) -> None:
    df = centerwise.sort_values("paper_center_no")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, df["colposcopy_image_count"], width=0.36, label="Colposcopy", color="#54A24B")
    ax.bar(x + 0.18, df["oct_image_or_bscan_count"], width=0.36, label="OCT/B-scan", color="#B279A2")
    ax.set_xticks(x)
    ax.set_xticklabels(center_labels(df), rotation=25, ha="right")
    ax.set_ylabel("Images or B-scans")
    ax.set_title("Image volume by centre")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "centerwise_image_volume.png", dpi=300)
    plt.close(fig)


def plot_split_sizes(split_summary: pd.DataFrame) -> None:
    df = split_summary[split_summary["split_name"].eq("all_center_patient_holdout_70_10_20")].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(df["subset"], df["negative"], label="Negative", color="#4C78A8")
    ax.bar(df["subset"], df["positive"], bottom=df["negative"], label="Positive", color="#F58518")
    ax.set_ylabel("Patients")
    ax.set_title("Patient-level 70/10/20 split")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "all_center_split_distribution.png", dpi=300)
    plt.close(fig)


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(y_prob))[::-1], -np.inf]
    tpr, fpr = [], []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tp = ((y_true == 1) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        fp = ((y_true == 0) & (pred == 1)).sum()
        tn = ((y_true == 0) & (pred == 0)).sum()
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    return np.asarray(fpr), np.asarray(tpr)


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.sort(np.unique(y_prob))[::-1], -np.inf]
    precision, recall = [], []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tp = ((y_true == 1) & (pred == 1)).sum()
        fp = ((y_true == 0) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / max(tp + fn, 1))
    return np.asarray(recall), np.asarray(precision)


def plot_checkpoint_curves() -> None:
    pred_path = OUT_ROOT / "predictions" / "HyDRA-CoE_runreal50_corrected_seed2026_external_test_full.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    y = df["y_true"].astype(int).to_numpy()
    p = df["y_prob"].astype(float).to_numpy()
    metric = binary_metrics(y, p, threshold=0.5)
    fpr, tpr = roc_curve_points(y, p)
    recall, precision = pr_curve_points(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].plot(fpr, tpr, color="#4C78A8", linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="#A0A0A0", linewidth=1, linestyle="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC AUC={metric.auc:.3f}")
    axes[1].plot(recall, precision, color="#F58518", linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR AUC={metric.auprc:.3f}")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "checkpoint_external_roc_pr.png", dpi=300)
    plt.close(fig)


def plot_feature_auc(feature_metrics: pd.DataFrame) -> None:
    df = feature_metrics[feature_metrics["split"].eq("external_test")].copy()
    df = df.sort_values("auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(df["method"], df["auc"], color="#72B7B2")
    ax.set_xlabel("External-test AUC")
    ax.set_title("50-epoch feature-space models, corrected 403-case external test")
    ax.set_xlim(0.5, max(0.95, float(df["auc"].max()) + 0.03))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "feature_space_external_auc_seed2026.png", dpi=300)
    plt.close(fig)


def write_key_results(feature_metrics: pd.DataFrame, checkpoint_external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not checkpoint_external.empty:
        for _, row in checkpoint_external.iterrows():
            rows.append({"result_family": "real_checkpoint_inference", **row.to_dict()})
    for _, row in feature_metrics[feature_metrics["split"].eq("external_test")].iterrows():
        rows.append({"result_family": "50epoch_feature_space", **row.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "key_results_corrected_5center.csv", index=False, encoding="utf-8-sig")
    return out


def write_summary(key_results: pd.DataFrame) -> None:
    centerwise = read_csv(TABLE_OUT / "centerwise_aligned_case_summary.csv")
    image_summary = read_csv(TABLE_OUT / "image_volume_summary.csv")
    split_summary = read_csv(TABLE_OUT / "split_summary_1897.csv")
    checkpoint = key_results[key_results["result_family"].eq("real_checkpoint_inference")]
    feature = key_results[key_results["result_family"].eq("50epoch_feature_space")]
    best_feature = feature.sort_values("auc", ascending=False).head(1)

    def fmt_metric(df: pd.DataFrame) -> str:
        if df.empty:
            return "not available"
        row = df.iloc[0]
        return (
            f"AUC {row['auc']:.3f}, AUPRC {row['auprc']:.3f}, "
            f"sensitivity {row['sensitivity']:.3f}, specificity {row['specificity']:.3f}, n={int(row['n'])}"
        )

    text = f"""# Corrected Five-Centre No-Report Results

Generated from the locked HyDRA-CoE no-report experiment after correcting raw OCT-prefix centre mapping to five participating hospitals.

## Cohort

- Raw registry cohort: 3010 patients.
- Final tri-modal aligned cohort: 1897 patients.
- Centres: {', '.join(centerwise['center_name'].astype(str).tolist())}.
- Total image/B-scan volume: {int(image_summary.loc[image_summary['modality'].eq('total'), 'total_images_or_bscans'].iloc[0])}.
- Colposcopy images: {int(image_summary.loc[image_summary['modality'].eq('colposcopy'), 'total_images_or_bscans'].iloc[0])}.
- OCT images/B-scans: {int(image_summary.loc[image_summary['modality'].eq('oct'), 'total_images_or_bscans'].iloc[0])}.
- All-center external_test size: {int(split_summary[split_summary['subset'].eq('external_test')]['n_cases'].iloc[0])}.

## Scope

Inputs are restricted to colposcopy images, OCT images/B-scans, and HPV/TCT/Age clinical text variables. Examination reports, diagnosis reports, generated reports, and VLM evidence caches are not used.

## Results

- Real 50-epoch HyDRA-CoE checkpoint inference, locked threshold: {fmt_metric(checkpoint)}.
- Best 50-epoch feature-space external-test model: {fmt_metric(best_feature)}.

## Files

- Tables: `tables/`
- Figures: `figures/`
- Real checkpoint predictions: `predictions/`
- Training log: `logs/feature_space_50epoch_seed2026.log`
"""
    (OUT_ROOT / "CORRECTED_5CENTER_RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_key_tables()
    feature_metrics = build_feature_metrics()
    checkpoint_external = checkpoint_metrics()
    centerwise = read_csv(TABLE_OUT / "centerwise_aligned_case_summary.csv")
    split_summary = read_csv(TABLE_OUT / "split_summary_1897.csv")

    plot_center_cases(centerwise)
    plot_center_images(centerwise)
    plot_split_sizes(split_summary)
    plot_checkpoint_curves()
    plot_feature_auc(feature_metrics)

    key_results = write_key_results(feature_metrics, checkpoint_external)
    write_summary(key_results)
    manifest = {
        "bundle": str(OUT_ROOT),
        "tables": sorted(path.name for path in TABLE_OUT.glob("*")),
        "figures": sorted(path.name for path in FIG_OUT.glob("*")),
    }
    (OUT_ROOT / "result_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
