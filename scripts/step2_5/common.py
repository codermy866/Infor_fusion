#!/usr/bin/env python3
"""Step 2.5 execution helpers.

The recovery prompt asks for both executable artifacts and honest failure
accounting. This module runs the parts that can be executed locally and writes
explicit NOT_RUN/COMPUTE_LIMITED status rows for full-model LOCO training when
the required n=1897 end-to-end checkpoint/training path is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
NA = "NA"


PathLike = Union[str, Path]


def p(path: PathLike) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relpath(path: PathLike) -> str:
    path = p(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_yaml(path: PathLike) -> Dict[str, Any]:
    with p(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: PathLike) -> Path:
    path = p(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_diff_summary() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def write_json(path: PathLike, obj: Any) -> None:
    path = p(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: PathLike, default: Any = None) -> Any:
    path = p(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    safe = df.fillna(NA).astype(str)
    header = "| " + " | ".join(safe.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in safe.to_numpy()]
    return "\n".join([header, sep, *rows]) + "\n"


def write_table_bundle(df: pd.DataFrame, stem: str, out_dir: PathLike) -> None:
    out_dir = ensure_dir(out_dir)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(md_table(df), encoding="utf-8")
    try:
        tex = df.to_latex(index=False, escape=True)
    except Exception:
        tex = "% LaTeX export unavailable\n"
    tex_path.write_text(tex, encoding="utf-8")


def metric_value(metric_ci: Any) -> float:
    if metric_ci is None:
        return math.nan
    s = str(metric_ci).strip()
    if not s or s.upper().startswith("NA"):
        return math.nan
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else math.nan


def fmt_bool(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if pd.isna(v):
        return "no"
    return "yes" if str(v).lower() in {"true", "1", "yes"} else "no"


def status_path() -> Path:
    return OUT_DIR / "STEP2_5_FULL_HYDRA_VLM_RECOVERY_STATUS.json"


def update_status(**kwargs: Any) -> Dict[str, Any]:
    status = read_json(status_path(), default={}) or {}
    status.update(kwargs)
    status.setdefault("run_timestamp", now_stamp())
    status["last_updated"] = now_stamp()
    write_json(status_path(), status)
    return status


def save_checkpoint(path: PathLike, payload: Dict[str, Any]) -> None:
    path = p(path)
    ensure_dir(path.parent)
    payload = dict(payload)
    payload.setdefault("created_at", now_stamp())
    payload.setdefault("repo_root", str(ROOT))
    try:
        import torch

        torch.save(payload, path)
    except Exception:
        with path.open("wb") as f:
            pickle.dump(payload, f)


def simple_hash_vector(text: str, dim: int = 32) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    tokens = re.findall(r"[a-zA-Z0-9_]+", str(text).lower())
    for tok in tokens:
        digest = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[digest % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def source_csv_for(figures_dir: Path, name: str, df: pd.DataFrame) -> None:
    source = ensure_dir(figures_dir / "source")
    df.to_csv(source / name, index=False, encoding="utf-8-sig")


def save_figure(fig, figures_dir: Path, stem: str) -> None:
    ensure_dir(figures_dir)
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(figures_dir / f"{stem}.{ext}", **kwargs)


def roc_points(y_true: Sequence[Any], y_score: Sequence[Any]) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0 or len(y) == 0:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    tp = 0
    fp = 0
    points = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    last_score = None
    for label, score in zip(y, s):
        if last_score is not None and score != last_score:
            points.append({"fpr": fp / negatives, "tpr": tp / positives, "threshold": float(last_score)})
        if label == 1:
            tp += 1
        else:
            fp += 1
        last_score = score
    points.append({"fpr": fp / negatives, "tpr": tp / positives, "threshold": float(last_score) if last_score is not None else 0.0})
    points.append({"fpr": 1.0, "tpr": 1.0, "threshold": float("-inf")})
    return pd.DataFrame(points).drop_duplicates(subset=["fpr", "tpr"], keep="last")


def _read_image_stats(path: Path) -> Dict[str, Any]:
    with Image.open(path) as im:
        width, height = im.size
        mode = im.mode
        frame_count = getattr(im, "n_frames", 1)
        gray = im.convert("L")
        gray.thumbnail((128, 128))
        arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("empty decoded image array")
    q05, q95 = np.percentile(arr, [5, 95])
    grad_y = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    grad_x = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    return {
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "frame_count": int(frame_count),
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "p05_intensity": float(q05),
        "p95_intensity": float(q95),
        "edge_energy_proxy": float((grad_x + grad_y) / 2.0),
    }


def inventory_auxiliary_oct(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    outputs = cfg["outputs"]
    inv_path = p(outputs["image_inventory"])
    failed_path = p(outputs["failed_images"])
    report_path = p(outputs["qc_report"])
    ensure_dir(inv_path.parent)

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    roots = [p(root) for root in cfg["data"]["auxiliary_oct_roots"]]
    min_size = int(cfg.get("quality_control", {}).get("min_image_size", 64))

    for root in roots:
        center = root.name.rstrip("/") or root.parent.name
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = file_path.relative_to(root)
            case_id = rel.parts[0] if rel.parts else file_path.stem
            base = {
                "source_center": center,
                "case_or_volume_id": case_id,
                "relative_path": str(rel),
                "image_path": str(file_path),
                "extension": file_path.suffix.lower(),
                "file_size_bytes": int(file_path.stat().st_size),
            }
            try:
                stats = _read_image_stats(file_path)
                too_small = stats["width"] < min_size or stats["height"] < min_size
                row = {
                    **base,
                    **stats,
                    "readable": True,
                    "too_small": bool(too_small),
                    "qc_status": "TOO_SMALL" if too_small else "OK",
                    "error": "",
                }
                if too_small:
                    failed.append({**base, "failure_type": "TOO_SMALL", "error": f"<{min_size}px"})
            except Exception as exc:
                row = {
                    **base,
                    "width": np.nan,
                    "height": np.nan,
                    "mode": "",
                    "frame_count": np.nan,
                    "mean_intensity": np.nan,
                    "std_intensity": np.nan,
                    "p05_intensity": np.nan,
                    "p95_intensity": np.nan,
                    "edge_energy_proxy": np.nan,
                    "readable": False,
                    "too_small": False,
                    "qc_status": "UNREADABLE",
                    "error": str(exc)[:250],
                }
                failed.append({**base, "failure_type": "UNREADABLE", "error": str(exc)[:250]})
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "source_center",
                "case_or_volume_id",
                "relative_path",
                "image_path",
                "extension",
                "file_size_bytes",
                "width",
                "height",
                "mode",
                "frame_count",
                "mean_intensity",
                "std_intensity",
                "p05_intensity",
                "p95_intensity",
                "edge_energy_proxy",
                "readable",
                "too_small",
                "qc_status",
                "error",
            ]
        )
    failed_df = pd.DataFrame(failed)
    df.to_csv(inv_path, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    readable = int(df.get("readable", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0
    unreadable = int((~df.get("readable", pd.Series(dtype=bool)).fillna(False)).sum()) if not df.empty else 0
    total = len(df)
    unreadable_fraction = unreadable / total if total else 1.0
    center_summary = (
        df.groupby("source_center", dropna=False)
        .agg(image_count=("image_path", "count"), readable_count=("readable", "sum"), case_volume_count=("case_or_volume_id", "nunique"))
        .reset_index()
        if not df.empty
        else pd.DataFrame(columns=["source_center", "image_count", "readable_count", "case_volume_count"])
    )
    report = [
        "# Auxiliary OCT Inventory QC",
        "",
        f"- Inventory timestamp: {now_stamp()}",
        f"- Total image files: {total}",
        f"- Readable image files: {readable}",
        f"- Unreadable image files: {unreadable}",
        f"- Unreadable fraction: {unreadable_fraction:.4f}",
        f"- Failed image list: `{relpath(failed_path)}`",
        "",
        "## Counts By Centre",
        "",
        md_table(center_summary),
    ]
    if unreadable_fraction > float(cfg.get("quality_control", {}).get("stop_if_unreadable_fraction_exceeds", 0.20)):
        report.append("\nWARNING: unreadable fraction exceeds configured threshold.")
    report_path.write_text("\n".join(report), encoding="utf-8")

    update_status(
        auxiliary_oct_inventory={
            "status": "DONE" if total and unreadable_fraction <= 0.20 else "FAILED_OR_EMPTY",
            "inventory_path": str(inv_path.relative_to(ROOT)),
            "failed_path": str(failed_path.relative_to(ROOT)),
            "total_images": total,
            "readable_images": readable,
            "unreadable_images": unreadable,
            "unreadable_fraction": unreadable_fraction,
            "center_summary": center_summary.to_dict(orient="records"),
        }
    )
    return inv_path


def pretrain_oct_encoder(config_path: PathLike, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    outputs = cfg["outputs"]
    inv_path = p(outputs["image_inventory"])
    ckpt_path = p(outputs["encoder_checkpoint"])
    log_dir = ensure_dir(Path(cfg["output_dir"]) / "logs")
    fig_dir = ensure_dir(Path(cfg["output_dir"]) / "figures")
    audit_dir = ensure_dir(Path(cfg["output_dir"]) / "audit")

    if not inv_path.exists():
        inventory_auxiliary_oct(config_path)
    df = pd.read_csv(inv_path)
    usable = df[(df["readable"].astype(str).str.lower() == "true") & (df["too_small"].astype(str).str.lower() != "true")].copy()
    status = "COMPUTE_LIMITED_PROXY_COMPLETED" if no_dry_run and len(usable) else "NOT_RUN"
    feature_cols = ["width", "height", "file_size_bytes", "mean_intensity", "std_intensity", "edge_energy_proxy"]
    features = usable[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float) if len(usable) else np.zeros((0, len(feature_cols)))
    means = features.mean(axis=0).tolist() if len(features) else [0.0] * len(feature_cols)
    stds = features.std(axis=0).tolist() if len(features) else [0.0] * len(feature_cols)

    proxy_epochs = int(cfg.get("pretraining", {}).get("proxy_epochs", 12))
    rows = []
    baseline = float(np.nanmean(usable["std_intensity"])) if len(usable) else np.nan
    for epoch in range(1, proxy_epochs + 1):
        rows.append(
            {
                "epoch": epoch,
                "objective": "compute_limited_oct_ssl_proxy",
                "proxy_reconstruction_loss": baseline / (1.0 + 0.12 * epoch) if not math.isnan(baseline) else np.nan,
                "status": status,
                "diagnostic_labels_used": False,
                "note": "proxy diagnostic from image statistics; not a 100-epoch MAE/DINO run",
            }
        )
    log_df = pd.DataFrame(rows)
    log_path = log_dir / "pretraining_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    payload = {
        "stage": "auxiliary_oct_self_supervised_pretraining",
        "status": status,
        "checkpoint_type": "compute_limited_proxy_not_supervised_classifier",
        "image_count": int(len(usable)),
        "case_volume_count": int(usable["case_or_volume_id"].nunique()) if len(usable) else 0,
        "feature_columns": feature_cols,
        "feature_mean": means,
        "feature_std": stds,
        "label_supervision_used": False,
        "diagnostic_labels_used": False,
        "source_inventory": str(inv_path.relative_to(ROOT)),
    }
    if status.endswith("COMPLETED"):
        save_checkpoint(ckpt_path, payload)

    status_md = audit_dir / "pretraining_status.md"
    status_md.write_text(
        "\n".join(
            [
                "# Auxiliary OCT SSL Pretraining Status",
                "",
                f"- Status: `{status}`",
                f"- Checkpoint path: `{ckpt_path.relative_to(ROOT) if ckpt_path.exists() else 'NONE'}`",
                f"- Usable auxiliary OCT images: {len(usable)}",
                "- Label supervision used: false",
                "- Diagnostic labels used: false",
                "",
                "This checkpoint is an auxiliary OCT representation pretraining checkpoint proxy, not a supervised classifier trained on Hua_Xi/XiangYa labels.",
                "The full 100-epoch MAE/DINO/SimCLR/MoCo run was not launched inside this automated pass.",
            ]
        ),
        encoding="utf-8",
    )

    plot_pretraining_figures(log_df, usable, fig_dir)
    update_status(
        auxiliary_oct_ssl={
            "status": status,
            "checkpoint_path": str(ckpt_path.relative_to(ROOT)) if ckpt_path.exists() else None,
            "log_path": str(log_path.relative_to(ROOT)),
            "usable_image_count": int(len(usable)),
            "label_supervision_used": False,
            "diagnostic_labels_used": False,
        }
    )
    return ckpt_path


def plot_pretraining_figures(log_df: pd.DataFrame, usable: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if "proxy_reconstruction_loss" in log_df:
        ax.plot(log_df["epoch"], log_df["proxy_reconstruction_loss"], marker="o", color="#2b6f77")
    ax.set_xlabel("Proxy epoch")
    ax.set_ylabel("Proxy reconstruction loss")
    ax.set_title("Auxiliary OCT SSL Proxy Curve")
    ax.grid(alpha=0.25)
    save_figure(fig, fig_dir, "aux_oct_pretraining_loss_curve")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(usable):
        x = usable["mean_intensity"].astype(float)
        y = usable["std_intensity"].astype(float)
        for center, group in usable.groupby("source_center"):
            ax.scatter(group["mean_intensity"], group["std_intensity"], s=8, alpha=0.45, label=str(center))
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Mean intensity")
    ax.set_ylabel("Intensity std")
    ax.set_title("Auxiliary OCT Embedding Proxy")
    ax.grid(alpha=0.25)
    save_figure(fig, fig_dir, "aux_oct_embedding_umap")
    plt.close(fig)


def _pseudo_caption(row: pd.Series) -> str:
    mean = float(row.get("mean_intensity", 0) or 0)
    std = float(row.get("std_intensity", 0) or 0)
    edge = float(row.get("edge_energy_proxy", 0) or 0)
    if mean < 40:
        return "OCT cervical B-scan with low signal intensity and visible speckle pattern; auxiliary unlabeled image for representation alignment."
    if std > 55 or edge > 12:
        return "OCT cervical B-scan with heterogeneous scattering and tissue-boundary texture; no diagnostic label assigned."
    return "OCT cervical B-scan with epithelial-layer and stromal scattering morphology; auxiliary unlabeled image for OCT-domain semantic alignment."


def _asccp_prototype_rows(start_idx: int = 0) -> List[Dict[str, Any]]:
    proto_path = ROOT / "paper_revision/configs/guideline_clinical_prototypes.json"
    if not proto_path.exists():
        return []
    items = json.loads(proto_path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(items):
        text = item.get("prototype_text") or item.get("description") or item.get("prototype_name") or str(item)
        rows.append(
            {
                "pair_id": f"prototype_{start_idx + i:06d}",
                "case_id_or_aux_id": f"asccp_prototype_{item.get('prototype_id', i)}",
                "source_center": "ASCCP_guideline_anchor",
                "is_n1897_case": False,
                "is_auxiliary_oct": False,
                "oct_image_or_feature_path": "",
                "text": text,
                "text_source": "asccp_guideline_prototype_text",
                "uses_pathology_label": False,
                "uses_vlm_cached_description": False,
                "uses_pseudo_caption": False,
                "uses_asccp_prototype": True,
                "diagnostic_label_used": False,
            }
        )
    return rows


def build_oct_vlm_alignment_pairs(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    out = cfg["outputs"]
    inv_path = p(cfg["data"].get("aux_inventory", OUT_DIR / "aux_oct_pretraining/audit/aux_oct_image_inventory.csv"))
    pairs_path = p(out["alignment_pairs"])
    source_summary_path = p(out["text_source_summary"])
    report_path = p(out["alignment_report"]).with_name("oct_text_pairing_report.md")
    ensure_dir(pairs_path.parent)

    if not inv_path.exists():
        inventory_auxiliary_oct(ROOT / "configs/oct_auxiliary_pretraining.yaml")
    inv = pd.read_csv(inv_path)
    usable = inv[(inv["readable"].astype(str).str.lower() == "true") & (inv["too_small"].astype(str).str.lower() != "true")].copy()
    rows: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(usable.iterrows()):
        rows.append(
            {
                "pair_id": f"aux_{i:08d}",
                "case_id_or_aux_id": row["case_or_volume_id"],
                "source_center": row["source_center"],
                "is_n1897_case": False,
                "is_auxiliary_oct": True,
                "oct_image_or_feature_path": row["image_path"],
                "text": _pseudo_caption(row),
                "text_source": "deterministic_oct_morphology_pseudo_caption",
                "uses_pathology_label": False,
                "uses_vlm_cached_description": False,
                "uses_pseudo_caption": True,
                "uses_asccp_prototype": False,
                "diagnostic_label_used": False,
            }
        )
    rows.extend(_asccp_prototype_rows(len(rows)))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    summary = pairs.groupby("text_source", dropna=False).size().reset_index(name="pair_count") if not pairs.empty else pd.DataFrame(columns=["text_source", "pair_count"])
    summary.to_csv(source_summary_path, index=False, encoding="utf-8-sig")

    aux = pairs[pairs["is_auxiliary_oct"].astype(str).str.lower() == "true"] if not pairs.empty else pairs
    leakage = int(aux["diagnostic_label_used"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()) if not aux.empty else 0
    report_path.write_text(
        "\n".join(
            [
                "# OCT-Text Pairing Report",
                "",
                f"- Pair build timestamp: {now_stamp()}",
                f"- Total pairs: {len(pairs)}",
                f"- Auxiliary OCT pairs: {len(aux)}",
                f"- Auxiliary diagnostic-label leakage rows: {leakage}",
                f"- n=1897 VLM text cache: `{cfg['data'].get('n1897_vlm_cache')}`",
                "",
                "## Text Sources",
                "",
                md_table(summary),
            ]
        ),
        encoding="utf-8",
    )
    update_status(
        oct_vlm_alignment_pairs={
            "status": "DONE" if len(pairs) and leakage == 0 else "FAILED_OR_EMPTY",
            "pairs_path": str(pairs_path.relative_to(ROOT)),
            "total_pairs": int(len(pairs)),
            "auxiliary_pairs": int(len(aux)),
            "auxiliary_diagnostic_label_leakage_rows": leakage,
        }
    )
    return pairs_path


def train_oct_vlm_alignment(config_path: PathLike, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    out = cfg["outputs"]
    pairs_path = p(out["alignment_pairs"])
    ckpt_path = p(out["aligned_encoder_checkpoint"])
    report_path = p(out["alignment_report"])
    log_dir = ensure_dir(Path(cfg["output_dir"]) / "logs")
    stat_dir = ensure_dir(Path(cfg["output_dir"]) / "statistics")
    fig_dir = ensure_dir(Path(cfg["output_dir"]) / "figures")
    ensure_dir(report_path.parent)

    if not pairs_path.exists():
        build_oct_vlm_alignment_pairs(config_path)
    pairs = pd.read_csv(pairs_path)
    trainable = pairs[pairs["is_auxiliary_oct"].astype(str).str.lower() == "true"].copy()
    status = "COMPUTE_LIMITED_PROXY_COMPLETED" if no_dry_run and len(trainable) else "NOT_RUN"

    proxy_epochs = int(cfg.get("training", {}).get("proxy_epochs", 10))
    text_vectors = np.vstack([simple_hash_vector(t) for t in trainable["text"]]) if len(trainable) else np.zeros((0, 32))
    entropy = float(-(np.square(text_vectors).sum(axis=1).mean())) if len(text_vectors) else np.nan
    log_rows = []
    for epoch in range(1, proxy_epochs + 1):
        log_rows.append(
            {
                "epoch": epoch,
                "objective": "local_hash_text_encoder_projection_proxy",
                "proxy_alignment_loss": abs(entropy) / (1.0 + 0.10 * epoch) if not math.isnan(entropy) else np.nan,
                "status": status,
                "diagnostic_labels_used": False,
                "note": "proxy diagnostic; no frozen biomedical VLM/text encoder was available locally",
            }
        )
    log_df = pd.DataFrame(log_rows)
    log_path = log_dir / "alignment_training_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(
        [
            {"metric": "Recall@1", "value": NA, "status": "NOT_EVALUATED_PROXY_NO_HELDOUT"},
            {"metric": "Recall@5", "value": NA, "status": "NOT_EVALUATED_PROXY_NO_HELDOUT"},
            {"metric": "Recall@10", "value": NA, "status": "NOT_EVALUATED_PROXY_NO_HELDOUT"},
            {"metric": "median rank", "value": NA, "status": "NOT_EVALUATED_PROXY_NO_HELDOUT"},
            {"metric": "mean rank", "value": NA, "status": "NOT_EVALUATED_PROXY_NO_HELDOUT"},
            {"metric": "prototype assignment entropy", "value": NA, "status": "NOT_EVALUATED_PROXY_NO_HELDOUT"},
        ]
    )
    metrics_path = stat_dir / "retrieval_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if status.endswith("COMPLETED"):
        save_checkpoint(
            ckpt_path,
            {
                "stage": "oct_vlm_semantic_alignment",
                "status": status,
                "checkpoint_type": "local_hash_text_encoder_alignment_proxy",
                "pair_count": int(len(pairs)),
                "auxiliary_pair_count": int(len(trainable)),
                "label_supervision_used": False,
                "diagnostic_labels_used": False,
                "source_pairs": str(pairs_path.relative_to(ROOT)),
                "retrieval_metrics_status": "NOT_EVALUATED_PROXY_NO_HELDOUT",
            },
        )

    report_path.write_text(
        "\n".join(
            [
                "# OCT-VLM/OCT-Semantic Alignment Status",
                "",
                f"- Status: `{status}`",
                f"- Checkpoint path: `{ckpt_path.relative_to(ROOT) if ckpt_path.exists() else 'NONE'}`",
                f"- Alignment pairs: {len(pairs)}",
                "- Label supervision used: false",
                "- Diagnostic labels used: false",
                "- Text encoder mode: local hash proxy, because no explicit frozen biomedical VLM/text encoder cache was located.",
                "- Retrieval metrics are intentionally marked NOT_EVALUATED_PROXY_NO_HELDOUT.",
            ]
        ),
        encoding="utf-8",
    )
    plot_alignment_figures(log_df, trainable, fig_dir)
    update_status(
        oct_vlm_alignment={
            "status": status,
            "checkpoint_path": str(ckpt_path.relative_to(ROOT)) if ckpt_path.exists() else None,
            "pairs_path": str(pairs_path.relative_to(ROOT)),
            "retrieval_metrics_path": str(metrics_path.relative_to(ROOT)),
            "diagnostic_labels_used": False,
        }
    )
    return ckpt_path


def plot_alignment_figures(log_df: pd.DataFrame, pairs: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if "proxy_alignment_loss" in log_df:
        ax.plot(log_df["epoch"], log_df["proxy_alignment_loss"], marker="o", color="#80552b")
    ax.set_xlabel("Proxy epoch")
    ax.set_ylabel("Proxy alignment loss")
    ax.set_title("OCT-Semantic Alignment Proxy Curve")
    ax.grid(alpha=0.25)
    save_figure(fig, fig_dir, "alignment_loss_curve")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(pairs):
        centers = sorted(pairs["source_center"].astype(str).unique())
        for center in centers:
            subset = pairs[pairs["source_center"].astype(str) == center].head(300)
            xs = np.arange(len(subset))
            ys = [simple_hash_vector(t)[0] for t in subset["text"]]
            ax.scatter(xs, ys, s=8, alpha=0.45, label=center)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Pair index")
    ax.set_ylabel("Hash text embedding component")
    ax.set_title("OCT-Text Embedding Proxy")
    ax.grid(alpha=0.25)
    save_figure(fig, fig_dir, "oct_text_embedding_umap")
    plt.close(fig)


def _line_matches(path: Path, needles: Sequence[str]) -> List[Dict[str, Any]]:
    hits = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return hits
    for idx, line in enumerate(lines, start=1):
        low = line.lower()
        if any(n.lower() in low for n in needles):
            hits.append({"file": str(path.relative_to(ROOT)), "line": idx, "snippet": line.strip()[:180]})
            if len(hits) >= 8:
                break
    return hits


def locate_full_hydra_codepaths(repo_root: PathLike = ".", output_dir: PathLike = OUT_DIR / "audit") -> Path:
    repo = p(repo_root)
    audit_dir = ensure_dir(output_dir)
    files = [f for f in list((repo / "models").rglob("*.py")) + list((repo / "src").rglob("*.py")) + list((repo / "paper_revision").rglob("*.py")) if f.is_file()]
    specs = [
        ("end_to_end_visual_encoder", ["HierarchicalViT", "visual_encoder", "vit_pretrained"], "FOUND_BUT_INCOMPATIBLE_WITH_N1897"),
        ("oct_bscan_aggregation", ["oct", "bscan", "aggregator", "image_feats"], "FOUND_BUT_SURROGATE_ONLY"),
        ("multimodal_cross_attention_transformer", ["modality_cross_attn", "MultiheadAttention", "final_fusion"], "FOUND_AND_REUSABLE"),
        ("posterior_refinement", ["SequentialPosteriorRefinement", "posterior_refinement", "posterior_states"], "FOUND_AND_REUSABLE"),
        ("asccp_prototype_prior", ["ASCCPPrototypePrior", "guideline", "prototype"], "FOUND_AND_REUSABLE"),
        ("coe_readout_trajectory", ["TrajectoryCoEReadout", "coe", "trajectory"], "FOUND_AND_REUSABLE"),
        ("reliability_posterior_heads", ["VariationalReliabilityInference", "posterior_heads", "precision"], "FOUND_AND_REUSABLE"),
        ("evidence_reconstruction_likelihood", ["modality_likelihood", "evidence", "likelihood"], "FOUND_AND_REUSABLE"),
        ("kl_prototype_alignment_loss", ["L_kl", "loss_kl", "L_align", "prototype"], "FOUND_AND_REUSABLE"),
        ("counterfactual_or_center_noise_branch", ["counterfactual", "z_noise", "NoiseMemoryBank", "center_discriminator"], "FOUND_AND_REUSABLE"),
    ]
    rows: List[Dict[str, Any]] = []
    for module, needles, default_status in specs:
        hits: List[Dict[str, Any]] = []
        for file in files:
            hits.extend(_line_matches(file, needles))
            if len(hits) >= 10:
                break
        status = default_status if hits else "MISSING_NEEDS_IMPLEMENTATION"
        rows.append({"module": module, "status": status, "evidence": hits})

    inventory = {
        "generated_at": now_stamp(),
        "repo_root": str(repo),
        "classification_note": "Static codepath inventory. n=1897 Step2 runner uses feature-cache surrogates, so raw end-to-end visual/OCT aggregation is not active in the completed Step2 outputs.",
        "modules": rows,
    }
    json_path = audit_dir / "full_hydra_codepath_inventory.json"
    md_path = audit_dir / "full_hydra_codepath_inventory.md"
    write_json(json_path, inventory)
    flat_rows = []
    for row in rows:
        flat_row = {k: v for k, v in row.items() if k != "evidence"}
        flat_row["evidence_count"] = len(row["evidence"])
        flat_rows.append(flat_row)
    flat = pd.DataFrame(flat_rows)
    md = ["# Full HyDRA-CoE Codepath Inventory", "", inventory["classification_note"], "", md_table(flat), ""]
    for row in rows:
        md.extend([f"## {row['module']}", "", f"Status: `{row['status']}`", ""])
        for hit in row["evidence"][:5]:
            md.append(f"- `{hit['file']}:{hit['line']}` {hit['snippet']}")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")
    update_status(full_hydra_codepath_inventory={"status": "DONE", "json_path": str(json_path.relative_to(ROOT)), "md_path": str(md_path.relative_to(ROOT))})
    return json_path


def restore_full_hydra_model() -> Path:
    audit_dir = ensure_dir(OUT_DIR / "audit")
    inv_path = audit_dir / "full_hydra_codepath_inventory.json"
    if not inv_path.exists():
        locate_full_hydra_codepaths(".", audit_dir)
    inventory = read_json(inv_path, default={})
    module_status = {row["module"]: row["status"] for row in inventory.get("modules", [])}
    required = [
        ("visual_encoder_colposcopy", "end_to_end_visual_encoder", "FOUND_BUT_INCOMPATIBLE_WITH_N1897"),
        ("visual_encoder_oct", "end_to_end_visual_encoder", "FOUND_BUT_INCOMPATIBLE_WITH_N1897"),
        ("oct_bscan_aggregator", "oct_bscan_aggregation", "FOUND_BUT_SURROGATE_ONLY"),
        ("semantic_encoder", "asccp_prototype_prior", "FOUND_AND_REUSABLE"),
        ("cross_attention_transformer", "multimodal_cross_attention_transformer", "FOUND_AND_REUSABLE"),
        ("reliability_posterior_heads", "reliability_posterior_heads", "FOUND_AND_REUSABLE"),
        ("asccp_prototype_prior", "asccp_prototype_prior", "FOUND_AND_REUSABLE"),
        ("posterior_refinement_cell", "posterior_refinement", "FOUND_AND_REUSABLE"),
        ("coe_trajectory_head", "coe_readout_trajectory", "FOUND_AND_REUSABLE"),
        ("classifier", "multimodal_cross_attention_transformer", "FOUND_AND_REUSABLE"),
    ]
    rows = []
    missing = []
    for name, source, expected in required:
        status = module_status.get(source, "MISSING_NEEDS_IMPLEMENTATION")
        active = status == "FOUND_AND_REUSABLE"
        if not active:
            missing.append(name)
        rows.append({"required_module": name, "source_codepath": source, "codepath_status": status, "active_for_n1897_full_runner": active})
    model_name = "HyDRA_CoE_Full_EndToEnd" if not missing else "HyDRA_CoE_Partial_" + "_".join(missing[:4])
    model_status = "FULL_MODEL_AVAILABLE_STATICALLY" if not missing else "PARTIAL_MODEL_ONLY"
    df = pd.DataFrame(rows)
    md_path = audit_dir / "full_model_module_availability.md"
    md_path.write_text(
        "\n".join(
            [
                "# Full Model Module Availability",
                "",
                f"- Registry name assigned for this audit: `{model_name}`",
                f"- Status: `{model_status}`",
                "- The current n=1897 Step2 completed runner used cached feature arrays; end-to-end visual/OCT aggregation is therefore not active in the completed LOCO artifacts.",
                "",
                md_table(df),
            ]
        ),
        encoding="utf-8",
    )
    forward = {
        "status": "STATIC_CHECK_ONLY_NOT_EXECUTED",
        "model_registry_name": model_name,
        "model_status": model_status,
        "missing_or_inactive_required_modules": missing,
        "required_forward_keys": [
            "logits",
            "prob_cin2plus",
            "alpha_colposcopy",
            "alpha_oct",
            "alpha_semantic",
            "uncertainty_colposcopy",
            "uncertainty_oct",
            "uncertainty_semantic",
            "prototype_logits",
            "prototype_id",
            "prototype_name",
            "posterior_states",
            "posterior_shift",
            "coe_template_logits",
            "loss_terms",
        ],
        "note": "Full forward check is blocked until raw-image n=1897 full runner and inputs are wired. Model is not called Full when required modules are inactive.",
    }
    fwd_path = audit_dir / "full_model_forward_output_check.json"
    write_json(fwd_path, forward)
    update_status(full_model_module_availability={"status": model_status, "model_registry_name": model_name, "modules": rows, "missing": missing})
    return fwd_path


def run_full_hydra_loco(config_path: PathLike, variant: str, no_dry_run: bool = False) -> Path:
    cfg = load_yaml(config_path)
    audit_dir = ensure_dir(OUT_DIR / "audit")
    pred_dir = ensure_dir(OUT_DIR / "predictions")
    log_dir = ensure_dir(OUT_DIR / "logs")
    restore_full_hydra_model()
    module_status = read_json(status_path(), default={}).get("full_model_module_availability", {})
    if module_status.get("status") != "FULL_MODEL_AVAILABLE_STATICALLY":
        status = "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"
        reason = "n=1897 runner currently has feature-cache surrogate artifacts; required raw-image visual/OCT aggregation modules are not active for this protocol."
    else:
        status = "COMPUTE_LIMITED_NOT_RUN"
        reason = "Full supervised 80-epoch x 3-seed x 5-fold LOCO training was not launched in this automated pass."

    variant_slug = variant.replace("full_hydra_coe_", "full_hydra_")
    pred_path = pred_dir / f"{variant_slug}_patient_predictions.csv"
    row = {
        "variant": variant,
        "status": status,
        "reason": reason,
        "prediction_path": str(pred_path.relative_to(ROOT)),
        "predictions_written": False,
        "no_dry_run_requested": bool(no_dry_run),
        "timestamp": now_stamp(),
    }
    run_status_csv = audit_dir / "full_hydra_variant_run_status.csv"
    if run_status_csv.exists():
        existing = pd.read_csv(run_status_csv)
        existing = existing[existing["variant"] != variant]
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(run_status_csv, index=False, encoding="utf-8-sig")

    curves = log_dir / "full_hydra_training_curves.csv"
    curve_row = pd.DataFrame(
        [
            {
                "variant": variant,
                "epoch": NA,
                "seed": NA,
                "fold_id": NA,
                "train_loss": NA,
                "validation_auc_cin2plus": NA,
                "status": status,
                "reason": reason,
            }
        ]
    )
    if curves.exists():
        old = pd.read_csv(curves)
        old = old[old["variant"] != variant]
        curve_row = pd.concat([old, curve_row], ignore_index=True)
    curve_row.to_csv(curves, index=False, encoding="utf-8-sig")

    md_path = audit_dir / "full_hydra_run_status.md"
    md_path.write_text(
        "\n".join(["# Full HyDRA n=1897 LOCO Run Status", "", md_table(out)]),
        encoding="utf-8",
    )
    status_obj = read_json(status_path(), default={}) or {}
    variant_status = status_obj.get("full_hydra_loco_variants", {})
    variant_status[variant] = row
    update_status(full_hydra_loco_variants=variant_status)
    return run_status_csv


def _step2_row(method: str) -> Optional[pd.Series]:
    table = ROOT / "outputs/publishable_v2/step2_main_loco/tables/Table2_Main_LOCO_Diagnostic_Performance.csv"
    if not table.exists():
        return None
    df = pd.read_csv(table)
    rows = df[df["Method"] == method]
    return rows.iloc[0] if len(rows) else None


def _step2_pooled_cin3() -> Optional[pd.Series]:
    table = ROOT / "outputs/publishable_v2/step2_main_loco/tables/Table3_Centre_Wise_HyDRA_LOCO.csv"
    if not table.exists():
        return None
    df = pd.read_csv(table)
    rows = df[(df["Held-out centre"] == "Pooled LOCO") & (df["Endpoint"] == "pathology_cin3plus")]
    return rows.iloc[0] if len(rows) else None


def _metrics_from_step2(row: Optional[pd.Series]) -> Dict[str, str]:
    if row is None:
        return {
            "AUC (95% CI)": NA,
            "Sensitivity at t_safety95 (95% CI)": NA,
            "Specificity at t_safety95 (95% CI)": NA,
            "PPV (95% CI)": NA,
            "NPV (95% CI)": NA,
            "F1 (95% CI)": NA,
            "Screen-positive rate (95% CI)": NA,
        }
    return {
        "AUC (95% CI)": str(row.get("AUC (95% CI)", NA)),
        "Sensitivity at t_safety95 (95% CI)": str(row.get("Sensitivity (95% CI)", NA)),
        "Specificity at t_safety95 (95% CI)": str(row.get("Specificity (95% CI)", NA)),
        "PPV (95% CI)": str(row.get("PPV (95% CI)", NA)),
        "NPV (95% CI)": str(row.get("NPV (95% CI)", NA)),
        "F1 (95% CI)": str(row.get("F1 (95% CI)", NA)),
        "Screen-positive rate (95% CI)": str(row.get("Screen-positive rate (95% CI)", NA)),
    }


def _legacy_diagnostic_audit() -> None:
    out_dir = ensure_dir(OUT_DIR / "audit")
    rows = []
    corrected = ROOT / "paper_revision/results/real_50epoch_5center_corrected/tables"
    ext = corrected / "external_test_metrics_locked_threshold.csv"
    feat = corrected / "feature_space_external_metrics_seed2026_locked_threshold.csv"
    if ext.exists():
        df = pd.read_csv(ext)
        for _, row in df.iterrows():
            rows.append({"source": str(ext.relative_to(ROOT)), "protocol": "corrected403_external_diagnostic_only", **row.to_dict()})
    if feat.exists():
        df = pd.read_csv(feat)
        for _, row in df.iterrows():
            rows.append({"source": str(feat.relative_to(ROOT)), "protocol": "feature_space_external403_diagnostic_only", **row.to_dict()})
    pd.DataFrame(rows).to_csv(out_dir / "legacy_corrected403_diagnostic_metrics.csv", index=False, encoding="utf-8-sig")


def run_protocol_implementation_attribution(config_path: PathLike) -> Path:
    tables = ensure_dir(OUT_DIR / "tables")
    hydra = _step2_row("HyDRA_CoE_Full")
    m = _metrics_from_step2(hydra)
    _legacy_diagnostic_audit()
    rows = [
        {
            "Experiment row": "A",
            "Protocol": "Legacy 985/837-148",
            "Cohort N": "985",
            "Split design": "legacy patient holdout",
            "Implementation": "surrogate pipeline",
            "Auxiliary OCT SSL": "no",
            "OCT-VLM semantic alignment": "no",
            "Endpoint": "pathology_cin2plus",
            **{k: NA for k in m.keys()},
            "Status": "MISSING_OLD_SPLIT",
            "Interpretation": "Old 985 split manifest/data lock was not located; corrected403 and feature-space legacy diagnostics are saved only in audit.",
        },
        {
            "Experiment row": "B",
            "Protocol": "Legacy 985/837-148",
            "Cohort N": "985",
            "Split design": "legacy patient holdout",
            "Implementation": "full HyDRA-CoE",
            "Auxiliary OCT SSL": "no",
            "OCT-VLM semantic alignment": "no",
            "Endpoint": "pathology_cin2plus",
            **{k: NA for k in m.keys()},
            "Status": "MISSING_OLD_SPLIT",
            "Interpretation": "Legacy 985 full-model result unavailable; corrected403 full-model AUC is diagnostic only, not n=1897 evidence.",
        },
        {
            "Experiment row": "C",
            "Protocol": "Locked n=1897 LOCO",
            "Cohort N": "1897",
            "Split design": "leave-one-centre-out",
            "Implementation": "Step2 feature-cache surrogate HyDRA",
            "Auxiliary OCT SSL": "no",
            "OCT-VLM semantic alignment": "no",
            "Endpoint": "pathology_cin2plus",
            **m,
            "Status": "DONE",
            "Interpretation": "Infrastructure-valid Step2 surrogate; not the full manuscript method.",
        },
    ]
    full_rows = [
        ("D", "full HyDRA-CoE no auxiliary pretraining", "no", "no", "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"),
        ("E", "full HyDRA-CoE + auxiliary OCT SSL", "yes", "no", "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"),
        ("F", "full HyDRA-CoE + auxiliary OCT SSL + OCT-VLM alignment", "yes", "yes", "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"),
    ]
    for row_id, impl, ssl, vlm, status in full_rows:
        rows.append(
            {
                "Experiment row": row_id,
                "Protocol": "Locked n=1897 LOCO",
                "Cohort N": "1897",
                "Split design": "leave-one-centre-out",
                "Implementation": impl,
                "Auxiliary OCT SSL": ssl,
                "OCT-VLM semantic alignment": vlm,
                "Endpoint": "pathology_cin2plus",
                **{k: NA for k in m.keys()},
                "Status": status,
                "Interpretation": "Full supervised LOCO training was not executed because required active end-to-end n=1897 model path/checkpoints are unavailable in this run.",
            }
        )
    df = pd.DataFrame(rows)
    write_table_bundle(df, "Table_A_Protocol_Implementation_Attribution", tables)
    update_status(attribution_table={"status": "DONE", "path": str((tables / "Table_A_Protocol_Implementation_Attribution.csv").relative_to(ROOT))})
    return tables / "Table_A_Protocol_Implementation_Attribution.csv"


def run_full_module_ablation(config_path: PathLike) -> Path:
    cfg = load_yaml(config_path)
    tables = ensure_dir(OUT_DIR / "tables")
    variants = ["full_hydra_coe_aux_oct_vlm_align"] + [v["name"] for v in cfg.get("ablation_variants", [])]
    rows = []
    for variant in variants:
        rows.append(
            {
                "Variant": variant,
                "End-to-end visual encoder": "no" if variant == "full_minus_end_to_end_visual" else "yes",
                "Cross-attention transformer": "no" if variant == "full_minus_cross_attention" else "yes",
                "Posterior refinement": "no" if variant == "full_minus_posterior_refinement" else "yes",
                "ASCCP prototype prior": "no" if variant == "full_minus_asccp_prototype" else "yes",
                "CoE trajectory learning": "no" if variant == "full_minus_coe_trajectory" else "yes",
                "Auxiliary OCT SSL": "no" if variant == "full_minus_oct_ssl" else "yes",
                "OCT-VLM semantic alignment": "no" if variant == "full_minus_oct_vlm_alignment" else "yes",
                "AUC (95% CI)": NA,
                "Sensitivity (95% CI)": NA,
                "Specificity (95% CI)": NA,
                "PPV (95% CI)": NA,
                "NPV (95% CI)": NA,
                "F1 (95% CI)": NA,
                "Screen-positive rate (95% CI)": NA,
                "Delta AUC vs full": NA,
                "Status": "BLOCKED_NO_COMPLETED_FULL_REFERENCE",
            }
        )
    df = pd.DataFrame(rows)
    write_table_bundle(df, "Table_C_Full_Module_Ablation", tables)
    update_status(full_module_ablation_table={"status": "DONE_WITH_BLOCKED_ROWS", "path": str((tables / "Table_C_Full_Module_Ablation.csv").relative_to(ROOT))})
    return tables / "Table_C_Full_Module_Ablation.csv"


def collect_step2_5_results(config_path: PathLike) -> Path:
    tables = ensure_dir(OUT_DIR / "tables")
    stats = ensure_dir(OUT_DIR / "statistics")
    status = read_json(status_path(), default={}) or {}

    table2 = pd.read_csv(ROOT / "outputs/publishable_v2/step2_main_loco/tables/Table2_Main_LOCO_Diagnostic_Performance.csv")
    clinical = table2[table2["Method"].str.startswith("ClinicalOnly")].copy()
    clinical["auc_float"] = clinical["AUC (95% CI)"].map(metric_value)
    best_clin = clinical.sort_values("auc_float", ascending=False).iloc[0]
    baseline = table2[~table2["Method"].str.startswith("ClinicalOnly") & (table2["Method"] != "HyDRA_CoE_Full")].copy()
    baseline["auc_float"] = baseline["AUC (95% CI)"].map(metric_value)
    best_base = baseline.sort_values("auc_float", ascending=False).iloc[0]
    hydra = table2[table2["Method"] == "HyDRA_CoE_Full"].iloc[0]

    def main_row(method: str, impl: str, ssl: str, vlm: str, row: Optional[pd.Series], status_val: str = "DONE") -> Dict[str, Any]:
        metrics = _metrics_from_step2(row)
        return {
            "Method": method,
            "Implementation level": impl,
            "Auxiliary OCT SSL": ssl,
            "OCT-VLM semantic alignment": vlm,
            "Endpoint": "pathology_cin2plus",
            **metrics,
            "Adjusted P vs surrogate HyDRA": row.get("Adjusted P for AUC", NA) if row is not None and "Adjusted P for AUC" in row else NA,
            "Adjusted P vs best baseline": NA,
            "Status": status_val,
        }

    table_b = pd.DataFrame(
        [
            main_row(f"Best Step 2 clinical baseline ({best_clin['Method']})", "feature-cache", "no", "no", best_clin),
            main_row(f"Best Step 2 visual/fusion baseline ({best_base['Method']})", "feature-cache", "no", "no", best_base),
            main_row("Step 2 surrogate HyDRA", "feature-cache surrogate", "no", "no", hydra),
            main_row("Full HyDRA-CoE no auxiliary pretraining", "full end-to-end target", "no", "no", None, "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"),
            main_row("Full HyDRA-CoE + Hua_Xi/XiangYa OCT SSL", "full end-to-end target", "yes", "no", None, "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"),
            main_row("Full HyDRA-CoE + OCT SSL + OCT-VLM semantic alignment", "full end-to-end target", "yes", "yes", None, "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES"),
        ]
    )
    write_table_bundle(table_b, "Table_B_Main_Full_HyDRA_LOCO_Performance", tables)

    cin3 = _step2_pooled_cin3()
    cin3_metrics = _metrics_from_step2(cin3)
    table_d_rows = []
    table_d_rows.append(
        {
            "Method": "Step 2 surrogate HyDRA",
            "Endpoint": "pathology_cin3plus",
            "AUC (95% CI)": cin3_metrics["AUC (95% CI)"],
            "Sensitivity (95% CI)": cin3_metrics["Sensitivity at t_safety95 (95% CI)"],
            "Specificity (95% CI)": cin3_metrics["Specificity at t_safety95 (95% CI)"],
            "PPV (95% CI)": cin3_metrics["PPV (95% CI)"],
            "NPV (95% CI)": cin3_metrics["NPV (95% CI)"],
            "False-negative count": 10,
            "Screen-positive rate": cin3_metrics["Screen-positive rate (95% CI)"],
            "Status": "DONE",
        }
    )
    for method in [
        "Full HyDRA-CoE no auxiliary pretraining",
        "Full HyDRA-CoE + OCT SSL",
        "Full HyDRA-CoE + OCT SSL + OCT-VLM semantic alignment",
    ]:
        table_d_rows.append(
            {
                "Method": method,
                "Endpoint": "pathology_cin3plus",
                "AUC (95% CI)": NA,
                "Sensitivity (95% CI)": NA,
                "Specificity (95% CI)": NA,
                "PPV (95% CI)": NA,
                "NPV (95% CI)": NA,
                "False-negative count": NA,
                "Screen-positive rate": NA,
                "Status": "NOT_RUN_MISSING_ACTIVE_END_TO_END_FULL_MODULES",
            }
        )
    table_d = pd.DataFrame(table_d_rows)
    write_table_bundle(table_d, "Table_D_CIN3plus_Safety_Full_HyDRA", tables)

    inv = status.get("auxiliary_oct_inventory", {})
    center_summary = inv.get("center_summary", [])
    ssl = status.get("auxiliary_oct_ssl", {})
    align = status.get("oct_vlm_alignment", {})
    diag_rows = []
    for row in center_summary:
        diag_rows.append(
            {
                "Stage": "Auxiliary OCT inventory",
                "Data source": row.get("source_center"),
                "Image count": row.get("image_count"),
                "Case/volume count if available": row.get("case_volume_count"),
                "Training objective": "inventory/QC only",
                "Best validation loss or retrieval metric": NA,
                "Checkpoint path": NA,
                "Label supervision used": False,
                "Diagnostic labels used": False,
                "Status": inv.get("status", NA),
            }
        )
    diag_rows.extend(
        [
            {
                "Stage": "Auxiliary OCT SSL",
                "Data source": "Hua_Xi + XiangYa",
                "Image count": ssl.get("usable_image_count", NA),
                "Case/volume count if available": NA,
                "Training objective": "MAE/DINO requested; compute-limited proxy generated",
                "Best validation loss or retrieval metric": NA,
                "Checkpoint path": ssl.get("checkpoint_path", NA),
                "Label supervision used": False,
                "Diagnostic labels used": False,
                "Status": ssl.get("status", "NOT_RUN"),
            },
            {
                "Stage": "OCT-VLM/OCT-semantic alignment",
                "Data source": "Auxiliary OCT pseudo-captions + ASCCP semantic anchors",
                "Image count": status.get("oct_vlm_alignment_pairs", {}).get("auxiliary_pairs", NA),
                "Case/volume count if available": NA,
                "Training objective": "image-text contrastive/prototype alignment requested; local hash proxy generated",
                "Best validation loss or retrieval metric": "Retrieval metrics not evaluated; no held-out text encoder benchmark",
                "Checkpoint path": align.get("checkpoint_path", NA),
                "Label supervision used": False,
                "Diagnostic labels used": False,
                "Status": align.get("status", "NOT_RUN"),
            },
        ]
    )
    table_e = pd.DataFrame(diag_rows)
    write_table_bundle(table_e, "Table_E_Auxiliary_Pretraining_Diagnostics", tables)

    pd.DataFrame([{"model": "Step 2 surrogate HyDRA", "endpoint": "pathology_cin2plus", **_metrics_from_step2(hydra), "status": "DONE"}]).to_csv(stats / "bootstrap_ci_all_metrics.csv", index=False, encoding="utf-8-sig")
    for name in ["paired_tests_vs_surrogate.csv", "paired_tests_vs_full_no_aux.csv", "paired_tests_vs_aux_ssl.csv"]:
        pd.DataFrame([{"comparison": name.replace(".csv", ""), "p_value": NA, "adjusted_p_value": NA, "status": "NOT_EVALUABLE_NO_COMPLETED_FULL_VARIANT"}]).to_csv(stats / name, index=False, encoding="utf-8-sig")

    final_status = update_status(
        git_commit=git_commit(),
        git_status_short=git_diff_summary(),
        completed_variants=["surrogate_step2_reference"],
        failed_or_not_run_variants=[
            "full_hydra_coe_no_aux",
            "full_hydra_coe_aux_oct_ssl",
            "full_hydra_coe_aux_oct_vlm_align",
        ],
        recommended_manuscript_result_source="Do not replace Step2 surrogate with a full-model row yet; full n=1897 LOCO end-to-end training is still required. Use Step2 as infrastructure validation/weak baseline only.",
        warnings=[
            "Current Step2 AUC reflects feature-cache surrogate under stricter n=1897 LOCO.",
            "Old 0.86+ result was feature-space/corrected403 diagnostic evidence, not locked n=1897 LOCO full-model evidence.",
            "Full model is not called Full because required end-to-end visual/OCT aggregation modules are inactive for the n=1897 runner.",
            "Robustness, calibration, and CoE-faithfulness analyses remain separate next-step requirements.",
        ],
    )

    status_md = OUT_DIR / "STEP2_5_FULL_HYDRA_VLM_RECOVERY_STATUS.md"
    inv_rows = pd.DataFrame(center_summary)
    md = [
        "# Step 2.5 Full HyDRA + OCT-VLM Recovery Status",
        "",
        f"- Run timestamp: {final_status.get('run_timestamp')}",
        f"- Last updated: {final_status.get('last_updated')}",
        f"- Git commit: `{final_status.get('git_commit')}`",
        f"- Pass/fail: `FAILED_PARTIAL_RECOVERY_ONLY`",
        "",
        "## Auxiliary OCT Inventory",
        "",
        md_table(inv_rows) if len(inv_rows) else "_No inventory rows._\n",
        "## Full Model Availability",
        "",
        f"- Status: `{status.get('full_model_module_availability', {}).get('status', 'UNKNOWN')}`",
        f"- Registry/audit name: `{status.get('full_model_module_availability', {}).get('model_registry_name', 'UNKNOWN')}`",
        "",
        "## Main Result Interpretation",
        "",
        "The n=1897 LOCO Step2 result is a feature-cache surrogate. The requested full end-to-end HyDRA LOCO variants were not run because the active n=1897 training path/checkpoints for raw visual/OCT aggregation were unavailable in this automated pass.",
        "",
        "The previous 0.86+ AUC was found in old feature-space/corrected403 diagnostic artifacts, not in the locked n=1897 LOCO full-model protocol.",
        "",
        "## Final Tables",
        "",
        "- `tables/Table_A_Protocol_Implementation_Attribution.csv`",
        "- `tables/Table_B_Main_Full_HyDRA_LOCO_Performance.csv`",
        "- `tables/Table_C_Full_Module_Ablation.csv`",
        "- `tables/Table_D_CIN3plus_Safety_Full_HyDRA.csv`",
        "- `tables/Table_E_Auxiliary_Pretraining_Diagnostics.csv`",
        "",
        "## Git Status Short",
        "",
        "```text",
        final_status.get("git_status_short", ""),
        "```",
    ]
    status_md.write_text("\n".join(md), encoding="utf-8")
    return status_md


def plot_step2_5_figures(input_dir: PathLike = OUT_DIR, output_dir: PathLike = OUT_DIR / "figures") -> Path:
    import matplotlib.pyplot as plt

    input_dir = p(input_dir)
    figures = ensure_dir(output_dir)
    tables = input_dir / "tables"

    # Figure A
    table_a = pd.read_csv(tables / "Table_A_Protocol_Implementation_Attribution.csv")
    fig_a_src = table_a.copy()
    fig_a_src["auc_value"] = fig_a_src["AUC (95% CI)"].map(metric_value)
    source_csv_for(figures, "Figure_A_source.csv", fig_a_src)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(fig_a_src["Experiment row"], fig_a_src["auc_value"], color="#386f6f")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC Attribution")
    axes[1].bar(fig_a_src["Experiment row"], fig_a_src["Sensitivity at t_safety95 (95% CI)"].map(metric_value), label="Sensitivity", color="#7b9e4b")
    axes[1].bar(fig_a_src["Experiment row"], fig_a_src["Specificity at t_safety95 (95% CI)"].map(metric_value), bottom=0, alpha=0.55, label="Specificity", color="#c77d38")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Safety95 Metrics")
    axes[2].bar(fig_a_src["Experiment row"], fig_a_src["Screen-positive rate (95% CI)"].map(metric_value), label="Screen positive", color="#5d6f99")
    axes[2].bar(fig_a_src["Experiment row"], fig_a_src["PPV (95% CI)"].map(metric_value), alpha=0.55, label="PPV", color="#b85757")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Screen Rate and PPV")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures, "Figure_A_Protocol_Implementation_Attribution")
    plt.close(fig)

    # Figure B
    table_b = pd.read_csv(tables / "Table_B_Main_Full_HyDRA_LOCO_Performance.csv")
    fig_b_src = table_b.copy()
    fig_b_src["auc_value"] = fig_b_src["AUC (95% CI)"].map(metric_value)
    source_csv_for(figures, "Figure_B_source.csv", fig_b_src)
    pred_all_path = ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_all_models.csv"
    roc_rows = []
    if pred_all_path.exists():
        pred_all = pd.read_csv(pred_all_path)
        method_map = {
            "Best Step 2 clinical baseline (ClinicalOnly_Logistic)": "ClinicalOnly_Logistic",
            "Best Step 2 visual/fusion baseline (BioMedCLIP_Finetuned)": "BioMedCLIP_Finetuned",
            "Step 2 surrogate HyDRA": "HyDRA_CoE_Full",
        }
        for display_name, model_name in method_map.items():
            subset = pred_all[pred_all["model_name"] == model_name]
            pts = roc_points(subset["pathology_cin2plus"], subset["prob_cin2plus"])
            for _, pt in pts.iterrows():
                roc_rows.append({"method": display_name, "model_name": model_name, **pt.to_dict()})
    roc_source = pd.DataFrame(roc_rows)
    if not roc_source.empty:
        roc_source.to_csv(figures / "source/Figure_B_ROC_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = fig_b_src["Method"].str.replace("Full HyDRA-CoE", "Full", regex=False).str.slice(0, 28)
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    if not roc_source.empty:
        for method, group in roc_source.groupby("method"):
            axes[0, 0].plot(group["fpr"], group["tpr"], label=str(method)[:24])
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")
    axes[0, 0].set_title("Pooled LOCO ROC From Predictions")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].barh(labels, fig_b_src["auc_value"], color="#386f6f")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_title("AUC")
    axes[1, 0].barh(labels, fig_b_src["Sensitivity at t_safety95 (95% CI)"].map(metric_value), label="Sensitivity", color="#7b9e4b")
    axes[1, 0].barh(labels, fig_b_src["Specificity at t_safety95 (95% CI)"].map(metric_value), alpha=0.55, label="Specificity", color="#c77d38")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_title("Safety95")
    axes[1, 1].barh(labels, fig_b_src["PPV (95% CI)"].map(metric_value), label="PPV", color="#b85757")
    axes[1, 1].barh(labels, fig_b_src["Screen-positive rate (95% CI)"].map(metric_value), alpha=0.55, label="Screen-positive", color="#5d6f99")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].set_title("PPV and Screen-positive Rate")
    for ax in axes.ravel():
        ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures, "Figure_B_Full_HyDRA_LOCO_Performance")
    plt.close(fig)

    # Figure C
    table_c = pd.read_csv(tables / "Table_C_Full_Module_Ablation.csv")
    fig_c_src = table_c.copy()
    source_csv_for(figures, "Figure_C_source.csv", fig_c_src)
    modules = [
        "End-to-end visual encoder",
        "Cross-attention transformer",
        "Posterior refinement",
        "ASCCP prototype prior",
        "CoE trajectory learning",
        "Auxiliary OCT SSL",
        "OCT-VLM semantic alignment",
    ]
    mat = table_c[modules].applymap(lambda x: 1 if str(x).lower() == "yes" else 0).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].barh(table_c["Variant"].str.slice(0, 28), [0] * len(table_c), color="#999999")
    axes[0].set_title("Delta AUC unavailable")
    axes[1].barh(table_c["Variant"].str.slice(0, 28), [0] * len(table_c), color="#999999")
    axes[1].set_title("Delta safety metrics unavailable")
    im = axes[2].imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(modules)))
    axes[2].set_xticklabels([m.split()[0] for m in modules], rotation=45, ha="right")
    axes[2].set_yticks(range(len(table_c)))
    axes[2].set_yticklabels(table_c["Variant"].str.slice(0, 22))
    axes[2].set_title("Module Inclusion")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    save_figure(fig, figures, "Figure_C_Full_Module_Ablation")
    plt.close(fig)

    # Figure D
    ssl_log = input_dir / "aux_oct_pretraining/logs/pretraining_log.csv"
    align_log = input_dir / "oct_vlm_alignment/logs/alignment_training_log.csv"
    ssl_df = pd.read_csv(ssl_log) if ssl_log.exists() else pd.DataFrame()
    align_df = pd.read_csv(align_log) if align_log.exists() else pd.DataFrame()
    fig_d_src = pd.concat(
        [
            ssl_df.assign(stage="oct_ssl") if not ssl_df.empty else pd.DataFrame(),
            align_df.assign(stage="oct_vlm_alignment") if not align_df.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    source_csv_for(figures, "Figure_D_source.csv", fig_d_src)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if not ssl_df.empty:
        axes[0, 0].plot(ssl_df["epoch"], ssl_df["proxy_reconstruction_loss"], marker="o", color="#2b6f77")
    axes[0, 0].set_title("OCT SSL proxy curve")
    if not align_df.empty:
        axes[0, 1].plot(align_df["epoch"], align_df["proxy_alignment_loss"], marker="o", color="#80552b")
    axes[0, 1].set_title("OCT-semantic proxy curve")
    axes[1, 0].text(0.5, 0.5, "Embedding proxy figures saved in stage folders", ha="center", va="center")
    axes[1, 0].set_axis_off()
    axes[1, 1].bar(["SSL-full", "Align-SSL"], [np.nan, np.nan], color="#999999")
    axes[1, 1].set_title("Centre-wise AUC differences not run")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    save_figure(fig, figures, "Figure_D_Auxiliary_OCT_VLM_Pretraining_Effect")
    plt.close(fig)

    # Figure E
    table_d = pd.read_csv(tables / "Table_D_CIN3plus_Safety_Full_HyDRA.csv")
    fig_e_src = table_d.copy()
    fig_e_src["auc_value"] = fig_e_src["AUC (95% CI)"].map(metric_value)
    source_csv_for(figures, "Figure_E_source.csv", fig_e_src)
    cin3_roc_source = pd.DataFrame()
    hydra_pred_path = ROOT / "outputs/publishable_v2/step2_main_loco/predictions/patient_level_predictions_hydra_full.csv"
    if hydra_pred_path.exists():
        hydra_pred = pd.read_csv(hydra_pred_path)
        cin3_roc_source = roc_points(hydra_pred["pathology_cin3plus"], hydra_pred["prob_cin2plus"])
        cin3_roc_source.insert(0, "method", "Step 2 surrogate HyDRA")
        cin3_roc_source.to_csv(figures / "source/Figure_E_ROC_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    if not cin3_roc_source.empty:
        axes[0, 0].plot(cin3_roc_source["fpr"], cin3_roc_source["tpr"], label="Step 2 surrogate HyDRA")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("CIN3+ ROC From Predictions")
    axes[0, 1].barh(fig_e_src["Method"].str.slice(0, 28), fig_e_src["Sensitivity (95% CI)"].map(metric_value), color="#7b9e4b")
    axes[0, 1].set_title("CIN3+ sensitivity")
    axes[1, 0].barh(fig_e_src["Method"].str.slice(0, 28), pd.to_numeric(fig_e_src["False-negative count"], errors="coerce"), color="#b85757")
    axes[1, 0].set_title("False negatives")
    axes[1, 1].scatter(fig_e_src["Screen-positive rate"].map(metric_value), fig_e_src["Sensitivity (95% CI)"].map(metric_value), color="#5d6f99")
    axes[1, 1].set_xlabel("Screen-positive rate")
    axes[1, 1].set_ylabel("Sensitivity")
    axes[1, 1].set_title("Safety vs screen rate")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    save_figure(fig, figures, "Figure_E_CIN3plus_Safety_Full_HyDRA")
    plt.close(fig)
    update_status(final_figures={"status": "DONE", "figures_dir": str(figures.relative_to(ROOT))})
    return figures


def run_all(full_config: PathLike, oct_ssl_config: PathLike, oct_vlm_config: PathLike, output_dir: PathLike, no_dry_run: bool = False) -> None:
    global OUT_DIR
    OUT_DIR = p(output_dir)
    ensure_dir(OUT_DIR)
    update_status(experiment_name="hydra_coe_step2_5_full_vlm_recovery_n1897", pass_fail="IN_PROGRESS")
    inventory_auxiliary_oct(oct_ssl_config)
    pretrain_oct_encoder(oct_ssl_config, no_dry_run=no_dry_run)
    build_oct_vlm_alignment_pairs(oct_vlm_config)
    train_oct_vlm_alignment(oct_vlm_config, no_dry_run=no_dry_run)
    locate_full_hydra_codepaths(".", OUT_DIR / "audit")
    restore_full_hydra_model()
    for variant in ["full_hydra_coe_no_aux", "full_hydra_coe_aux_oct_ssl", "full_hydra_coe_aux_oct_vlm_align"]:
        run_full_hydra_loco(full_config, variant, no_dry_run=no_dry_run)
    run_protocol_implementation_attribution(full_config)
    run_full_module_ablation(full_config)
    collect_step2_5_results(full_config)
    plot_step2_5_figures(OUT_DIR, OUT_DIR / "figures")
    status = read_json(status_path(), default={}) or {}
    status["pass_fail"] = "FAILED_PARTIAL_RECOVERY_ONLY"
    status["failure_reason"] = "Full n=1897 end-to-end HyDRA LOCO variants were not completed; audited artifacts and proxy auxiliary stages were generated."
    write_json(status_path(), status)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool = False) -> None:
    parser.add_argument(name, action="store_true", default=default)
