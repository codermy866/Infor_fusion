#!/usr/bin/env python3
"""Run the formal Shared-LoRA BioCOT image-level LOCO experiment.

The smoke test only verifies wiring. This runner creates auditable splits,
optionally pretrains an OCT+clinical expert on external OCT-text pools, adapts
Shared-LoRA on four source centres, evaluates the held-out centre, and writes
tables/figures under outputs/publishable_v2/shared_lora_biocot/formal_loco.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python")
if not DEFAULT_PYTHON.exists():
    DEFAULT_PYTHON = Path(sys.executable)

CENTER_ORDER = [
    "恩施州中心医院",
    "武大人民医院",
    "襄阳市中心医院",
    "荆州市第一人民医院",
    "十堰市人民医院",
]
CENTER_TO_GROUP = {name: idx for idx, name in enumerate(CENTER_ORDER)}
PALETTE = [
    "#8b98b3",
    "#abb8cc",
    "#dbb98c",
    "#edd6b8",
    "#b57979",
    "#dea3a2",
    "#b3b0b0",
    "#d9d8d8",
]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(value), flags=re.UNICODE)
    return value.strip("_") or "fold"


def old_format_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ID": df["case_id"].astype(str),
            "patient_id": df["patient_id"].astype(str),
            "oct_id": df["case_id"].astype(str),
            "OCT": df["case_id"].astype(str),
            "age": df["age"],
            "hpv": df["hpv_status_harmonized"],
            "tct": df["tct_status_harmonized"],
            "label": df["pathology_cin2plus"].astype(int),
            "pathology_cin2plus": df["pathology_cin2plus"].astype(int),
            "pathology_cin3plus": df["pathology_cin3plus"].astype(int),
            "center_name": df["center_name"].astype(str),
            "center_group_id": df["center_name"].map(CENTER_TO_GROUP).fillna(0).astype(int),
            "oct_paths": df["oct_paths"].fillna("").astype(str),
            "col_paths": df["colposcopy_paths"].fillna("").astype(str),
            "colposcopy_paths": df["colposcopy_paths"].fillna("").astype(str),
            "oct_count": df["oct_num_bscans"].fillna(0).astype(int),
            "col_count": df["colposcopy_num_images"].fillna(0).astype(int),
            "is_positive_patient": df["pathology_cin2plus"].astype(int).eq(1),
        }
    )
    return out


def _choose_hard_validation_center(source_pool: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for center_name in sorted(source_pool["center_name"].dropna().astype(str).unique()):
        val_part = source_pool[source_pool["center_name"].astype(str).eq(center_name)]
        train_part = source_pool[~source_pool["center_name"].astype(str).eq(center_name)]
        if len(val_part) == 0 or len(train_part) == 0:
            continue
        val_rate = float(val_part["pathology_cin2plus"].mean())
        train_rate = float(train_part["pathology_cin2plus"].mean())
        has_both_classes = bool(val_part["pathology_cin2plus"].astype(int).nunique() == 2)
        support_penalty = 1.0 / max(len(val_part), 1)
        rows.append(
            {
                "candidate_val_center": center_name,
                "n_val": len(val_part),
                "n_train": len(train_part),
                "val_cin2plus_rate": val_rate,
                "train_cin2plus_rate": train_rate,
                "prevalence_gap": abs(val_rate - train_rate),
                "has_both_cin2_classes": has_both_classes,
                "score": abs(val_rate - train_rate) - 0.25 * support_penalty,
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("Cannot choose hard validation center: no valid source-center candidates.")
    selectable = candidates[candidates["has_both_cin2_classes"]].copy()
    if selectable.empty:
        selectable = candidates.copy()
    chosen = selectable.sort_values(["score", "n_val"], ascending=[False, False]).iloc[0]
    return str(chosen["candidate_val_center"]), candidates


def build_loco_splits(
    manifest_path: Path,
    data_lock_path: Path,
    split_root: Path,
    validation_policy: str = "manifest",
) -> pd.DataFrame:
    manifest = read_csv(manifest_path)
    data_lock = read_csv(data_lock_path)
    keep_cols = [
        "case_id",
        "patient_id",
        "center_id",
        "center_name",
        "age",
        "hpv_status_harmonized",
        "tct_status_harmonized",
        "pathology_cin2plus",
        "pathology_cin3plus",
        "oct_num_bscans",
        "colposcopy_num_images",
        "colposcopy_paths",
        "oct_paths",
    ]
    merged = manifest.merge(
        data_lock[keep_cols],
        on=["case_id", "patient_id", "center_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"],
        how="left",
        validate="many_to_one",
    )
    if merged["oct_paths"].isna().any():
        raise ValueError("Some split rows are missing oct_paths after data-lock merge.")

    split_root.mkdir(parents=True, exist_ok=True)
    rows = []
    role_to_file = {
        "train": "train_labels.csv",
        "validation": "val_labels.csv",
        "test": "external_test_labels.csv",
    }
    audit_rows = []
    hard_candidate_rows = []
    for fold_id, fold_df in merged.groupby("fold_id", sort=True):
        fold_dir = split_root / safe_name(fold_id)
        fold_dir.mkdir(parents=True, exist_ok=True)
        held_out = sorted(fold_df.loc[fold_df["split_role"].eq("test"), "center_name"].unique())
        hard_val_center = ""
        if validation_policy == "hard-center":
            test_df = fold_df[fold_df["split_role"].eq("test")].copy()
            source_pool = fold_df[~fold_df["split_role"].eq("test")].copy()
            hard_val_center, candidates = _choose_hard_validation_center(source_pool)
            candidates.insert(0, "fold_id", fold_id)
            candidates.insert(1, "held_out_center", ";".join(held_out))
            hard_candidate_rows.extend(candidates.to_dict("records"))
            raw_parts = {
                "train": source_pool[~source_pool["center_name"].astype(str).eq(hard_val_center)].copy(),
                "validation": source_pool[source_pool["center_name"].astype(str).eq(hard_val_center)].copy(),
                "test": test_df,
            }
        elif validation_policy == "manifest":
            raw_parts = {role: fold_df[fold_df["split_role"].eq(role)].copy() for role in role_to_file}
        else:
            raise ValueError(f"Unsupported validation_policy={validation_policy}")

        for role, file_name in role_to_file.items():
            raw_part = raw_parts[role]
            part = old_format_rows(raw_part.copy())
            out_path = fold_dir / file_name
            part.to_csv(out_path, index=False, encoding="utf-8-sig")
            rows.append(
                {
                    "fold_id": fold_id,
                    "held_out_center": ";".join(held_out),
                    "split_role": role,
                    "validation_policy": validation_policy,
                    "hard_val_center": hard_val_center,
                    "n": len(part),
                    "cin2plus_rate": float(part["pathology_cin2plus"].mean()) if len(part) else np.nan,
                    "cin3plus_rate": float(part["pathology_cin3plus"].mean()) if len(part) else np.nan,
                    "csv": str(out_path.relative_to(EXP_ROOT)),
                }
            )
            for center_name, center_df in raw_part.groupby("center_name", sort=True):
                y = center_df["pathology_cin2plus"].astype(int)
                audit_rows.append(
                    {
                        "fold_id": fold_id,
                        "held_out_center": ";".join(held_out),
                        "split_role": role,
                        "validation_policy": validation_policy,
                        "hard_val_center": hard_val_center,
                        "center_name": center_name,
                        "n": len(center_df),
                        "cin2plus_rate": float(y.mean()) if len(y) else np.nan,
                        "cin3plus_rate": float(center_df["pathology_cin3plus"].astype(int).mean()) if len(center_df) else np.nan,
                        "has_both_cin2_classes": bool(y.nunique() == 2),
                    }
                )
    manifest_out = pd.DataFrame(rows)
    manifest_out.to_csv(split_root / "formal_loco_split_manifest.csv", index=False, encoding="utf-8-sig")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(split_root / "formal_loco_split_audit.csv", index=False, encoding="utf-8-sig")
    if hard_candidate_rows:
        pd.DataFrame(hard_candidate_rows).to_csv(split_root / "hard_center_validation_candidates.csv", index=False, encoding="utf-8-sig")
    warnings = audit[~audit["has_both_cin2_classes"]].copy()
    report = [
        "# Formal LOCO Split Audit",
        "",
        f"validation_policy: `{validation_policy}`",
        "",
        "## Split Manifest",
        "",
        manifest_out.to_markdown(index=False),
        "",
        "## Per-Centre Label Support Warnings",
        "",
        warnings.to_markdown(index=False) if len(warnings) else "No single-class centre partitions detected.",
        "",
        "## Output Files",
        "",
        f"- split manifest: `{(split_root / 'formal_loco_split_manifest.csv').relative_to(EXP_ROOT)}`",
        f"- split audit: `{(split_root / 'formal_loco_split_audit.csv').relative_to(EXP_ROOT)}`",
    ]
    if hard_candidate_rows:
        report.append(f"- hard-centre candidates: `{(split_root / 'hard_center_validation_candidates.csv').relative_to(EXP_ROOT)}`")
    (split_root / "formal_loco_split_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest_out


def expand_oct_ids(raw_id: object, root: Path) -> list[Path]:
    text = str(raw_id).strip()
    if not text or text.lower() == "nan":
        return []
    candidates: list[str] = [text]
    if "/" in text:
        prefix, suffix = text.rsplit("/", 1)
        match = re.search(r"(.*?P)(\d+)$", prefix)
        if match and suffix.isdigit():
            stem, number = match.groups()
            width = len(number)
            candidates = [prefix, f"{stem}{int(suffix):0{width}d}"]
    padded = []
    for candidate in candidates:
        padded.append(candidate)
        match = re.search(r"(.*?P)(\d+)$", candidate)
        if match:
            stem, number = match.groups()
            padded.append(f"{stem}{int(number):07d}")
    paths = []
    for candidate in dict.fromkeys(padded):
        folder = root / candidate
        if folder.exists():
            paths.append(folder)
    return paths


def image_files(folder: Path) -> list[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files: list[Path] = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return [str(path) for path in sorted(files)]


def build_pretrain_splits(out_dir: Path, seed: int) -> pd.DataFrame:
    specs = [
        ("HuaXi", Path("/data2/10center_datas/HuaXi_dataset.csv"), Path("/data2/10center_datas/Hua_Xi")),
        ("XiangYa", Path("/data2/10center_datas/XiangYa_dataset.csv"), Path("/data2/10center_datas/XiangYa")),
    ]
    rows = []
    for center_name, csv_path, oct_root in specs:
        df = read_csv(csv_path)
        for _, row in df.iterrows():
            if pd.isna(row.get("Final_Label")):
                continue
            folders = expand_oct_ids(row.get("OCT_ID"), oct_root)
            paths = [file_name for folder in folders for file_name in image_files(folder)]
            if not paths:
                continue
            case_id = str(row.get("OCT_ID")).replace("/", "_")
            rows.append(
                {
                    "ID": f"{center_name}_{case_id}",
                    "patient_id": f"{center_name}_{case_id}",
                    "oct_id": f"{center_name}_{case_id}",
                    "OCT": f"{center_name}_{case_id}",
                    "age": row.get("Age"),
                    "hpv": row.get("HPV_Result"),
                    "tct": row.get("TCT_Result"),
                    "label": int(row.get("Final_Label")),
                    "pathology_cin2plus": int(row.get("Final_Label")),
                    "pathology_cin3plus": int(row.get("Final_Label")),
                    "center_name": center_name,
                    "center_group_id": 0 if center_name == "HuaXi" else 1,
                    "oct_paths": ";".join(paths),
                    "col_paths": "",
                    "colposcopy_paths": "",
                    "oct_count": len(paths),
                    "col_count": 0,
                    "is_positive_patient": bool(int(row.get("Final_Label"))),
                }
            )
    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("No Huaxi/Xiangya OCT-text pretraining rows could be constructed.")

    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for (_, label), group in all_df.groupby(["center_name", "label"], sort=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * 0.2)))
        val_indices.extend(indices[:n_val].tolist())
    val_mask = all_df.index.isin(val_indices)
    train_df = all_df.loc[~val_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = all_df.loc[val_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_labels.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val_labels.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "external_test_labels.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(train_df),
                "cin2plus_rate": float(train_df["label"].mean()),
                "csv": str((out_dir / "train_labels.csv").relative_to(EXP_ROOT)),
            },
            {
                "split": "val",
                "n": len(val_df),
                "cin2plus_rate": float(val_df["label"].mean()),
                "csv": str((out_dir / "val_labels.csv").relative_to(EXP_ROOT)),
            },
        ]
    )
    summary.to_csv(out_dir / "pretrain_split_manifest.csv", index=False, encoding="utf-8-sig")
    return summary


def write_config(
    config_path: Path,
    data_root: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    vit_batch_size: int,
    seed_note: str,
    *,
    pretrain_without_colpo: bool,
    load_expert_checkpoint: Path | None = None,
    colpo_pretrained: bool = True,
    pass_raw_oct_to_model: bool = False,
    use_hierarchical: bool = False,
    vit_pretrained: bool = False,
    vit_model_name: str = "vit_base_patch16_224",
    vit_checkpoint_path: str | None = None,
    colpo_encoder_name: str = "vit_base_patch16_224",
    colpo_encoder_checkpoint_path: str | None = None,
    use_oct_encoder_lora: bool = False,
    use_colpo_encoder_lora: bool = False,
    use_fusion_layer_lora: bool = False,
    encoder_lora_rank: int = 8,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    use_hierarchical = bool(use_hierarchical or pass_raw_oct_to_model)
    load_line = (
        f"    load_expert_base_checkpoint_path: str = r\"{load_expert_checkpoint}\"\n"
        if load_expert_checkpoint
        else "    load_expert_base_checkpoint_path = None\n"
    )
    vit_ckpt_line = (
        f"    vit_checkpoint_path: str = r\"{vit_checkpoint_path}\"\n"
        if vit_checkpoint_path
        else "    vit_checkpoint_path = None\n"
    )
    colpo_ckpt_line = (
        f"    colpo_encoder_checkpoint_path: str = r\"{colpo_encoder_checkpoint_path}\"\n"
        if colpo_encoder_checkpoint_path
        else "    colpo_encoder_checkpoint_path = None\n"
    )
    config_path.write_text(
        f'''#!/usr/bin/env python3
from dataclasses import dataclass
from configs.shared_lora_loco_template import SharedLoRALOCOConfig


@dataclass
class SharedLoRAFormalConfig(SharedLoRALOCOConfig):
    """Auto-generated formal Shared-LoRA config: {seed_note}."""

    data_root: str = r"{data_root}"
    output_dir: str = r"{out_dir}"
    checkpoint_dir: str = r"{out_dir / "checkpoints"}"
    log_dir: str = r"{out_dir / "logs"}"

    num_epochs: int = {int(epochs)}
    batch_size: int = {int(batch_size)}
    num_workers: int = {int(num_workers)}
    vit_batch_size: int = {int(vit_batch_size)}
    oct_frames: int = 8
    colposcopy_images: int = 3
    use_hierarchical: bool = {bool(use_hierarchical)}
    pass_raw_oct_to_model: bool = {bool(pass_raw_oct_to_model)}
    vit_model_name: str = "{vit_model_name}"
    vit_pretrained: bool = {bool(vit_pretrained)}
{vit_ckpt_line}    raw_oct_encoder_batch_size: int = {int(vit_batch_size)}
    center_balanced_sampling: bool = True
    checkpoint_metric: str = "auc_minus_ece"
    ece_penalty: float = 0.15
    calibration_bins: int = 10

    pretrain_without_colpo: bool = {bool(pretrain_without_colpo)}
    pass_raw_colpo_to_model: bool = {not bool(pretrain_without_colpo)}
    enable_colpo_encoder: bool = {not bool(pretrain_without_colpo)}
    colpo_encoder_name: str = "{colpo_encoder_name}"
    colpo_encoder_pretrained: bool = {bool(colpo_pretrained)}
{colpo_ckpt_line}    use_oct_encoder_lora: bool = {bool(use_oct_encoder_lora)}
    use_colpo_encoder_lora: bool = {bool(use_colpo_encoder_lora)}
    use_fusion_layer_lora: bool = {bool(use_fusion_layer_lora)}
    encoder_lora_rank: int = {int(encoder_lora_rank)}
    encoder_lora_alpha: float = {float(encoder_lora_rank) * 2.0}
    encoder_lora_dropout: float = 0.05
    encoder_lora_targets: tuple = ("attn.qkv", "attn.proj")
    train_colpo_encoder: bool = False
    freeze_expert_base_for_lora: bool = {not bool(pretrain_without_colpo)}
    freeze_colpo_encoder_for_lora: bool = True
{load_line}    use_colpo_lora_bridge: bool = {not bool(pretrain_without_colpo)}
    shared_lora_rank: int = 8
    shared_lora_alpha: float = 16.0
    shared_lora_dropout: float = 0.05
    lambda_colpo_bridge_ot: float = 0.2
    lambda_colpo_bridge_align: float = 0.05
    use_adversarial: bool = False
    lambda_adv: float = 0.0
''',
        encoding="utf-8",
    )


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(EXP_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)


def newest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("best_model_v3_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No best_model_v3_*.pth found under {checkpoint_dir}")
    return candidates[0]


def threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    if candidates.size == 0:
        return 0.5
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        pred = y_prob >= thr
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def threshold_for_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target: float = 0.95) -> float:
    positives = y_prob[y_true == 1]
    if positives.size == 0:
        return 0.5
    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    feasible = []
    for thr in candidates:
        pred = y_prob >= thr
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        sens = tp / (tp + fn) if tp + fn > 0 else 0.0
        if sens >= target:
            feasible.append(float(thr))
    if feasible:
        return max(feasible)
    return float(np.min(positives))


def target_free_logit_median_match(source_prob: Iterable[float], target_prob: Iterable[float]) -> tuple[np.ndarray, float]:
    """Unlabeled target calibration: match target median logit to source validation median logit."""
    source_arr = np.clip(np.asarray(list(source_prob), dtype=float), 1e-6, 1.0 - 1e-6)
    target_arr = np.clip(np.asarray(list(target_prob), dtype=float), 1e-6, 1.0 - 1e-6)
    if source_arr.size == 0 or target_arr.size == 0:
        return target_arr, 0.0
    source_logit = np.log(source_arr / (1.0 - source_arr))
    target_logit = np.log(target_arr / (1.0 - target_arr))
    shift = float(np.median(source_logit) - np.median(target_logit))
    calibrated = 1.0 / (1.0 + np.exp(-(target_logit + shift)))
    return calibrated, shift


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    pred = y_prob_arr >= float(threshold)
    tp = float(np.sum((pred == 1) & (y_true_arr == 1)))
    tn = float(np.sum((pred == 0) & (y_true_arr == 0)))
    fp = float(np.sum((pred == 1) & (y_true_arr == 0)))
    fn = float(np.sum((pred == 0) & (y_true_arr == 1)))
    auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if len(np.unique(y_true_arr)) > 1 else np.nan
    ap = float(average_precision_score(y_true_arr, y_prob_arr)) if len(np.unique(y_true_arr)) > 1 else np.nan
    return {
        "auc": auc,
        "average_precision": ap,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1.0)),
        "sensitivity": float(tp / max(tp + fn, 1.0)),
        "specificity": float(tn / max(tn + fp, 1.0)),
        "ppv": float(tp / max(tp + fp, 1.0)),
        "npv": float(tn / max(tn + fn, 1.0)),
        "false_negative_count": int(fn),
        "false_positive_count": int(fp),
        "threshold": float(threshold),
    }


def aggregate_results(out_root: Path, fold_rows: list[dict[str, object]]) -> None:
    pred_dir = out_root / "predictions"
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    report_dir = out_root / "reports"
    for path in [table_dir, fig_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    patient_predictions = []
    for fold in fold_rows:
        fold_id = str(fold["fold_id"])
        run_id = safe_name(fold_id)
        val_path = pred_dir / f"SharedLoRA_BioCOT_run{run_id}_seed{fold['seed']}_val_full.csv"
        test_path = pred_dir / f"SharedLoRA_BioCOT_run{run_id}_seed{fold['seed']}_external_test_full.csv"
        if not val_path.exists() or not test_path.exists():
            continue
        val = read_csv(val_path)
        test = read_csv(test_path)
        test_split = read_csv(Path(fold["test_csv"]))
        labels = test_split[["ID", "pathology_cin3plus", "center_name"]].rename(columns={"ID": "case_id"})
        test = test.merge(labels, on="case_id", how="left", suffixes=("", "_split"))
        val_split = read_csv(Path(fold["val_csv"]))
        val_labels = val_split[["ID", "pathology_cin3plus"]].rename(columns={"ID": "case_id"})
        val = val.merge(val_labels, on="case_id", how="left")

        thr_cin2 = threshold_max_f1(val["y_true"].to_numpy(), val["y_prob"].to_numpy())
        thr_cin3 = threshold_for_sensitivity(
            val["pathology_cin3plus"].fillna(0).astype(int).to_numpy(),
            val["y_prob"].to_numpy(),
            target=0.95,
        )
        test["y_prob_target_free_calibrated"], target_free_shift = target_free_logit_median_match(
            val["y_prob"].to_numpy(),
            test["y_prob"].to_numpy(),
        )
        cin2 = binary_metrics(test["y_true"], test["y_prob"], thr_cin2)
        cin3 = binary_metrics(test["pathology_cin3plus"].fillna(0).astype(int), test["y_prob"], thr_cin3)
        cin2_tfc = binary_metrics(test["y_true"], test["y_prob_target_free_calibrated"], thr_cin2)
        cin3_tfc = binary_metrics(
            test["pathology_cin3plus"].fillna(0).astype(int),
            test["y_prob_target_free_calibrated"],
            thr_cin3,
        )
        row = {
            "fold_id": fold_id,
            "held_out_center": fold["held_out_center"],
            "n_test": len(test),
            "cin2plus_prevalence": float(test["y_true"].mean()),
            "cin3plus_prevalence": float(test["pathology_cin3plus"].fillna(0).mean()),
            "target_free_logit_shift": target_free_shift,
            **{f"cin2plus_{k}": v for k, v in cin2.items()},
            **{f"cin3plus_{k}": v for k, v in cin3.items()},
            **{f"cin2plus_tfc_{k}": v for k, v in cin2_tfc.items()},
            **{f"cin3plus_tfc_{k}": v for k, v in cin3_tfc.items()},
            "checkpoint": fold["checkpoint"],
            "train_log": fold["train_log"],
            "val_predictions": str(val_path.relative_to(EXP_ROOT)),
            "test_predictions": str(test_path.relative_to(EXP_ROOT)),
        }
        rows.append(row)
        test["fold_id"] = fold_id
        test["held_out_center"] = fold["held_out_center"]
        test["cin2_threshold_from_val"] = thr_cin2
        test["cin3_safety_threshold_from_val"] = thr_cin3
        patient_predictions.append(test)

    if not rows:
        raise RuntimeError("No fold predictions found to aggregate.")

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(table_dir / "Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    if patient_predictions:
        pd.concat(patient_predictions, ignore_index=True).to_csv(
            pred_dir / "SharedLoRA_Formal_LOCO_All_Patient_Predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_rows = []
    metric_cols = [c for c in fold_metrics.columns if c.startswith(("cin2plus_", "cin3plus_")) and pd.api.types.is_numeric_dtype(fold_metrics[c])]
    for metric in metric_cols:
        values = fold_metrics[metric].dropna()
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "Table_SharedLoRA_Formal_LOCO_Aggregate_Metrics.csv", index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", palette=PALETTE, font="DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plot_df = fold_metrics.sort_values("cin2plus_auc")
    sns.pointplot(data=plot_df, x="cin2plus_auc", y="held_out_center", ax=ax, color=PALETTE[0], join=False, scale=0.9)
    ax.axvline(plot_df["cin2plus_auc"].mean(), color=PALETTE[4], linestyle="--", linewidth=1.4)
    ax.set_xlabel("Held-out CIN2+ AUROC")
    ax.set_ylabel("Held-out centre")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_SharedLoRA_Formal_LOCO_CIN2_AUROC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=plot_df, x="cin3plus_sensitivity", y="held_out_center", ax=ax, color=PALETTE[2])
    sns.stripplot(data=plot_df, x="cin3plus_sensitivity", y="held_out_center", ax=ax, color=PALETTE[4], size=6)
    ax.set_xlabel("CIN3+ sensitivity at validation-selected safety threshold")
    ax.set_ylabel("Held-out centre")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_SharedLoRA_Formal_LOCO_CIN3_Safety.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Shared-LoRA BioCOT Formal LOCO Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Main Output Files",
        "",
        f"- Fold metrics: `{(table_dir / 'Table_SharedLoRA_Formal_LOCO_Fold_Metrics.csv').relative_to(EXP_ROOT)}`",
        f"- Aggregate metrics: `{(table_dir / 'Table_SharedLoRA_Formal_LOCO_Aggregate_Metrics.csv').relative_to(EXP_ROOT)}`",
        f"- Patient predictions: `{(pred_dir / 'SharedLoRA_Formal_LOCO_All_Patient_Predictions.csv').relative_to(EXP_ROOT)}`",
        f"- CIN2+ AUROC figure: `{(fig_dir / 'Figure_SharedLoRA_Formal_LOCO_CIN2_AUROC.png').relative_to(EXP_ROOT)}`",
        f"- CIN3+ safety figure: `{(fig_dir / 'Figure_SharedLoRA_Formal_LOCO_CIN3_Safety.png').relative_to(EXP_ROOT)}`",
        "",
        "## Fold Metrics",
        "",
        fold_metrics.to_markdown(index=False),
        "",
        "## Aggregate Metrics",
        "",
        summary.to_markdown(index=False),
    ]
    (report_dir / "SharedLoRA_Formal_LOCO_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


@dataclass
class RunState:
    fold_id: str
    held_out_center: str
    seed: int
    config: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    checkpoint: Path
    train_log: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/formal_loco"))
    parser.add_argument("--manifest", default=str(EXP_ROOT / "outputs/publishable_v2/splits/split_manifest_v2.csv"))
    parser.add_argument("--data-lock", default=str(EXP_ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv"))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--loco-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vit-batch-size", type=int, default=8)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--expert-checkpoint", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Build/audit splits and exit before training.")
    parser.add_argument("--fold", action="append", default=None, help="Optional fold_id subset.")
    parser.add_argument("--colpo-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-policy", choices=["manifest", "hard-center"], default="manifest")
    parser.add_argument("--pass-raw-oct-to-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-hierarchical", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vit-pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vit-model-name", default="vit_base_patch16_224")
    parser.add_argument("--vit-checkpoint-path", default=None)
    parser.add_argument("--colpo-encoder-name", default=None)
    parser.add_argument("--colpo-encoder-checkpoint-path", default=None)
    parser.add_argument("--use-oct-encoder-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-colpo-encoder-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-fusion-layer-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-lora-rank", type=int, default=8)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = EXP_ROOT / out_root
    split_root = out_root / "splits"
    config_root = out_root / "configs"
    logs_root = out_root / "logs"
    pred_root = out_root / "predictions"
    for path in [split_root, config_root, logs_root, pred_root]:
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("Building formal LOCO splits...")
    split_manifest = build_loco_splits(
        Path(args.manifest),
        Path(args.data_lock),
        split_root / "loco",
        validation_policy=args.validation_policy,
    )
    print(split_manifest.to_string(index=False))
    if args.prepare_only:
        print(f"Prepared splits and audit files under {split_root / 'loco'}")
        return

    expert_checkpoint = Path(args.expert_checkpoint) if args.expert_checkpoint else None
    if not args.skip_pretrain and expert_checkpoint is None:
        pretrain_split_dir = split_root / "huaxi_xiangya_pretrain"
        print("Building Huaxi/Xiangya OCT+clinical pretraining split...")
        print(build_pretrain_splits(pretrain_split_dir, seed=args.seed).to_string(index=False))
        pretrain_out = out_root / "pretrain_oct_text_expert"
        pretrain_config = config_root / "shared_lora_pretrain_oct_text_config.py"
        write_config(
            pretrain_config,
            pretrain_split_dir,
            pretrain_out,
            args.pretrain_epochs,
            args.batch_size,
            args.num_workers,
            args.vit_batch_size,
            "Huaxi/Xiangya OCT+clinical expert pretraining",
            pretrain_without_colpo=True,
            load_expert_checkpoint=None,
            colpo_pretrained=False,
            pass_raw_oct_to_model=args.pass_raw_oct_to_model,
            use_hierarchical=args.use_hierarchical,
            vit_pretrained=args.vit_pretrained,
            vit_model_name=args.vit_model_name,
            vit_checkpoint_path=args.vit_checkpoint_path,
            colpo_encoder_name=args.colpo_encoder_name or args.vit_model_name,
            colpo_encoder_checkpoint_path=None,
            use_oct_encoder_lora=args.use_oct_encoder_lora,
            use_colpo_encoder_lora=False,
            use_fusion_layer_lora=args.use_fusion_layer_lora,
            encoder_lora_rank=args.encoder_lora_rank,
        )
        if not args.skip_train:
            run_command(
                [
                    str(args.python),
                    "training/train_bio_cot_v3.2.py",
                    "--config",
                    str(pretrain_config),
                    "--train-csv",
                    str(pretrain_split_dir / "train_labels.csv"),
                    "--val-csv",
                    str(pretrain_split_dir / "val_labels.csv"),
                    "--seed",
                    str(args.seed),
                ],
                logs_root / "pretrain_oct_text_expert.log",
                env,
            )
        expert_checkpoint = newest_checkpoint(pretrain_out / "checkpoints")
        print(f"Expert checkpoint: {expert_checkpoint}")

    fold_states: list[RunState] = []
    fold_filter = set(args.fold or [])
    for fold_id in sorted(split_manifest["fold_id"].unique()):
        if fold_filter and fold_id not in fold_filter and safe_name(fold_id) not in fold_filter:
            continue
        rows = split_manifest[split_manifest["fold_id"].eq(fold_id)]
        held_out_center = str(rows["held_out_center"].iloc[0])
        fold_safe = safe_name(fold_id)
        data_root = split_root / "loco" / fold_safe
        fold_out = out_root / "folds" / fold_safe
        fold_config = config_root / f"{fold_safe}_config.py"
        write_config(
            fold_config,
            data_root,
            fold_out,
            args.loco_epochs,
            args.batch_size,
            args.num_workers,
            args.vit_batch_size,
            f"LOCO adaptation {fold_id}",
            pretrain_without_colpo=False,
            load_expert_checkpoint=expert_checkpoint,
            colpo_pretrained=args.colpo_pretrained,
            pass_raw_oct_to_model=args.pass_raw_oct_to_model,
            use_hierarchical=args.use_hierarchical,
            vit_pretrained=args.vit_pretrained,
            vit_model_name=args.vit_model_name,
            vit_checkpoint_path=args.vit_checkpoint_path,
            colpo_encoder_name=args.colpo_encoder_name or args.vit_model_name,
            colpo_encoder_checkpoint_path=args.colpo_encoder_checkpoint_path,
            use_oct_encoder_lora=args.use_oct_encoder_lora,
            use_colpo_encoder_lora=args.use_colpo_encoder_lora,
            use_fusion_layer_lora=args.use_fusion_layer_lora,
            encoder_lora_rank=args.encoder_lora_rank,
        )
        train_log = logs_root / f"{fold_safe}_train.log"
        if not args.skip_train:
            run_command(
                [
                    str(args.python),
                    "training/train_bio_cot_v3.2.py",
                    "--config",
                    str(fold_config),
                    "--train-csv",
                    str(data_root / "train_labels.csv"),
                    "--val-csv",
                    str(data_root / "val_labels.csv"),
                    "--seed",
                    str(args.seed),
                ],
                train_log,
                env,
            )
        checkpoint = newest_checkpoint(fold_out / "checkpoints")
        fold_states.append(
            RunState(
                fold_id=fold_id,
                held_out_center=held_out_center,
                seed=args.seed,
                config=fold_config,
                train_csv=data_root / "train_labels.csv",
                val_csv=data_root / "val_labels.csv",
                test_csv=data_root / "external_test_labels.csv",
                checkpoint=checkpoint,
                train_log=train_log,
            )
        )

        if not args.skip_eval:
            for split_name, csv_path in [("val", data_root / "val_labels.csv"), ("external_test", data_root / "external_test_labels.csv")]:
                run_command(
                    [
                        str(args.python),
                        "paper_revision/scripts/evaluate_checkpoint_predictions.py",
                        "--config",
                        str(fold_config),
                        "--checkpoint",
                        str(checkpoint),
                        "--csv",
                        str(csv_path),
                        "--split",
                        split_name,
                        "--method",
                        "SharedLoRA_BioCOT",
                        "--run_id",
                        fold_safe,
                        "--seed",
                        str(args.seed),
                        "--output-dir",
                        str(pred_root),
                        "--batch-size",
                        str(args.batch_size),
                    ],
                    logs_root / f"{fold_safe}_eval_{split_name}.log",
                    env,
                )

    state_rows = [
        {
            "fold_id": state.fold_id,
            "held_out_center": state.held_out_center,
            "seed": state.seed,
            "config": str(state.config),
            "train_csv": str(state.train_csv),
            "val_csv": str(state.val_csv),
            "test_csv": str(state.test_csv),
            "checkpoint": str(state.checkpoint),
            "train_log": str(state.train_log),
        }
        for state in fold_states
    ]
    (out_root / "formal_loco_run_state.json").write_text(json.dumps(state_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_results(out_root, state_rows)
    print(f"Formal LOCO outputs written under {out_root}")


if __name__ == "__main__":
    main()
