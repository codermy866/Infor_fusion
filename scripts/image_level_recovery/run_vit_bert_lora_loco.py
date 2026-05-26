#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import timm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/publishable_v2/vit_bert_lora_loco"
DATA_LOCK = ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv"
SPLIT_MANIFEST = ROOT / "outputs/publishable_v2/splits/split_manifest_v2.csv"

FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
PRED_DIR = OUT / "predictions"
CKPT_DIR = OUT / "checkpoints"
AUDIT_DIR = OUT / "audit"
LOG_DIR = OUT / "logs"

VIT_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
BERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
SCI = ["#8b98b3", "#abb8cc", "#dbb98c", "#edd6b8", "#b57979", "#dea3a2", "#b3b0b0", "#d9d8d8"]


@dataclass
class FoldResult:
    fold_id: str
    held_out_center: str
    inner_val_center: str
    best_epoch: int
    best_inner_auc: float
    threshold_cin3_safety95: float


def ensure_dirs() -> None:
    for path in [OUT, FIG_DIR, TABLE_DIR, PRED_DIR, CKPT_DIR, AUDIT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def gpu_snapshot(stage: str, run_tag: str) -> None:
    path = AUDIT_DIR / f"gpu_telemetry_{run_tag}.csv"
    if not path.exists():
        path.write_text(
            "time,stage,torch_cuda_available,torch_device_count,gpu_index,gpu_name,"
            "memory_used_mib,memory_total_mib,util_gpu_pct,util_mem_pct,"
            "torch_allocated_mib,torch_reserved_mib\n",
            encoding="utf-8",
        )
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cuda_ok = torch.cuda.is_available()
    ndev = torch.cuda.device_count() if cuda_ok else 0
    allocated = torch.cuda.memory_allocated(0) / 1024**2 if cuda_ok and ndev else 0.0
    reserved = torch.cuda.memory_reserved(0) / 1024**2 if cuda_ok and ndev else 0.0
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        lines = []
    with path.open("a", encoding="utf-8") as f:
        if not lines:
            f.write(f"{now},{stage},{cuda_ok},{ndev},NA,NA,NA,NA,NA,NA,{allocated:.1f},{reserved:.1f}\n")
        for line in lines:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 6:
                f.write(
                    f"{now},{stage},{cuda_ok},{ndev},{parts[0]},{parts[1]},"
                    f"{parts[2]},{parts[3]},{parts[4]},{parts[5]},{allocated:.1f},{reserved:.1f}\n"
                )


def split_paths(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [x for x in str(value).split(";") if x and os.path.exists(x)]


def pick_path(paths: list[str], train: bool, eval_view_index: int = 0, eval_views: int = 1) -> str | None:
    if not paths:
        return None
    if train:
        return random.choice(paths)
    if eval_views <= 1 or len(paths) == 1:
        return paths[len(paths) // 2]
    idx = int(round(eval_view_index * (len(paths) - 1) / max(eval_views - 1, 1)))
    return paths[max(0, min(idx, len(paths) - 1))]


def read_image(path: str | None) -> Image.Image:
    if path is None:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    return text if text else "missing"


def clinical_prompt(row: pd.Series) -> str:
    return (
        f"Age {clean_value(row.get('age'))}. "
        f"HPV status {clean_value(row.get('hpv_status_harmonized'))}. "
        f"HPV16 or HPV18 status {clean_value(row.get('hpv16_18_status'))}. "
        f"Cytology TCT status {clean_value(row.get('tct_status_harmonized'))}."
    )


class CervixImageTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Any, train: bool, eval_view_index: int = 0, eval_views: int = 1):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.train = train
        self.eval_view_index = int(eval_view_index)
        self.eval_views = int(eval_views)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        col = self.transform(
            read_image(pick_path(split_paths(row.get("colposcopy_paths", "")), self.train, self.eval_view_index, self.eval_views))
        )
        oct_img = self.transform(
            read_image(pick_path(split_paths(row.get("oct_paths", "")), self.train, self.eval_view_index, self.eval_views))
        )
        return {
            "col_image": col,
            "oct_image": oct_img,
            "text": clinical_prompt(row),
            "y2": int(row["pathology_cin2plus"]),
            "y3": int(row["pathology_cin3plus"]),
            "case_id": str(row["case_id"]),
            "patient_id": str(row["patient_id"]),
            "center_name": str(row["center_name"]),
        }


def collate_fn(batch: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    texts = [b["text"] for b in batch]
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
    return {
        "col_image": torch.stack([b["col_image"] for b in batch]),
        "oct_image": torch.stack([b["oct_image"] for b in batch]),
        "text_inputs": tokenized,
        "y2": torch.tensor([b["y2"] for b in batch], dtype=torch.float32),
        "y3": torch.tensor([b["y3"] for b in batch], dtype=torch.long),
        "case_id": [b["case_id"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
        "center_name": [b["center_name"] for b in batch],
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def inject_vit_lora(vit: nn.Module, rank: int, alpha: float, dropout: float, last_n: int) -> list[str]:
    freeze_all(vit)
    changed: list[str] = []
    blocks = getattr(vit, "blocks")
    start = max(0, len(blocks) - last_n)
    for i in range(start, len(blocks)):
        block = blocks[i]
        if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha, dropout)
            changed.append(f"vit.blocks.{i}.attn.qkv")
        if hasattr(block.attn, "proj") and isinstance(block.attn.proj, nn.Linear):
            block.attn.proj = LoRALinear(block.attn.proj, rank, alpha, dropout)
            changed.append(f"vit.blocks.{i}.attn.proj")
    return changed


def inject_bert_lora(bert: nn.Module, rank: int, alpha: float, dropout: float, last_n: int) -> list[str]:
    freeze_all(bert)
    changed: list[str] = []
    layers = bert.encoder.layer
    start = max(0, len(layers) - last_n)
    for i in range(start, len(layers)):
        attn = layers[i].attention.self
        for name in ["query", "value"]:
            base = getattr(attn, name)
            if isinstance(base, nn.Linear):
                setattr(attn, name, LoRALinear(base, rank, alpha, dropout))
                changed.append(f"bert.encoder.layer.{i}.attention.self.{name}")
    return changed


class VitBertLoraClassifier(nn.Module):
    def __init__(self, rank: int, alpha: float, dropout: float, vit_last_n: int, bert_last_n: int):
        super().__init__()
        self.vit = timm.create_model(VIT_MODEL, pretrained=True, num_classes=0)
        self.bert = AutoModel.from_pretrained(BERT_MODEL, local_files_only=True, use_safetensors=True)
        self.vit_lora_modules = inject_vit_lora(self.vit, rank, alpha, dropout, vit_last_n)
        self.bert_lora_modules = inject_bert_lora(self.bert, rank, alpha, dropout, bert_last_n)
        vdim = int(getattr(self.vit, "num_features", 768))
        tdim = int(self.bert.config.hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(vdim * 2 + tdim),
            nn.Linear(vdim * 2 + tdim, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 1),
        )

    def forward(self, col: torch.Tensor, oct_img: torch.Tensor, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        col_feat = self.vit(col)
        oct_feat = self.vit(oct_img)
        text_out = self.bert(**text_inputs)
        text_feat = text_out.pooler_output if getattr(text_out, "pooler_output", None) is not None else text_out.last_hidden_state[:, 0, :]
        feat = torch.cat([col_feat, oct_feat, text_feat], dim=1)
        return self.head(feat).squeeze(1)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable}


def make_transform() -> Any:
    tmp = timm.create_model(VIT_MODEL, pretrained=False, num_classes=0)
    return create_transform(**resolve_model_data_config(tmp))


def choose_inner_val_center(source: pd.DataFrame) -> str:
    rows = []
    for center, g in source.groupby("center_name"):
        if g["pathology_cin2plus"].nunique() < 2:
            continue
        rows.append((center, int(g["pathology_cin3plus"].sum()), int(g["pathology_cin2plus"].sum()), len(g)))
    if not rows:
        return str(source["center_name"].value_counts().idxmax())
    rows.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return rows[0][0]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, p))


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> float:
    if (y == 1).sum() == 0:
        return float(np.quantile(p, 0.5))
    for thr in sorted(np.unique(p), reverse=True):
        if ((p >= thr) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target:
            return float(thr)
    return float(np.min(p) - 1e-8)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else math.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else math.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else math.nan,
        "npv": tn / (tn + fn) if (tn + fn) else math.nan,
        "screen_positive_rate": float(pred.mean()),
        "fn": fn,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    col = batch["col_image"].to(device, non_blocking=True)
    oct_img = batch["oct_image"].to(device, non_blocking=True)
    text = {k: v.to(device, non_blocking=True) for k, v in batch["text_inputs"].items()}
    y = batch["y2"].to(device, non_blocking=True)
    return col, oct_img, text, y


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            col, oct_img, text, _ = move_batch(batch, device)
            logits = model(col, oct_img, text)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for i, prob in enumerate(probs):
                rows.append(
                    {
                        "case_id": batch["case_id"][i],
                        "patient_id": batch["patient_id"][i],
                        "center_name": batch["center_name"][i],
                        "pathology_cin2plus": int(batch["y2"][i].item()),
                        "pathology_cin3plus": int(batch["y3"][i].item()),
                        "prob_cin2plus": float(prob),
                    }
                )
    return pd.DataFrame(rows)


def train_one_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    transform: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    run_tag: str,
    fold_id: str,
    seed: int,
    final_epochs: int | None = None,
) -> tuple[VitBertLoraClassifier, pd.DataFrame]:
    set_seed(seed)
    model = VitBertLoraClassifier(args.lora_rank, args.lora_alpha, args.dropout, args.vit_lora_last_n, args.bert_lora_last_n).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    pos = max(int(train_df["pathology_cin2plus"].sum()), 1)
    neg = max(len(train_df) - pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    train_loader = DataLoader(
        CervixImageTextDataset(train_df, transform, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=False,
    )
    val_loader = DataLoader(
        CervixImageTextDataset(val_df, transform, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    n_epochs = int(final_epochs or args.epochs)
    logs: list[dict[str, Any]] = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        gpu_snapshot(f"{fold_id}_epoch{epoch}_start", run_tag)
        for batch in train_loader:
            col, oct_img, text, y = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(col, oct_img, text)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_pred = run_eval(model, val_loader, device)
        val_auc = safe_auc(val_pred["pathology_cin2plus"].to_numpy(int), val_pred["prob_cin2plus"].to_numpy(float))
        logs.append(
            {
                "fold_id": fold_id,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else math.nan,
                "inner_val_auc": val_auc,
                "elapsed_sec": time.time() - t0,
                "n_train": len(train_df),
                "n_val": len(val_df),
            }
        )
        gpu_snapshot(f"{fold_id}_epoch{epoch}_end", run_tag)
    return model, pd.DataFrame(logs)


def run_fold(fold_id: str, fold_df: pd.DataFrame, data: pd.DataFrame, transform: Any, tokenizer: Any, args: argparse.Namespace, device: torch.device, run_tag: str) -> tuple[pd.DataFrame, FoldResult]:
    test_cases = fold_df[fold_df["split_role"] == "test"]["case_id"].astype(str)
    source_cases = fold_df[fold_df["split_role"] != "test"]["case_id"].astype(str)
    source = data[data["case_id"].astype(str).isin(source_cases)].copy()
    test = data[data["case_id"].astype(str).isin(test_cases)].copy()
    held_out = str(test["center_name"].iloc[0])
    inner_center = choose_inner_val_center(source)
    inner_train = source[source["center_name"] != inner_center].copy()
    inner_val = source[source["center_name"] == inner_center].copy()
    if args.max_train_cases and len(inner_train) > args.max_train_cases:
        inner_train = inner_train.sample(args.max_train_cases, random_state=args.seed)
    if args.max_val_cases and len(inner_val) > args.max_val_cases:
        inner_val = inner_val.sample(args.max_val_cases, random_state=args.seed)

    print(f"[fold] {fold_id} held_out={held_out} inner_val={inner_center} train={len(inner_train)} val={len(inner_val)}", flush=True)
    model, logs = train_one_model(inner_train, inner_val, transform, tokenizer, args, device, run_tag, fold_id, args.seed)
    best_idx = logs["inner_val_auc"].astype(float).fillna(-1).idxmax()
    best_epoch = int(logs.loc[best_idx, "epoch"])
    best_auc = float(logs.loc[best_idx, "inner_val_auc"])
    # Quick but auditable final model: retrain on all source for the selected epoch.
    final_source = source.copy()
    if args.max_train_cases and len(final_source) > args.max_train_cases:
        final_source = final_source.sample(args.max_train_cases, random_state=args.seed + 17)
    final_model, final_logs = train_one_model(final_source, inner_val, transform, tokenizer, args, device, run_tag, fold_id, args.seed + 100, final_epochs=best_epoch)
    final_logs["stage"] = "final_source_retrain"
    logs["stage"] = "inner_selection"
    all_logs = pd.concat([logs, final_logs], ignore_index=True)
    fold_dir = CKPT_DIR / run_tag / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    all_logs.to_csv(fold_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    torch.save(trainable_state_dict(final_model), fold_dir / "trainable_lora_state.pt")
    config = {
        "fold_id": fold_id,
        "held_out_center": held_out,
        "inner_val_center": inner_center,
        "vit_model": VIT_MODEL,
        "bert_model": BERT_MODEL,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "vit_lora_last_n": args.vit_lora_last_n,
        "bert_lora_last_n": args.bert_lora_last_n,
        "vit_lora_modules": final_model.vit_lora_modules,
        "bert_lora_modules": final_model.bert_lora_modules,
        "best_epoch": best_epoch,
        "best_inner_auc": best_auc,
        "target_labels_used_for_training": False,
        "checkpoint_type": "trainable_lora_state_only",
    }
    (fold_dir / "lora_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    val_loader = DataLoader(
        CervixImageTextDataset(inner_val, transform, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    test_loader = DataLoader(
        CervixImageTextDataset(test, transform, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_pred = run_eval(final_model, val_loader, device)
    test_pred = run_eval(final_model, test_loader, device)
    threshold = threshold_for_sensitivity(val_pred["pathology_cin3plus"].to_numpy(int), val_pred["prob_cin2plus"].to_numpy(float), 0.95)
    test_pred["fold_id"] = fold_id
    test_pred["held_out_center"] = held_out
    test_pred["inner_val_center"] = inner_center
    test_pred["model"] = "ViT-PubMedBERT-LoRA"
    test_pred["threshold_cin3_safety95"] = threshold
    test_pred["pred_cin3_safety95"] = (test_pred["prob_cin2plus"] >= threshold).astype(int)
    return test_pred, FoldResult(fold_id, held_out, inner_center, best_epoch, best_auc, threshold)


def summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y2 = pred["pathology_cin2plus"].to_numpy(int)
    y3 = pred["pathology_cin3plus"].to_numpy(int)
    p = pred["prob_cin2plus"].to_numpy(float)
    threshold = float(pred["threshold_cin3_safety95"].median())
    m3 = binary_metrics(y3, p, threshold)
    agg = pd.DataFrame(
        [
            {
                "model": "ViT-PubMedBERT-LoRA",
                "n": len(pred),
                "CIN2+ AUC": safe_auc(y2, p),
                "CIN2+ AP": float(average_precision_score(y2, p)),
                "CIN3+ AUC": safe_auc(y3, p),
                "CIN3+ sensitivity": m3["sensitivity"],
                "CIN3+ FN": m3["fn"],
                "specificity_at_CIN3_safety_threshold": m3["specificity"],
                "PPV_at_CIN3_safety_threshold": m3["ppv"],
                "NPV_at_CIN3_safety_threshold": m3["npv"],
                "screen_positive_rate": m3["screen_positive_rate"],
            }
        ]
    )
    centre_rows = []
    for center, g in pred.groupby("center_name"):
        pc = g["prob_cin2plus"].to_numpy(float)
        y2c = g["pathology_cin2plus"].to_numpy(int)
        y3c = g["pathology_cin3plus"].to_numpy(int)
        m3c = binary_metrics(y3c, pc, threshold)
        centre_rows.append(
            {
                "center_name": center,
                "n": len(g),
                "CIN2+ AUC": safe_auc(y2c, pc),
                "CIN3+ sensitivity": m3c["sensitivity"],
                "CIN3+ FN": m3c["fn"],
                "screen_positive_rate": m3c["screen_positive_rate"],
            }
        )
    centre = pd.DataFrame(centre_rows)
    return agg, centre


def write_outputs(pred: pd.DataFrame, fold_results: list[FoldResult], run_tag: str) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        safe = df.copy()
        for col in safe.columns:
            if pd.api.types.is_float_dtype(safe[col]):
                safe[col] = safe[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
            else:
                safe[col] = safe[col].astype(str)
        lines = [
            "| " + " | ".join(safe.columns) + " |",
            "| " + " | ".join(["---"] * len(safe.columns)) + " |",
        ]
        for _, row in safe.iterrows():
            lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in safe.columns) + " |")
        return "\n".join(lines)

    pred_path = PRED_DIR / "vit_bert_lora_loco_patient_predictions.csv"
    pred_tag_path = PRED_DIR / f"vit_bert_lora_loco_patient_predictions__{run_tag}.csv"
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    pred.to_csv(pred_tag_path, index=False, encoding="utf-8-sig")
    fold_table = pd.DataFrame([r.__dict__ for r in fold_results])
    fold_table.to_csv(TABLE_DIR / "vit_bert_lora_fold_selection.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(TABLE_DIR / f"vit_bert_lora_fold_selection__{run_tag}.csv", index=False, encoding="utf-8-sig")
    agg, centre = summarize(pred)
    agg.to_csv(TABLE_DIR / "vit_bert_lora_aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(TABLE_DIR / f"vit_bert_lora_aggregate_metrics__{run_tag}.csv", index=False, encoding="utf-8-sig")
    centre.to_csv(TABLE_DIR / "vit_bert_lora_centre_metrics.csv", index=False, encoding="utf-8-sig")
    centre.to_csv(TABLE_DIR / f"vit_bert_lora_centre_metrics__{run_tag}.csv", index=False, encoding="utf-8-sig")
    sns.set_theme(style="whitegrid", context="paper", palette=SCI)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    sns.barplot(data=agg, x="model", y="CIN2+ AUC", ax=axes[0], color=SCI[0])
    axes[0].axhline(0.8, ls="--", color=SCI[4], lw=1)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")
    sns.barplot(data=centre, y="center_name", x="CIN2+ AUC", ax=axes[1], color=SCI[1])
    axes[1].axvline(0.8, ls="--", color=SCI[4], lw=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_ViT_BERT_LoRA_LOCO.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure_ViT_BERT_LoRA_LOCO.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"Figure_ViT_BERT_LoRA_LOCO__{run_tag}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"Figure_ViT_BERT_LoRA_LOCO__{run_tag}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
    report = [
        "# ViT + PubMedBERT LoRA LOCO Report",
        "",
        f"Run tag: `{run_tag}`",
        "",
        "Status: `COMPLETED`" if len(pred) else "Status: `NO_PREDICTIONS`",
        "",
        "This run performs true trainable LoRA updates, not cached-feature adapter fitting.",
        "",
        "- Visual backbone: timm ViT-B/16 (`vit_base_patch16_224.augreg2_in21k_ft_in1k`).",
        "- Semantic backbone: local PubMedBERT.",
        "- LoRA insertion: ViT attention qkv/proj and PubMedBERT query/value in the last configured layers.",
        "- Target labels are not used for training; held-out centres are evaluated only after fold training.",
        "- GPU telemetry is recorded under `audit/`.",
        "",
        "## Aggregate Metrics",
        "",
        md_table(agg),
        "",
        "## Centre Metrics",
        "",
        md_table(centre),
        "",
        "## Fold Selection",
        "",
        md_table(fold_table),
    ]
    (OUT / "ViT_BERT_LoRA_LOCO_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / f"ViT_BERT_LoRA_LOCO_Report__{run_tag}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vit-lora-last-n", type=int, default=4)
    parser.add_argument("--bert-lora-last-n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-val-cases", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    ensure_dirs()
    run_tag = args.run_tag or time.strftime("vitbert_lora_%Y%m%d_%H%M%S")
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[start] run_tag={run_tag} device={device} vit={VIT_MODEL} bert={BERT_MODEL}", flush=True)
    gpu_snapshot("run_start", run_tag)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, local_files_only=True)
    transform = make_transform()
    data = pd.read_csv(DATA_LOCK)
    split = pd.read_csv(SPLIT_MANIFEST)
    data["case_id"] = data["case_id"].astype(str)
    preds: list[pd.DataFrame] = []
    fold_results: list[FoldResult] = []
    fold_items = list(split.groupby("fold_id"))
    if args.fold_limit:
        fold_items = fold_items[: args.fold_limit]
    for fold_id, fold_df in fold_items:
        fold_pred, fold_result = run_fold(str(fold_id), fold_df, data, transform, tokenizer, args, device, run_tag)
        preds.append(fold_pred)
        fold_results.append(fold_result)
        partial = pd.concat(preds, ignore_index=True)
        partial.to_csv(PRED_DIR / "vit_bert_lora_loco_patient_predictions.partial.csv", index=False, encoding="utf-8-sig")
        partial.to_csv(PRED_DIR / f"vit_bert_lora_loco_patient_predictions__{run_tag}.partial.csv", index=False, encoding="utf-8-sig")
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    write_outputs(pred, fold_results, run_tag)
    gpu_snapshot("run_end", run_tag)
    print("[done]", flush=True)
    if len(pred):
        agg, _ = summarize(pred)
        print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
