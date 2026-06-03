"""Shared helpers for supplementary experiment completion scripts."""

from __future__ import annotations

import hashlib
import io
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    brier_score_loss = None
    roc_auc_score = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = ROOT / "paper_revisions/if_supplementary_experiments"
PRIMARY_MODEL = "HyDRA_CoE_Full"
HASH_SALT = "if_supplementary_experiments_2026"

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

MODEL_DISPLAY = {
    "HyDRA_CoE_Full": "HyDRA-CoE full",
    "BioMedCLIP_Finetuned": "BioMedCLIP",
    "ColposcopyOCTText_CrossAttention": "Cross-attention fusion",
    "ColposcopyOCT_LateFusion": "Late score fusion",
    "ColposcopyOCT_EarlyConcat": "Early concat",
    "OCTOnly_ViT": "OCT only",
    "ColposcopyOnly_ViT": "Colposcopy only",
    "ClinicalOnly_Logistic": "Clinical logistic",
    "ClinicalOnly_XGBoost": "Clinical XGBoost",
}

CENTER_DISPLAY = {
    "\u6b66\u5927\u4eba\u6c11\u533b\u9662": "Wuhan Renmin",
    "\u6069\u65bd\u5dde\u4e2d\u5fc3\u533b\u9662": "Enshi",
    "\u8944\u9633\u5e02\u4e2d\u5fc3\u533b\u9662": "Xiangyang",
    "\u5341\u5830\u5e02\u4eba\u6c11\u533b\u9662": "Shiyan",
    "\u8346\u5dde\u5e02\u7b2c\u4e00\u4eba\u6c11\u533b\u9662": "Jingzhou",
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def str_rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def preferred_font_family() -> str:
    names = {Path(f.fname).stem.lower(): f.name for f in font_manager.fontManager.ttflist}
    for wanted in ["arial", "arialbd", "times new roman", "times"]:
        for stem, name in names.items():
            if wanted in stem:
                return name
    return "DejaVu Sans"


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk", palette=PALETTE)
    plt.rcParams.update(
        {
            "font.family": preferred_font_family(),
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#30335f",
            "axes.titlecolor": "#30335f",
        }
    )


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def hash_id(value: object) -> str:
    raw = "" if value is None else str(value)
    return hashlib.sha256(f"{HASH_SALT}|{raw}".encode("utf-8")).hexdigest()[:16]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def logit(p: Iterable[float] | float) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    return np.log(arr / (1 - arr))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and not pd.isna(b) else float("nan")


def safe_auc(y: Iterable[int], score: Iterable[float]) -> float:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy()
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2 or roc_auc_score is None:
        return float("nan")
    try:
        return float(roc_auc_score(y_arr.astype(int), s_arr))
    except Exception:
        return float("nan")


def safe_auprc(y: Iterable[int], score: Iterable[float]) -> float:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy()
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2 or average_precision_score is None:
        return float("nan")
    try:
        return float(average_precision_score(y_arr.astype(int), s_arr))
    except Exception:
        return float("nan")


def safe_brier(y: Iterable[int], score: Iterable[float]) -> float:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy()
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    if mask.sum() == 0 or brier_score_loss is None:
        return float("nan")
    try:
        return float(brier_score_loss(y_arr[mask].astype(int), np.clip(s_arr[mask], 0, 1)))
    except Exception:
        return float("nan")


def expected_calibration_error(y: Iterable[int], score: Iterable[float], bins: int = 10) -> float:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy()
    mask = np.isfinite(y_arr) & np.isfinite(s_arr)
    y_arr = y_arr[mask].astype(int)
    s_arr = np.clip(s_arr[mask], 0, 1)
    if len(y_arr) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (s_arr >= lo) & (s_arr < hi)
        if hi == 1.0:
            in_bin = (s_arr >= lo) & (s_arr <= hi)
        if not in_bin.any():
            continue
        ece += float(in_bin.mean()) * abs(float(s_arr[in_bin].mean()) - float(y_arr[in_bin].mean()))
    return ece


def binary_metrics(y: Iterable[int], score: Iterable[float], threshold: float) -> dict[str, float]:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).astype(int).to_numpy()
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce").fillna(0.0).to_numpy()
    pred = (s_arr >= threshold).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    return {
        "threshold": float(threshold),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": np.nanmean([sens, spec]),
        "referral_rate": float(pred.mean()) if len(pred) else float("nan"),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def metric_row(
    df: pd.DataFrame,
    *,
    score_col: str = "pred_cin2_score",
    y_col: str = "y_cin2",
    threshold_col: str = "threshold_cin2_locked",
) -> dict[str, float]:
    threshold = 0.5
    if threshold_col in df.columns and df[threshold_col].notna().any():
        threshold = float(pd.to_numeric(df[threshold_col], errors="coerce").median())
    row = {
        "n": int(len(df)),
        "n_positive": int(pd.to_numeric(df[y_col], errors="coerce").fillna(0).sum()),
        "auroc": safe_auc(df[y_col], df[score_col]),
        "auprc": safe_auprc(df[y_col], df[score_col]),
        "brier": safe_brier(df[y_col], df[score_col]),
        "ece": expected_calibration_error(df[y_col], df[score_col]),
    }
    row.update(binary_metrics(df[y_col], df[score_col], threshold))
    if "y_cin3" in df.columns:
        th3 = 0.5
        if "threshold_cin3_locked" in df.columns and df["threshold_cin3_locked"].notna().any():
            th3 = float(pd.to_numeric(df["threshold_cin3_locked"], errors="coerce").median())
        cin3 = binary_metrics(df["y_cin3"], df[score_col], th3)
        row.update(
            {
                "cin3_sensitivity": cin3["sensitivity"],
                "cin3_specificity": cin3["specificity"],
                "cin3_false_negatives": cin3["false_negative"],
                "cin3_referral_rate": cin3["referral_rate"],
            }
        )
    return row


def grouped_metrics(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row.update(metric_row(g))
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_entropy(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum(axis=1, keepdims=True)
    return -(w * np.log(w)).sum(axis=1) / np.log(w.shape[1])


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("if_diverging", ["#8b98b3", "#f7f3ef", "#b57979"])


def ensure_out_dirs(out_root: Path) -> None:
    for sub in [
        "partial_completion",
        "05_modality_ablation_and_missingness",
        "09_reliability_validation",
        "10_coe_faithfulness",
        "12_final_tables_figures",
        "13_submission_audit",
    ]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def environment_snapshot() -> str:
    lines = [
        f"created_at: {now()}",
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"hostname: {socket.gethostname()}",
        f"working_directory: {ROOT}",
    ]
    for pkg in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn", "scipy", "torch"]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            lines.append(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{pkg}: unavailable ({exc})")
    try:
        import torch

        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
    except Exception as exc:
        lines.append(f"cuda_available: unavailable ({exc})")
    for cmd, label in [(["git", "rev-parse", "HEAD"], "git_hash"), (["git", "status", "--short"], "git_status_short")]:
        try:
            res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            lines.append(f"{label}: {res.stdout.strip()}")
        except Exception as exc:
            lines.append(f"{label}: unavailable ({exc})")
    return "\n".join(lines) + "\n"


def make_zip_package(out_root: Path, package_path: Path | None = None) -> Path:
    package_path = package_path or (out_root / "IF_SUPPLEMENTARY_EXPERIMENTS_PACKAGE.zip")
    include_suffix = {".csv", ".md", ".txt", ".json", ".yaml", ".yml", ".png", ".pdf", ".py"}
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in out_root.rglob("*"):
            if file == package_path or not file.is_file():
                continue
            if file.suffix.lower() in include_suffix:
                arcname = str(file.relative_to(out_root))
                if file.suffix.lower() == ".csv":
                    payload = sanitized_csv_bytes_for_package(file)
                    if payload is not None:
                        zf.writestr(arcname, payload)
                    else:
                        zf.write(file, arcname=arcname)
                else:
                    zf.write(file, arcname=arcname)
        for file in sorted((ROOT / "scripts/if_supplementary").glob("*.py")):
            zf.write(file, arcname=f"reproducibility/{file.relative_to(ROOT)}")
        for file in sorted((ROOT / "src/if_supplementary").glob("*.py")):
            zf.write(file, arcname=f"reproducibility/{file.relative_to(ROOT)}")
        config = ROOT / "configs/if_supplementary_same_backbone_baselines.yaml"
        if config.exists():
            zf.write(config, arcname=f"reproducibility/{config.relative_to(ROOT)}")
    return package_path


def sanitized_csv_bytes_for_package(path: Path) -> bytes | None:
    """Drop raw identifier columns from CSVs before adding them to the package.

    This does not alter locked files on disk. It only prevents raw identifiers
    from being copied into the portable supplementary archive.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    raw_cols = [c for c in df.columns if c in {"patient_id", "case_id", "raw_patient_id", "raw_case_id"}]
    if not raw_cols:
        return None
    df = df.drop(columns=raw_cols)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")


def copy_pair(src_stem: Path, dst_stem: Path) -> list[Path]:
    outputs: list[Path] = []
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        src = src_stem.with_suffix(suffix)
        dst = dst_stem.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, dst)
            outputs.append(dst)
    return outputs


def display_center(value: object) -> str:
    return CENTER_DISPLAY.get(str(value), str(value))


def validate_no_raw_id_columns(df: pd.DataFrame) -> None:
    forbidden = {"patient_id", "case_id", "raw_patient_id", "raw_case_id"}
    present = forbidden.intersection(df.columns)
    if present:
        raise ValueError(f"Raw identifier columns are not allowed in this output: {sorted(present)}")


def input_paths(out_root: Path) -> dict[str, Path]:
    return {
        "test_predictions": out_root / "01_prediction_registry/standardized_patient_mean_test_predictions.csv",
        "thresholds": out_root / "02_cin3_safety/validation_locked_thresholds.csv",
        "reliability": out_root / "09_reliability_validation/reliability_weights_patient_level.csv",
        "coe_proxy": out_root / "10_coe_faithfulness/coe_intervention_patient_level.csv",
        "loco_folds": ROOT / "outputs/publishable_v2/splits/loco_folds_v2.json",
        "feature_npz": ROOT / "outputs/publishable_v2/step2_main_loco/audit/step2_locked_feature_arrays.npz",
        "data_lock": ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv",
        "checkpoint_root": ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897/ablations",
    }
