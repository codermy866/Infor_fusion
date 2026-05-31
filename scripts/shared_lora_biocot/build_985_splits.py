#!/usr/bin/env python3
"""Build auditable 985-case splits with oct_paths for BioCOT v3.2."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/data2/hmy/5Center_datas/5centers_multi_leave_centers_out")
DEFAULT_OUT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/cached_985/splits"

CENTER_MAP = {
    "M22105": ("恩施州中心医院", 0),
    "M20105": ("武大人民医院", 1),
    "M20203": ("武大人民医院", 1),
    "M22102": ("襄阳市中心医院", 2),
    "M0008": ("荆州市第一人民医院", 3),
    "M22101": ("十堰市人民医院", 4),
    "M22104": ("十堰市人民医院", 4),
}

LOCO_EXCLUDE_CENTERS = {"武大人民医院"}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def collect_paths(base: Path, patterns: list[str]) -> list[str]:
    if not base.exists():
        return []
    files: list[str] = []
    for pattern in patterns:
        files.extend(str(p) for p in sorted(base.glob(pattern)))
    return files


def center_info(oct_id: str) -> tuple[str, int, str]:
    code = str(oct_id).split("_")[0]
    name, group_id = CENTER_MAP.get(code, (code, -1))
    return name, group_id, code


def resolve_media_paths(data_root: Path, patient_id: str, oct_id: str) -> tuple[str, str, int, int]:
    oct_bases = [
        data_root / "internal_train" / "train" / "oct",
        data_root / "internal_train" / "val" / "oct",
        data_root / "external_validation" / "oct",
    ]
    col_bases = [
        data_root / "internal_train" / "train" / "col",
        data_root / "internal_train" / "val" / "col",
        data_root / "external_validation" / "col",
    ]
    oct_paths: list[str] = []
    for base in oct_bases:
        oct_paths = collect_paths(base / oct_id, ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"])
        if oct_paths:
            break
    col_paths: list[str] = []
    for base in col_bases:
        col_paths = collect_paths(base / patient_id, ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"])
        if col_paths:
            break
    return ";".join(oct_paths), ";".join(col_paths), len(oct_paths), len(col_paths)


def enrich_rows(df: pd.DataFrame, data_root: Path, source_split: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        patient_id = str(row["ID"])
        oct_id = str(row["OCT"])
        center_name, center_group_id, center_code = center_info(oct_id)
        oct_paths, col_paths, oct_count, col_count = resolve_media_paths(data_root, patient_id, oct_id)
        label = int(row["label"])
        rows.append(
            {
                "ID": patient_id,
                "patient_id": patient_id,
                "oct_id": oct_id,
                "OCT": oct_id,
                "AGE": row.get("AGE", ""),
                "age": row.get("AGE", ""),
                "HPV清洗": row.get("HPV清洗", ""),
                "hpv": row.get("HPV清洗", ""),
                "TCT清洗": row.get("TCT清洗", ""),
                "tct": row.get("TCT清洗", ""),
                "label": label,
                "pathology_cin2plus": label,
                "pathology_cin3plus": label,
                "center_name": center_name,
                "center_code": center_code,
                "center_group_id": center_group_id,
                "source_split": source_split,
                "oct_paths": oct_paths,
                "col_paths": col_paths,
                "colposcopy_paths": col_paths,
                "oct_count": oct_count,
                "col_count": col_count,
                "is_positive_patient": bool(label),
            }
        )
    return pd.DataFrame(rows)


def write_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def stratified_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for _, group in df.groupby(["center_name", "label"], dropna=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            continue
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(indices[:n_val].tolist())
    val_set = set(val_indices)
    val_df = df.loc[sorted(val_set)].copy()
    train_df = df.drop(index=sorted(val_set)).copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def summarize(df: pd.DataFrame, split_name: str, held_out: str = "") -> dict[str, object]:
    pos = int(df["label"].sum()) if len(df) else 0
    return {
        "split_name": split_name,
        "held_out_center": held_out,
        "n": int(len(df)),
        "positive": pos,
        "negative": int(len(df) - pos),
        "positive_rate": float(df["label"].mean()) if len(df) else 0.0,
        "binary_auc_valid": bool(df["label"].nunique() == 2) if len(df) else False,
        "centers": ",".join(sorted(df["center_name"].astype(str).unique())) if len(df) else "",
        "missing_oct": int((df["oct_count"] <= 0).sum()) if len(df) else 0,
        "missing_col": int((df["col_count"] <= 0).sum()) if len(df) else 0,
    }


def build_official_split(all_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    official_dir = out_dir / "official"
    for split_name, file_name in [
        ("train", "train_labels.csv"),
        ("val", "val_labels.csv"),
        ("external_test", "external_test_labels.csv"),
    ]:
        part = all_df[all_df["source_split"].eq(split_name)].copy()
        write_split_csv(part, official_dir / file_name)
    rows = [
        summarize(all_df[all_df["source_split"].eq("train")], "train"),
        summarize(all_df[all_df["source_split"].eq("val")], "val"),
        summarize(all_df[all_df["source_split"].eq("external_test")], "external_test"),
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(official_dir / "official_split_audit.csv", index=False, encoding="utf-8-sig")
    (official_dir / "README.md").write_text(
        "# Official 985 leave-two-centers-out split\n\n"
        "Train/val from internal centres; external_test is Shiyan + Jingzhou (148 cases).\n",
        encoding="utf-8",
    )
    return audit


def build_loco_splits(all_df: pd.DataFrame, out_dir: Path, seed: int) -> pd.DataFrame:
    loco_root = out_dir / "loco"
    rows = []
    for center_name in sorted(all_df["center_name"].astype(str).unique()):
        if center_name in LOCO_EXCLUDE_CENTERS:
            continue
        fold_id = f"loco_{center_name}"
        fold_dir = loco_root / fold_id
        external = all_df[all_df["center_name"].astype(str).eq(center_name)].copy()
        development = all_df[~all_df["center_name"].astype(str).eq(center_name)].copy()
        train, val = stratified_val_split(development, val_fraction=0.2, seed=seed)
        write_split_csv(train, fold_dir / "train_labels.csv")
        write_split_csv(val, fold_dir / "val_labels.csv")
        write_split_csv(external, fold_dir / "external_test_labels.csv")
        for split_name, split_df in [("train", train), ("val", val), ("external_test", external)]:
            rows.append(summarize(split_df, split_name, held_out=center_name))
        (fold_dir / "README.md").write_text(
            f"# LOCO fold: {center_name}\n\nHeld-out centre only appears in external_test.\n",
            encoding="utf-8",
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(loco_root / "loco_split_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def build_manifest(all_df: pd.DataFrame, out_dir: Path) -> Path:
    manifest_path = out_dir / "case_manifest_985.csv"
    all_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def build_985_splits(source_root: Path, out_root: Path, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_root.mkdir(parents=True, exist_ok=True)

    frames = []
    for split_name, file_name in [
        ("train", "train_labels.csv"),
        ("val", "val_labels.csv"),
        ("external_test", "external_test_labels.csv"),
    ]:
        raw = read_csv(source_root / file_name)
        frames.append(enrich_rows(raw, source_root, split_name))
    all_df = pd.concat(frames, ignore_index=True)
    if (all_df["oct_count"] <= 0).any() or (all_df["col_count"] <= 0).any():
        bad = all_df[(all_df["oct_count"] <= 0) | (all_df["col_count"] <= 0)]
        raise RuntimeError(
            f"{len(bad)} cases missing OCT/col paths. Examples: {bad[['ID','OCT']].head(3).to_dict('records')}"
        )

    manifest = build_manifest(all_df, out_root)
    official_audit = build_official_split(all_df, out_root)
    loco_audit = build_loco_splits(all_df, out_root, seed=seed)

    print(f"Wrote manifest: {manifest}")
    print("\nOfficial split audit:")
    print(official_audit.to_string(index=False))
    print("\nLOCO split audit:")
    print(loco_audit.to_string(index=False))
    return official_audit, loco_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    build_985_splits(Path(args.source_root), Path(args.out_root), seed=args.seed)


if __name__ == "__main__":
    main()
