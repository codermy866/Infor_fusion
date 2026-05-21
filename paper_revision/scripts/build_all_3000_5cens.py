#!/usr/bin/env python3
"""Build All_3000_5cens: train/test/oct|col layout aligned to 3000_nums.xlsx.

Uses symlinks only (no image copy). Sources:
- OCT: data/cervix_oct_original (SSHFS) with fallback to data/5centers_multi
- Colposcopy: data/colposcopy_3000 with fallback to data/5centers_multi
- Labels/splits: preserve data/5centers_multi train|test where available
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "data"
OUT = DATA / "All_3000_5cens"
REGISTRY_XLSX = DATA / "colposcopy_3000" / "3000_nums.xlsx"
LEGACY_ROOT = DATA / "5centers_multi"
OCT_REMOTE = DATA / "cervix_oct_original"
COL_RAW = DATA / "colposcopy_3000"
OCT_FLAT_CACHE = OUT / "_oct_flat_cache"

SEED = 2026
TRAIN_RATIO = 0.8

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DATE8_RE = re.compile(r"(20\d{6})")
DATE_DASH_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
DOT_DATE_RE = re.compile(r"(?:阴道镜)?(\d{2})\.(\d{2})\.(\d{2})")

HOSPITAL_COL_DIR = {
    "荆州市第一人民医院": "阴道镜图像（荆州市第一人民医院）",
    "武大人民医院": "阴道镜图像（武大人民医院）",
    "十堰市人民医院": "阴道镜图像（十堰市人民医院）",
    "襄阳市中心医院": "阴道镜图像（襄阳市中心医院）",
    "恩施州中心医院": "阴道镜图像（恩施州中心医院）",
}

HOSPITAL_OCT_CENTER = {
    "荆州市第一人民医院": "jingzhou",
    "武大人民医院": "WuDa",
    "十堰市人民医院": "ShiYan",
    "襄阳市中心医院": "XiangYang",
    "恩施州中心医院": "Enshi",
}

OCT_PREFIX_CENTER = {
    "M0008": "jingzhou",
    "M22101": "ShiYan",
    "M20105": "WuDa",
    "M20203": "WuDa",
    "M22104": "ShiYan",
    "M22102": "XiangYang",
    "M22105": "Enshi",
}

ColIndex = dict  # (hospital, YYYYMMDD) -> [col folder paths]


def date_key(dt) -> str | None:
    if pd.isna(dt):
        return None
    return str(dt)[:10].replace("-", "")


def extract_date_keys(text: str) -> set[str]:
    keys: set[str] = set(DATE8_RE.findall(text))
    for y, mo, d in DATE_DASH_RE.findall(text):
        keys.add(f"{y}{mo}{d}")
    for y, mo, d in DOT_DATE_RE.findall(text):
        keys.add(f"20{y}{mo}{d}")
    return keys


def infer_label(oct_reading) -> int | None:
    if pd.isna(oct_reading) or str(oct_reading).strip() in ("", "nan"):
        return None
    s = str(oct_reading)
    if "高级别" in s or "疑似" in s:
        return 1
    if "未发现" in s or "低级别" in s:
        return 0
    return None


def resolve_label(row: pd.Series, legacy: dict | None) -> int | None:
    if legacy is not None:
        return int(legacy["label"])
    for col in (
        "OCT二次判读_img",
        "OCT二次判读",
        "OCT实时判读_img",
        "OCT实时判读",
    ):
        if col not in row.index:
            continue
        label = infer_label(row[col])
        if label is not None:
            return label
    if "二次判读高级别" in row.index and pd.notna(row["二次判读高级别"]):
        try:
            if float(row["二次判读高级别"]) > 0:
                return 1
        except (TypeError, ValueError):
            pass
    if "二次判读疑似" in row.index and pd.notna(row["二次判读疑似"]):
        try:
            if float(row["二次判读疑似"]) > 0:
                return 1
        except (TypeError, ValueError):
            pass
    return None


def dir_has_images(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in IMAGE_EXTS
    try:
        for p in path.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return True
    except OSError:
        return False
    return False


def safe_symlink(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return True
            dst.unlink()
        except OSError:
            return False
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return True
    except OSError:
        return False


def build_col_index() -> ColIndex:
    index: ColIndex = {}
    for hospital, subdir in HOSPITAL_COL_DIR.items():
        root = COL_RAW / subdir
        if not root.exists():
            continue
        for folder in root.rglob("*"):
            if not folder.is_dir() or not dir_has_images(folder):
                continue
            rel = str(folder.relative_to(root))
            dates = extract_date_keys(rel) | extract_date_keys(folder.name)
            for dk in dates:
                index.setdefault((hospital, dk), []).append(folder)
    return index


def find_legacy_pair(oct_id: str, patient_id: str) -> tuple[Path | None, Path | None]:
    for split in ("train", "test"):
        oct_p = LEGACY_ROOT / split / "oct" / oct_id
        col_p = LEGACY_ROOT / split / "col" / patient_id
        if dir_has_images(oct_p) and dir_has_images(col_p):
            return oct_p, col_p
    return None, None


def _oct_centers(oct_id: str, hospital: str) -> list[str]:
    centers: list[str] = []
    hc = HOSPITAL_OCT_CENTER.get(hospital)
    if hc:
        centers.append(hc)
    prefix = oct_id.split("_")[0]
    if prefix in OCT_PREFIX_CENTER:
        centers.append(OCT_PREFIX_CENTER[prefix])
    return list(dict.fromkeys(centers))


def resolve_oct_path(oct_id: str, hospital: str) -> Path | None:
    for center in _oct_centers(oct_id, hospital):
        cand = OCT_REMOTE / center / oct_id
        if dir_has_images(cand):
            return cand
        center_dir = OCT_REMOTE / center
        if not center_dir.exists():
            continue
        flat_files = sorted(
            p
            for p in center_dir.glob(f"{oct_id}*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not flat_files:
            continue
        bundle = OCT_FLAT_CACHE / center / oct_id
        bundle.mkdir(parents=True, exist_ok=True)
        for src in flat_files:
            safe_symlink(src, bundle / src.name)
        if dir_has_images(bundle):
            return bundle
    for split in ("train", "test"):
        legacy = LEGACY_ROOT / split / "oct" / oct_id
        if dir_has_images(legacy):
            return legacy
    return None


def normalize_patient_id(folder_name: str, exam_dt) -> str:
    dk = date_key(exam_dt)
    dash = re.match(r"^(20\d{2})-(\d{2})-(\d{2})\s+.*?\s+(.+)$", folder_name)
    if dash and dk:
        return f"{dk}_{dash.group(4).strip()}"
    m = re.match(r"^(\d{8,11})[_-]?(.*)$", folder_name)
    if m and dk:
        name = m.group(2).lstrip("_-").strip()
        if name:
            return f"{dk}_{name}"
    if dk and "_" not in folder_name and folder_name:
        return f"{dk}_{folder_name}"
    return folder_name


def pick_col_folder(
    candidates: list[Path],
    patient_hint: str | None = None,
) -> Path | None:
    if not candidates:
        return None
    if patient_hint:
        for c in candidates:
            if c.name == patient_hint:
                return c
        hint_name = patient_hint.split("_", 1)[-1] if "_" in patient_hint else patient_hint
        filtered = [c for c in candidates if hint_name in c.name]
        if len(filtered) == 1:
            return filtered[0]
        if filtered:
            candidates = filtered
    if len(candidates) == 1:
        return candidates[0]

    def score(p: Path) -> tuple[int, int]:
        imgs = sum(1 for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        return (-imgs, len(p.name))

    return sorted(candidates, key=score)[0]


def find_col_from_index(
    col_index: ColIndex,
    hospital: str,
    exam_dt,
    patient_hint: str | None = None,
) -> Path | None:
    dk = date_key(exam_dt)
    if not dk:
        return None
    candidates = list(col_index.get((hospital, dk), []))
    if not candidates:
        return None
    return pick_col_folder(candidates, patient_hint)


def load_existing_labels() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for split, fname in (("train", "train_labels.csv"), ("test", "test_labels.csv")):
        df = pd.read_csv(LEGACY_ROOT / fname)
        for _, row in df.iterrows():
            out[str(row["OCT"])] = {
                "ID": str(row["ID"]),
                "label": int(row["label"]),
                "split": split,
                "AGE": row.get("AGE", ""),
                "HPV清洗": row.get("HPV清洗", ""),
                "TCT清洗": row.get("TCT清洗", ""),
            }
    return out


def stratified_assign_splits(rows: list[dict]) -> None:
    rng = np.random.default_rng(SEED)
    by_group: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        by_group.setdefault((r["center_name"], r["label"]), []).append(i)
    for idxs in by_group.values():
        rng.shuffle(idxs)
        n_train = max(1, int(round(len(idxs) * TRAIN_RATIO)))
        if len(idxs) <= 1:
            n_train = 1
        elif n_train >= len(idxs):
            n_train = len(idxs) - 1
        for j, i in enumerate(idxs):
            rows[i]["split"] = "train" if j < n_train else "test"


def clear_split_links() -> None:
    for split in ("train", "test"):
        for mod in ("oct", "col"):
            base = OUT / split / mod
            if not base.exists():
                continue
            for p in base.iterdir():
                if p.is_symlink():
                    p.unlink()


def main() -> None:
    if not REGISTRY_XLSX.exists():
        raise SystemExit(f"Missing registry: {REGISTRY_XLSX}")
    if not OCT_REMOTE.exists():
        raise SystemExit(
            f"OCT mount missing: {OCT_REMOTE}. Mount SSHFS first (data/mnt/cervix_oct_remote)."
        )

    print("Indexing colposcopy folders ...")
    col_index = build_col_index()
    print(f"  date buckets: {len(col_index)}")

    mi = pd.read_excel(REGISTRY_XLSX, sheet_name="MedicalInfo")
    oct_img = pd.read_excel(REGISTRY_XLSX, sheet_name="OCTImages")
    mi = mi.drop_duplicates(subset=["OCT图像Id"], keep="first")
    oct_cols = [
        c
        for c in (
            "OCT图像Id",
            "OCT二次判读",
            "OCT实时判读",
            "二次判读疑似",
            "二次判读高级别",
        )
        if c in oct_img.columns
    ]
    oct_img = oct_img[oct_cols].rename(
        columns={
            "OCT二次判读": "OCT二次判读_img",
            "OCT实时判读": "OCT实时判读_img",
        }
    )
    mi = mi.merge(oct_img, on="OCT图像Id", how="left", suffixes=("", "_dup"))

    existing = load_existing_labels()

    rows: list[dict] = []
    stats = {
        "registry_unique": int(len(mi)),
        "legacy_label_rows": len(existing),
        "col_index_buckets": len(col_index),
        "oct_found": 0,
        "col_found": 0,
        "both_found": 0,
        "used_legacy_paths": 0,
        "used_remote_oct": 0,
        "symlink_ok": 0,
        "symlink_fail": 0,
        "skipped_no_label": 0,
        "skipped_no_modality": 0,
    }

    for _, r in mi.iterrows():
        oct_id = str(r["OCT图像Id"]).strip()
        hospital = str(r["医院"]).strip()
        legacy = existing.get(oct_id)

        patient_id = legacy["ID"] if legacy else None
        label = resolve_label(r, legacy)
        split = legacy["split"] if legacy else None

        oct_src = col_src = None
        if patient_id:
            leg_oct, leg_col = find_legacy_pair(oct_id, patient_id)
            if leg_oct and leg_col:
                oct_src, col_src = leg_oct, leg_col
                stats["used_legacy_paths"] += 1

        if oct_src is None:
            oct_src = resolve_oct_path(oct_id, hospital)
            if oct_src is not None and str(LEGACY_ROOT) not in str(oct_src):
                stats["used_remote_oct"] += 1

        if col_src is None:
            col_src = find_col_from_index(
                col_index, hospital, r["OCT检查日期时间"], patient_id
            )
            if col_src is not None and patient_id is None:
                patient_id = normalize_patient_id(col_src.name, r["OCT检查日期时间"])

        if oct_src is not None:
            stats["oct_found"] += 1
        if col_src is not None:
            stats["col_found"] += 1
        if oct_src is None or col_src is None or patient_id is None:
            stats["skipped_no_modality"] += 1
            continue
        if label is None:
            stats["skipped_no_label"] += 1
            continue

        center_code = oct_id.split("_")[0]
        rows.append(
            {
                "ID": patient_id,
                "OCT": oct_id,
                "AGE": legacy["AGE"] if legacy else r.get("年龄", ""),
                "HPV清洗": legacy["HPV清洗"] if legacy else r.get("HPV清洗（高亮表示阳性）", ""),
                "TCT清洗": legacy["TCT清洗"] if legacy else r.get("TCT清洗（高亮表示阳性）", ""),
                "label": int(label),
                "split": split,
                "center_name": hospital,
                "center_code": center_code,
                "oct_src": str(oct_src.resolve()),
                "col_src": str(col_src.resolve()),
            }
        )

    stats["both_found"] = len(rows)

    need_split = [i for i, row in enumerate(rows) if row["split"] is None]
    if need_split:
        pending = [rows[i] for i in need_split]
        stratified_assign_splits(pending)
        for i, row in zip(need_split, pending):
            rows[i]["split"] = row["split"]

    for sub in ("train", "test"):
        for mod in ("oct", "col"):
            (OUT / sub / mod).mkdir(parents=True, exist_ok=True)

    clear_split_links()

    xlsx_link = OUT / "3000_num.xlsx"
    if xlsx_link.is_symlink() or xlsx_link.exists():
        xlsx_link.unlink()
    os.symlink(REGISTRY_XLSX, xlsx_link)

    train_rows, test_rows = [], []
    for row in rows:
        split = row.pop("split")
        oct_dst = OUT / split / "oct" / row["OCT"]
        col_dst = OUT / split / "col" / row["ID"]
        ok_o = safe_symlink(Path(row["oct_src"]), oct_dst)
        ok_c = safe_symlink(Path(row["col_src"]), col_dst)
        if ok_o and ok_c:
            stats["symlink_ok"] += 1
        else:
            stats["symlink_fail"] += 1
        row.pop("oct_src")
        row.pop("col_src")
        (train_rows if split == "train" else test_rows).append(row)

    cols = ["ID", "OCT", "AGE", "HPV清洗", "TCT清洗", "label", "center_name", "center_code"]
    pd.DataFrame(train_rows)[cols].to_csv(OUT / "train_labels.csv", index=False, encoding="utf-8")
    pd.DataFrame(test_rows)[cols].to_csv(OUT / "test_labels.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "build_manifest.csv", index=False, encoding="utf-8")

    stats["train_n"] = len(train_rows)
    stats["test_n"] = len(test_rows)
    stats["train_pos"] = int(sum(r["label"] == 1 for r in train_rows))
    stats["test_pos"] = int(sum(r["label"] == 1 for r in test_rows))

    with open(OUT / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": "All_3000_5cens",
                "description": "Five-centre 3000 registry cohort with OCT+colposcopy symlinks",
                "registry": str(REGISTRY_XLSX),
                "layout": "train|test / oct|col + train_labels.csv + test_labels.csv",
                "stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
