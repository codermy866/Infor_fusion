#!/usr/bin/env python3
"""Audit the 3010-to-1897 HyDRA-CoE no-report cohort alignment.

This script intentionally excludes examination-report fields. Pathology labels
are audited and carried only as supervised training/evaluation labels; they are
not composed into clinical text inputs.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_COHORT_ROOT = DATA_ROOT / "All_3000_5cens"
DEFAULT_RAW_XLSX = DATA_ROOT / "colposcopy_3000" / "3000_nums.xlsx"
DEFAULT_OUTPUT_DIR = EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit"
TABLE_DIR = EXP_ROOT / "paper_revision" / "tables"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}
FORBIDDEN_REPORT_FIELDS = {
    "report_text",
    "clinical_report",
    "diagnosis_report",
    "generated_report",
    "examination_report",
}

# Fixed five-hospital mapping for the current no-report experiment. Some
# hospitals have multiple raw OCT_ID prefixes; those prefixes are provenance,
# not additional participating centres.
CENTER_NAME_TO_CANONICAL_CODE = {
    "武大人民医院": "C01_WUHAN_RENMIN",
    "恩施州中心医院": "C02_ENSHI",
    "襄阳市中心医院": "C03_XIANGYANG",
    "十堰市人民医院": "C04_SHIYAN",
    "荆州市第一人民医院": "C05_JINGZHOU",
}
CENTER_NAME_TO_PAPER_NO = {
    "武大人民医院": 1,
    "恩施州中心医院": 2,
    "襄阳市中心医院": 3,
    "十堰市人民医院": 4,
    "荆州市第一人民医院": 5,
}
CENTER_NAME_TO_IDX = {
    name: paper_no - 1 for name, paper_no in CENTER_NAME_TO_PAPER_NO.items()
}


def canonical_center_code(center_name: Any, raw_center_code: Any = "") -> str:
    name = clean_text(center_name)
    if name in CENTER_NAME_TO_CANONICAL_CODE:
        return CENTER_NAME_TO_CANONICAL_CODE[name]
    # Fallback only for malformed rows; the audit will still expose the raw
    # value and downstream split code will require known five-centre names.
    return clean_text(raw_center_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rigorous no-report 3010-to-1897 data alignment audit."
    )
    parser.add_argument("--raw-registry-csv", default=None)
    parser.add_argument("--raw-registry-xlsx", default=None)
    parser.add_argument("--colposcopy-root", default=str(DEFAULT_COHORT_ROOT))
    parser.add_argument("--oct-root", default=str(DEFAULT_COHORT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--expected-raw-n", type=int, default=3010)
    parser.add_argument("--expected-aligned-n", type=int, default=1897)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | None, *, default: Path | None = None, base: Path = EXP_ROOT) -> Path | None:
    if value is None or str(value).strip() == "":
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported registry table format: {path}")


def nonempty(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null", "-", "—"}


def nonempty_series(series: pd.Series) -> pd.Series:
    return series.apply(nonempty)


def clean_text(value: Any) -> str:
    return "" if not nonempty(value) else str(value).strip()


def choose_col(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        hit = normalized.get(candidate.strip().lower())
        if hit is not None:
            return hit
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(candidate.strip().lower() in col_lower for candidate in candidates):
            return col
    return None


def report_field_absence_check(*frames: pd.DataFrame) -> dict[str, Any]:
    present: list[str] = []
    for frame in frames:
        for col in frame.columns:
            if str(col).strip().lower() in FORBIDDEN_REPORT_FIELDS:
                present.append(str(col))
    return {
        "forbidden_report_fields_present": sorted(set(present)),
        "report_field_absence_check": "pass" if not present else "present_but_ignored",
    }


def drop_forbidden_report_fields(df: pd.DataFrame) -> pd.DataFrame:
    forbidden = {col for col in df.columns if str(col).strip().lower() in FORBIDDEN_REPORT_FIELDS}
    if forbidden:
        return df.drop(columns=sorted(forbidden))
    return df


def load_raw_registry(raw_csv: Path | None, raw_xlsx: Path | None) -> tuple[pd.DataFrame, Path]:
    if raw_csv is not None and raw_csv.exists():
        path = raw_csv
    elif raw_xlsx is not None and raw_xlsx.exists():
        path = raw_xlsx
    else:
        path = DEFAULT_RAW_XLSX
    if not path.exists():
        raise FileNotFoundError(f"Raw registry table not found: {path}")
    return drop_forbidden_report_fields(read_table(path)), path


def load_aligned_manifest(cohort_root: Path) -> pd.DataFrame:
    train_path = cohort_root / "train_labels.csv"
    test_path = cohort_root / "test_labels.csv"
    if train_path.exists() and test_path.exists():
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        train["original_split"] = "train"
        test["original_split"] = "test"
        return drop_forbidden_report_fields(pd.concat([train, test], ignore_index=True))

    manifest_path = cohort_root / "build_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, low_memory=False)
        manifest["original_split"] = "unknown"
        return drop_forbidden_report_fields(manifest)

    raise FileNotFoundError(f"No train/test labels or build_manifest.csv found under {cohort_root}")


def image_file(path: Path, modality: str) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    if modality == "colposcopy" and path.name.lower() in {"report.jpg", "report.jpeg", "report.png"}:
        return False
    if modality == "colposcopy" and ("检查报告" in path.name or "诊断报告" in path.name):
        return False
    return True


def collect_image_paths(case_dir: Path | None, modality: str) -> list[str]:
    if case_dir is None or not case_dir.exists():
        return []
    paths = [p for p in case_dir.rglob("*") if p.is_file() and image_file(p, modality)]
    return [str(p) for p in sorted(paths)]


def find_case_dir(root: Path, split: str, modality_dir: str, key: str) -> Path | None:
    candidates = [
        root / split / modality_dir / key,
        root / modality_dir / key,
        root / split / key,
        root / key,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def registry_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = list(df.columns)
    return {
        "center": choose_col(cols, ["医院", "center_name", "center", "hospital"]),
        "oct_id": choose_col(cols, ["OCT图像Id", "OCT", "oct_id", "oct"]),
        "age": choose_col(cols, ["年龄", "AGE", "age"]),
        "hpv": choose_col(cols, ["HPV清洗", "HPV", "hpv"]),
        "tct": choose_col(cols, ["TCT清洗", "TCT", "tct"]),
        "pathology_result": choose_col(cols, ["病理结果", "pathology", "histology"]),
        "pathology_grade": choose_col(cols, ["病理级别", "pathology_grade", "histology_grade"]),
        "followup": choose_col(cols, ["后续治疗", "treatment", "followup"]),
    }


def pathology_available(row: pd.Series, cols: dict[str, str | None]) -> bool:
    return any(nonempty(row.get(cols[key])) for key in ["pathology_result", "pathology_grade", "followup"] if cols.get(key))


def normalize_label(value: Any) -> int | float:
    if not nonempty(value):
        return np.nan
    try:
        numeric = int(float(str(value).strip()))
    except ValueError:
        return np.nan
    return numeric if numeric in {0, 1} else np.nan


def build_final_index(manifest: pd.DataFrame, col_root: Path, oct_root: Path) -> pd.DataFrame:
    required = ["ID", "OCT", "AGE", "HPV清洗", "TCT清洗", "label", "center_name", "center_code"]
    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise ValueError(f"Aligned manifest missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        patient_id = clean_text(row["ID"])
        oct_id = clean_text(row["OCT"])
        split = clean_text(row.get("original_split", "unknown")) or "unknown"
        col_dir = find_case_dir(col_root, split, "col", patient_id)
        oct_dir = find_case_dir(oct_root, split, "oct", oct_id)
        col_paths = collect_image_paths(col_dir, "colposcopy")
        oct_paths = collect_image_paths(oct_dir, "oct")
        label = normalize_label(row["label"])
        center_name = clean_text(row["center_name"])
        raw_center_code = clean_text(row["center_code"])
        center_idx = CENTER_NAME_TO_IDX.get(center_name, "")
        rows.append(
            {
                "case_id": patient_id,
                "patient_id": patient_id,
                "oct_id": oct_id,
                "center_name": center_name,
                "center_code": canonical_center_code(center_name, raw_center_code),
                "raw_center_code": raw_center_code,
                "paper_center_no": CENTER_NAME_TO_PAPER_NO.get(center_name, ""),
                "center_idx": center_idx,
                "center_group_id": center_idx,
                "age": row.get("AGE", ""),
                "hpv_raw": row.get("HPV清洗", ""),
                "hpv_clean": row.get("HPV清洗", ""),
                "tct_raw": row.get("TCT清洗", ""),
                "tct_clean": row.get("TCT清洗", ""),
                "label": label,
                "colposcopy_paths": ";".join(col_paths),
                "oct_paths": ";".join(oct_paths),
                "col_count": len(col_paths),
                "oct_count": len(oct_paths),
                "age_available": nonempty(row.get("AGE")),
                "hpv_available": nonempty(row.get("HPV清洗")),
                "tct_available": nonempty(row.get("TCT清洗")),
                "pathology_label_available": not pd.isna(label),
                "original_split": split,
                "full_multimodal_complete": bool(len(col_paths) > 0 and len(oct_paths) > 0 and not pd.isna(label)),
            }
        )
    return pd.DataFrame(rows)


def build_all_cases_audit(raw: pd.DataFrame, final_index: pd.DataFrame, reg_cols: dict[str, str | None]) -> pd.DataFrame:
    final_by_oct = final_index.drop_duplicates("oct_id").set_index("oct_id", drop=False)
    raw_oct_col = reg_cols["oct_id"]
    raw_center_col = reg_cols["center"]
    raw_age_col = reg_cols["age"]
    raw_hpv_col = reg_cols["hpv"]
    raw_tct_col = reg_cols["tct"]
    rows: list[dict[str, Any]] = []
    duplicated_oct = raw[raw_oct_col].astype(str).duplicated(keep=False) if raw_oct_col else pd.Series(False, index=raw.index)

    for raw_idx, row in raw.iterrows():
        raw_oct_id = clean_text(row.get(raw_oct_col)) if raw_oct_col else ""
        matched = raw_oct_id in final_by_oct.index
        aligned = final_by_oct.loc[raw_oct_id] if matched else None
        rows.append(
            {
                "raw_row_index": raw_idx,
                "raw_center_name": clean_text(row.get(raw_center_col)) if raw_center_col else "uncertain",
                "raw_oct_id": raw_oct_id,
                "raw_oct_id_duplicated": bool(duplicated_oct.loc[raw_idx]),
                "in_final_1897": bool(matched),
                "case_id": aligned["case_id"] if matched else "",
                "patient_id": aligned["patient_id"] if matched else "",
                "oct_id": aligned["oct_id"] if matched else raw_oct_id,
                "center_name": aligned["center_name"] if matched else (clean_text(row.get(raw_center_col)) if raw_center_col else "uncertain"),
                "center_code": aligned["center_code"] if matched else "",
                "raw_center_code": aligned.get("raw_center_code", "") if matched else "",
                "age_available": bool(aligned["age_available"]) if matched else nonempty(row.get(raw_age_col)),
                "hpv_available": bool(aligned["hpv_available"]) if matched else nonempty(row.get(raw_hpv_col)),
                "tct_available": bool(aligned["tct_available"]) if matched else nonempty(row.get(raw_tct_col)),
                "pathology_label_available": bool(aligned["pathology_label_available"]) if matched else pathology_available(row, reg_cols),
                "colposcopy_available": bool(aligned["col_count"] > 0) if matched else False,
                "oct_available": bool(aligned["oct_count"] > 0) if matched else False,
                "col_count": int(aligned["col_count"]) if matched else 0,
                "oct_count": int(aligned["oct_count"]) if matched else 0,
                "label": aligned["label"] if matched else "",
                "full_multimodal_complete": bool(aligned["full_multimodal_complete"]) if matched else False,
            }
        )
    return pd.DataFrame(rows)


def build_centerwise_summary(final_index: pd.DataFrame) -> pd.DataFrame:
    grouped = final_index.groupby(["center_name"], dropna=False)
    rows: list[dict[str, Any]] = []
    for center_name, group in grouped:
        if isinstance(center_name, tuple):
            center_name = center_name[0]
        positive = int((group["label"] == 1).sum())
        negative = int((group["label"] == 0).sum())
        center_code = canonical_center_code(center_name)
        raw_code_col = "raw_center_code" if "raw_center_code" in group.columns else "center_code"
        raw_center_codes = ";".join(sorted(str(code) for code in group[raw_code_col].dropna().unique()))
        rows.append(
            {
                "paper_center_no": CENTER_NAME_TO_PAPER_NO.get(str(center_name), ""),
                "center_name": center_name,
                "center_code": center_code,
                "raw_center_codes": raw_center_codes,
                "n_cases": int(len(group)),
                "positive_cases": positive,
                "negative_cases": negative,
                "positive_rate": round(positive / len(group), 4) if len(group) else np.nan,
                "age_available": int(group["age_available"].sum()),
                "hpv_available": int(group["hpv_available"].sum()),
                "tct_available": int(group["tct_available"].sum()),
                "pathology_label_available": int(group["pathology_label_available"].sum()),
                "full_multimodal_complete": int(group["full_multimodal_complete"].sum()),
                "colposcopy_image_count": int(group["col_count"].sum()),
                "oct_image_or_bscan_count": int(group["oct_count"].sum()),
                "total_image_count": int(group["col_count"].sum() + group["oct_count"].sum()),
                "median_colposcopy_images_per_case": float(group["col_count"].median()),
                "median_oct_images_per_case": float(group["oct_count"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["paper_center_no", "center_name"]).reset_index(drop=True)


def build_image_volume_summary(final_index: pd.DataFrame, expected_total: int = 130_000) -> pd.DataFrame:
    col_total = int(final_index["col_count"].sum())
    oct_total = int(final_index["oct_count"].sum())
    total = col_total + oct_total
    lower = int(expected_total * 0.9)
    upper = int(expected_total * 1.1)
    status = "approximately_130k" if lower <= total <= upper else "outside_approx_130k_window"
    rows = [
        {
            "modality": "colposcopy",
            "case_count": int((final_index["col_count"] > 0).sum()),
            "total_images_or_bscans": col_total,
            "median_per_case": float(final_index["col_count"].median()),
            "min_per_case": int(final_index["col_count"].min()),
            "max_per_case": int(final_index["col_count"].max()),
            "mean_per_case": round(float(final_index["col_count"].mean()), 2),
            "approx_130k_total_status": status,
        },
        {
            "modality": "oct",
            "case_count": int((final_index["oct_count"] > 0).sum()),
            "total_images_or_bscans": oct_total,
            "median_per_case": float(final_index["oct_count"].median()),
            "min_per_case": int(final_index["oct_count"].min()),
            "max_per_case": int(final_index["oct_count"].max()),
            "mean_per_case": round(float(final_index["oct_count"].mean()), 2),
            "approx_130k_total_status": status,
        },
        {
            "modality": "total",
            "case_count": int(len(final_index)),
            "total_images_or_bscans": total,
            "median_per_case": float((final_index["col_count"] + final_index["oct_count"]).median()),
            "min_per_case": int((final_index["col_count"] + final_index["oct_count"]).min()),
            "max_per_case": int((final_index["col_count"] + final_index["oct_count"]).max()),
            "mean_per_case": round(float((final_index["col_count"] + final_index["oct_count"]).mean()), 2),
            "approx_130k_total_status": status,
        },
    ]
    return pd.DataFrame(rows)


def build_cohort_flow(raw: pd.DataFrame, final_index: pd.DataFrame, all_cases: pd.DataFrame, expected_raw_n: int) -> pd.DataFrame:
    raw_n = len(raw)
    aligned_n = len(final_index)
    raw_unique_oct = int(all_cases["raw_oct_id"].replace("", np.nan).nunique())
    raw_duplicate_oct_rows = int(all_cases["raw_oct_id_duplicated"].sum())
    complete_n = int(final_index["full_multimodal_complete"].sum())
    return pd.DataFrame(
        [
            {
                "step": "raw_registry_cases",
                "n": raw_n,
                "removed_from_previous_step": 0,
                "note": f"Expected raw registry N={expected_raw_n}.",
            },
            {
                "step": "raw_registry_unique_oct_ids",
                "n": raw_unique_oct,
                "removed_from_previous_step": raw_n - raw_unique_oct,
                "note": f"Rows with duplicated OCT_ID: {raw_duplicate_oct_rows}. patient_id unavailable in raw registry unless supplied by input columns.",
            },
            {
                "step": "final_tri_modal_aligned_modeling_cohort",
                "n": aligned_n,
                "removed_from_previous_step": raw_n - aligned_n,
                "note": "Aligned colposcopy images, OCT images/B-scans, and HPV/TCT/Age clinical variable rows; examination reports not used.",
            },
            {
                "step": "full_multimodal_complete_flag_true",
                "n": complete_n,
                "removed_from_previous_step": aligned_n - complete_n,
                "note": "Flag requires colposcopy image(s), OCT image/B-scan(s), and supervised label availability.",
            },
        ]
    )


def write_readme(output_dir: Path) -> None:
    text = """# Full Multimodal Resplit Cohort Alignment Audit

The current main experiment uses 1897 patients with aligned colposcopy images, OCT images, and HPV/TCT/Age clinical text variables from an original registry of 3010 cases. Examination reports are not used.

Pathology labels are retained only as supervised training and evaluation labels. They are not inserted into clinical_info or any clinical text input.

Generated audit files:
- `full_multimodal_all_cases_audit.csv`: 3010-row registry-to-aligned-cohort audit.
- `final_1897_case_index.csv`: final no-report 1897-case modeling index.
- `../../tables/cohort_flow_3010_to_1897.csv`: cohort flow summary.
- `../../tables/image_volume_summary.csv`: image and B-scan volume summary.
- `../../tables/centerwise_aligned_case_summary.csv`: center-wise aligned cohort summary.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def print_dry_run(
    raw_n: int,
    aligned_n: int,
    image_summary: pd.DataFrame,
    centerwise: pd.DataFrame,
    final_index: pd.DataFrame,
    absence: dict[str, Any],
) -> None:
    total_col = int(final_index["col_count"].sum())
    total_oct = int(final_index["oct_count"].sum())
    total_images = total_col + total_oct
    missing = {
        "missing_colposcopy": int((final_index["col_count"] == 0).sum()),
        "missing_oct": int((final_index["oct_count"] == 0).sum()),
        "missing_age": int((~final_index["age_available"]).sum()),
        "missing_hpv": int((~final_index["hpv_available"]).sum()),
        "missing_tct": int((~final_index["tct_available"]).sum()),
        "missing_pathology_label": int((~final_index["pathology_label_available"]).sum()),
    }
    print(f"raw_n: {raw_n}")
    print(f"aligned_n: {aligned_n}")
    print(f"total_colposcopy_images: {total_col}")
    print(f"total_oct_images: {total_oct}")
    print(f"total_images: {total_images}")
    print("centerwise summary:")
    print(centerwise.to_string(index=False))
    print("missing modality counts:")
    for key, value in missing.items():
        print(f"  {key}: {value}")
    print("report-field absence check:")
    print(f"  {absence['report_field_absence_check']}")
    print(f"  forbidden_report_fields_present: {absence['forbidden_report_fields_present']}")
    print("image volume summary:")
    print(image_summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    raw_csv = resolve_path(args.raw_registry_csv, default=None)
    raw_xlsx = resolve_path(args.raw_registry_xlsx, default=DEFAULT_RAW_XLSX)
    col_root = resolve_path(args.colposcopy_root, default=DEFAULT_COHORT_ROOT)
    oct_root = resolve_path(args.oct_root, default=DEFAULT_COHORT_ROOT)
    output_dir = resolve_path(args.output_dir, default=DEFAULT_OUTPUT_DIR)
    assert col_root is not None and oct_root is not None and output_dir is not None

    raw, raw_path = load_raw_registry(raw_csv, raw_xlsx)
    cohort_root = DEFAULT_COHORT_ROOT
    # If either modality root is the All_3000_5cens root, use it as manifest root.
    for candidate in [col_root, oct_root]:
        if (candidate / "train_labels.csv").exists() and (candidate / "test_labels.csv").exists():
            cohort_root = candidate
            break
    manifest = load_aligned_manifest(cohort_root)
    absence = report_field_absence_check(raw, manifest)
    reg_cols = registry_columns(raw)
    final_index = build_final_index(manifest, col_root, oct_root)
    all_cases = build_all_cases_audit(raw, final_index, reg_cols)
    centerwise = build_centerwise_summary(final_index)
    image_summary = build_image_volume_summary(final_index)
    cohort_flow = build_cohort_flow(raw, final_index, all_cases, args.expected_raw_n)

    raw_n = len(raw)
    aligned_n = int(final_index["full_multimodal_complete"].sum())
    total_images = int(final_index["col_count"].sum() + final_index["oct_count"].sum())

    if raw_n != args.expected_raw_n:
        warnings.warn(f"raw_n={raw_n} does not match expected {args.expected_raw_n}", RuntimeWarning)
    if aligned_n != args.expected_aligned_n:
        message = f"final aligned N={aligned_n} does not match expected {args.expected_aligned_n}"
        if args.allow_count_mismatch:
            warnings.warn(message, RuntimeWarning)
        else:
            raise RuntimeError(message)
    if not (117_000 <= total_images <= 143_000):
        warnings.warn(f"total image count {total_images} is outside the approximate 130k window", RuntimeWarning)

    print(f"Using raw registry: {raw_path}")
    print(f"Using aligned manifest root: {cohort_root}")
    print(f"Total image count: {total_images}")

    if args.dry_run:
        print_dry_run(raw_n, aligned_n, image_summary, centerwise, final_index, absence)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_cases.to_csv(output_dir / "full_multimodal_all_cases_audit.csv", index=False, encoding="utf-8-sig")
    final_columns = [
        "case_id",
        "patient_id",
        "oct_id",
        "center_name",
        "center_code",
        "raw_center_code",
        "paper_center_no",
        "center_idx",
        "center_group_id",
        "age",
        "hpv_raw",
        "hpv_clean",
        "tct_raw",
        "tct_clean",
        "label",
        "colposcopy_paths",
        "oct_paths",
        "col_count",
        "oct_count",
        "full_multimodal_complete",
    ]
    final_index[final_columns].to_csv(output_dir / "final_1897_case_index.csv", index=False, encoding="utf-8-sig")
    cohort_flow.to_csv(TABLE_DIR / "cohort_flow_3010_to_1897.csv", index=False, encoding="utf-8-sig")
    image_summary.to_csv(TABLE_DIR / "image_volume_summary.csv", index=False, encoding="utf-8-sig")
    centerwise.to_csv(TABLE_DIR / "centerwise_aligned_case_summary.csv", index=False, encoding="utf-8-sig")
    write_readme(output_dir)

    print(f"Wrote: {output_dir / 'full_multimodal_all_cases_audit.csv'}")
    print(f"Wrote: {output_dir / 'final_1897_case_index.csv'}")
    print(f"Wrote: {TABLE_DIR / 'cohort_flow_3010_to_1897.csv'}")
    print(f"Wrote: {TABLE_DIR / 'image_volume_summary.csv'}")
    print(f"Wrote: {TABLE_DIR / 'centerwise_aligned_case_summary.csv'}")
    print(f"Wrote: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
