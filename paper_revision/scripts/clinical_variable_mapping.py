#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize HPV, TCT, and Age variables for the no-report HyDRA-CoE cohort.

Current experiments use only OCT, colposcopy, and HPV/TCT/Age clinical text
variables as model inputs. Pathology labels are supervised targets only and are
never inserted into clinical_info.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = EXP_ROOT / "paper_revision" / "splits" / "full_multimodal_resplit" / "final_1897_case_index.csv"
DEFAULT_OUTPUT = EXP_ROOT / "paper_revision" / "tables" / "clinical_variable_mapping_audit.csv"

HPV_CATEGORIES = [
    "hpv_negative",
    "hrhpv_positive",
    "hpv16_18_positive",
    "other_hrhpv_positive",
    "hpv_unknown",
]

TCT_CATEGORIES = [
    "NILM",
    "ASC-US",
    "LSIL",
    "ASC-H",
    "HSIL",
    "AGC",
    "SCC_or_cancer_suspicious",
    "TCT_unknown",
]

HR_HPV_GENOTYPES = {
    "16",
    "18",
    "31",
    "33",
    "35",
    "39",
    "45",
    "51",
    "52",
    "53",
    "56",
    "58",
    "59",
    "66",
    "68",
    "73",
    "81",
    "82",
}
HPV16_18 = {"16", "18"}

AGE_UNKNOWN = "age_unknown"

HPV_COLUMN_CANDIDATES = [
    "hpv_clean",
    "hpv_raw",
    "hpv",
    "HPV清洗",
    "HPV",
    "HPV结果",
    "hrhpv",
]
TCT_COLUMN_CANDIDATES = [
    "tct_clean",
    "tct_raw",
    "tct",
    "TCT清洗",
    "TCT",
    "cytology",
    "cytology_raw",
]
AGE_COLUMN_CANDIDATES = ["age", "AGE", "Age", "年龄"]

REPORT_LIKE_COLUMNS = {
    "report",
    "report_text",
    "clinical_report",
    "diagnosis_report",
    "generated_report",
    "exam_report",
    "examination_report",
    "检查报告",
    "诊断报告",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a", "unknown", "未查", "未知"}


def _first_present(row: Mapping[str, Any] | pd.Series, candidates: list[str]) -> Any:
    getter = row.get if hasattr(row, "get") else row.__getitem__
    for column in candidates:
        try:
            value = getter(column)
        except (KeyError, TypeError):
            continue
        if not _is_missing(value):
            return value
    return None


def _clean_text(value: Any) -> str:
    return str(value).strip().replace("（", "(").replace("）", ")").replace("，", ",")


def normalize_age(value: Any) -> int | str:
    """Return integer age or age_unknown."""
    if _is_missing(value):
        return AGE_UNKNOWN
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        age = int(round(float(value)))
        return age if 0 < age < 120 else AGE_UNKNOWN

    text = _clean_text(value)
    match = re.search(r"(?<!\d)(\d{1,3})(?:\s*岁|y|yr|years?)?", text, flags=re.IGNORECASE)
    if not match:
        return AGE_UNKNOWN
    age = int(match.group(1))
    return age if 0 < age < 120 else AGE_UNKNOWN


def _extract_hr_hpv_genotypes(text: str) -> set[str]:
    """Extract explicit high-risk HPV genotypes.

    This deliberately does not treat a lone "1" as positive. Only recognized
    genotype numbers such as 16, 18, 52, etc. count as genotype evidence.
    """
    found: set[str] = set()
    for genotype in HR_HPV_GENOTYPES:
        pattern = rf"(?<!\d){re.escape(genotype)}(?!\d)"
        if re.search(pattern, text):
            found.add(genotype)
    return found


def normalize_hpv(value: Any) -> str:
    """Normalize HPV to one of the fixed HPV_CATEGORIES."""
    if _is_missing(value):
        return "hpv_unknown"

    text = _clean_text(value)
    lower = text.lower()

    # Explicit negative evidence wins when no positive genotype is present.
    negative_patterns = [
        r"\bnegative\b",
        r"\bneg\b",
        r"\bnot\s+detected\b",
        r"\bundetected\b",
        r"\bnone\s+detected\b",
        r"阴性",
        r"未检出",
        r"未见",
        r"未检测到",
        r"无感染",
        r"(-|－)\s*$",
    ]
    has_negative = any(re.search(pattern, lower) for pattern in negative_patterns)

    genotypes = _extract_hr_hpv_genotypes(lower)
    positive_patterns = [
        r"\bpositive\b",
        r"\bpos\b",
        r"\bdetected\b",
        r"\bhr\s*-?\s*hpv\b.*\bpos",
        r"\bhpv\b.*\bpos",
        r"阳性",
        r"检出",
        r"高危",
        r"hrhpv",
    ]
    has_positive = any(re.search(pattern, lower) for pattern in positive_patterns)

    if genotypes and not has_negative:
        if genotypes & HPV16_18:
            return "hpv16_18_positive"
        return "other_hrhpv_positive"

    if has_negative and not has_positive:
        return "hpv_negative"

    if has_positive:
        return "hrhpv_positive"

    # A numeric high-risk genotype by itself is explicit genotype evidence. A
    # lone "1" is not a valid high-risk genotype and remains unknown.
    compact = lower.strip()
    if compact in HR_HPV_GENOTYPES:
        return "hpv16_18_positive" if compact in HPV16_18 else "other_hrhpv_positive"
    if re.fullmatch(r"\d+(?:\.0+)?", compact):
        number = str(int(float(compact)))
        if number in HR_HPV_GENOTYPES:
            return "hpv16_18_positive" if number in HPV16_18 else "other_hrhpv_positive"

    return "hpv_unknown"


def normalize_tct(value: Any) -> str:
    """Normalize TCT/cytology text to one of the fixed TCT_CATEGORIES."""
    if _is_missing(value):
        return "TCT_unknown"

    text = _clean_text(value)
    upper = text.upper().replace("_", "-")

    if re.search(r"\bNILM\b|未见上皮内病变|未见.*恶性|阴性", upper):
        return "NILM"
    if re.search(r"SCC|鳞癌|癌|恶性|CANCER|CARCINOMA", upper):
        return "SCC_or_cancer_suspicious"
    if re.search(r"\bAGC\b|腺细胞", upper):
        return "AGC"
    if re.search(r"\bASC[\s-]*H\b|不能除外高级别|不除外高级别", upper):
        return "ASC-H"
    if re.search(r"\bHSIL\b|高级别鳞状上皮内病变", upper):
        return "HSIL"
    if re.search(r"\bLSIL\b|低级别鳞状上皮内病变", upper):
        return "LSIL"
    if re.search(r"\bASC[\s-]*US\b|意义不明确|非典型鳞状细胞", upper):
        return "ASC-US"
    return "TCT_unknown"


def clinical_info_from_row(row: Mapping[str, Any] | pd.Series) -> str:
    """Return the exact no-report clinical_info string used by the model."""
    hpv = normalize_hpv(_first_present(row, HPV_COLUMN_CANDIDATES))
    tct = normalize_tct(_first_present(row, TCT_COLUMN_CANDIDATES))
    age = normalize_age(_first_present(row, AGE_COLUMN_CANDIDATES))
    return f"HPV: {hpv}, TCT: {tct}, Age: {age}"


def clinical_features_from_row(row: Mapping[str, Any] | pd.Series) -> list[float]:
    """Return a 14-D structured vector: age + HPV one-hot + TCT one-hot."""
    age = normalize_age(_first_present(row, AGE_COLUMN_CANDIDATES))
    age_scaled = float(age) / 100.0 if isinstance(age, int) else 0.0
    hpv = normalize_hpv(_first_present(row, HPV_COLUMN_CANDIDATES))
    tct = normalize_tct(_first_present(row, TCT_COLUMN_CANDIDATES))
    hpv_features = [1.0 if hpv == category else 0.0 for category in HPV_CATEGORIES]
    tct_features = [1.0 if tct == category else 0.0 for category in TCT_CATEGORIES]
    return [age_scaled, *hpv_features, *tct_features]


def assert_no_report_columns(columns: list[str] | pd.Index) -> None:
    report_cols = [
        str(col)
        for col in columns
        if str(col).strip().lower() in REPORT_LIKE_COLUMNS or "report" in str(col).strip().lower()
    ]
    if report_cols:
        raise ValueError(f"Report-like columns are not allowed in current experiments: {report_cols}")


def build_mapping_audit(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input cohort index not found: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    assert_no_report_columns(df.columns)

    rows: list[dict[str, Any]] = []
    for variable, candidates, normalizer in [
        ("HPV", HPV_COLUMN_CANDIDATES, normalize_hpv),
        ("TCT", TCT_COLUMN_CANDIDATES, normalize_tct),
        ("Age", AGE_COLUMN_CANDIDATES, normalize_age),
    ]:
        raw_values = df.apply(lambda row: _first_present(row, candidates), axis=1)
        normalized = raw_values.apply(normalizer)
        temp = pd.DataFrame({"raw_value": raw_values.fillna(""), "normalized": normalized})
        for (raw_value, normalized_value), group in temp.groupby(["raw_value", "normalized"], dropna=False):
            rows.append(
                {
                    "variable": variable,
                    "raw_value": str(raw_value),
                    "normalized": str(normalized_value),
                    "n": int(len(group)),
                }
            )

    audit = pd.DataFrame(rows).sort_values(["variable", "normalized", "n"], ascending=[True, True, False])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return audit


def run_tests() -> None:
    assert normalize_hpv("HPV16 positive") == "hpv16_18_positive"
    assert normalize_hpv("HPV negative") == "hpv_negative"
    assert normalize_hpv("阴性") == "hpv_negative"
    assert normalize_hpv("1") == "hpv_unknown"
    assert normalize_tct("NILM") == "NILM"
    assert normalize_tct("未见上皮内病变") == "NILM"
    assert normalize_tct("未见上皮内病变或恶性病变") == "NILM"
    assert normalize_tct("ASC-US") == "ASC-US"
    assert normalize_tct("LSIL") == "LSIL"
    assert normalize_tct("ASC-H") == "ASC-H"
    assert normalize_tct("HSIL") == "HSIL"
    assert normalize_tct("AGC") == "AGC"
    clinical_info = clinical_info_from_row({"hpv_raw": "HPV16 positive", "tct_raw": "HSIL", "age": 42, "label": 1})
    assert clinical_info == "HPV: hpv16_18_positive, TCT: HSIL, Age: 42"
    assert "label" not in clinical_info.lower()
    assert len(clinical_features_from_row({"hpv_raw": "阴性", "tct_raw": "NILM", "age": 50})) == 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_tests:
        run_tests()
    audit = build_mapping_audit(args.input, args.output)
    print(f"Wrote {len(audit)} clinical mapping audit rows to {args.output}")


if __name__ == "__main__":
    main()
