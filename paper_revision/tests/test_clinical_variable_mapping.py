#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for HPV/TCT/Age no-report clinical variable mapping."""

from __future__ import annotations

import sys
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.clinical_variable_mapping import (  # noqa: E402
    clinical_info_from_row,
    normalize_hpv,
    normalize_tct,
)


def test_hpv16_positive_maps_to_hpv16_18_positive() -> None:
    assert normalize_hpv("HPV16 positive") == "hpv16_18_positive"


def test_hpv_negative_maps_to_negative() -> None:
    assert normalize_hpv("HPV negative") == "hpv_negative"
    assert normalize_hpv("阴性") == "hpv_negative"


def test_lone_one_is_not_hpv_positive() -> None:
    assert normalize_hpv("1") == "hpv_unknown"


def test_nilm_maps_to_nilm() -> None:
    assert normalize_tct("NILM") == "NILM"
    assert normalize_tct("未见上皮内病变") == "NILM"
    assert normalize_tct("未见上皮内病变或恶性病变") == "NILM"


def test_tct_abnormal_categories() -> None:
    assert normalize_tct("ASC-US") == "ASC-US"
    assert normalize_tct("LSIL") == "LSIL"
    assert normalize_tct("ASC-H") == "ASC-H"
    assert normalize_tct("HSIL") == "HSIL"
    assert normalize_tct("AGC") == "AGC"


def test_labels_never_enter_clinical_info() -> None:
    text = clinical_info_from_row({"hpv_raw": "HPV16 positive", "tct_raw": "HSIL", "age": 42, "label": 1})
    assert text == "HPV: hpv16_18_positive, TCT: HSIL, Age: 42"
    assert "label" not in text.lower()
    assert "cin" not in text.lower()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("clinical_variable_mapping tests passed")
