#!/usr/bin/env python3
"""Sanitize paper_revisions CSV files before publishing them to git.

The locked working outputs may contain raw identifier columns inherited from
upstream registries. This script preserves hash identifiers and removes raw
patient/case columns in-place for repository publication.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.if_supplementary.common import hash_id, save_csv


HEX16 = re.compile(r"^[0-9a-f]{16}$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "paper_revisions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(args.root.rglob("*.csv")):
        changed, details = sanitize_csv(path)
        if changed:
            try:
                display_path = str(path.resolve().relative_to(ROOT.resolve()))
            except ValueError:
                display_path = str(path)
            rows.append({"path": display_path, "details": details})
    report = pd.DataFrame(rows)
    if len(report):
        save_csv(report, args.root / "if_supplementary_experiments/13_submission_audit/GIT_SANITIZATION_REPORT.csv")
    print(f"sanitized_csv_files={len(report)}")
    if len(report):
        print(report.to_string(index=False))
    return 0


def sanitize_csv(path: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return False, "read_failed"
    original_cols = list(df.columns)
    notes = []

    if "patient_id" in df.columns:
        if "patient_id_hash" not in df.columns:
            sample = df["patient_id"].dropna().astype(str).head(50)
            if len(sample) and sample.map(lambda x: bool(HEX16.match(x))).all():
                df = df.rename(columns={"patient_id": "patient_id_hash"})
                notes.append("renamed hashed patient_id to patient_id_hash")
            else:
                df["patient_id_hash"] = df["patient_id"].map(hash_id)
                df = df.drop(columns=["patient_id"])
                notes.append("hashed and removed patient_id")
        else:
            df = df.drop(columns=["patient_id"])
            notes.append("removed patient_id because patient_id_hash exists")

    if "case_id" in df.columns:
        if "case_id_hash" not in df.columns:
            sample = df["case_id"].dropna().astype(str).head(50)
            if len(sample) and sample.map(lambda x: bool(HEX16.match(x))).all():
                df = df.rename(columns={"case_id": "case_id_hash"})
                notes.append("renamed hashed case_id to case_id_hash")
            else:
                df["case_id_hash"] = df["case_id"].map(hash_id)
                df = df.drop(columns=["case_id"])
                notes.append("hashed and removed case_id")
        else:
            df = df.drop(columns=["case_id"])
            notes.append("removed case_id because case_id_hash exists")

    for col in ["raw_patient_id", "raw_case_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            notes.append(f"removed {col}")

    if list(df.columns) == original_cols:
        return False, ""
    save_csv(df, path)
    return True, "; ".join(notes)


if __name__ == "__main__":
    raise SystemExit(main())
