#!/usr/bin/env python3
"""R0: Archive legacy mixed-n result tables (external_n != 403)."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [
    ROOT / "paper_revision" / "results",
    ROOT / "paper_revision" / "tables",
]
ARCHIVE_DIR = ROOT / "paper_revision" / "results" / "archive_legacy_mixed_n"
ALLOWED_EXTERNAL_N = {403}
METRIC_COLS = ("n", "external_n", "heldout_test_n", "test_n")


def infer_external_n(df: pd.DataFrame) -> set[int]:
    values: set[int] = set()
    if "split" in df.columns:
        ext = df[df["split"].astype(str).str.contains("external", case=False, na=False)]
        if "n" in ext.columns:
            values.update(int(x) for x in pd.to_numeric(ext["n"], errors="coerce").dropna().unique())
    for col in METRIC_COLS:
        if col in df.columns:
            values.update(int(x) for x in pd.to_numeric(df[col], errors="coerce").dropna().unique())
    return values


def is_legacy_file(path: Path) -> tuple[bool, str]:
    if path.suffix.lower() not in {".csv", ".json"}:
        return False, ""
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5000)
            ns = infer_external_n(df)
            bad = [n for n in ns if n not in ALLOWED_EXTERNAL_N and n > 0]
            if bad:
                return True, f"external_n in {sorted(bad)}"
        elif path.suffix == ".json":
            payload = pd.read_json(path)
            if isinstance(payload, dict):
                for key in ("external_n", "external_test_n", "n"):
                    if key in payload and int(payload[key]) not in ALLOWED_EXTERNAL_N:
                        return True, f"{key}={payload[key]}"
    except Exception as exc:
        return False, f"unreadable:{exc}"
    return False, ""


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_rows: list[dict] = []
    status_rows: list[dict] = []

    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if "archive_legacy_mixed_n" in str(rel) or "real_50epoch_5center_corrected" in str(rel):
                continue
            legacy, reason = is_legacy_file(path)
            status_rows.append(
                {
                    "path": str(rel),
                    "legacy_not_for_main_paper": legacy,
                    "reason": reason,
                    "scanned_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if not legacy:
                continue
            dest = ARCHIVE_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(path, dest)
            legacy_rows.append({"source": str(rel), "archive_copy": str(dest.relative_to(ROOT)), "reason": reason})

    scope = ROOT / "paper_revision" / "results" / "CLEAN_RERUN_SCOPE.md"
    scope.write_text(
        "# Clean Rerun Scope\n\n"
        "Only results generated from the corrected 1897-patient cohort with "
        "**403-case external test** are eligible for main-paper tables.\n\n"
        "Older 148/196/283 external-test outputs are archived under "
        "`paper_revision/results/archive_legacy_mixed_n/` and must not be used "
        "in main manuscript tables.\n",
        encoding="utf-8",
    )

    manifest = ARCHIVE_DIR / "legacy_results_manifest.csv"
    status = ROOT / "paper_revision" / "results" / "current_result_status_before_clean_rerun.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "archive_copy", "reason"])
        writer.writeheader()
        writer.writerows(legacy_rows)
    pd.DataFrame(status_rows).to_csv(status, index=False, encoding="utf-8-sig")
    print(f"Archived {len(legacy_rows)} legacy files -> {ARCHIVE_DIR}")
    print(f"Wrote {manifest} and {status}")


if __name__ == "__main__":
    main()
