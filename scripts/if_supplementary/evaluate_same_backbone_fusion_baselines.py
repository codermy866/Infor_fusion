#!/usr/bin/env python3
"""Evaluate available same-backbone fusion baseline predictions.

This wrapper delegates to the unified supplementary orchestrator, which reads
locked predictions, computes patient-level metrics, and writes P05 outputs.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "scripts/if_supplementary/run_all_if_supplementary_experiments.py"


def main() -> int:
    runpy.run_path(str(ORCHESTRATOR), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
