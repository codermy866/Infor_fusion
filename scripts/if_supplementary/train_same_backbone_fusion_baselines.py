#!/usr/bin/env python3
"""Stub entry point for exact same-backbone fusion baseline training.

The current supplementary package evaluates all variants with available locked
patient-level predictions. Exact retraining for the full ten-variant
same-backbone suite requires a raw/frozen-feature training run with identical
encoders, preprocessing, optimizer, schedule, and LOCO splits.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "paper_revisions/if_supplementary_experiments/04_same_backbone_fusion_baselines/SAME_BACKBONE_TRAINING_STUB.md"


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Same-Backbone Training Stub\n\n"
        "No new training was launched by this stub. The orchestrator reuses locked "
        "patient-level predictions and records unavailable exact controls in "
        "`same_backbone_ablation_table.csv` and `MISSING_REQUIREMENTS.md`.\n\n"
        "To make this a full training script, implement the ten variants listed in "
        "`configs/if_supplementary_same_backbone_baselines.yaml` with identical "
        "backbones, preprocessing, optimizer, schedule, and strict LOCO splits.\n",
        encoding="utf-8",
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
