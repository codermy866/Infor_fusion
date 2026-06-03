#!/usr/bin/env python3
"""Complete P11 CoE intervention-logit audit outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.if_supplementary.coe_interventions import run_p11_completion
from src.if_supplementary.common import DEFAULT_OUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="strict_loco")
    parser.add_argument("--clinical-interventions", action="store_true")
    parser.add_argument("--visual-interventions", action="store_true")
    parser.add_argument("--random-control-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_p11_completion(
        out_root=args.out_root,
        protocol=args.protocol,
        clinical_interventions=args.clinical_interventions,
        visual_interventions=args.visual_interventions,
        random_control_repeats=args.random_control_repeats,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
