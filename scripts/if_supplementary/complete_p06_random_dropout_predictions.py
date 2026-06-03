#!/usr/bin/env python3
"""Complete P06 patient-level random modality dropout predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.if_supplementary.common import DEFAULT_OUT_ROOT
from src.if_supplementary.random_dropout import run_p06_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="strict_loco")
    parser.add_argument("--dropout-rates", type=float, nargs="+", default=[0.10, 0.30, 0.50])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[202601, 202602, 202603, 202604, 202605])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_p06_completion(
        out_root=args.out_root,
        dropout_rates=args.dropout_rates,
        repeats=args.repeats,
        seeds=args.seeds,
        protocol=args.protocol,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
