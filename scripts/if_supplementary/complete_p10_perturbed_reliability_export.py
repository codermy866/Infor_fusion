#!/usr/bin/env python3
"""Complete P10 clean-vs-perturbed reliability weight export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.if_supplementary.common import DEFAULT_OUT_ROOT
from src.if_supplementary.reliability_perturbations import run_p10_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="strict_loco")
    parser.add_argument("--save-clean", action="store_true")
    parser.add_argument("--save-perturbed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    save_clean = args.save_clean or not args.save_perturbed
    save_perturbed = args.save_perturbed or not args.save_clean
    outputs = run_p10_completion(
        out_root=args.out_root,
        protocol=args.protocol,
        save_clean=save_clean,
        save_perturbed=save_perturbed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
