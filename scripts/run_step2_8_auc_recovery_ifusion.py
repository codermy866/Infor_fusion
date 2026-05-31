#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "step2_8"))

from common import OUT_DIR, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    run_all(args.config, args.output_dir, no_dry_run=args.no_dry_run)


if __name__ == "__main__":
    main()
