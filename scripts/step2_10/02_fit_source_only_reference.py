#!/usr/bin/env python3
import argparse

from common import fit_source_only_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    fit_source_only_reference(args.config, no_dry_run=args.no_dry_run)


if __name__ == "__main__":
    main()
