#!/usr/bin/env python3
import argparse

from common import build_dg_ensembles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_dg_ensembles(args.config)


if __name__ == "__main__":
    main()
