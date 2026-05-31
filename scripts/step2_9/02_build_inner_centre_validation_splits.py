#!/usr/bin/env python3
import argparse

from common import build_inner_centre_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_inner_centre_splits(args.config)


if __name__ == "__main__":
    main()
