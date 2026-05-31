#!/usr/bin/env python3
import argparse

from common import generate_dg_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate_dg_tables(args.config)


if __name__ == "__main__":
    main()
