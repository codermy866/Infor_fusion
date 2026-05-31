#!/usr/bin/env python3
import argparse

from common import validation_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    validation_search(args.config)


if __name__ == "__main__":
    main()
