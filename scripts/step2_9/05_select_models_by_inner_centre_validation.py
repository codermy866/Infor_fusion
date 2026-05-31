#!/usr/bin/env python3
import argparse

from common import select_models_by_inner_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    select_models_by_inner_validation(args.config)


if __name__ == "__main__":
    main()
