#!/usr/bin/env python3
import argparse

from common import evaluate_tta_and_safety


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_tta_and_safety(args.config)


if __name__ == "__main__":
    main()
