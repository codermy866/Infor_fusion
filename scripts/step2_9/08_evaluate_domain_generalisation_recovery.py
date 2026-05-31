#!/usr/bin/env python3
import argparse

from common import evaluate_domain_generalisation_recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_domain_generalisation_recovery(args.config)


if __name__ == "__main__":
    main()
