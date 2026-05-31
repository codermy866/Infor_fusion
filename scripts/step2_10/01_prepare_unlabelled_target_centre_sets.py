#!/usr/bin/env python3
import argparse

from common import prepare_unlabelled_target_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare_unlabelled_target_sets(args.config)


if __name__ == "__main__":
    main()
