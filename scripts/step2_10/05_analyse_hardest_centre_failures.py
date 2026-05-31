#!/usr/bin/env python3
import argparse

from common import analyse_hardest_centre_failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyse_hardest_centre_failures(args.config)


if __name__ == "__main__":
    main()
