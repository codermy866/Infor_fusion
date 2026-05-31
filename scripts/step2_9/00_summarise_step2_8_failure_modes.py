#!/usr/bin/env python3
import argparse

from common import summarise_step2_8_failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summarise_step2_8_failure(args.config)


if __name__ == "__main__":
    main()
