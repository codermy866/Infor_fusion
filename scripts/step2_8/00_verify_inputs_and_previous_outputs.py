#!/usr/bin/env python3
import argparse

from common import verify_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    verify_inputs(args.config)


if __name__ == "__main__":
    main()
