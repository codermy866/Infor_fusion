#!/usr/bin/env python3
import argparse

from common import overfit_and_sanity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(overfit_and_sanity(args.config))


if __name__ == "__main__":
    main()
