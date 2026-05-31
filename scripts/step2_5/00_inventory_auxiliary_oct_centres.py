#!/usr/bin/env python3
import argparse

from common import inventory_auxiliary_oct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = inventory_auxiliary_oct(args.config)
    print(path)


if __name__ == "__main__":
    main()
