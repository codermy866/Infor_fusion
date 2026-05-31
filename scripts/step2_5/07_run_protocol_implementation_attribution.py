#!/usr/bin/env python3
import argparse

from common import run_protocol_implementation_attribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_protocol_implementation_attribution(args.config)
    print(path)


if __name__ == "__main__":
    main()
