#!/usr/bin/env python3
import argparse

from common import collect_step2_5_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = collect_step2_5_results(args.config)
    print(path)


if __name__ == "__main__":
    main()
