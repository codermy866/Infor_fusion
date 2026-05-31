#!/usr/bin/env python3
import argparse

from common import run_full_module_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    path = run_full_module_ablation(args.config)
    print(path)


if __name__ == "__main__":
    main()
