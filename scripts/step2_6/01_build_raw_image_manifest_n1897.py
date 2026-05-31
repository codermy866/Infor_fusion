#!/usr/bin/env python3
import argparse

from common import build_raw_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(build_raw_manifest(args.data_lock, args.output_dir))


if __name__ == "__main__":
    main()
