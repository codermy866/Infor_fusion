#!/usr/bin/env python3
import argparse

from common import reaudit_endpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-lock", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(reaudit_endpoint(args.data_lock, args.split_manifest, args.output_dir))


if __name__ == "__main__":
    main()
