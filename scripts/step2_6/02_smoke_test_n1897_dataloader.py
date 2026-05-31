#!/usr/bin/env python3
import argparse

from common import smoke_test_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(smoke_test_dataloader(args.manifest, args.split_manifest, args.output_dir))


if __name__ == "__main__":
    main()
