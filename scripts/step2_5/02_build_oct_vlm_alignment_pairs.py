#!/usr/bin/env python3
import argparse

from common import build_oct_vlm_alignment_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = build_oct_vlm_alignment_pairs(args.config)
    print(path)


if __name__ == "__main__":
    main()
