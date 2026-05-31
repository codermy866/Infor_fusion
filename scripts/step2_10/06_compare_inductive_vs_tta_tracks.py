#!/usr/bin/env python3
import argparse

from common import compare_inductive_vs_tta_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    compare_inductive_vs_tta_tracks(args.config)


if __name__ == "__main__":
    main()
