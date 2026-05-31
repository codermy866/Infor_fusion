#!/usr/bin/env python3
import argparse

from common import pretrain_oct_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    path = pretrain_oct_encoder(args.config, no_dry_run=args.no_dry_run)
    print(path)


if __name__ == "__main__":
    main()
