#!/usr/bin/env python3
import argparse

from common import verify_step2_9_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    verify_step2_9_outputs(args.config)


if __name__ == "__main__":
    main()
