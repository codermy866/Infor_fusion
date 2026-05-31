#!/usr/bin/env python3
import argparse

from common import plot_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plot_figures(args.config)


if __name__ == "__main__":
    main()
