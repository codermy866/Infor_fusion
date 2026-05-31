#!/usr/bin/env python3
import argparse

from common import plot_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(plot_figures(args.input_dir, args.output_dir))


if __name__ == "__main__":
    main()
