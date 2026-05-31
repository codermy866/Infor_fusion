#!/usr/bin/env python3
import argparse

from common import plot_step2_5_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = plot_step2_5_figures(args.input_dir, args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
