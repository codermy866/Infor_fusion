#!/usr/bin/env python3
import argparse

from common import locate_full_hydra_codepaths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = locate_full_hydra_codepaths(args.repo_root, args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
