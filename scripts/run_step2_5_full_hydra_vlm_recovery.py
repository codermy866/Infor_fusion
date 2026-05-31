#!/usr/bin/env python3
import argparse

from step2_5.common import run_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-config", required=True)
    parser.add_argument("--oct-ssl-config", required=True)
    parser.add_argument("--oct-vlm-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    run_all(
        full_config=args.full_config,
        oct_ssl_config=args.oct_ssl_config,
        oct_vlm_config=args.oct_vlm_config,
        output_dir=args.output_dir,
        no_dry_run=args.no_dry_run,
    )


if __name__ == "__main__":
    main()
