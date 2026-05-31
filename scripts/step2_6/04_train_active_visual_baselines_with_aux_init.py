#!/usr/bin/env python3
import argparse

from common import train_active_visual_baselines_with_aux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--oct-ssl-checkpoint", required=True)
    parser.add_argument("--oct-vlm-checkpoint", required=True)
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    print(
        train_active_visual_baselines_with_aux(
            args.config,
            args.oct_ssl_checkpoint,
            args.oct_vlm_checkpoint,
            no_dry_run=args.no_dry_run,
        )
    )


if __name__ == "__main__":
    main()
