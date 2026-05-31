#!/usr/bin/env python3
import argparse

from common import OUT_DIR, rerun_top_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-k-configs", default=str(OUT_DIR / "search/top_k_model_configs.json"))
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    rerun_top_models(args.config, args.top_k_configs, no_dry_run=args.no_dry_run)


if __name__ == "__main__":
    main()
