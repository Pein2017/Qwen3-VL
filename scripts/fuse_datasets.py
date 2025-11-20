#!/usr/bin/env python3
"""Offline fusion builder for dense-caption training."""

import argparse
import logging
import sys
from pathlib import Path

from src.datasets.fusion import FusionConfig, build_fused_jsonl

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mix target and auxiliary datasets into a single fused train JSONL"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to fusion config (YAML/JSON) describing target + auxiliary datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSONL path. Defaults to <target_name>_train_fused.jsonl alongside the target JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for shuffling and sampling (default: 2025)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of the fused output (default: shuffle enabled)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = FusionConfig.from_file(args.config)
    if args.output:
        output_path = Path(args.output)
    else:
        target_path = Path(config.target.train_jsonl)
        output_path = target_path.with_name(f"{config.target.name}_train_fused.jsonl")

    fused_path = build_fused_jsonl(
        config,
        str(output_path),
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    target_count = sum(
        1 for _ in Path(config.target.train_jsonl).open("r", encoding="utf-8")
    )
    aux_quota = sum(round(source.ratio * target_count) for source in config.sources)
    fused_count = sum(1 for _ in fused_path.open("r", encoding="utf-8"))
    LOGGER.info(
        "Fused JSONL written to %s (%d records: target=%d, aux=%d)",
        fused_path,
        fused_count,
        target_count,
        aux_quota,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logging.exception("Fusion builder failed")
        sys.exit(1)
