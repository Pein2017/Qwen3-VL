#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for generating per-group Stage-B bundle reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.stage_b.io.group_report import build_group_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Stage-B outputs into per-group reports")
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to a Stage-B mission run directory")
    parser.add_argument(
        "--stage-a",
        dest="stage_a",
        type=Path,
        nargs="+",
        default=None,
        help="Optional Stage-A JSONL path(s) to annotate labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = build_group_report(args.run_dir, args.stage_a)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
