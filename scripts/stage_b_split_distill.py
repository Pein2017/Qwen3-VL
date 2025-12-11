#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a distill_chatml.jsonl corpus into train/val JSONL files.

Usage:
  python scripts/stage_b_split_distill.py \
      --input output_post/stage_b/run/mission/distill_chatml.jsonl \
      --train-out data/stage_b/distill_chatml.train.jsonl \
      --val-out data/stage_b/distill_chatml.val.jsonl \
      --val-ratio 0.1 --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Mapping, Sequence


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input distill file not found: {path}")
    records: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")


def split_distill_corpus(
    input_path: Path,
    train_out: Path,
    val_out: Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 17,
) -> None:
    records = _read_jsonl(input_path)
    rng = random.Random(seed)
    rng.shuffle(records)

    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in (0,1)")
    val_count = max(1, int(len(records) * val_ratio))
    if val_count >= len(records):
        val_count = len(records) // 2 or 1

    val_split = records[:val_count]
    train_split = records[val_count:]
    if not train_split:
        raise ValueError("Train split would be empty; reduce val_ratio or add data.")

    _write_jsonl(train_out, train_split)
    _write_jsonl(val_out, val_split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split distill_chatml.jsonl into train/val")
    parser.add_argument("--input", required=True, help="Path to distill_chatml.jsonl")
    parser.add_argument(
        "--train-out",
        default=None,
        help="Output path for train split (default: alongside input, distill_chatml.train.jsonl)",
    )
    parser.add_argument(
        "--val-out",
        default=None,
        help="Output path for val split (default: alongside input, distill_chatml.val.jsonl)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=17, help="Shuffle seed")

    args = parser.parse_args()
    input_path = Path(args.input)
    base_dir = input_path.parent
    train_out = Path(args.train_out) if args.train_out else base_dir / "distill_chatml.train.jsonl"
    val_out = Path(args.val_out) if args.val_out else base_dir / "distill_chatml.val.jsonl"

    split_distill_corpus(
        input_path=input_path,
        train_out=train_out,
        val_out=val_out,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
