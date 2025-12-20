"""Merge paired pass/fail stage_a JSONL records.

Use case: each `group_id` may have one earlier `fail` submission and one later
`pass` submission (which currently only contains the latest images). The goal is
to append the fail images/per_image entries to the pass record so the pass
contains the full history, while dropping the standalone fail line from output.

Usage:
  conda run -n ms python scripts/merge_pass_fail.py \
    --input output_post/stage_a/BBU安装方式检查（正装）_stage_a.jsonl \
    --output output_post/stage_a/BBU安装方式检查（正装）_stage_a_merged.jsonl

Notes:
- Only merges groups that have exactly one pass and one fail. Other patterns
  are left unchanged (all original lines are kept).
- `per_image` keys from the fail line are renumbered to avoid collisions.
- `mission` mismatches are logged as warnings but the pass mission is retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_records(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def merge_pass_fail(records: List[dict]) -> Tuple[List[dict], List[str]]:
    """Merge fail content into pass for groups that have exactly one pass+fail.

    Output keeps the original fail record; the pass is augmented with fail
    images/per_image. Other patterns are untouched.
    """

    grouped: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        gid = rec.get("group_id")
        grouped[gid].append((idx, rec))

    warnings: List[str] = []
    output: List[Tuple[int, dict]] = []

    for gid, items in grouped.items():
        if len(items) != 2:
            # Keep all as-is when not exactly two entries.
            output.extend(items)
            continue

        pass_item = [it for it in items if it[1].get("label") == "pass"]
        fail_item = [it for it in items if it[1].get("label") == "fail"]

        if len(pass_item) != 1 or len(fail_item) != 1:
            output.extend(items)
            continue

        pass_idx, pass_rec = pass_item[0]
        _, fail_rec = fail_item[0]

        # Warn on mission mismatch but keep pass mission.
        if pass_rec.get("mission") != fail_rec.get("mission"):
            warnings.append(
                f"mission mismatch for {gid}: pass='{pass_rec.get('mission')}' "
                f"fail='{fail_rec.get('mission')}'"
            )

        merged = dict(pass_rec)  # shallow copy sufficient for replacement

        # Merge images.
        merged_images = list(pass_rec.get("images", [])) + list(
            fail_rec.get("images", [])
        )
        merged["images"] = merged_images

        # Merge per_image with renumbering to avoid key collisions.
        merged_pi = dict(pass_rec.get("per_image", {}))
        start = len(merged_pi)
        for offset, (k, v) in enumerate(fail_rec.get("per_image", {}).items(), 1):
            merged_pi[f"image_{start + offset}"] = v
        merged["per_image"] = merged_pi

        # Keep augmented pass AND original fail (stable order: based on original indices)
        output.append((pass_idx, merged))
        output.append((fail_item[0][0], fail_rec))

    # Keep original order by the earliest index of each kept/merged record.
    output_sorted = [rec for _, rec in sorted(output, key=lambda x: x[0])]
    return output_sorted, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input stage_a JSONL path")
    parser.add_argument("--output", required=True, help="Output merged JSONL path")
    args = parser.parse_args()

    records = load_records(args.input)
    merged, warnings = merge_pass_fail(records)
    save_records(args.output, merged)

    print(f"merged records written to {args.output}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(" -", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
