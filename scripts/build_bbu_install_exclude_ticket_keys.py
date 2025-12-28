#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a combined Stage-B ticket-key exclusion list for BBU install.

Direction-B workflow: exclude tickets that are "unlearnable" for Stage-B given
current Stage-A summaries (i.e., GT=fail but Stage-A evidence looks like pass).

For BBU安装方式检查（正装）, the common unlearnable pattern (after removing known
noise-by-remark tickets) is:
  - label=fail
  - Stage-A summaries contain both:
      * "BBU设备"
      * "BBU安装螺丝,符合"

This script merges:
  1) existing noise filter file (optional)
  2) unlearnable fail keys derived from Stage-A evidence

Example:
  conda run -n ms python scripts/build_bbu_install_exclude_ticket_keys.py \\
    --stage-a output_post/stage_a_bbu_rru_summary_12-22/BBU安装方式检查（正装）_stage_a.jsonl \\
    --noise-file output_post/stage_b/filters/bbu_install_noise_ticket_keys.txt \\
    --out output_post/stage_b/filters/bbu_install_exclude_ticket_keys.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def _load_ticket_keys_txt(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            keys.add(stripped)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", required=True, type=Path)
    parser.add_argument("--noise-file", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    stage_a_path: Path = args.stage_a
    noise_file: Path | None = args.noise_file
    out_path: Path = args.out

    if not stage_a_path.exists():
        raise FileNotFoundError(stage_a_path)
    if noise_file is not None and not noise_file.exists():
        raise FileNotFoundError(noise_file)

    excluded = _load_ticket_keys_txt(noise_file) if noise_file is not None else set()
    noise_count = len(excluded)

    unlearnable = set()
    scanned = 0
    for obj in _iter_jsonl(stage_a_path):
        scanned += 1
        group_id = str(obj.get("group_id", "")).strip()
        label = str(obj.get("label", "")).strip().lower()
        if not group_id or label != "fail":
            continue
        ticket_key = f"{group_id}::{label}"
        if ticket_key in excluded:
            continue
        per_image = obj.get("per_image") or {}
        if not isinstance(per_image, dict):
            continue
        text = " ".join(str(v) for v in per_image.values())
        # Unlearnable for Stage-B given current evidence vocabulary.
        if ("BBU设备" in text) and ("BBU安装螺丝,符合" in text):
            unlearnable.add(ticket_key)

    excluded |= unlearnable

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Combined Stage-B exclusion list for BBU安装方式检查（正装）\n")
        fh.write(f"# stage_a: {stage_a_path}\n")
        if noise_file is not None:
            fh.write(f"# noise_file: {noise_file} (n={noise_count})\n")
        fh.write(
            "# unlearnable_rule: label=fail AND contains 'BBU设备' AND contains 'BBU安装螺丝,符合'\n"
        )
        fh.write(f"# unlearnable_added: {len(unlearnable)}\n")
        fh.write(f"# total_excluded: {len(excluded)}\n")
        for key in sorted(excluded):
            fh.write(key + "\n")

    print(
        json.dumps(
            {
                "scanned_stage_a_records": scanned,
                "noise_keys": noise_count,
                "unlearnable_added": len(unlearnable),
                "total_excluded": len(excluded),
                "out": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

