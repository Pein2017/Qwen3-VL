#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Stage-B ticket-key exclusion lists (noise filters).

This is a pragmatic helper for filtering out known-noisy tickets where the GT
fail reasons are not observable in Stage-A summaries (and therefore not learnable
by Stage-B rule_search).

Typical usage (BBU install):
  conda run -n ms python scripts/build_stage_b_noise_ticket_filter.py \
    --stage-a output_post/stage_a_bbu_rru_summary_12-22/BBU安装方式检查（正装）_stage_a.jsonl \
    --hard-samples output_post/stage_b/BBU安装方式检查（正装）/12-19-bbu-line-rule-search-train-eval/rule_search_hard_samples.jsonl \
    --out output_post/stage_b/filters/bbu_install_noise_ticket_keys.txt

Additional optional filters:
- Exclude GT=pass tickets whose useful-image count is too small after removing irrelevant
  ones (aligned with `scripts/flip_stage_a_pass_by_irrelevant.py`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_REASON_PATTERNS = (
    # Station name / site name (not available in current Stage-A summaries for BBU)
    r"站点名|站名|本站|写上站|写上本站|站名写",
    # Photo completeness / clarity (not available in Stage-A)
    r"整体照|拍全|拍完整|拍清晰|清晰|被遮挡|无遮挡",
    # Screw clarity/photo constraints (Stage-A has screw existence, not clarity)
    r"螺丝",
    # Fiber tube / yellow fiber (Stage-A is unreliable for this mission)
    r"尾纤|套管|黄尾纤|黄色|黄纤",
)


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


def _load_stage_a_ticket_keys(stage_a_path: Path) -> set[str]:
    keys: set[str] = set()
    for obj in _iter_jsonl(stage_a_path):
        group_id = str(obj.get("group_id", "")).strip()
        label = str(obj.get("label", "")).strip().lower()
        if not group_id or label not in {"pass", "fail"}:
            continue
        keys.add(f"{group_id}::{label}")
    if not keys:
        raise RuntimeError(f"No valid ticket keys found in Stage-A JSONL: {stage_a_path}")
    return keys


def _iter_stage_a_ticket_rows(stage_a_path: Path):
    for obj in _iter_jsonl(stage_a_path):
        group_id = str(obj.get("group_id", "")).strip()
        label = str(obj.get("label", "")).strip().lower()
        per_image = obj.get("per_image")
        images = obj.get("images")
        if not group_id or label not in {"pass", "fail"}:
            continue
        if not isinstance(per_image, dict) or not per_image:
            continue
        total_images = len(images) if isinstance(images, list) and images else len(per_image)
        yield f"{group_id}::{label}", total_images, per_image


def _count_irrelevant(per_image: dict[str, str], token: str) -> int:
    return sum(1 for value in per_image.values() if token in str(value))


def _match_reason(payload: dict, patterns: list[re.Pattern[str]]) -> bool:
    texts: list[str] = []
    for k in ("gt_fail_reason_text", "gt_fail_reason_texts"):
        value = payload.get(k)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(str(item) for item in value)
    joined = " | ".join(t.strip() for t in texts if str(t).strip())
    if not joined:
        return False
    return any(p.search(joined) for p in patterns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", required=True, type=Path, help="Stage-A JSONL path")
    parser.add_argument(
        "--hard-samples",
        default=None,
        type=Path,
        help="Postprocessed rule_search_hard_samples.jsonl path (optional)",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output txt path")
    parser.add_argument(
        "--reason-pattern",
        action="append",
        default=list(DEFAULT_REASON_PATTERNS),
        help="Regex (can be repeated). Defaults cover station-name / photo / screw / fiber-tube noises.",
    )
    parser.add_argument(
        "--irrelevant-token",
        type=str,
        default="无关图片",
        help="Substring that marks an image as irrelevant (default: 无关图片).",
    )
    parser.add_argument(
        "--exclude-pass-min-useful",
        type=int,
        default=None,
        help=(
            "If set, exclude tickets with GT=pass where useful_images < min_useful after removing irrelevant images. "
            "Aligned with `scripts/flip_stage_a_pass_by_irrelevant.py` default rule (min_useful=3). "
            "Example: --exclude-pass-min-useful 3"
        ),
    )
    args = parser.parse_args()

    stage_a_path: Path = args.stage_a
    hard_samples_path: Path | None = args.hard_samples
    out_path: Path = args.out
    raw_patterns: list[str] = list(args.reason_pattern)
    irrelevant_token: str = str(args.irrelevant_token or "无关图片")
    exclude_pass_min_useful: int | None = args.exclude_pass_min_useful

    if not stage_a_path.exists():
        raise FileNotFoundError(stage_a_path)
    if hard_samples_path is not None and not hard_samples_path.exists():
        raise FileNotFoundError(hard_samples_path)

    if hard_samples_path is None and exclude_pass_min_useful is None:
        raise ValueError(
            "Nothing to do: provide --hard-samples and/or --exclude-pass-min-useful"
        )

    stage_a_keys = _load_stage_a_ticket_keys(stage_a_path)
    matched: set[str] = set()  # matched from hard samples
    unmatched: set[str] = set()  # hard-samples keys missing from stage-a
    pass_irrel: set[str] = set()  # derived from stage-a (pass with too few useful images)
    total = 0  # hard samples total rows

    if hard_samples_path is not None:
        patterns = [re.compile(pat) for pat in raw_patterns if str(pat).strip()]
        if not patterns:
            raise ValueError("At least one --reason-pattern must be provided")

        for obj in _iter_jsonl(hard_samples_path):
            total += 1
            ticket_key = str(obj.get("ticket_key", "")).strip()
            if not ticket_key:
                continue
            if not _match_reason(obj, patterns):
                continue
            if ticket_key in stage_a_keys:
                matched.add(ticket_key)
            else:
                unmatched.add(ticket_key)

    if exclude_pass_min_useful is not None:
        min_useful = int(exclude_pass_min_useful)
        if min_useful <= 0:
            raise ValueError("--exclude-pass-min-useful must be > 0")

        for ticket_key, total_images, per_image in _iter_stage_a_ticket_rows(stage_a_path):
            if not ticket_key.endswith("::pass"):
                continue
            irrelevant_count = _count_irrelevant(per_image, irrelevant_token)
            if irrelevant_count <= 0:
                # Match flip_stage_a_pass_by_irrelevant.py default: only flip/exclude when >=1 irrelevant exists.
                continue
            useful_images = int(total_images) - int(irrelevant_count)
            if useful_images < min_useful:
                pass_irrel.add(ticket_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Auto-generated Stage-B noise filter (ticket_key per line)\n")
        fh.write(f"# stage_a: {stage_a_path}\n")
        if hard_samples_path is not None:
            fh.write(f"# hard_samples: {hard_samples_path}\n")
            fh.write("# reason_patterns:\n")
            for pat in raw_patterns:
                fh.write(f"# - {pat}\n")
        if exclude_pass_min_useful is not None:
            fh.write(f"# irrelevant_token: {irrelevant_token}\n")
            fh.write(f"# exclude_pass_min_useful: {exclude_pass_min_useful}\n")
            fh.write(
                "# rule: label=pass AND irrelevant_count>=1 AND (total_images-irrelevant_count)<min_useful\n"
            )

        for key in sorted(matched | pass_irrel):
            fh.write(key + "\n")

    print(
        json.dumps(
            {
                "hard_samples_total": total,
                "matched_excluded_keys": len(matched),
                "unmatched_excluded_keys": len(unmatched),
                "pass_irrel_excluded_keys": len(pass_irrel),
                "out": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
