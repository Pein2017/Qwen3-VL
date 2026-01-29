#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Stage-A JSONL with gt_fail_reason_texts and suggest skip filters.

This is a lightweight, deterministic helper for reviewing Stage-A outputs after
`scripts/add_gt_fail_reason_to_stage_a.py` has injected:
  - gt_fail_reason_text (str|None)
  - gt_fail_reason_texts (list[str])

It produces:
  1) A markdown audit report with label stats, top fail reasons, and samples.
  2) (Optional) Stage-B exclude_ticket_keys files for "need extra info" fails,
     based on *GT remarks* (not Stage-A ratios).

NOTE: The authoritative rule for "irrelevant-image noise" is
`scripts/flip_stage_a_pass_by_irrelevant.py`. This script does NOT do ratio
filtering; it only uses GT fail reason remarks when asked to write filters.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_NEED_EXTRA_PATTERNS: tuple[str, ...] = (
    r"补拍",
    r"拍摄",
    r"请拍",
    r"未拍",
    r"拍全",
    r"拍完整",
    r"拍清晰",
    r"看不清",
    r"模糊",
    r"遮挡",
    r"反光",
    r"信息不足",
    r"角度不对|拍摄角度",
    r"无法判断|无法确认|不能判断|不确定",
    r"未提供|无图|无照片|无图片",
)


DEFAULT_CATEGORIES: tuple[tuple[str, str], ...] = (
    (
        "need_extra",
        r"补拍|拍摄|请拍|未拍|建议.*补|拍全|拍完整|拍清晰|看不清|模糊|遮挡|反光|信息不足|角度",
    ),
    ("station_name", r"站名|站点名|写上站|写.*站点|写.*站名"),
    ("label", r"标签|挂牌|挂标签|两端标签|标签一致|标签统一|标签内容"),
    ("cable_protect", r"套管|尾纤|ODF|保护|光纤"),
    ("ground", r"接地|接地排|地排|漏铜"),
)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in {"pass", "通过", "1"}:
        return "pass"
    if s in {"fail", "不通过", "2"}:
        return "fail"
    return s


def _join_reasons(obj: dict) -> str:
    texts: list[str] = []
    for k in ("gt_fail_reason_text", "gt_fail_reason_texts"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            texts.append(v.strip())
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
    return " | ".join(texts)


def _ticket_key(group_id: str, label: str) -> str:
    return f"{group_id}::{label}"


def _shorten(text: str, max_chars: int) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, int(max_chars) - 1)] + "…"


@dataclass(frozen=True)
class AuditConfig:
    seed: int
    samples_per_category: int
    max_per_image_chars: int
    max_images_per_sample: int


def audit_stage_a(
    stage_a_path: Path,
    categories: list[tuple[str, re.Pattern[str]]],
    cfg: AuditConfig,
):
    label_counts: Counter[str] = Counter()
    fail_reason_counts: Counter[str] = Counter()
    missing_reason_in_fail = 0

    # Per-category pools for sampling.
    pools: dict[str, list[dict]] = defaultdict(list)
    category_record_counts: Counter[str] = Counter()
    fail_total = 0

    required = {"mission", "group_id", "label", "images", "per_image"}
    missing_required = 0

    mission_name: str | None = None

    for obj in _iter_jsonl(stage_a_path):
        label = _normalize_label(obj.get("label"))
        label_counts[label] += 1

        if mission_name is None:
            mission_name = str(obj.get("mission", "")).strip() or None

        if not required.issubset(obj.keys()):
            missing_required += 1

        if label != "fail":
            continue
        fail_total += 1

        joined = _join_reasons(obj)
        if not joined:
            missing_reason_in_fail += 1
            continue
        for part in (p.strip() for p in joined.split("|")):
            if part:
                fail_reason_counts[part] += 1

        matched_any = False
        for cat, rx in categories:
            if rx.search(joined):
                pools[cat].append(obj)
                category_record_counts[cat] += 1
                matched_any = True
        if not matched_any:
            pools["other"].append(obj)
            category_record_counts["other"] += 1

    return {
        "mission": mission_name or stage_a_path.stem,
        "stage_a_path": stage_a_path,
        "label_counts": label_counts,
        "missing_required": missing_required,
        "fail_total": fail_total,
        "missing_reason_in_fail": missing_reason_in_fail,
        "fail_reason_counts": fail_reason_counts,
        "category_record_counts": category_record_counts,
        "pools": pools,
        "cfg": cfg,
    }


def _format_markdown_samples(
    *,
    title: str,
    records: list[dict],
    cfg: AuditConfig,
) -> str:
    rnd = random.Random(cfg.seed)
    chosen = rnd.sample(records, k=min(cfg.samples_per_category, len(records)))
    lines: list[str] = []
    lines.append(f"### {title}")
    if not chosen:
        lines.append("")
        lines.append("_No samples._")
        lines.append("")
        return "\n".join(lines)

    for obj in chosen:
        gid = str(obj.get("group_id", "")).strip()
        label = _normalize_label(obj.get("label"))
        reasons = obj.get("gt_fail_reason_texts")
        reasons_list = reasons if isinstance(reasons, list) else []
        joined = _join_reasons(obj)

        per_image = (
            obj.get("per_image") if isinstance(obj.get("per_image"), dict) else {}
        )
        per_items = list(per_image.items())[: cfg.max_images_per_sample]

        lines.append(f"- ticket_key: `{_ticket_key(gid, label)}`")
        if reasons_list:
            lines.append(
                f"  - gt_fail_reason_texts: `{_shorten(' | '.join(map(str, reasons_list)), 240)}`"
            )
        elif joined:
            lines.append(f"  - gt_fail_reason_text: `{_shorten(joined, 240)}`")
        else:
            lines.append("  - gt_fail_reason: `_missing_`")
        lines.append("  - per_image (partial):")
        for k, v in per_items:
            lines.append(f"    - `{k}`: `{_shorten(str(v), cfg.max_per_image_chars)}`")
    lines.append("")
    return "\n".join(lines)


def _write_exclude_keys(
    *,
    out_path: Path,
    stage_a_path: Path,
    patterns: list[str],
    keys: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Auto-generated Stage-B exclude_ticket_keys (ticket_key per line)\n")
        fh.write(f"# stage_a: {stage_a_path}\n")
        fh.write(
            "# rule: label=fail AND any(gt_fail_reason_text{,s} matches pattern)\n"
        )
        fh.write("# patterns:\n")
        for pat in patterns:
            fh.write(f"# - {pat}\n")
        for key in keys:
            fh.write(key + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Stage-A JSONL gt_fail_reason_texts and propose skips."
    )
    parser.add_argument(
        "--stage-a",
        action="append",
        required=True,
        type=Path,
        help="Path to *_stage_a.jsonl (repeatable)",
    )
    parser.add_argument(
        "--out-md", required=True, type=Path, help="Output markdown path"
    )
    parser.add_argument(
        "--seed", type=int, default=20251225, help="Sampling seed (default: 20251225)"
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=3,
        help="Samples per category (default: 3)",
    )
    parser.add_argument(
        "--max-per-image-chars",
        type=int,
        default=220,
        help="Max chars per per_image entry (default: 220)",
    )
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=3,
        help="Max images shown per sample (default: 3)",
    )
    parser.add_argument(
        "--write-need-extra-filters",
        action="store_true",
        help="Write Stage-B exclude_ticket_keys for need_extra fail tickets (based on GT remarks).",
    )
    parser.add_argument(
        "--filters-dir",
        type=Path,
        default=Path("output_post/stage_b/filters"),
        help="Where to write filter files (default: output_post/stage_b/filters)",
    )
    parser.add_argument(
        "--need-extra-pattern",
        action="append",
        default=list(DEFAULT_NEED_EXTRA_PATTERNS),
        help="Regex for 'need extra info' (repeatable). Defaults are conservative and remark-driven.",
    )
    args = parser.parse_args()

    cfg = AuditConfig(
        seed=int(args.seed),
        samples_per_category=int(args.samples_per_category),
        max_per_image_chars=int(args.max_per_image_chars),
        max_images_per_sample=int(args.max_images_per_sample),
    )

    categories = [(name, re.compile(rx)) for name, rx in DEFAULT_CATEGORIES]
    need_extra_patterns = [str(p) for p in args.need_extra_pattern if str(p).strip()]
    need_extra_rx = (
        re.compile("|".join(f"(?:{p})" for p in need_extra_patterns))
        if need_extra_patterns
        else None
    )

    report_lines: list[str] = []
    report_lines.append("# Stage-A gt_fail_reason_texts 审计报告")
    report_lines.append("")
    report_lines.append(f"- seed: `{cfg.seed}`")
    report_lines.append(f"- samples_per_category: `{cfg.samples_per_category}`")
    report_lines.append("")

    for stage_a_path in args.stage_a:
        if not stage_a_path.exists():
            raise FileNotFoundError(stage_a_path)

        res = audit_stage_a(stage_a_path, categories, cfg)
        mission = str(res["mission"])
        label_counts: Counter[str] = res["label_counts"]
        fail_total: int = int(res["fail_total"])
        missing_required: int = int(res["missing_required"])
        missing_reason_in_fail: int = int(res["missing_reason_in_fail"])
        fail_reason_counts: Counter[str] = res["fail_reason_counts"]
        category_record_counts: Counter[str] = res["category_record_counts"]
        pools: dict[str, list[dict]] = res["pools"]

        report_lines.append(f"## {mission}")
        report_lines.append("")
        report_lines.append(f"- stage_a: `{stage_a_path}`")
        report_lines.append(f"- labels: `{dict(label_counts)}`")
        report_lines.append(f"- missing_required_records: `{missing_required}`")
        report_lines.append(f"- fail_total: `{fail_total}`")
        report_lines.append(
            f"- fail_missing_gt_fail_reason: `{missing_reason_in_fail}`"
        )
        report_lines.append("")

        report_lines.append("### fail reason top")
        for t, c in fail_reason_counts.most_common(20):
            report_lines.append(f"- `{c}`: `{_shorten(t, 120)}`")
        report_lines.append("")

        report_lines.append("### category coverage (fail records)")
        for cat, c in category_record_counts.most_common():
            report_lines.append(
                f"- `{cat}`: `{c}` ({(c / fail_total) if fail_total else 0:.1%})"
            )
        report_lines.append("")

        for cat in [
            "need_extra",
            "station_name",
            "label",
            "cable_protect",
            "ground",
            "other",
        ]:
            if cat not in pools:
                continue
            report_lines.append(
                _format_markdown_samples(
                    title=f"samples: {cat}", records=pools[cat], cfg=cfg
                )
            )

        if args.write_need_extra_filters and need_extra_rx is not None:
            keys: list[str] = []
            for obj in pools.get("need_extra", []):
                gid = str(obj.get("group_id", "")).strip()
                if not gid:
                    continue
                label = _normalize_label(obj.get("label"))
                if label != "fail":
                    continue
                joined = _join_reasons(obj)
                if not joined or not need_extra_rx.search(joined):
                    continue
                keys.append(_ticket_key(gid, label))
            keys = sorted(set(keys))

            out_name = stage_a_path.stem.replace("_stage_a", "")
            out_path = (
                args.filters_dir / f"{out_name}_need_extra_exclude_ticket_keys.txt"
            )
            _write_exclude_keys(
                out_path=out_path,
                stage_a_path=stage_a_path,
                patterns=need_extra_patterns,
                keys=keys,
            )
            report_lines.append("### generated filter (need_extra)")
            report_lines.append(f"- out: `{out_path}`")
            report_lines.append(f"- excluded_keys: `{len(keys)}`")
            report_lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
