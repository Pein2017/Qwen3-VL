#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A postprocess utilities (optional cleanup)."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from ..utils import get_logger

logger = get_logger(__name__)

_GROUP_PREFIX_RE = re.compile(r"^(组\d+[:：])+")
_GROUP_PREFIX_OF_RE = re.compile(r"^组(\d+)的")
_REMARK_RE = re.compile(r"(,)?备注:.*$")


def _strip_remark(item: str) -> str:
    return _REMARK_RE.sub("", item).strip(" ，")


def sanitize_summary_by_dataset(text: str, dataset: str) -> str:
    """Normalize summary text with dataset-specific constraints.

    For RRU, drop non-domain items (e.g., BBU-only objects). If nothing remains,
    return "无关图片". Preserve label/station OCR content.
    """
    summary_text = text.strip()
    if not summary_text:
        return summary_text

    parts = [p.strip() for p in summary_text.split("，") if p.strip()]
    if not parts:
        return summary_text

    def _normalize_irrelevant_only(items: list[str]) -> list[str]:
        cleaned = []
        for item in items:
            if item.startswith("无关图片"):
                continue
            cleaned.append(item)
        return cleaned

    cleaned: list[str] = []
    if dataset.lower() != "rru":
        for item in _normalize_irrelevant_only(parts):
            trimmed = _strip_remark(item)
            if trimmed:
                cleaned.append(trimmed)
        return "无关图片" if not cleaned else "，".join(cleaned)

    allowed_prefixes = (
        "标签/",
        "站点距离/",
        "RRU设备",
        "RRU接地端",
        "地排接地端螺丝",
        "紧固件",
        "尾纤",
        "接地线",
    )

    for item in parts:
        if item.startswith("无关图片"):
            continue
        normalized = _GROUP_PREFIX_OF_RE.sub(r"组\1:", item.replace("：", ":"))
        group_prefix = ""
        match = _GROUP_PREFIX_RE.match(normalized)
        core = normalized
        if match:
            group_prefix = match.group(0)
            core = normalized[len(group_prefix) :]

        core = _strip_remark(core)
        if not core:
            continue

        if core.startswith("RRU设备/"):
            count = ""
            mcount = re.search(r"(×\d+)$", core)
            if mcount:
                count = mcount.group(1)
            normalized_core = f"RRU设备{count}"
            cleaned.append(
                f"{group_prefix}{normalized_core}" if group_prefix else normalized_core
            )
            continue

        if core.startswith(allowed_prefixes):
            cleaned.append(f"{group_prefix}{core}" if group_prefix else core)
            continue

    if not cleaned:
        return "无关图片"
    return "，".join(cleaned)


def clean_stage_a_record(record: dict[str, object], dataset: str) -> dict[str, object]:
    per_image = record.get("per_image")
    if not isinstance(per_image, dict):
        return record
    cleaned: dict[str, str] = {}
    for key, value in per_image.items():
        if not isinstance(value, str):
            continue
        cleaned[key] = sanitize_summary_by_dataset(value, dataset)
    record["per_image"] = cleaned
    return record


def postprocess_jsonl(input_path: Path, output_path: Path, dataset: str) -> None:
    logger.info("Postprocessing Stage-A JSONL: %s", input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    # NOTE: input_path == output_path (in-place) must not open the same path for
    # write while reading, otherwise the file is truncated to empty.
    inplace = input_path.resolve() == output_path.resolve()
    tmp_path: Path | None = None
    if inplace:
        fd, tmp_name = tempfile.mkstemp(
            prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent)
        )
        # Close the low-level fd; we'll reopen via Path for text IO.
        try:
            Path(tmp_name).write_text("", encoding="utf-8")
        finally:
            try:
                import os

                os.close(fd)
            except Exception:
                pass
        tmp_path = Path(tmp_name)
        output_target = tmp_path
    else:
        output_target = output_path

    try:
        with input_path.open("r", encoding="utf-8") as f_in, output_target.open(
            "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)
                cleaned = clean_stage_a_record(record, dataset)
                f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

        if inplace and tmp_path is not None:
            tmp_path.replace(output_path)

        logger.info("Postprocess done: %d records -> %s", total, output_path)
    finally:
        if inplace and tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-A JSONL postprocess cleaner")
    parser.add_argument("--input", required=True, help="Stage-A JSONL path")
    parser.add_argument("--output", default="", help="Output JSONL path")
    parser.add_argument("--dataset", default="bbu", choices=["bbu", "rru"])
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if args.inplace:
        output_path = input_path
    else:
        output_path = (
            Path(args.output)
            if args.output
            else input_path.with_suffix(".cleaned.jsonl")
        )
    postprocess_jsonl(input_path, output_path, args.dataset)


if __name__ == "__main__":
    main()
