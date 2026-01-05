#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A postprocess utilities (optional cleanup)."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ..utils import get_logger, require_mapping, require_mutable_mapping
from ..utils.unstructured import UnstructuredMapping
from .types import StageAGroupRecord

logger = get_logger(__name__)

_GROUP_PREFIX_RE = re.compile(r"^(组\d+[:：])+")
_GROUP_PREFIX_OF_RE = re.compile(r"^组(\d+)的")
_REMARK_RE = re.compile(r"(,)?备注:.*$")

_BBU_CATEGORIES = {
    "BBU设备",
    "挡风板",
    "光纤",
    "电线",
    "标签",
    "BBU安装螺丝",
    "机柜处接地螺丝",
    "地排处接地螺丝",
    "ODF端光纤插头",
    "BBU端光纤插头",
}

_RRU_CATEGORIES = {
    "RRU设备",
    "紧固件",
    "RRU接地端",
    "地排接地端螺丝",
    "尾纤",
    "接地线",
    "标签",
    "站点距离",
}


def _format_summary_json(obj: UnstructuredMapping) -> str:
    """Format an intentionally unstructured summary JSON mapping."""
    obj = require_mapping(obj, context="stage_a.summary")
    ordered: dict[str, object] = {}
    for key in ("统计", "备注", "分组统计"):
        if key in obj:
            ordered[key] = obj[key]
    for key, value in obj.items():
        if key not in ordered:
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def _extract_summary_json_line(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    def _maybe_parse_obj(candidate: str) -> UnstructuredMapping | None:
        c = candidate.strip()
        if not (c.startswith("{") and c.endswith("}")):
            return None
        try:
            obj = json.loads(c)
        except Exception:
            return None
        try:
            return require_mapping(obj, context="stage_a.summary_json")
        except TypeError:
            return None

    def _is_summary(obj: UnstructuredMapping) -> bool:
        return "统计" in obj

    obj = _maybe_parse_obj(stripped)
    if obj is not None and _is_summary(obj):
        return _format_summary_json(obj)

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        obj = _maybe_parse_obj(line)
        if obj is not None and _is_summary(obj):
            return _format_summary_json(obj)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _maybe_parse_obj(stripped[start : end + 1])
        if obj is not None and _is_summary(obj):
            return _format_summary_json(obj)

    return None


def _strip_remark(item: str) -> str:
    return _REMARK_RE.sub("", item).strip(" ，")


def sanitize_summary_by_dataset(text: str, dataset: str) -> str:
    """Return summary text without dataset-specific injection or filtering."""
    _ = dataset
    summary_text = (text or "").strip()
    if not summary_text:
        return summary_text
    if summary_text.startswith("无关图片"):
        return "无关图片"
    extracted = _extract_summary_json_line(summary_text)
    if extracted is not None:
        return extracted
    if summary_text.startswith("{") and summary_text.endswith("}"):
        try:
            obj = json.loads(summary_text)
        except Exception:
            obj = None
        if isinstance(obj, dict) and "统计" in obj:
            return _format_summary_json(obj)
    return summary_text


def clean_stage_a_record(record: StageAGroupRecord, dataset: str) -> StageAGroupRecord:
    """Clean a Stage-A group record in-place."""
    record = cast(
        StageAGroupRecord, require_mutable_mapping(record, context="stage_a.record")
    )
    per_image = record.get("per_image")
    if not isinstance(per_image, Mapping):
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
