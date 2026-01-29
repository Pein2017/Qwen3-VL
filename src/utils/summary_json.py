"""Shared Stage-A / Stage-B summary-JSON helpers.

Stage-A "summary mode" models often emit a single JSON object containing the key
`统计` (and optionally `备注` / `分组统计`). Stage-B needs to:
- strip optional non-JSON header lines (e.g. "<DOMAIN=...><TASK=...>")
- extract the summary JSON object (prefer the last JSON line when multiple exist)
- format it canonically (stable ordering + separators) for consistent prompting

These helpers are intentionally lightweight and MUST NOT import Stage-A/Stage-B
pipeline modules.

See `tests/test_summary_json_utils.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import cast

from .unstructured import UnstructuredMapping, require_mapping

_CANONICAL_KEYS: tuple[str, ...] = ("统计", "备注", "分组统计")


def strip_summary_headers(text: str) -> str:
    """Strip known non-JSON header lines from a summary output.

    Some runtimes prepend one "domain/task" line like:
      <DOMAIN=...><TASK=...>

    We drop any non-empty lines matching that exact pattern and return the remaining
    text joined by newlines.
    """
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("<DOMAIN=")
            and "<TASK=" in stripped
            and stripped.endswith(">")
        ):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def _maybe_parse_obj(candidate: str, *, context: str) -> UnstructuredMapping | None:
    c = (candidate or "").strip()
    if not (c.startswith("{") and c.endswith("}")):
        return None
    try:
        parsed = json.loads(c)
    except Exception:
        return None
    try:
        return require_mapping(parsed, context=context)
    except TypeError:
        return None


def is_summary_json(obj: UnstructuredMapping) -> bool:
    return "统计" in obj


def format_summary_json(obj: UnstructuredMapping, *, context: str) -> str:
    """Canonical, stable formatting for a summary JSON mapping.

    Ordering:
    - preferred keys first: 统计, 备注, 分组统计
    - preserve any additional top-level keys (in their existing order)
    """
    obj = require_mapping(obj, context=context)
    ordered: dict[str, object] = {}
    for key in _CANONICAL_KEYS:
        if key in obj:
            ordered[key] = obj[key]
    for key, value in obj.items():
        if key not in ordered:
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def extract_summary_json_obj(text: str, *, context: str) -> UnstructuredMapping | None:
    """Extract the summary JSON object (mapping) from model output text.

    Returns:
    - mapping if a JSON object containing "统计" can be found
    - None otherwise
    """
    cleaned = strip_summary_headers(text)
    stripped = (cleaned or "").strip()
    if not stripped or stripped.startswith("无关图片"):
        return None

    obj = _maybe_parse_obj(stripped, context=context)
    if obj is not None and is_summary_json(obj):
        return obj

    # Prefer the last JSON-line summary when multiple are present.
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        obj = _maybe_parse_obj(line, context=context)
        if obj is not None and is_summary_json(obj):
            return obj

    # Fallback: try to parse the widest {...} substring.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _maybe_parse_obj(stripped[start : end + 1], context=context)
        if obj is not None and is_summary_json(obj):
            return obj

    # Minimal syntactic recovery for list-assignment style outputs:
    #   统计=[{...}]
    #   分组统计=[{...}]
    #
    # This is not valid JSON, but it is unambiguous to repair into:
    #   {"统计": [...], "分组统计": [...]}
    def _extract_balanced_json_list(s: str, start_index: int) -> str | None:
        if start_index < 0 or start_index >= len(s) or s[start_index] != "[":
            return None
        depth = 0
        in_str = False
        escape = False
        for idx in range(start_index, len(s)):
            ch = s[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "[":
                depth += 1
                continue
            if ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start_index : idx + 1]
        return None

    def _maybe_parse_list_assignment(key: str) -> list[object] | None:
        for sep in ("=", ":"):
            marker = f"{key}{sep}"
            pos = stripped.find(marker)
            if pos == -1:
                continue
            bracket = stripped.find("[", pos + len(marker))
            if bracket == -1:
                continue
            list_text = _extract_balanced_json_list(stripped, bracket)
            if list_text is None:
                continue
            try:
                parsed = json.loads(list_text)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(parsed, list):
                return cast(list[object], parsed)
        return None

    stats_list = _maybe_parse_list_assignment("统计")
    group_stats_list = _maybe_parse_list_assignment("分组统计")
    if stats_list is not None or group_stats_list is not None:
        repaired: dict[str, object] = {}
        if stats_list is not None:
            repaired["统计"] = stats_list
        if group_stats_list is not None:
            repaired["分组统计"] = group_stats_list
        if "统计" in repaired:
            return cast(UnstructuredMapping, repaired)

    return None


def extract_summary_json_line(text: str, *, context: str) -> str | None:
    """Extract and format a summary JSON object as a canonical single-line string."""
    obj = extract_summary_json_obj(text, context=context)
    if obj is None:
        return None
    return format_summary_json(obj, context=context)


def summary_entries(obj: UnstructuredMapping) -> list[UnstructuredMapping]:
    """Return `统计` entries as a list of mapping items (best-effort)."""
    entries_raw: object = obj.get("统计")
    if not isinstance(entries_raw, list):
        return []
    entries_list = cast(list[object], entries_raw)
    entries: list[UnstructuredMapping] = []
    for item in entries_list:
        if isinstance(item, Mapping):
            entries.append(cast(UnstructuredMapping, item))
    return entries


def entry_by_category(
    entries: Iterable[UnstructuredMapping], category: str
) -> UnstructuredMapping | None:
    for entry in entries:
        if entry.get("类别") == category:
            return entry
    return None


__all__ = [
    "entry_by_category",
    "extract_summary_json_line",
    "extract_summary_json_obj",
    "format_summary_json",
    "is_summary_json",
    "strip_summary_headers",
    "summary_entries",
]
