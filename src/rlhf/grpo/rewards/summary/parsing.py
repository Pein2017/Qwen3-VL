"""Shared parsing helpers for summary GRPO rewards."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_IRRELEVANT_SOURCE = "irrelevant_summary"
_IRRELEVANT_TEXT = "无关图片"
_HEADER_PATTERN = re.compile(r"^<DOMAIN=[A-Z]+>, <TASK=[A-Z]+>$")
_SPECIAL_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>")
_WHITESPACE_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")

_BBU_OCR_TRANSLATION = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "－": "-",
        "—": "-",
        "–": "-",
        "−": "-",
        "‐": "-",
        "／": "/",
        "：": ":",
        "，": ",",
    }
)


def normalize_text(text: str) -> str:
    for token in _SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def extract_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return normalize_text(completion)
    if isinstance(completion, list):
        parts: list[str] = []
        for part in completion:
            if isinstance(part, Mapping):
                text = part.get("text")
                if text is not None:
                    parts.append(str(text))
            elif isinstance(part, str):
                parts.append(part)
        if parts:
            return normalize_text("".join(parts))
        return ""
    if isinstance(completion, Mapping):
        text = completion.get("text")
        if text is not None:
            return normalize_text(str(text))
    return ""


def ensure_list(value: Any, n: int) -> list[Any]:
    if isinstance(value, list):
        if len(value) < n:
            return value + [None] * (n - len(value))
        return value
    return [value] * n


def split_lines(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return stripped.splitlines()


def normalize_free_text(value: str) -> str:
    """Normalize OCR / free text for more stable matching."""

    return _WHITESPACE_RE.sub("", value or "").strip()


def normalize_digit_text(value: str) -> str:
    """Normalize free-form digit text (e.g. 站点距离) to canonical digits."""

    raw = normalize_free_text(value)
    match = _DIGITS_RE.search(raw)
    if not match:
        return raw
    digits = match.group(0)
    try:
        return str(int(digits))
    except Exception:
        return digits


def normalize_bbu_ocr_text(value: str) -> str:
    """Normalize BBU OCR strings for stable matching."""

    normalized = normalize_free_text(value).translate(_BBU_OCR_TRANSLATION)
    return normalized.upper()


def is_irrelevant(meta: Any) -> bool:
    return isinstance(meta, Mapping) and meta.get("_fusion_source") == _IRRELEVANT_SOURCE


def get_domain_token(meta: Any) -> str | None:
    if not isinstance(meta, Mapping):
        return None
    if meta.get("_fusion_source") == _IRRELEVANT_SOURCE:
        return None
    template = meta.get("_fusion_template")
    if not isinstance(template, str) or not template.strip():
        return None
    if template == "summary_bbu":
        return "BBU"
    if template == "summary_rru":
        return "RRU"
    raise ValueError(
        "Summary template must be summary_bbu or summary_rru "
        f"for domain token mapping; got {template!r}."
    )


def get_summary_ref(meta: Any) -> str | None:
    if not isinstance(meta, Mapping):
        return None
    summary = meta.get("summary_ref")
    if isinstance(summary, str):
        summary = summary.strip()
        if summary:
            return summary
    return None


def extract_strict_json_line(
    *,
    meta: Any,
    text: str,
    lines: list[str] | None = None,
    domain_token: str | None = None,
) -> str | None:
    """Return the JSON line only when the output follows the strict 2-line contract."""

    if is_irrelevant(meta):
        return None
    if lines is None:
        lines = split_lines(text)
    if len(lines) != 2:
        return None
    if domain_token is None:
        domain_token = get_domain_token(meta)
    if domain_token is None:
        return None
    expected = f"<DOMAIN={domain_token}>, <TASK=SUMMARY>"
    if lines[0].strip() != expected:
        return None
    return lines[1].strip()


def canonicalize(obj: Any, *, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        items: dict[str, Any] = {}
        for k, v in obj.items():
            items[str(k)] = canonicalize(v, key=str(k))
        return {k: items[k] for k in sorted(items.keys())}
    if isinstance(obj, list):
        if key in {"统计", "备注"}:
            canonical_items = [canonicalize(item) for item in obj]
            serialized = [
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for item in canonical_items
            ]
            return sorted(serialized)
        return [canonicalize(item) for item in obj]
    return obj


def normalize_summary(obj: Any, domain_token: str | None) -> tuple[Any | None, bool]:
    if not isinstance(obj, dict):
        return None, False
    if domain_token == "RRU" and "备注" in obj:
        return None, False
    if domain_token == "BBU" and "分组统计" in obj:
        return None, False
    return canonicalize(obj), True


def parse_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value.is_integer():
            return int(value) if value > 0 else None
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def parse_non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value.is_integer():
            int_value = int(value)
            return int_value if int_value >= 0 else None
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


class JsonDuplicateKeyError(ValueError):
    pass


def loads_json_rejecting_duplicate_keys(text: str) -> Any:
    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for k, v in pairs:
            if k in obj:
                raise JsonDuplicateKeyError(f"Duplicate key: {k}")
            obj[k] = v
        return obj

    return json.loads(text, object_pairs_hook=_hook)


__all__ = [
    "_HEADER_PATTERN",
    "_IRRELEVANT_SOURCE",
    "_IRRELEVANT_TEXT",
    "JsonDuplicateKeyError",
    "canonicalize",
    "ensure_list",
    "extract_strict_json_line",
    "extract_text",
    "get_domain_token",
    "get_summary_ref",
    "is_irrelevant",
    "loads_json_rejecting_duplicate_keys",
    "normalize_bbu_ocr_text",
    "normalize_digit_text",
    "normalize_free_text",
    "normalize_summary",
    "normalize_text",
    "parse_non_negative_int",
    "parse_positive_int",
    "split_lines",
]
