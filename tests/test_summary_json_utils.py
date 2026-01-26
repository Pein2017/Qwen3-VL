from __future__ import annotations

import json

from src.utils.summary_json import (
    extract_summary_json_line,
    extract_summary_json_obj,
    format_summary_json,
    strip_summary_headers,
)


def test_strip_summary_headers_removes_domain_task_line() -> None:
    text = '<DOMAIN=BBU>, <TASK=SUMMARY>\n{"统计": []}\n'
    assert strip_summary_headers(text) == '{"统计": []}'


def test_extract_summary_json_line_returns_canonical_format() -> None:
    text = '<DOMAIN=BBU>, <TASK=SUMMARY>\n{"统计": [{"类别": "RRU设备"}], "extra": 1}\n'
    extracted = extract_summary_json_line(text, context="t")
    assert extracted is not None
    # Ensure it's valid JSON.
    parsed = json.loads(extracted)
    assert "统计" in parsed
    assert parsed.get("extra") == 1


def test_format_summary_json_orders_canonical_keys_first_and_preserves_extras() -> None:
    obj: dict[str, object] = {"extra": 1, "备注": "x", "统计": []}
    formatted = format_summary_json(obj, context="t")
    assert '"统计"' in formatted
    assert '"备注"' in formatted
    assert '"extra"' in formatted
    # Canonical keys come first.
    assert (
        formatted.index('"统计"')
        < formatted.index('"备注"')
        < formatted.index('"extra"')
    )


def test_extract_summary_json_prefers_last_json_line() -> None:
    text = '<DOMAIN=BBU>, <TASK=SUMMARY>\n{"统计": [1]}\n{"统计": [2]}\n'
    extracted = extract_summary_json_line(text, context="t")
    assert extracted is not None
    parsed = json.loads(extracted)
    assert parsed["统计"] == [2]


def test_extract_summary_json_returns_none_when_missing() -> None:
    assert extract_summary_json_obj("hello", context="t") is None
    assert extract_summary_json_obj("无关图片", context="t") is None
