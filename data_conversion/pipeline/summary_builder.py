#!/usr/bin/env python3
"""
Summary builder for BBU/RRU datasets (key=value desc mode).

Outputs a JSON-string summary with per-category statistics and optional
remarks/group breakdowns.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from data_conversion.utils.sanitizers import sanitize_free_text_value


NEGATIVE_MARKERS = ("不符合", "不合规", "不合格", "不合理", "错误", "不能")
FREE_TEXT_KEYS = {"备注", "文本"}

BBU_CATEGORIES = {
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

RRU_CATEGORIES = {
    "RRU设备",
    "紧固件",
    "RRU接地端",
    "地排接地端螺丝",
    "尾纤",
    "接地线",
    "标签",
    "站点距离",
}


def _is_negative_value(value: str) -> bool:
    return any(marker in value for marker in NEGATIVE_MARKERS)


def _parse_desc(desc: str) -> Tuple[str, Dict[str, List[str]], Dict[str, bool]]:
    """Parse key=value desc into category, kv map, and error flags."""
    kv: Dict[str, List[str]] = {}
    errors = {
        "invalid": False,
        "conflict": False,
    }

    tokens = [t.strip() for t in desc.split(",") if t.strip()]
    current_key = None
    current_value = ""
    stray_tokens: List[str] = []

    def _append_value(key: str, value: str) -> None:
        if not key or not value:
            errors["invalid"] = True
            return
        if key in FREE_TEXT_KEYS:
            sanitized = sanitize_free_text_value(value)
            if not sanitized:
                return
            kv.setdefault(key, []).append(sanitized)
        else:
            values = [v for v in value.split("|") if v]
            if not values:
                return
            kv.setdefault(key, []).extend(values)

    for token in tokens:
        if "=" in token:
            if current_key is not None:
                _append_value(current_key, current_value)
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                errors["invalid"] = True
                current_key = None
                current_value = ""
                stray_tokens.append(token)
                continue
            current_key = key
            current_value = value
        else:
            if current_key in FREE_TEXT_KEYS:
                current_value = f"{current_value},{token}" if current_value else token
            else:
                stray_tokens.append(token)

    if current_key is not None:
        _append_value(current_key, current_value)

    if stray_tokens:
        remark_value = sanitize_free_text_value(",".join(stray_tokens))
        if remark_value:
            kv.setdefault("备注", []).append(remark_value)

    category = ""
    if "类别" in kv and kv["类别"]:
        category = kv["类别"][0]

    for values in kv.values():
        if len(values) < 2:
            continue
        has_negative = any(_is_negative_value(v) for v in values)
        has_positive = any(not _is_negative_value(v) for v in values)
        if has_negative and has_positive:
            errors["conflict"] = True
            break

    if not category:
        errors["invalid"] = True

    return category, kv, errors


def build_summary_from_objects(
    objects: List[Dict[str, Any]], *, dataset: str = "BBU"
) -> str:
    """Build JSON-string summary from objects with key=value descs."""
    if not objects:
        raise ValueError("build_summary_from_objects: no objects provided (fail-fast)")

    summary_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    remarks: List[str] = []
    remark_seen: set[str] = set()
    group_stats: Dict[str, int] = {}

    invalid = 0
    unknown_category = 0
    conflicts = 0
    error_examples: Dict[str, str] = {}

    expected_categories = BBU_CATEGORIES if dataset.upper() == "BBU" else RRU_CATEGORIES

    for obj in objects:
        desc = obj.get("desc", "")
        if not isinstance(desc, str):
            desc = str(desc)
        desc = desc.strip()
        if not desc:
            invalid += 1
            error_examples.setdefault("无法解析", "<empty>")
            continue

        category, kv, errors = _parse_desc(desc)
        if errors["invalid"]:
            invalid += 1
            error_examples.setdefault("无法解析", desc)
        if errors["conflict"]:
            conflicts += 1
            error_examples.setdefault("冲突值", desc)

        if category and category not in expected_categories:
            unknown_category += 1
            error_examples.setdefault("未知类别", desc)

        if not category:
            continue

        cat_stats = summary_stats.setdefault(category, {})

        for key, values in kv.items():
            if key == "类别":
                continue
            if key == "备注":
                if dataset.upper() == "BBU":
                    for value in values:
                        if value and value not in remark_seen:
                            remark_seen.add(value)
                            remarks.append(value)
                continue
            if key == "组":
                for gid in values:
                    if not gid:
                        continue
                    group_stats[gid] = group_stats.get(gid, 0) + 1
                    cat_group_stats = cat_stats.setdefault("组", {})
                    cat_group_stats[gid] = cat_group_stats.get(gid, 0) + 1
                continue

            value_counts = cat_stats.setdefault(key, {})
            for value in values:
                if not value:
                    continue
                value_counts[value] = value_counts.get(value, 0) + 1

    summary_obj: Dict[str, Any] = {
        "统计": [],
    }

    dataset_key = dataset.upper()

    def _category_sort_key(category: str) -> tuple[int, str]:
        # RRU contract: "站点距离" must be the first stats entry when present.
        # This aligns with Stage-A prompt constraints and stabilizes training.
        if dataset_key == "RRU" and category == "站点距离":
            return (0, "")
        return (1, category)

    for category in sorted(summary_stats.keys(), key=_category_sort_key):
        entry: Dict[str, Any] = {"类别": category}
        for key in sorted(summary_stats[category].keys()):
            entry[key] = summary_stats[category][key]
        summary_obj["统计"].append(entry)

    if dataset_key == "BBU" and remarks:
        summary_obj["备注"] = remarks
    elif dataset_key != "BBU" and group_stats:
        summary_obj["分组统计"] = group_stats

    error_fields: Dict[str, Any] = {}
    if invalid:
        error_fields["无法解析"] = invalid
    if unknown_category:
        error_fields["未知类别"] = unknown_category
    if conflicts:
        error_fields["冲突值"] = conflicts
    if error_examples:
        error_fields["示例"] = error_examples
    if error_fields:
        raise ValueError(f"summary anomalies detected: {error_fields}")

    return json.dumps(summary_obj, ensure_ascii=False, separators=(", ", ": "))
