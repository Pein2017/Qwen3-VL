"""Summary reward fact extraction and scoring helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Tuple

from .parsing import (
    normalize_digit_text,
    normalize_free_text,
    parse_non_negative_int,
    parse_positive_int,
)


def extract_summary_fact_counts(
    obj: Any, domain_token: str | None
) -> dict[Tuple[str, ...], int] | None:
    """Extract count-weighted "facts" for partial content similarity."""
    if not isinstance(obj, dict):
        return None

    if domain_token == "RRU" and "备注" in obj:
        return None
    if domain_token == "BBU" and "分组统计" in obj:
        return None

    counts: dict[Tuple[str, ...], int] = {}

    notes = obj.get("备注")
    if isinstance(notes, list):
        for note in {normalize_free_text(str(n)) for n in notes if n is not None}:
            if note:
                counts[("备注", note)] = 1

    group_stats = obj.get("分组统计")
    if isinstance(group_stats, dict):
        for group_id_raw, count_raw in group_stats.items():
            group_id = normalize_digit_text(str(group_id_raw))
            if not group_id:
                continue
            count = parse_positive_int(count_raw)
            if count is None:
                continue
            key = ("分组统计", group_id)
            counts[key] = counts.get(key, 0) + count

    stats = obj.get("统计")
    if isinstance(stats, list):
        for entry in stats:
            if not isinstance(entry, dict):
                continue
            category_raw = entry.get("类别")
            if category_raw is None:
                continue
            category = str(category_raw).strip()
            if not category:
                continue
            for attr_raw, val in entry.items():
                if attr_raw in {"类别", "异常"}:
                    continue
                attr = str(attr_raw).strip()
                if not attr:
                    continue
                if isinstance(val, dict):
                    for value_raw, count_raw in val.items():
                        value_str = str(value_raw).strip()
                        if attr == "文本":
                            value_str = normalize_free_text(value_str)
                        elif attr in {"站点距离", "组"}:
                            value_str = normalize_digit_text(value_str)
                        if not value_str:
                            continue
                        count = parse_positive_int(count_raw)
                        if count is None:
                            continue
                        key = ("统计", category, attr, value_str)
                        counts[key] = counts.get(key, 0) + count
                elif isinstance(val, list):
                    for value_raw in val:
                        value_str = str(value_raw).strip()
                        if attr == "文本":
                            value_str = normalize_free_text(value_str)
                        elif attr in {"站点距离", "组"}:
                            value_str = normalize_digit_text(value_str)
                        if not value_str:
                            continue
                        key = ("统计", category, attr, value_str)
                        counts[key] = counts.get(key, 0) + 1
                else:
                    value_str = str(val).strip()
                    if attr == "文本":
                        value_str = normalize_free_text(value_str)
                    elif attr in {"站点距离", "组"}:
                        value_str = normalize_digit_text(value_str)
                    if not value_str:
                        continue
                    key = ("统计", category, attr, value_str)
                    counts[key] = counts.get(key, 0) + 1

    return counts


def f1_from_sets(pred: set[str], ref: set[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def f1_from_fact_counts(
    pred_counts: Mapping[Tuple[str, ...], int],
    ref_counts: Mapping[Tuple[str, ...], int],
) -> float:
    pred_total = sum(int(v) for v in pred_counts.values() if v)
    ref_total = sum(int(v) for v in ref_counts.values() if v)
    if pred_total <= 0 and ref_total <= 0:
        return 1.0
    if pred_total <= 0 or ref_total <= 0:
        return 0.0

    overlap = 0
    for key in pred_counts.keys() & ref_counts.keys():
        overlap += min(int(pred_counts[key]), int(ref_counts[key]))

    if overlap <= 0:
        return 0.0

    precision = overlap / pred_total
    recall = overlap / ref_total
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def tversky_from_fact_counts(
    pred_counts: Mapping[Tuple[str, ...], int],
    ref_counts: Mapping[Tuple[str, ...], int],
    *,
    alpha: float,
    beta: float,
) -> float:
    """Recall-biased similarity for count-weighted facts."""

    pred_total = sum(int(v) for v in pred_counts.values() if v)
    ref_total = sum(int(v) for v in ref_counts.values() if v)
    if pred_total <= 0 and ref_total <= 0:
        return 1.0
    if ref_total <= 0:
        return 0.0

    overlap = 0
    for key in pred_counts.keys() & ref_counts.keys():
        overlap += min(int(pred_counts[key]), int(ref_counts[key]))

    fp = max(0, pred_total - overlap)
    fn = max(0, ref_total - overlap)
    denom = overlap + alpha * fp + beta * fn
    if denom <= 0:
        return 0.0
    return float(overlap) / float(denom)


def score_objects_total(pred_value: Any, ref_value: Any) -> float:
    pred = parse_non_negative_int(pred_value)
    ref = parse_non_negative_int(ref_value)
    if pred is None or ref is None:
        return 0.0
    diff = abs(pred - ref)
    return 1.0 / (1.0 + float(diff))


def score_objects_total_lower_bound(pred_value: Any, ref_value: Any) -> float:
    """Lower-bound objects_total score with +2 free slack."""

    pred = parse_non_negative_int(pred_value)
    ref = parse_non_negative_int(ref_value)
    if pred is None or ref is None:
        return 0.0

    if pred < ref:
        under = ref - pred
        scale_under = max(3.0, 0.3 * float(ref))
        return float(math.exp(-float(under) / scale_under))

    if pred <= ref + 2:
        return 1.0

    over = pred - (ref + 2)
    scale_over = max(6.0, 0.5 * float(ref))
    return float(math.exp(-float(over) / scale_over))


def filter_fact_counts(
    counts: Mapping[Tuple[str, ...], int],
    *,
    exclude_notes: bool,
    exclude_text: bool,
) -> dict[Tuple[str, ...], int]:
    filtered: dict[Tuple[str, ...], int] = {}
    for key, value in counts.items():
        if not value:
            continue
        if exclude_notes and key and key[0] == "备注":
            continue
        if exclude_text and len(key) >= 4 and key[0] == "统计" and key[2] == "文本":
            continue
        filtered[key] = int(value)
    return filtered


__all__ = [
    "extract_summary_fact_counts",
    "f1_from_fact_counts",
    "f1_from_sets",
    "filter_fact_counts",
    "score_objects_total",
    "score_objects_total_lower_bound",
    "tversky_from_fact_counts",
]
