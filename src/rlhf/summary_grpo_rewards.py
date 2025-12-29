"""Reward functions for summary-mode GRPO."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Iterable, List, Tuple

from swift.plugin import orms as _orms
from swift.plugin.orm import ORM

_IRRELEVANT_SOURCE = "irrelevant_summary"
_IRRELEVANT_TEXT = "无关图片"
_HEADER_PATTERN = re.compile(r"^<DOMAIN=[A-Z]+>, <TASK=[A-Z]+>$")
_SPECIAL_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>")


def _normalize_text(text: str) -> str:
    for token in _SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def _extract_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return _normalize_text(completion)
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
            return _normalize_text("".join(parts))
        return ""
    if isinstance(completion, Mapping):
        text = completion.get("text")
        if text is not None:
            return _normalize_text(str(text))
    return ""


def _ensure_list(value: Any, n: int) -> list[Any]:
    if isinstance(value, list):
        if len(value) < n:
            return value + [None] * (n - len(value))
        return value
    return [value] * n


def _split_lines(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return stripped.splitlines()


def _is_irrelevant(meta: Any) -> bool:
    return isinstance(meta, Mapping) and meta.get("_fusion_source") == _IRRELEVANT_SOURCE


def _get_domain_token(meta: Any) -> str | None:
    if not isinstance(meta, Mapping):
        return None
    token = meta.get("_fusion_domain_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    template = meta.get("_fusion_template")
    if isinstance(template, str):
        lowered = template.lower()
        if "bbu" in lowered:
            return "BBU"
        if "rru" in lowered:
            return "RRU"
    return None


def _get_summary_ref(meta: Any) -> str | None:
    if not isinstance(meta, Mapping):
        return None
    summary = meta.get("summary_ref")
    if isinstance(summary, str):
        summary = summary.strip()
        if summary:
            return summary
    return None


def _canonicalize(obj: Any, *, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        items: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "异常":
                continue
            items[str(k)] = _canonicalize(v, key=str(k))
        return {k: items[k] for k in sorted(items.keys())}
    if isinstance(obj, list):
        if key in {"统计", "备注"}:
            canonical_items = [_canonicalize(item) for item in obj]
            serialized = [
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for item in canonical_items
            ]
            return sorted(serialized)
        return [_canonicalize(item) for item in obj]
    return obj


def _normalize_summary(obj: Any, domain_token: str | None) -> tuple[Any | None, bool]:
    if not isinstance(obj, dict):
        return None, False
    if domain_token == "RRU" and "备注" in obj:
        return None, False
    if domain_token == "BBU" and "分组统计" in obj:
        return None, False
    return _canonicalize(obj), True


def _parse_positive_int(value: Any) -> int | None:
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


def _parse_non_negative_int(value: Any) -> int | None:
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


def _extract_summary_fact_counts(
    obj: Any, domain_token: str | None
) -> dict[Tuple[str, ...], int] | None:
    """Extract count-weighted "facts" for partial content similarity.

    Facts are derived from:
    - 顶层 "备注" (list of strings; de-duplicated)
    - 顶层 "统计" (list of per-category dicts, each containing 值→次数 mappings)

    Domain constraints (consistent with SummaryContentReward normalization):
    - RRU summaries MUST NOT include 顶层 "备注"
    - BBU summaries MUST NOT include 顶层 "分组统计"
    """
    if not isinstance(obj, dict):
        return None

    if domain_token == "RRU" and "备注" in obj:
        return None
    if domain_token == "BBU" and "分组统计" in obj:
        return None

    counts: dict[Tuple[str, ...], int] = {}

    notes = obj.get("备注")
    if isinstance(notes, list):
        for note in {str(n).strip() for n in notes if n is not None}:
            if note:
                counts[("备注", note)] = 1

    group_stats = obj.get("分组统计")
    if isinstance(group_stats, dict):
        for group_id_raw, count_raw in group_stats.items():
            group_id = str(group_id_raw).strip()
            if not group_id:
                continue
            count = _parse_positive_int(count_raw)
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
                        if not value_str:
                            continue
                        count = _parse_positive_int(count_raw)
                        if count is None:
                            continue
                        key = ("统计", category, attr, value_str)
                        counts[key] = counts.get(key, 0) + count
                elif isinstance(val, list):
                    for value_raw in val:
                        value_str = str(value_raw).strip()
                        if not value_str:
                            continue
                        key = ("统计", category, attr, value_str)
                        counts[key] = counts.get(key, 0) + 1
                else:
                    value_str = str(val).strip()
                    if not value_str:
                        continue
                    key = ("统计", category, attr, value_str)
                    counts[key] = counts.get(key, 0) + 1

    return counts


def _f1_from_fact_counts(
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


def _score_objects_total(pred_value: Any, ref_value: Any) -> float:
    pred = _parse_non_negative_int(pred_value)
    ref = _parse_non_negative_int(ref_value)
    if pred is None or ref is None:
        return 0.0
    diff = abs(pred - ref)
    return 1.0 / (1.0 + float(diff))


class SummaryFormatReward(ORM):
    """Checks summary format: header+JSON for non-irrelevant, single-line for irrelevant."""

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            text = _extract_text(completion)
            if _is_irrelevant(meta):
                rewards.append(1.0 if text == _IRRELEVANT_TEXT else 0.0)
                continue
            lines = _split_lines(text)
            if len(lines) != 2:
                rewards.append(0.0)
                continue
            header = lines[0].strip()
            json_line = lines[1].strip()
            if not _HEADER_PATTERN.match(header):
                rewards.append(0.0)
                continue
            if not json_line.endswith("}"):
                rewards.append(0.0)
                continue
            rewards.append(1.0)
        return rewards


class SummaryHeaderReward(ORM):
    """Validates <DOMAIN> and <TASK=SUMMARY> for non-irrelevant samples."""

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            text = _extract_text(completion)
            lines = _split_lines(text)
            if _is_irrelevant(meta):
                if lines and lines[0].strip().startswith("<DOMAIN="):
                    rewards.append(-1.0)
                elif "<TASK=" in text:
                    rewards.append(-1.0)
                else:
                    rewards.append(0.0)
                continue
            if not lines:
                rewards.append(0.0)
                continue
            domain_token = _get_domain_token(meta)
            if domain_token is None:
                rewards.append(0.0)
                continue
            expected = f"<DOMAIN={domain_token}>, <TASK=SUMMARY>"
            rewards.append(1.0 if lines[0].strip() == expected else 0.0)
        return rewards


class SummaryParsePenalty(ORM):
    """Applies a negative reward only when JSON parsing fails."""

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            if _is_irrelevant(meta):
                rewards.append(0.0)
                continue
            text = _extract_text(completion)
            lines = _split_lines(text)
            if len(lines) < 2:
                rewards.append(-1.0)
                continue
            json_line = lines[1].strip()
            try:
                json.loads(json_line)
            except Exception:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        return rewards


class SummaryContentReward(ORM):
    """Checks order-invariant JSON equivalence against the ground-truth summary."""

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            if _is_irrelevant(meta):
                rewards.append(0.0)
                continue
            text = _extract_text(completion)
            lines = _split_lines(text)
            if len(lines) < 2:
                rewards.append(0.0)
                continue
            json_line = lines[1].strip()
            try:
                pred_json = json.loads(json_line)
            except Exception:
                rewards.append(0.0)
                continue
            summary_ref = _get_summary_ref(meta)
            if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
                rewards.append(0.0)
                continue
            try:
                ref_json = json.loads(summary_ref)
            except Exception:
                rewards.append(0.0)
                continue
            domain_token = _get_domain_token(meta)
            norm_pred, pred_ok = _normalize_summary(pred_json, domain_token)
            norm_ref, ref_ok = _normalize_summary(ref_json, domain_token)
            if not (pred_ok and ref_ok):
                rewards.append(0.0)
                continue
            rewards.append(1.0 if norm_pred == norm_ref else 0.0)
        return rewards


class SummaryDatasetReward(ORM):
    """Checks JSON `dataset` matches the expected domain token (BBU/RRU)."""

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            if _is_irrelevant(meta):
                rewards.append(0.0)
                continue

            domain_token = _get_domain_token(meta)
            if domain_token is None:
                rewards.append(0.0)
                continue

            text = _extract_text(completion)
            lines = _split_lines(text)
            if len(lines) < 2:
                rewards.append(0.0)
                continue

            json_line = lines[1].strip()
            try:
                pred_json = json.loads(json_line)
            except Exception:
                rewards.append(0.0)
                continue
            if not isinstance(pred_json, dict):
                rewards.append(0.0)
                continue

            dataset_raw = pred_json.get("dataset")
            if not isinstance(dataset_raw, str) or not dataset_raw.strip():
                rewards.append(0.0)
                continue
            rewards.append(
                1.0 if dataset_raw.strip().upper() == domain_token.strip().upper() else 0.0
            )
        return rewards


class SummaryObjectsTotalReward(ORM):
    """Dense reward for matching JSON `objects_total` (1.0 when equal, decays with abs diff)."""

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            if _is_irrelevant(meta):
                rewards.append(0.0)
                continue

            text = _extract_text(completion)
            lines = _split_lines(text)
            if len(lines) < 2:
                rewards.append(0.0)
                continue

            json_line = lines[1].strip()
            try:
                pred_json = json.loads(json_line)
            except Exception:
                rewards.append(0.0)
                continue
            if not isinstance(pred_json, dict):
                rewards.append(0.0)
                continue

            summary_ref = _get_summary_ref(meta)
            if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
                rewards.append(0.0)
                continue
            try:
                ref_json = json.loads(summary_ref)
            except Exception:
                rewards.append(0.0)
                continue
            if not isinstance(ref_json, dict):
                rewards.append(0.0)
                continue

            rewards.append(
                float(
                    _score_objects_total(
                        pred_json.get("objects_total"), ref_json.get("objects_total")
                    )
                )
            )

        return rewards


class SummaryContentF1Reward(ORM):
    """Partial-match content reward based on count-weighted fact overlap.

    This is intentionally *not* exact-match: it gives a dense signal in [0, 1],
    where:
    - 1.0 means predicted summary facts fully match `metadata.summary_ref`
    - 0.0 means no fact overlap at all
    - intermediate values scale with correctness (and penalize spurious extras)
    """

    def __call__(self, completions: Iterable[Any], metadata=None, **kwargs) -> List[float]:
        completions_list = list(completions)
        metas = _ensure_list(metadata, len(completions_list))
        rewards: list[float] = []
        for completion, meta in zip(completions_list, metas):
            if _is_irrelevant(meta):
                rewards.append(0.0)
                continue

            text = _extract_text(completion)
            lines = _split_lines(text)
            if len(lines) < 2:
                rewards.append(0.0)
                continue

            json_line = lines[1].strip()
            try:
                pred_json = json.loads(json_line)
            except Exception:
                rewards.append(0.0)
                continue

            summary_ref = _get_summary_ref(meta)
            if not summary_ref or summary_ref == _IRRELEVANT_TEXT:
                rewards.append(0.0)
                continue

            try:
                ref_json = json.loads(summary_ref)
            except Exception:
                rewards.append(0.0)
                continue

            domain_token = _get_domain_token(meta)
            pred_counts = _extract_summary_fact_counts(pred_json, domain_token)
            ref_counts = _extract_summary_fact_counts(ref_json, domain_token)
            if pred_counts is None or ref_counts is None:
                rewards.append(0.0)
                continue

            rewards.append(float(_f1_from_fact_counts(pred_counts, ref_counts)))

        return rewards


def register_summary_grpo_rewards() -> None:
    _orms.setdefault("summary_format", SummaryFormatReward)
    _orms.setdefault("summary_header", SummaryHeaderReward)
    _orms.setdefault("summary_parse", SummaryParsePenalty)
    _orms.setdefault("summary_content", SummaryContentReward)
    _orms.setdefault("summary_dataset", SummaryDatasetReward)
    _orms.setdefault("summary_objects_total", SummaryObjectsTotalReward)
    _orms.setdefault("summary_content_f1", SummaryContentF1Reward)


register_summary_grpo_rewards()

__all__ = [
    "register_summary_grpo_rewards",
    "SummaryFormatReward",
    "SummaryHeaderReward",
    "SummaryParsePenalty",
    "SummaryContentReward",
    "SummaryDatasetReward",
    "SummaryObjectsTotalReward",
    "SummaryContentF1Reward",
]
