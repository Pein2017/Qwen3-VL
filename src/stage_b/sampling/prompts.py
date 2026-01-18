#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the Stage-B rule-search pipeline."""

from __future__ import annotations

import json
import re

from src.prompts.stage_b_verdict import build_stage_b_system_prompt
from src.utils import require_mapping
from src.utils.unstructured import UnstructuredMapping

from ..types import GroupTicket, MissionGuidance
from ..utils.chinese import normalize_spaces, to_simplified

_INDEX_RE = re.compile(r"(\d+)$")
_STATION_DISTANCE_RE = re.compile(r"站点距离[=/](\d+)")


def _parse_summary_json(text: str) -> UnstructuredMapping | None:
    if not text:
        return None

    stripped = text.strip()
    if not stripped or stripped.startswith("无关图片"):
        return None

    lines = [line for line in stripped.splitlines() if line.strip()]
    stripped = "\n".join(
        [
            line
            for line in lines
            if not (
                line.strip().startswith("<DOMAIN=")
                and "<TASK=" in line
                and line.strip().endswith(">")
            )
        ]
    ).strip()
    if not stripped:
        return None

    def _maybe_parse_obj(candidate: str) -> UnstructuredMapping | None:
        c = candidate.strip()
        if not (c.startswith("{") and c.endswith("}")):
            return None
        try:
            parsed = json.loads(c)
        except Exception:
            return None
        try:
            return require_mapping(parsed, context="stage_b.summary_json")
        except TypeError:
            return None

    def _is_summary(obj: UnstructuredMapping) -> bool:
        return "统计" in obj

    # Fast path: whole string is the JSON object.
    obj = _maybe_parse_obj(stripped)
    if obj is not None and _is_summary(obj):
        return obj

    # Fallback: extract the first {...} block from the full text.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = _maybe_parse_obj(stripped[start : end + 1])
        if obj is not None and _is_summary(obj):
            return obj

    return None


def _format_summary_json(obj: UnstructuredMapping) -> str:
    obj = require_mapping(obj, context="stage_b.summary")
    preferred_order = ["统计", "备注", "分组统计"]
    allowed = set(preferred_order)
    ordered: dict[str, object] = {}
    for key in preferred_order:
        if key in obj:
            ordered[key] = obj[key]
    for key, value in obj.items():
        if key == "format_version":
            continue
        if key not in ordered and key in allowed:
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def _summary_entries(obj: UnstructuredMapping) -> list[UnstructuredMapping]:
    entries = obj.get("统计")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def _entry_category(entry: UnstructuredMapping) -> str | None:
    """Extract a hashable category name from a Stage-A `统计` entry.

    Stage-A is expected to emit `{"类别": "<name>", ...}` but we have seen rare
    corrupted rows where `类别` becomes a mapping like `{"地线夹": 1}`.
    For Stage-B prompting we only need a stable category identifier; for such
    mapping-shaped values we conservatively take the first string key when the
    mapping has exactly one key, otherwise we drop it.
    """

    raw = entry.get("类别")
    if isinstance(raw, str):
        cat = raw.strip()
        return cat or None
    if isinstance(raw, dict):
        keys = [k for k in raw.keys() if isinstance(k, str) and k.strip()]
        if len(keys) == 1:
            return keys[0].strip()
    return None


def _entry_by_category(
    entries: list[UnstructuredMapping], category: str
) -> UnstructuredMapping | None:
    for entry in entries:
        if _entry_category(entry) == category:
            return entry
    return None


def _summary_distances(obj: UnstructuredMapping) -> list[str]:
    entry = _entry_by_category(_summary_entries(obj), "站点距离")
    if not entry:
        return []
    distances = entry.get("站点距离")
    if not isinstance(distances, dict):
        distances = entry.get("距离")
    if isinstance(distances, dict):
        return [str(k) for k in distances.keys() if str(k).strip().isdigit()]
    return []


def _summary_has_label_text(obj: UnstructuredMapping) -> bool:
    entry = _entry_by_category(_summary_entries(obj), "标签")
    if not entry:
        return False
    texts = entry.get("文本")
    if isinstance(texts, dict) and any(str(k).strip() for k in texts.keys()):
        return True
    readability = entry.get("可读性")
    if isinstance(readability, dict):
        return any(str(k).strip() and str(k) != "不可读" for k in readability.keys())
    return False


def _estimate_object_count_from_summary(obj: UnstructuredMapping) -> int:
    """Best-effort object count estimate derived from `统计`."""

    entries = _summary_entries(obj)

    def _to_int(value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                try:
                    return int(stripped)
                except Exception:
                    return None
        return None

    def _sum_counts(value: object) -> int:
        if isinstance(value, dict):
            total = 0
            for count_raw in value.values():
                count = _to_int(count_raw)
                if count is None or count <= 0:
                    continue
                total += int(count)
            return total
        if isinstance(value, list):
            return len([v for v in value if v is not None])
        if value is None:
            return 0
        return 1

    def _estimate_category(entry: UnstructuredMapping) -> int:
        category = entry.get("类别")
        cat = category.strip() if isinstance(category, str) else ""

        max_attr_total = 0
        label_text_total = 0
        label_readability_total = 0

        for key, val in entry.items():
            if key == "类别":
                continue
            total = _sum_counts(val)
            max_attr_total = max(max_attr_total, total)
            if key == "文本":
                label_text_total = total
            elif key == "可读性":
                label_readability_total = total

        if cat == "标签":
            combined = label_text_total + label_readability_total
            return max(1, max_attr_total, combined)
        return max(1, max_attr_total)

    return int(sum(_estimate_category(entry) for entry in entries))


def _sanitize_stage_a_summary_for_prompt(text: str) -> str:
    """Drop summary headers and return the payload unchanged for prompting."""

    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("无关图片"):
        return "无关图片"

    lines = [line for line in stripped.splitlines() if line.strip()]
    kept = [
        line
        for line in lines
        if not (
            line.strip().startswith("<DOMAIN=")
            and "<TASK=" in line
            and line.strip().endswith(">")
        )
    ]
    return "\n".join(kept).strip()


def _sorted_summaries(per_image: dict[str, str]) -> list[tuple[str, str]]:
    def _index(key: str) -> int:
        match = _INDEX_RE.search(key)
        if match:
            try:
                return int(match.group(1))
            except ValueError:  # pragma: no cover - defensive
                return 0
        return 0

    return sorted(per_image.items(), key=lambda item: _index(item[0]))


def _render_guidance_snippets(experiences: dict[str, str]) -> str:
    """Format experiences dict as numbered experiences text block."""

    if not experiences:
        raise ValueError("Experiences dict must be non-empty")
    formatted: list[str] = [
        f"{idx}. {value}"
        for idx, (_, value) in enumerate(sorted(experiences.items()), start=1)
    ]
    return "\n".join(formatted)


def _has_readable_label(text: str) -> bool:
    summary_obj = _parse_summary_json(text)
    if summary_obj is not None:
        return _summary_has_label_text(summary_obj)

    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified)
    if "标签/" not in simplified:
        return False
    if "无法识别" in simplified or "没有标签" in simplified:
        return False
    return True


def _aggregate_rru_install_points(stage_a_summaries: dict[str, str]) -> str:
    """Build a compact per-station aggregation for RRU install checks."""

    if not stage_a_summaries:
        return ""

    stats: dict[str, dict[str, bool]] = {}
    for _, text in _sorted_summaries(stage_a_summaries):
        summary_obj = _parse_summary_json(text)
        if summary_obj is not None:
            distances = _summary_distances(summary_obj)
            if not distances:
                continue
            entries = _summary_entries(summary_obj)
            categories = {_entry_category(entry) for entry in entries}
            categories.discard(None)
            has_rru = "RRU设备" in categories
            has_fix = "紧固件" in categories or "固定件" in categories
            for dist in distances:
                entry = stats.setdefault(dist, {"rru": False, "fix": False})
                entry["rru"] = entry["rru"] or has_rru
                entry["fix"] = entry["fix"] or has_fix
            continue

        simplified = _sanitize_stage_a_summary_for_prompt(text)
        distances = _STATION_DISTANCE_RE.findall(simplified)
        if not distances:
            continue
        has_rru = bool(re.search(r"\bRRU\b", simplified) or "RRU设备" in simplified)
        has_fix = "紧固件" in simplified or "固定件" in simplified
        for dist in distances:
            entry = stats.setdefault(dist, {"rru": False, "fix": False})
            entry["rru"] = entry["rru"] or has_rru
            entry["fix"] = entry["fix"] or has_fix

    if not stats:
        return ""

    def _dist_key(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0

    lines = [
        f"- 站点距离={dist}: RRU={'有' if flags['rru'] else '无'}, 紧固件={'有' if flags['fix'] else '无'}"
        for dist, flags in sorted(stats.items(), key=lambda item: _dist_key(item[0]))
    ]
    return "安装点汇总（按站点距离合并）：\n" + "\n".join(lines)


def _aggregate_rru_position_points(stage_a_summaries: dict[str, str]) -> str:
    """Build a compact per-station aggregation for RRU position checks."""

    if not stage_a_summaries:
        return ""

    stats: dict[str, dict[str, bool]] = {}
    for _, text in _sorted_summaries(stage_a_summaries):
        summary_obj = _parse_summary_json(text)
        if summary_obj is not None:
            distances = _summary_distances(summary_obj)
            if not distances:
                continue
            entries = _summary_entries(summary_obj)
            categories = {_entry_category(entry) for entry in entries}
            categories.discard(None)
            has_ground_terminal = "RRU接地端" in categories
            has_ground = "接地线" in categories
            has_label = _summary_has_label_text(summary_obj)
            has_ground_label = has_ground and has_label
            if not (has_ground_terminal or has_ground_label):
                continue
            for dist in distances:
                entry = stats.setdefault(
                    dist, {"ground_terminal": False, "ground_label": False}
                )
                entry["ground_terminal"] = (
                    entry["ground_terminal"] or has_ground_terminal
                )
                entry["ground_label"] = entry["ground_label"] or has_ground_label
            continue

        simplified = _sanitize_stage_a_summary_for_prompt(text)
        distances = _STATION_DISTANCE_RE.findall(simplified)
        if not distances:
            continue
        has_ground_terminal = "RRU接地端" in simplified
        has_ground = "接地线" in simplified
        has_label = _has_readable_label(simplified)
        # Only surface evidence-bearing points to reduce confusion:
        # - RRU evidence: has_ground_terminal
        # - Ground evidence: has_ground + readable label
        has_ground_label = has_ground and has_label
        if not (has_ground_terminal or has_ground_label):
            continue
        for dist in distances:
            entry = stats.setdefault(
                dist, {"ground_terminal": False, "ground_label": False}
            )
            entry["ground_terminal"] = entry["ground_terminal"] or has_ground_terminal
            entry["ground_label"] = entry["ground_label"] or has_ground_label

    if not stats:
        return ""

    def _dist_key(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0

    lines = [
        f"- 站点距离={dist}: RRU接地端={'有' if flags['ground_terminal'] else '无'}, 接地线(可读标签)={'有' if flags['ground_label'] else '无'}"
        for dist, flags in sorted(stats.items(), key=lambda item: _dist_key(item[0]))
    ]
    return "安装点汇总（按站点距离合并）：\n" + "\n".join(lines)


def _aggregate_rru_cable_points(stage_a_summaries: dict[str, str]) -> str:
    """Build a compact per-station aggregation for RRU cable checks."""

    if not stage_a_summaries:
        return ""

    stats: dict[str, dict[str, bool]] = {}
    for _, text in _sorted_summaries(stage_a_summaries):
        summary_obj = _parse_summary_json(text)
        if summary_obj is not None:
            distances = _summary_distances(summary_obj)
            if not distances:
                continue
            entries = _summary_entries(summary_obj)
            categories = {entry.get("类别") for entry in entries}
            has_tail = "尾纤" in categories
            tail_entry = _entry_by_category(entries, "尾纤")
            has_tube = False
            if tail_entry and isinstance(tail_entry.get("套管保护"), dict):
                tube_map = tail_entry.get("套管保护")
                if isinstance(tube_map, dict):
                    has_tube = any(
                        key not in {"没有保护", "无保护"} and count
                        for key, count in tube_map.items()
                    )
            has_label = _summary_has_label_text(summary_obj)
            if not (has_tail or has_tube or has_label):
                continue
            for dist in distances:
                entry = stats.setdefault(
                    dist,
                    {
                        "tail": False,
                        "tube": False,
                        "label": False,
                    },
                )
                entry["tail"] = entry["tail"] or has_tail
                entry["tube"] = entry["tube"] or has_tube
                entry["label"] = entry["label"] or has_label
            continue

        simplified = _sanitize_stage_a_summary_for_prompt(text)
        distances = _STATION_DISTANCE_RE.findall(simplified)
        if not distances:
            continue
        has_tail = "尾纤" in simplified
        has_tube = "套管" in simplified or "套管保护" in simplified
        has_label = _has_readable_label(simplified)
        # Avoid listing empty install points (RRU-only / no evidence).
        if not (has_tail or has_tube or has_label):
            continue
        for dist in distances:
            entry = stats.setdefault(
                dist,
                {
                    "tail": False,
                    "tube": False,
                    "label": False,
                },
            )
            entry["tail"] = entry["tail"] or has_tail
            entry["tube"] = entry["tube"] or has_tube
            entry["label"] = entry["label"] or has_label

    if not stats:
        return ""

    def _dist_key(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0

    lines = [
        f"- 站点距离={dist}: 尾纤={'有' if flags['tail'] else '无'}, 套管={'有' if flags['tube'] else '无'}, 标签文本={'有' if flags['label'] else '无'}"
        for dist, flags in sorted(stats.items(), key=lambda item: _dist_key(item[0]))
    ]
    return "安装点汇总（按站点距离合并）：\n" + "\n".join(lines)


def _render_summaries(stage_a_summaries: dict[str, str]) -> str:
    lines = [
        f"{idx}. {_sanitize_stage_a_summary_for_prompt(text)}"
        for idx, (_, text) in enumerate(_sorted_summaries(stage_a_summaries), start=1)
    ]
    return "\n".join(lines)


def _estimate_object_count(text: str) -> int:
    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified).strip()
    if simplified == "无关图片":
        return 0

    summary_obj = _parse_summary_json(text)
    if summary_obj is not None:
        return _estimate_object_count_from_summary(summary_obj)

    # Fallback: count coarse entries separated by Chinese comma.
    entries = [seg.strip() for seg in simplified.split("，") if seg.strip()]
    return len(entries) if entries else (1 if simplified else 0)


def _render_image_stats(stage_a_summaries: dict[str, str]) -> str:
    parts: list[str] = []
    for idx, (_, text) in enumerate(_sorted_summaries(stage_a_summaries), start=1):
        parts.append(f"Image{idx}(obj={_estimate_object_count(text)})")
    return "统计: " + ", ".join(parts)


def build_system_prompt(guidance: MissionGuidance, *, domain: str = "bbu") -> str:
    _ = guidance
    return build_stage_b_system_prompt(domain=domain)


def build_user_prompt(ticket: GroupTicket, guidance: MissionGuidance) -> str:
    """Build user prompt with experiences prepended."""

    if not guidance.experiences:
        raise ValueError("Experiences dict must be non-empty (no empty fallback)")

    scaffold_experiences = {
        k: v for k, v in guidance.experiences.items() if k.startswith("S")
    }
    mutable_experiences = {
        k: v for k, v in guidance.experiences.items() if k.startswith("G")
    }

    guidance_section = ""
    if scaffold_experiences:
        scaffold_block = _render_guidance_snippets(scaffold_experiences)
        guidance_section += f"结构不变量：\n{scaffold_block}\n\n"
    if mutable_experiences:
        mutable_block = _render_guidance_snippets(mutable_experiences)
        guidance_section += f"可学习规则：\n{mutable_block}\n\n"

    stage_a_summaries = ticket.summaries.as_dict()

    summaries_text = _render_summaries(stage_a_summaries)
    stats_text = _render_image_stats(stage_a_summaries)
    image_count = len(stage_a_summaries)
    stats_text = f"{stats_text}；图片数量: {image_count}"

    mission_label = guidance.experiences.get("G0") or ticket.mission

    aggregation_block = ""
    if ticket.mission == "RRU安装检查":
        aggregation_block = _aggregate_rru_install_points(stage_a_summaries)
    elif ticket.mission == "RRU位置检查":
        aggregation_block = _aggregate_rru_position_points(stage_a_summaries)
    elif ticket.mission == "RRU线缆":
        aggregation_block = _aggregate_rru_cable_points(stage_a_summaries)
    if aggregation_block:
        aggregation_block = aggregation_block + "\n\n"

    return (
        f"任务: {ticket.mission}\n"
        f"重点: {mission_label}\n"
        f"{guidance_section}"
        f"{stats_text}\n\n"
        f"{aggregation_block}"
        "图片摘要：\n"
        f"{summaries_text}"
    )


def build_messages(
    ticket: GroupTicket, guidance: MissionGuidance, *, domain: str = "bbu"
) -> list[dict[str, str]]:
    system_prompt = build_system_prompt(guidance, domain=domain)
    user_prompt = build_user_prompt(ticket, guidance)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


__all__ = ["build_messages", "build_system_prompt", "build_user_prompt"]
