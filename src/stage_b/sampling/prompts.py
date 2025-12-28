#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the Stage-B rule-search pipeline."""

from __future__ import annotations

import json
import re

from src.prompts.stage_b_verdict import build_stage_b_system_prompt

from ..types import GroupTicket, MissionGuidance
from ..utils.chinese import normalize_spaces, to_simplified

_INDEX_RE = re.compile(r"(\d+)$")
_NEED_REVIEW_MARKER_RE = re.compile(r"需复核\s*[，,]?\s*备注[:：]")
_STATION_DISTANCE_RE = re.compile(r"站点距离[=/](\d+)")


def _parse_summary_json(text: str) -> dict[str, object] | None:
    if not text:
        return None
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        obj = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    required = {"dataset", "统计", "objects_total"}
    if not required.issubset(obj.keys()):
        return None
    return obj


def _format_summary_json(obj: dict[str, object]) -> str:
    preferred_order = ["objects_total", "统计", "备注", "分组统计"]
    ordered: dict[str, object] = {}
    for key in preferred_order:
        if key in obj:
            ordered[key] = obj[key]
    for key, value in obj.items():
        if key == "format_version" or key == "dataset":
            continue
        if key not in ordered:
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def _summary_entries(obj: dict[str, object]) -> list[dict[str, object]]:
    entries = obj.get("统计")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def _entry_by_category(
    entries: list[dict[str, object]], category: str
) -> dict[str, object] | None:
    for entry in entries:
        if entry.get("类别") == category:
            return entry
    return None


def _summary_distances(obj: dict[str, object]) -> list[str]:
    entry = _entry_by_category(_summary_entries(obj), "站点距离")
    if not entry:
        return []
    distances = entry.get("站点距离")
    if not isinstance(distances, dict):
        distances = entry.get("距离")
    if isinstance(distances, dict):
        return [str(k) for k in distances.keys() if str(k).strip().isdigit()]
    return []


def _summary_has_label_text(obj: dict[str, object]) -> bool:
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


def _sanitize_stage_a_summary_for_prompt(text: str) -> str:
    """Sanitize Stage-A summary strings for Stage-B prompting.

    Stage-B inference output forbids third-state wording (e.g., "需复核").
    However, Stage-A summaries may include structured markers like "需复核,备注:".
    We remove the marker while preserving the remark content so the model can
    still use the evidence without echoing forbidden tokens.
    """

    summary_obj = _parse_summary_json(text)
    if summary_obj is not None:
        return _format_summary_json(summary_obj)

    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified)
    # Preserve the "need review" signal in a safe, non-forbidden phrasing for Stage-B.
    simplified = _NEED_REVIEW_MARKER_RE.sub("备注(待确认):", simplified)
    simplified = simplified.replace("需复核", "")
    simplified = normalize_spaces(simplified).strip()
    return simplified


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
            categories = {entry.get("类别") for entry in entries}
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
            categories = {entry.get("类别") for entry in entries}
            has_rru = "RRU设备" in categories
            has_ground = "接地线" in categories
            has_label = _summary_has_label_text(summary_obj)
            has_ground_label = has_ground and has_label
            if not (has_rru or has_ground_label):
                continue
            for dist in distances:
                entry = stats.setdefault(dist, {"rru": False, "ground_label": False})
                entry["rru"] = entry["rru"] or has_rru
                entry["ground_label"] = entry["ground_label"] or has_ground_label
            continue

        simplified = _sanitize_stage_a_summary_for_prompt(text)
        distances = _STATION_DISTANCE_RE.findall(simplified)
        if not distances:
            continue
        has_rru = bool(re.search(r"\bRRU\b", simplified) or "RRU设备" in simplified)
        has_ground = "接地线" in simplified
        has_label = _has_readable_label(simplified)
        # Only surface evidence-bearing points to reduce confusion:
        # - RRU evidence: has_rru
        # - Ground evidence: has_ground + readable label
        has_ground_label = has_ground and has_label
        if not (has_rru or has_ground_label):
            continue
        for dist in distances:
            entry = stats.setdefault(dist, {"rru": False, "ground_label": False})
            entry["rru"] = entry["rru"] or has_rru
            entry["ground_label"] = entry["ground_label"] or has_ground_label

    if not stats:
        return ""

    def _dist_key(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0

    lines = [
        f"- 站点距离={dist}: RRU={'有' if flags['rru'] else '无'}, 接地线(可识别标签)={'有' if flags['ground_label'] else '无'}"
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
            if not (has_tail or has_tube):
                continue
            for dist in distances:
                entry = stats.setdefault(dist, {"tail": False, "tube": False})
                entry["tail"] = entry["tail"] or has_tail
                entry["tube"] = entry["tube"] or has_tube
            continue

        simplified = _sanitize_stage_a_summary_for_prompt(text)
        distances = _STATION_DISTANCE_RE.findall(simplified)
        if not distances:
            continue
        has_tail = "尾纤" in simplified
        has_tube = "套管" in simplified or "套管保护" in simplified
        # Avoid listing empty install points (RRU-only / no evidence).
        if not (has_tail or has_tube):
            continue
        for dist in distances:
            entry = stats.setdefault(dist, {"tail": False, "tube": False})
            entry["tail"] = entry["tail"] or has_tail
            entry["tube"] = entry["tube"] or has_tube

    if not stats:
        return ""

    def _dist_key(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0

    lines = [
        f"- 站点距离={dist}: 尾纤={'有' if flags['tail'] else '无'}, 套管={'有' if flags['tube'] else '无'}"
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
    summary_obj = _parse_summary_json(text)
    if summary_obj is not None:
        count = summary_obj.get("objects_total")
        if isinstance(count, int):
            return count
        if isinstance(count, str) and count.isdigit():
            return int(count)

    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified)
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
