#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..types import GroupTicket, MissionGuidance
from ..utils.chinese import normalize_spaces, to_simplified

_INTRO_SYSTEM_PROMPT = """你是通信机房质检助手，请始终用简体中文回答。

【任务】
根据多张图片的文字摘要，对当前任务做出组级判定。最终只输出两行：
Verdict: 通过 / 不通过
Reason: ...

【任务要点（G0 是检查清单；补充提示必须遵守）】
- “检查什么”以 G0 为准；只围绕 G0 的检查要点给结论。
- “如何判/证据覆盖/例外边界”以补充提示（G1、G2...）为准，必须遵守；若与通用软信号规则冲突，以补充提示为准。
- 同一组工单/图片可能被不同 mission 审核；不同 mission 允许不同结论。与本 mission 的 G0 无关的内容不得影响本次判定。
"""

_FAIL_FIRST_SYSTEM_PROMPT = """【硬信号：任务相关的明确负项】
- 通用负项触发词（仅动词/形容词/短语，不含名词）：
  未按要求、错误、缺失、松动、损坏、方向不正确、反向、不符合要求、不合格、不合理、未安装、未配备
- Pattern-first：任何形如 `不符合要求/<issue>` 的描述均视为明确负项（无需穷举 <issue>），但仍仅对与当前 G0 相关的要点执行 fail-first。
"""

_SOFT_SIGNALS_SYSTEM_PROMPT = """【软信号：备注 + 待确认信号】
- 多图摘要可能包含 `备注:`：备注是补充信息，判定以 `备注:` 的具体内容与多图证据为准。
- "无法确认/无法判断/只显示部分/模糊"等属于待确认信号：本身不是明确负项，但也不能忽略。
- 输入中包含 `ImageN(obj=...)`（从摘要中 `×N` 求和得到），用于了解图片复杂度。
- 若 G0 关键要点在所有图片中都无法"明确确认"，判不通过；若同一要点多图矛盾，优先用显示完整的证据消解，无法消解则判不通过。
- 例外：若补充提示（G1、G2...）明确规定某类待确认信号或图片覆盖/视角要求需要判不通过，则以补充提示为准。
- 通用安全约束：若无法给出支持通过的依据（覆盖 G0 关键点），必须判不通过。
"""

_OUTPUT_SYSTEM_PROMPT = """【输出格式（必须严格遵守）】
- 严格两行输出；不得有多余空行/前后空格/第三行/JSON/Markdown。
- 第1行：`Verdict: 通过` 或 `Verdict: 不通过`（两种之一）。
- 第2行：`Reason: ...`（单行中文，建议 `Image1: ...; Image2: ...; ...; 总结: ...`，<=240字）。
- 严禁出现任何第三状态词面（如：需复核、need-review、证据不足、待定、通过但需复核等）。
"""

_INDEX_RE = re.compile(r"(\d+)$")
_NEED_REVIEW_MARKER_RE = re.compile(r"需复核[，,]\\s*备注[:：]")
_COUNT_RE = re.compile(r"×(\\d+)")


def _sanitize_stage_a_summary_for_prompt(text: str) -> str:
    """Sanitize Stage-A summary strings for Stage-B prompting.

    Stage-B inference output forbids third-state wording (e.g., "需复核").
    However, Stage-A summaries may include structured markers like "需复核,备注:".
    We remove the marker while preserving the remark content so the model can
    still use the evidence without echoing forbidden tokens.
    """

    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified)
    simplified = _NEED_REVIEW_MARKER_RE.sub("备注:", simplified)
    simplified = simplified.replace("需复核", "")
    simplified = normalize_spaces(simplified).strip()
    return simplified


def _sorted_summaries(per_image: Dict[str, str]) -> List[Tuple[str, str]]:
    def _index(key: str) -> int:
        match = _INDEX_RE.search(key)
        if match:
            try:
                return int(match.group(1))
            except ValueError:  # pragma: no cover - defensive
                return 0
        return 0

    return sorted(per_image.items(), key=lambda item: _index(item[0]))


def _render_guidance_snippets(experiences: Dict[str, str]) -> str:
    """Format experiences dict as numbered experiences text block."""

    if not experiences:
        raise ValueError("Experiences dict must be non-empty")
    formatted: List[str] = [
        f"[{key}]. {value}" for key, value in sorted(experiences.items())
    ]
    return "\n".join(formatted)


def _render_summaries(stage_a_summaries: Dict[str, str]) -> str:
    lines = [
        f"{idx}. {_sanitize_stage_a_summary_for_prompt(text)}"
        for idx, (_, text) in enumerate(_sorted_summaries(stage_a_summaries), start=1)
    ]
    return "\n".join(lines)


def _estimate_object_count(text: str) -> int:
    simplified = to_simplified(text or "")
    simplified = normalize_spaces(simplified)
    matches = _COUNT_RE.findall(simplified)
    if matches:
        total = 0
        for match in matches:
            try:
                total += int(match)
            except ValueError:  # pragma: no cover - defensive
                continue
        return total
    # Fallback: count coarse entries separated by Chinese comma.
    entries = [seg.strip() for seg in simplified.split("，") if seg.strip()]
    return len(entries) if entries else (1 if simplified else 0)


def _render_image_stats(stage_a_summaries: Dict[str, str]) -> str:
    parts: List[str] = []
    for idx, (_, text) in enumerate(_sorted_summaries(stage_a_summaries), start=1):
        parts.append(f"Image{idx}(obj={_estimate_object_count(text)})")
    return "统计: " + ", ".join(parts)


def build_system_prompt(guidance: MissionGuidance) -> str:
    clauses: List[str] = [_INTRO_SYSTEM_PROMPT]
    g0 = guidance.experiences.get("G0")
    if g0:
        clauses.append(f"【G0】\n{g0}")
    clauses.append(_FAIL_FIRST_SYSTEM_PROMPT)
    clauses.append(_SOFT_SIGNALS_SYSTEM_PROMPT)
    clauses.append(_OUTPUT_SYSTEM_PROMPT)
    return "\n\n".join(clauses)


def build_user_prompt(ticket: GroupTicket, guidance: MissionGuidance) -> str:
    """Build user prompt with experiences prepended."""

    if not guidance.experiences:
        raise ValueError("Experiences dict must be non-empty (no empty fallback)")
    # Do not repeat G0 (mission headline) inside the guidance block
    filtered_experiences = {k: v for k, v in guidance.experiences.items() if k != "G0"}
    guidance_section = ""
    if filtered_experiences:
        guidance_block = _render_guidance_snippets(filtered_experiences)
        guidance_section = f"补充提示：\n{guidance_block}\n\n"

    stage_a_summaries = ticket.summaries.as_dict()
    summaries_text = _render_summaries(stage_a_summaries)
    stats_text = _render_image_stats(stage_a_summaries)
    image_count = len(stage_a_summaries)
    stats_text = f"{stats_text}；图片数量: {image_count}"

    # Check if this mission requires global/local image distinction
    g1_text = guidance.experiences.get("G1", "")
    needs_global_local = "全局图" in g1_text and "局部图" in g1_text

    # If needed, identify and annotate the global image
    global_image_hint = ""
    if needs_global_local and stage_a_summaries:
        sorted_items = _sorted_summaries(stage_a_summaries)
        obj_counts = [
            (idx, _estimate_object_count(text))
            for idx, (_, text) in enumerate(sorted_items, start=1)
        ]
        if obj_counts:
            # Find the image with maximum obj count
            max_idx, max_obj = max(obj_counts, key=lambda x: x[1])
            global_image_hint = (
                f"\n提示: Image{max_idx}(obj={max_obj}) 是全局图，其余图片都是局部图。"
            )

    mission_label = guidance.experiences.get("G0") or ticket.mission

    return (
        f"任务: {ticket.mission}\n"
        f"重点: {mission_label}\n"
        f"{guidance_section}"
        f"{stats_text}{global_image_hint}\n"
        "按两行协议输出；Reason 用 Image1/Image2... 逐图写证据，最后总结。\n\n"
        "图片摘要：\n"
        f"{summaries_text}"
    )


def build_messages(
    ticket: GroupTicket, guidance: MissionGuidance
) -> List[Dict[str, str]]:
    system_prompt = build_system_prompt(guidance)
    user_prompt = build_user_prompt(ticket, guidance)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


__all__ = ["build_messages", "build_system_prompt", "build_user_prompt"]
