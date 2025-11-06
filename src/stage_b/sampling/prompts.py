#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..types import GroupTicket, MissionGuidance

BASE_SYSTEM_PROMPT = """你是通信机房质检助手，需要根据多张图片的描述给出工单级结论。

输出必须严格遵循三行格式：
第1行：Verdict: 通过 或 Verdict: 不通过
第2行：Reason: <简洁描述支撑结论的证据，引用摘要要点即可>
第3行：Confidence: <0到1之间的小数，最多保留两位，表示你对结论的自信度>

请专注于任务要点；若摘要之间互相矛盾或信息不足，请倾向“不通过”，并明确说明原因。
当遇到不确定或噪声摘要时，要反思是否需要在指导规则中补充“关键检查点”或“排除噪声”的提醒。"""

_INDEX_RE = re.compile(r"(\d+)$")


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
        f"{idx}. {text}"
        for idx, (_, text) in enumerate(_sorted_summaries(stage_a_summaries), start=1)
    ]
    return "\n".join(lines)


def build_system_prompt(guidance: MissionGuidance) -> str:
    clauses: List[str] = [BASE_SYSTEM_PROMPT]
    if guidance.focus:
        clauses.append(f"任务要点：{guidance.focus}")
    if guidance.experiences:
        clauses.append("若已有提示仍无法判断，请记录需要补充的规则，便于后续指导更新。")
    return "\n\n".join(clauses)


def build_user_prompt(ticket: GroupTicket, guidance: MissionGuidance) -> str:
    """Build user prompt with experiences prepended."""

    if not guidance.experiences:
        raise ValueError("Experiences dict must be non-empty (no empty fallback)")
    guidance_block = _render_guidance_snippets(guidance.experiences)
    guidance_section = f"补充提示：\n{guidance_block}\n\n"

    summaries_text = _render_summaries(ticket.summaries.as_dict())
    mission_label = guidance.focus or ticket.mission

    return (
        f"任务: {ticket.mission}\n"
        f"重点: {mission_label}\n"
        f"{guidance_section}"
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

