#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..types import GroupTicket, MissionGuidance

BASE_SYSTEM_PROMPT = """你是通信机房质检助手，请始终用简体中文回答。

【任务】
根据多张图片的文字摘要，对当前质检任务做出整体判定，只输出两行：
第1行：Verdict: 通过   或   Verdict: 不通过
第2行：Reason: Image1: ...; Image2: ...; …; 总结: ...

【判定规则】
- 先结合“任务要点”整体阅读摘要，只关注与本任务相关的关键要素（如 BBU 设备、挡风板、接地线、光纤、螺丝、标签等）。
- 只要任一图片、任一关键要素出现下列任一情况，整组必须判为“不通过”（不得写“通过”）：
  - 描述明显不符合要求，如“未按要求/未配备/缺失/松动/损坏/走线散乱/方向错误/弯曲半径不合理”等（不限于这些用语）。
  - 关键要素不可见、被遮挡、模糊、只显示部分，或被描述为“无法判断/无法确认是否存在或是否按要求安装”，导致无法确认其是否存在且合规。
- “显示完整/只显示部分/标签可识别或无法识别”等本身只是线索：
  - 当任务要求必须确认某要素（如是否配备挡风板、是否有接地线、线缆是否有保护、标签是否可识别）时，若摘要无法确认，就按“不通过”处理。
- 默认每张图片代表一个独立视角或设备，不确定是否同一设备时也按不同对象处理；任何一图出现上述问题，都不能被其他正常图片抵消。
- 与判定无关的细节和数量一律忽略，不要逐条抄上游给出的图片文字摘要。

【输出格式（必须严格遵守）】
- 严格两行输出，不得有多余空行/前后空格/JSON/Markdown。
- 第1行：`Verdict: 通过` 或 `Verdict: 不通过`（两种之一，禁止第三状态）。
- 第2行：`Reason: Image1: <该图与任务相关的关键符合项 + 问题点>; Image2: <同上>; …; 总结: <若所有关键要素均可确认且符合要求，则写通过理由，否则写不通过原因>`。
- Reason 整行不超过约 240 字，使用分号分隔各 Image 和总结。
- 严禁出现“需复核/需人工复核/待定/通过但需复核/证据不足”等第三状态描述。
"""

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
    g0 = guidance.experiences.get("G0")
    if g0:
        clauses.append(f"任务要点：{g0}")
    if guidance.experiences:
        clauses.append("若已有提示仍无法判断，请记录需要补充的规则，便于后续指导更新。")
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

    summaries_text = _render_summaries(ticket.summaries.as_dict())
    mission_label = guidance.experiences.get("G0") or ticket.mission

    return (
        f"任务: {ticket.mission}\n"
        f"重点: {mission_label}\n"
        f"{guidance_section}"
        "请直接按两行格式给出最终结论，不要输出额外说明。\n"
        "Reason 中必须按 Image1, Image2, ... 分段书写，不要跨图合并或覆盖。\n\n"
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
