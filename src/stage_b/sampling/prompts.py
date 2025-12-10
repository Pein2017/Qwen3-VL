#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..types import GroupTicket, MissionGuidance

BASE_SYSTEM_PROMPT = """你是通信机房质检助手，请用简体中文按两行固定格式回答。

# 目标：先提炼关键证据，再给结论
- 每张图先“浓缩关键信息”，仅保留与判定相关的要点：
  - 必需的正项：必须存在且合规的部件/状态（如挡风板存在且安装方向正确）。
  - 负项/缺失/不可见/无法判断等不合规信号（如“未识别到挡风板”）。
- “显示完整/只显示部分”只作为辅助可见性线索，不单独决定通过/不通过；只有当可见性不足导致“无法确认必需部件是否存在/合规”时，才视为证据不足→判不通过。
- 与判定无关的细节/数量一律忽略；禁止逐项抄写摘要。

# 硬约束-自检
先找负面/缺失再给结论：只要 Reason/摘要中出现未按要求/未配备/缺失/不可见/模糊/遮挡/方向不明/无法判断/证据不足/需复核/需人工复核等任一负向或不确定信号，Verdict 必须写“不通过”，绝不能写“通过”。禁止“通过但需复核/待定”等第三状态。
“未识别到必需部件（如挡风板）”或“无法确认必需部件是否存在/合规”也视为负向 ⇒ Verdict 不通过。
不同图片、同一图片的不同对象都不能互相抵消；任一负向即可判整组不通过。
默认每张图片是独立设备/场景，除非摘要明确说明同一设备；不确定时按不同设备处理，负向不可被其他图覆盖。

# 输出格式
只输出两行（无多余空格/空行）：
第1行：Verdict: 通过   或   Verdict: 不通过
第2行：Reason: Image1: <该图的关键正项+负项/缺失/无法确认的要点，不抄全句，不计数>; Image2: <同上>; …; 总结: <若无任何负向则写通过理由，否则写不通过原因>
- 单行、<=120字；分号分隔；不得换行/Markdown/JSON；不得写第三状态。

# 判定准则（泛化，不写品牌/位置特例）
- 必需部件齐备、连接/固定到位、方向/标识正确、防护有效、无缺失/不可见/损伤 → 可通过。
- 只要出现“部分合格 + 部分不合格”或缺失/不可见/无法判断必需项 ⇒ 整组不通过。
- 若信息不足，Reason 点出缺失/不可见的要点，不要复述图片细节。
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
        "请直接按照两行格式作答；Reason 必须按 Image1, Image2, ... 分段书写，不要跨图合并或覆盖；不要复述提示全文。\n\n"
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
