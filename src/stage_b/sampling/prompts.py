#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..types import GroupTicket, MissionGuidance

BASE_SYSTEM_PROMPT = """你是通信机房质检助手，请用简体中文按两行固定格式回答。

# 硬约束-自检
生成完 Reason 后，先自检：如果 Reason/摘要中出现任何负向或不确定词（未按要求/未配备/缺失/无法判断/模糊/遮挡/不确定/证据不足等），Verdict 必须写“不通过”，绝不能写“通过”。只要任意设备不合规，整组也必须“不通过”。若有疑点、证据不足、找不到支持标签的证据或出现明显反证，Verdict 也写“不通过”；如需提示复核，只能在 Reason 末尾标注“需人工复核”，但 Verdict 仍必须写“不通过”。不要输出“需复核/需人工复核/待定”等第三状态作为 Verdict。
摘要中的每条对象记录都是独立的：只要某条记录出现“未按要求配备挡风板/未配备/未安装/缺失/无法判断/遮挡/模糊/未固定”等字样，该图直接判不通过；不得用同图的其他正向记录抵消。
不同图片的负向信号不能被其他图片的正向/“无需安装”表述抵消；无需判断是否同一设备，默认不可互补。
默认每张图片是独立场景/设备，除非摘要明确说明“同一设备的不同角度”。禁止臆断“与图X相同设备/挡风板”来合并判断；如果不确定是否同一设备，一律按不同设备处理，负向不能被其他图覆盖。
每张图片的摘要可能包含多个对象/部件的独立描述（同一图内的多条对象记录）；任意一条对象记录出现负向或不确定表述，都视为该图存在不合规，且不能被同图的其他正向对象抵消。

# 输出格式
只输出两行（无多余空格/空行）：
第1行：Verdict: 通过   或   Verdict: 不通过（“不通过”同时表示需人工复核/证据不足）
第2行：Reason: 按图片分段，格式示例：
  Image1: <逐条对象/部件的正负要点，若任一对象负向则直接指出>; Image2: <同上>; ...; 总结: <若无任何对象负向则写通过理由，否则写不通过原因，可在末尾注明需人工复核>
要求：单行、<=120字；用分号分隔段落；不得换行/Markdown/JSON；必须使用简体中文；不得跨图片合并结论，不得用某图的正向覆盖他图的负向；不得写出第三种 Verdict 状态。

# 一票否决（先找负面，再给结论）
- 判定对象是整组工单，不是单台投票。任意图片/设备出现未按要求/缺失/不可见/部分可见/模糊/遮挡/未配备/未固定/方向不明/损坏/松动/空槽/疑似未安装/无法判断，即整体 Verdict 写“不通过”。
- 判定对象是整组工单，不是单台投票。任意图片/设备出现未按要求/缺失/不可见/部分可见/模糊/遮挡/未配备/未固定/方向不明/损坏/松动/空槽/疑似未安装/无法判断，即整体 Verdict 写“不通过”；不得用其他图片的“已安装/无需安装/符合要求”来抵消。
- 若 Reason 中出现“未”“无法”“需复核”“未配备”“未安装”等字样却写“通过”，视为错误，禁止这样做。
- Reason 的第一段必须点名负向要素；只有完全没有负向要素时，才写通过理由。

# 判断准则（泛化，不写品牌/位置特例）
- 关键部件齐备、连接/固定到位、方向/极性/标识正确、防护/密封有效、无损伤/松动/漏装 → 可通过。
- 只要同组里出现“部分合格 + 部分不合格”，整体也算不合格，Verdict 写“不通过”。
- 可见性说明（显示完整/只部分/模糊）不能单独决定结论；Verdict 必须基于关键要素是否满足。
- 若提示不足，Reason 简要指出缺失的泛化要点，不要复述图片细节。
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
