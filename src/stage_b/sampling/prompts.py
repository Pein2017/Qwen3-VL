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
根据多张图片的文字摘要，对当前质检任务做出整体判定。最终只输出两行：
第1行：Verdict: 通过   或   Verdict: 不通过
第2行：Reason: ...

【任务要点（仅以当前 mission 的 G0 为准）】
- 关键检查项只来自下方注入的 G0；只围绕这些要点给结论。
- 同一组工单/图片可能被不同 mission 审核；不同 mission 允许不同结论。与本 mission 的 G0 无关的内容不得影响本次判定。
"""

_FAIL_FIRST_SYSTEM_PROMPT = """【硬护栏：明确负项 fail-first（mission-scoped）】
- 若任一图片摘要中出现与当前 G0 相关的明确负项，整组必须判不通过（fail-first），且 Reason 需指出对应负项证据。
- 通用负项触发词（仅动词/形容词/短语，不含名词）：
  未按要求、错误、缺失、松动、损坏、方向不正确、反向、不符合要求、不合格、不合理、未安装、未配备
- Pattern-first：任何形如 `不符合要求/<issue>` 的描述均视为明确负项（无需穷举 <issue>），但仍仅对与当前 G0 相关的要点执行 fail-first。
"""

_SOFT_SIGNALS_SYSTEM_PROMPT = """【软信号：备注 + 待确认信号】
- Stage-A 摘要可能包含 `备注:`：备注是补充信息，判定以 `备注:` 的具体内容与多图证据为准。
- “无法确认/无法判断/只显示部分/模糊”等属于待确认信号：本身不是通用硬触发词，但也不能被忽略。
  - 视角/覆盖提示：若某张图摘要主要是螺丝/端口等局部细节，且 BBU/挡风板仅“只显示部分/无法判断”，可视为特写图；特写图用于补充细节，不应因为其无法识别品牌/空间而否决已由其它全局图明确确认的关键点（除非特写图出现明确负项）。
  - 若待确认信号涉及 G0 关键要点：你必须在其它图片中找到“明确确认”的证据，并在 Reason 中点名对应图片；找不到则判不通过。
  - 若同一句/同一条摘要同时出现待确认信号（如“只显示部分/模糊”）与肯定词（如“符合要求/安装方向正确/按要求配备”），优先视为“无法确认”，除非其它图片给出显示完整的确认。
  - 若多图对同一关键点给出矛盾描述（如既“无需安装”又“空间充足需要安装”），先尝试用“显示完整/覆盖关键点的全局图”消解矛盾；若无法消解（不存在任何一张图能明确确认关键点），判不通过。
- 例外：若补充提示（G1、G2...）明确规定某类待确认信号需要判不通过，则以补充提示为准。
- 通用安全约束：若你无法给出支持通过的依据（覆盖 G0 关键点），必须判不通过。
"""

_OUTPUT_SYSTEM_PROMPT = """【输出格式（必须严格遵守）】
- 严格两行输出，不得有多余空行/前后空格/JSON/Markdown。
- 第1行：`Verdict: 通过` 或 `Verdict: 不通过`（两种之一，禁止第三状态）。
- 第2行：`Reason: ...`（单行中文）。
- Reason 建议用分号分隔：`Image1: ...; Image2: ...; ...; 总结: ...`，整行不超过约 240 字，只写与 G0 相关的要点证据。
- 禁止无根据概括：只有当每张图片摘要都明确包含同一要点时，才可写“所有图像均……”；否则需点名哪几张图确认、哪几张仅为局部/无关/未覆盖要点。
- 最终输出中严禁出现任何第三状态词面（包括但不限于：需复核、需人工复核、need-review、证据不足、待定、通过但需复核、通过但需人工复核）。
"""

_INDEX_RE = re.compile(r"(\d+)$")
_NEED_REVIEW_MARKER_RE = re.compile(r"需复核[，,]\\s*备注[:：]")


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
