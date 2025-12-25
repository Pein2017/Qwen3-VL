#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A summary prompt composition (system + user)."""

from __future__ import annotations

from typing import Optional

from src.prompts.summary_core import (
    MISSION_SPECIFIC_PRIOR_RULES,
    USER_PROMPT_SUMMARY,
    build_summary_system_prompt_minimal,
)
from src.prompts.domain_packs import get_domain_pack
from src.prompts.summary_profiles import (
    DEFAULT_SUMMARY_PROFILE_RUNTIME,
    get_summary_profile,
)

_SUMMARY_GLOBAL_NON_SITE_RULES = (
    "【重要】图纸/文档/截图等非现场图片一律输出“无关图片”。"
    "即使出现BBU/RRU等文字或示意，也不得据此生成摘要。"
)

_BBU_SCENARIO_RULES = (
    "【BBU场景限定（强规则）】："
    "仅真实现场且可见BBU相关目标时输出摘要，否则输出“无关图片”。"
)

_RRU_SCENARIO_RULES = (
    "【RRU场景限定（强规则）】："
    "仅真实现场且可见RRU相关目标时输出摘要，否则输出“无关图片”。"
    "\n【RRU重点（强规则）】："
    "仅需强调物体之间的配对关系与站点距离；具体检查项以 guidance 为准，不在摘要中重复。"
    "\n【站点距离（强规则）】："
    "每张图片凡非“无关图片”，必须输出且只输出1个站点距离（左上角水印），格式：站点距离/<数字或无法识别>×1；建议作为第一项，不得遗漏。"
    "\n【RRU禁用项（强规则）】："
    "禁止BBU相关内容与品牌/机柜概念（如机柜/机房/挡风板/BBU端/ODF端）。"
)


def _render_mission_rules(mission: Optional[str]) -> str:
    if not mission:
        return ""
    mission_rules = MISSION_SPECIFIC_PRIOR_RULES.get(mission)
    if not mission_rules:
        return ""
    rules = (
        "".join(mission_rules)
        if isinstance(mission_rules, (tuple, list))
        else str(mission_rules)
    )
    return f"【{mission}任务特定规则】\n{rules}" if rules else ""


def _render_scenario_rules(domain: str) -> str:
    rules = ""
    if domain == "bbu":
        rules = _BBU_SCENARIO_RULES
    elif domain == "rru":
        rules = _RRU_SCENARIO_RULES
    return rules


def build_stage_a_system_prompt(
    *,
    domain: str = "bbu",
    mission: Optional[str] = None,
    profile_name: str = DEFAULT_SUMMARY_PROFILE_RUNTIME,
    summary_label_grouping: Optional[bool] = None,
) -> str:
    """Compose Stage-A system prompt = summary task base + scenario block."""

    profile = get_summary_profile(profile_name)
    parts = [
        build_summary_system_prompt_minimal(
            summary_label_grouping=summary_label_grouping
        ).strip()
    ]

    parts.append(_SUMMARY_GLOBAL_NON_SITE_RULES)

    if profile.include_mission_rules:
        mission_rules = _render_mission_rules(mission)
        if mission_rules:
            parts.append(mission_rules)

    return "\n\n".join(part for part in parts if part)


def build_stage_a_user_prompt(
    base_prompt: str,
    *,
    domain: str = "bbu",
    include_domain_pack: bool = True,
) -> str:
    """Compose Stage-A user prompt = base summary instruction + domain hints."""

    parts = [base_prompt]
    if include_domain_pack:
        pack = get_domain_pack(domain)
        parts.append(f"【领域提示（只读，不参与经验更新）】\n{pack.block.strip()}")
        scenario_rules = _render_scenario_rules(domain)
        if scenario_rules:
            parts.append(scenario_rules)
    return "\n".join(part for part in parts if part)


__all__ = ["build_stage_a_system_prompt", "build_stage_a_user_prompt"]
