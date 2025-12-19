#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mission-dependent prompt builder for Stage-A inference.

Stage-A runtime uses a richer system prompt to reduce hallucination and improve
rare/long-tail coverage, while summary-mode SFT training intentionally uses a
minimal format-only prompt to avoid injecting business priors into the model.
"""

from __future__ import annotations

from typing import Optional

# Import centralized mission definitions
from src.config.missions import (
    STAGE_A_MISSION_FOCUS as MISSION_FOCUS,
)
from src.config.missions import (
    validate_mission as _validate_mission,
)

# Import runtime summary prompt builder (not the training-minimal default)
from src.config.prompts import USER_PROMPT_SUMMARY
from src.prompts.summary_profiles import (
    DEFAULT_SUMMARY_PROFILE_RUNTIME,
    build_summary_system_prompt,
)

# Stage-A runtime system prompt (richer than training-minimal)
# Default: use runtime profile without mission (backward compat)
SUMMARY_SYSTEM_PROMPT = build_summary_system_prompt(
    DEFAULT_SUMMARY_PROFILE_RUNTIME,
    domain="bbu",
)


def build_system_prompt(
    mission: Optional[str] = None,
    dataset: Optional[str] = None,
    profile_name: str = DEFAULT_SUMMARY_PROFILE_RUNTIME,
) -> str:
    """Build mission-dependent system prompt.

    Args:
        mission: Mission name (one of SUPPORTED_MISSIONS) or None
        dataset: Dataset type ("bbu" or "rru"), defaults to "bbu"
        profile_name: Summary prompt profile name

    Returns:
        System prompt string with optional mission-specific prior rules
    """
    ds = dataset or "bbu"
    return build_summary_system_prompt(
        profile_name,
        domain=ds,
        mission=mission,
    )


# Base user prompt from training (summary-specific, no coordinates)
SUMMARY_USER_PROMPT_BASE = USER_PROMPT_SUMMARY


def build_user_prompt(mission: Optional[str] = None) -> str:
    """Build mission-dependent user prompt.

    Args:
        mission: Mission name (one of SUPPORTED_MISSIONS) or None

    Returns:
        User prompt string with optional mission focus appended
    """
    base = SUMMARY_USER_PROMPT_BASE

    # 强化图纸类图像的识别规则
    drawing_warning = (
        "\n【重要】图纸类图像识别规则："
        "如果图像是技术图纸、工程图纸、蓝图、示意图、CAD图纸、施工图纸、设计图纸等任何形式的图纸类文档，"
        "无论图纸内容是否涉及BBU相关设备，都必须严格判定为**无关图片**，"
        "不得将其误判为BBU室内场景或生成任何BBU相关的描述。"
        "图纸类图像的特征包括：线条图、符号标注、表格数据、技术规范文本等。"
        "此外：若仅能看到室内环境/机柜外观等背景，但无法确认任何任务相关目标存在，同样必须输出**无关图片**，禁止“按场景常识”补全对象。"
    )

    if mission and mission in MISSION_FOCUS:
        return f"{base}{drawing_warning}\n任务重点：{MISSION_FOCUS[mission]}"
    return f"{base}{drawing_warning}"


def validate_mission(mission: str) -> None:
    """Validate mission against supported values.

    Args:
        mission: Mission name to validate

    Raises:
        ValueError: If mission is not in SUPPORTED_MISSIONS
    """
    _validate_mission(mission)


# Export for use in inference and other modules
__all__ = [
    "build_user_prompt",
    "build_system_prompt",
    "validate_mission",
    "SUMMARY_SYSTEM_PROMPT",
    "MISSION_FOCUS",
]
