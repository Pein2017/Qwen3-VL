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

    # 强化图纸类图像的识别规则 + BBU 场景强约束
    drawing_warning = (
        "\n【重要】图纸类图像识别规则（强制，无例外）："
        "如果图像是技术图纸、工程图纸、蓝图、示意图、CAD图纸、施工图纸、设计图纸、设备平面布局图等任何形式的图纸类文档，"
        "无论其中是否出现“BBU/机柜/挡风板”等文字或草图，都必须直接输出**无关图片**，"
        "不得解析图纸内容、不得从图纸推断真实现场、不得生成任何BBU相关描述。"
        "图纸类图像特征包括但不限于：线条/符号/标注为主、尺寸线、坐标网格、表格、剖视/三视图、图框标题栏、技术规范文本等。"
        "若图像疑似图纸（无法确认是否为真实现场照片），也必须按**无关图片**处理。"
        "\n【BBU场景限定（强规则）】："
        "仅当图像为**真实现场照片**且能直接观察到BBU相关目标（如BBU设备、挡风板、螺丝、光纤插头、光纤、电线、标签等）时，才输出摘要；"
        "除BBU现场照片以外的任何内容（文档、截图、机房示意、其他设备/场景、仅有背景无法确认目标）一律输出**无关图片**。"
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
