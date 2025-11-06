#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized mission definitions for BBU quality control tasks.

This module provides canonical mission names and focus descriptions used across
both Stage-A (image-level summarization) and Stage-B (group-level judgment).
"""

from __future__ import annotations

from typing import Dict, List

# Canonical mission names (from actual data directory structure)
SUPPORTED_MISSIONS: List[str] = [
    "BBU安装方式检查（正装）",
    "BBU接地线检查",
    "BBU线缆布放要求",
    "挡风板安装检查",
]

# Stage-A mission focus (appended to user prompt for context)
# These are short hints to help the model focus on relevant objects
STAGE_A_MISSION_FOCUS: Dict[str, str] = {
    "BBU安装方式检查（正装）": "关注BBU设备、安装螺丝及其符合性",
    "BBU接地线检查": "关注机柜处/地排处接地螺丝、电线捆扎情况",
    "BBU线缆布放要求": "关注BBU端/ODF端光纤插头、光纤保护措施和弯曲半径",
    "挡风板安装检查": "关注BBU设备及挡风板需求与配置符合性",
}

# Stage-B mission focus (检查要点 for group-level judgment)
# These are detailed requirements used in Stage-B system prompt
STAGE_B_MISSION_FOCUS: Dict[str, str] = {
    "BBU安装方式检查（正装）": "至少需要检测到BBU设备，BBU安装螺丝且需要符合要求",
    "BBU接地线检查": "至少需要检测到机柜处接地螺丝和地排处接地螺丝且需要符合要求，还需要检测到电线且需要捆扎整齐",
    "BBU线缆布放要求": "至少需要检测到BBU端光纤插头和ODF端光纤插头且需要符合要求，还需要检测到光纤且有保护措施和弯曲半径合理",
    "挡风板安装检查": "至少需要检测到BBU设备并根据情况判断是否需要安装挡风板。若需要安装，则判断是否符合要求",
}


def validate_mission(mission: str) -> None:
    """Validate mission against supported values.

    Args:
        mission: Mission name to validate

    Raises:
        ValueError: If mission is not in SUPPORTED_MISSIONS
    """
    if mission not in SUPPORTED_MISSIONS:
        raise ValueError(
            f"Unsupported mission: {mission}. "
            f"Must be one of: {', '.join(SUPPORTED_MISSIONS)}"
        )


__all__ = [
    "SUPPORTED_MISSIONS",
    "STAGE_A_MISSION_FOCUS",
    "STAGE_B_MISSION_FOCUS",
    "validate_mission",
]
