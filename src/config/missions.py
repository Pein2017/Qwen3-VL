#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized mission definitions for BBU quality control tasks.

This module provides canonical mission names and focus descriptions used across
both Stage-A (image-level summarization) and Stage-B (group-level judgment).
"""

from __future__ import annotations

# Canonical mission names (from actual data directory structure)
SUPPORTED_MISSIONS: list[str] = [
    "BBU安装方式检查（正装）",
    "BBU接地线检查",
    "BBU线缆布放要求",
    "挡风板安装检查",
    "RRU安装检查",
    "RRU位置检查",
    "RRU线缆",
]

# Stage-B mission focus (检查要点 for group-level judgment)
# These are detailed requirements used in Stage-B system prompt
STAGE_B_MISSION_FOCUS: dict[str, str] = {
    "BBU安装方式检查（正装）": "至少需要检测到BBU设备，BBU安装螺丝且需要符合",
    "BBU接地线检查": "至少需要检测到机柜处接地螺丝和地排处接地螺丝且需要符合，还需要检测到电线且需要整齐",
    "BBU线缆布放要求": "至少需要检测到BBU端光纤插头和ODF端光纤插头且需要符合，还需要检测到光纤且有保护和半径合理",
    "挡风板安装检查": "至少需要检测到BBU设备并根据情况判断是否需要安装挡风板。若需要安装，则判断是否符合",
    "RRU安装检查": "每个安装点需同时覆盖RRU设备与固定件（合格）；任一缺失或紧固负项则不通过",
    "RRU位置检查": "每个安装点需覆盖固定件合格、接地线存在且标签清晰可识别；任一缺失则不通过",
    "RRU线缆": "必须检测到尾纤且有套管；出现无套管等负项则不通过",
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
            f"Unsupported mission: {mission}. Must be one of: {', '.join(SUPPORTED_MISSIONS)}"
        )


__all__ = [
    "SUPPORTED_MISSIONS",
    "STAGE_B_MISSION_FOCUS",
    "validate_mission",
]
