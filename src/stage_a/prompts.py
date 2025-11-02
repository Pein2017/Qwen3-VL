#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mission-dependent prompt builder for Stage-A inference.

Directly imports SYSTEM_PROMPT_SUMMARY from training to ensure alignment.
Adds mission-specific focus context to user prompt.
"""

from __future__ import annotations

from typing import Optional

# ✅ IMPORT DIRECTLY FROM TRAINING to ensure alignment
from src.config.prompts import SYSTEM_PROMPT_SUMMARY, USER_PROMPT_SUMMARY

# Import centralized mission definitions
from src.config.missions import (
    STAGE_A_MISSION_FOCUS as MISSION_FOCUS,
    validate_mission as _validate_mission,
)

# Use training's SYSTEM_PROMPT_SUMMARY directly
SUMMARY_SYSTEM_PROMPT = SYSTEM_PROMPT_SUMMARY

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

    if mission and mission in MISSION_FOCUS:
        return f"{base}\n任务重点：{MISSION_FOCUS[mission]}"
    return base


def validate_mission(mission: str) -> None:
    """Validate mission against supported values.

    Args:
        mission: Mission name to validate

    Raises:
        ValueError: If mission is not in SUPPORTED_MISSIONS
    """
    _validate_mission(mission)
