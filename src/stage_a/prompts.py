#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mission-dependent prompt builder for Stage-A inference.

Stage-A runtime uses a richer system prompt to reduce hallucination and improve
rare/long-tail coverage, while summary-mode SFT training intentionally uses a
minimal format-only prompt to avoid injecting business priors into the model.
"""

from __future__ import annotations

from typing import Optional

from src.config.missions import (
    validate_mission as _validate_mission,
)

# Import runtime summary prompt builder (not the training-minimal default)
from src.config.prompts import USER_PROMPT_SUMMARY
from src.prompts.stage_a_summary import (
    build_stage_a_system_prompt,
    build_stage_a_user_prompt,
)
from src.prompts.summary_profiles import (
    DEFAULT_SUMMARY_PROFILE_RUNTIME,
    get_summary_profile,
)

# Stage-A runtime system prompt (richer than training-minimal)
# Default: use runtime profile without mission (backward compat)
SUMMARY_SYSTEM_PROMPT = build_stage_a_system_prompt(
    domain="bbu",
    profile_name=DEFAULT_SUMMARY_PROFILE_RUNTIME,
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
    return build_stage_a_system_prompt(
        domain=ds,
        mission=mission,
        profile_name=profile_name,
    )


# Base user prompt from training (summary-specific, no coordinates)
SUMMARY_USER_PROMPT_BASE = USER_PROMPT_SUMMARY


def build_user_prompt(
    mission: Optional[str] = None,
    dataset: Optional[str] = None,
    profile_name: str = DEFAULT_SUMMARY_PROFILE_RUNTIME,
) -> str:
    """Build mission-dependent user prompt.

    Args:
        mission: Mission name (one of SUPPORTED_MISSIONS) or None

    Returns:
        User prompt string with domain guidance appended
    """
    base = SUMMARY_USER_PROMPT_BASE
    ds = dataset or "bbu"
    profile = get_summary_profile(profile_name)
    return build_stage_a_user_prompt(
        base,
        domain=ds,
        include_domain_pack=profile.include_domain_pack,
    )


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
]
