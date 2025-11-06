"""Sampling utilities for Stage-B (prompts + rollout)."""

from .prompts import build_messages, build_system_prompt, build_user_prompt
from .rollout import RolloutSampler

__all__ = [
    "RolloutSampler",
    "build_messages",
    "build_system_prompt",
    "build_user_prompt",
]

