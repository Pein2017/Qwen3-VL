"""Configuration management for YAML-based training setup"""
from .loader import ConfigLoader
from .prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, USER_PROMPT

__all__ = [
    "ConfigLoader",
    "SYSTEM_PROMPT_A",
    "SYSTEM_PROMPT_B",
    "USER_PROMPT",
]

