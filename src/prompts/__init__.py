"""Prompt composition utilities (summary profiles, domain packs)."""

from .domain_packs import DomainKnowledgePack, get_domain_pack
from .stage_a_summary import build_stage_a_system_prompt
from .stage_b_verdict import build_stage_b_system_prompt
from .summary_profiles import (
    SummaryPromptProfile,
    build_summary_system_prompt,
    get_summary_profile,
)

__all__ = [
    "DomainKnowledgePack",
    "SummaryPromptProfile",
    "build_stage_a_system_prompt",
    "build_stage_b_system_prompt",
    "build_summary_system_prompt",
    "get_domain_pack",
    "get_summary_profile",
]
