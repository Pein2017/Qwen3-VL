"""Summary prompt profiles for training vs inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.prompts.summary_core import (
    build_summary_system_prompt_minimal,
    MISSION_SPECIFIC_PRIOR_RULES,
)

from .domain_packs import get_domain_pack


@dataclass(frozen=True)
class SummaryPromptProfile:
    name: str
    include_domain_pack: bool
    include_mission_rules: bool

    def build(
        self,
        *,
        domain: Optional[str] = None,
        mission: Optional[str] = None,
    ) -> str:
        parts = [build_summary_system_prompt_minimal().strip()]

        if self.include_domain_pack:
            pack = get_domain_pack(domain)
            parts.append(pack.block.strip())

        if self.include_mission_rules and mission:
            mission_rules = MISSION_SPECIFIC_PRIOR_RULES.get(mission)
            if mission_rules:
                rules = (
                    "".join(mission_rules)
                    if isinstance(mission_rules, (tuple, list))
                    else str(mission_rules)
                )
                parts.append(f"【{mission}任务特定规则】\n{rules}")

        return "\n\n".join(part for part in parts if part)


SUMMARY_PROMPT_PROFILES = {
    "summary_runtime": SummaryPromptProfile(
        name="summary_runtime",
        include_domain_pack=True,
        include_mission_rules=True,
    ),
}


DEFAULT_SUMMARY_PROFILE_TRAIN = "summary_runtime"
DEFAULT_SUMMARY_PROFILE_RUNTIME = "summary_runtime"


def get_summary_profile(name: str | None) -> SummaryPromptProfile:
    if name is None:
        raise ValueError("prompt profile must be provided")
    normalized = str(name).strip().lower()
    profile = SUMMARY_PROMPT_PROFILES.get(normalized)
    if profile is None:
        available = ", ".join(sorted(SUMMARY_PROMPT_PROFILES.keys()))
        raise ValueError(f"Unknown prompt profile '{name}'. Available: {available}")
    return profile


def build_summary_system_prompt(
    profile_name: str | None,
    *,
    domain: Optional[str] = None,
    mission: Optional[str] = None,
) -> str:
    """Compose the summary system prompt based on profile and domain."""
    profile = get_summary_profile(profile_name)
    return profile.build(
        domain=domain,
        mission=mission,
    )


__all__ = [
    "DEFAULT_SUMMARY_PROFILE_RUNTIME",
    "DEFAULT_SUMMARY_PROFILE_TRAIN",
    "SUMMARY_PROMPT_PROFILES",
    "SummaryPromptProfile",
    "build_summary_system_prompt",
    "get_summary_profile",
]
