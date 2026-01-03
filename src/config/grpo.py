"""GRPO-specific configuration validation."""

from __future__ import annotations

from typing import Any

from src.rlhf.grpo.rewards.names import LEGACY_SUMMARY_REWARD_NAMES

from .schema import TrainingConfig


def _as_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{field_name} must be an integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if not stripped.isdigit():
            raise ValueError(f"{field_name} must be an integer, got {value!r}")
        return int(stripped)
    raise TypeError(f"{field_name} must be an integer")


def validate_grpo_config(config: TrainingConfig) -> None:
    rlhf = dict(config.rlhf or {})
    rlhf_type = rlhf.get("rlhf_type")

    chord = config.custom.grpo.chord
    if chord.enabled and rlhf_type != "grpo":
        raise ValueError(
            "custom.grpo.chord.enabled=true is only supported when rlhf.rlhf_type=grpo"
        )

    if rlhf_type != "grpo":
        return

    reward_funcs = rlhf.get("reward_funcs")
    reward_weights = rlhf.get("reward_weights")

    if reward_funcs is not None and not isinstance(reward_funcs, list):
        raise TypeError("rlhf.reward_funcs must be a list when provided")

    if reward_weights is not None and not isinstance(reward_weights, list):
        raise TypeError("rlhf.reward_weights must be a list when provided")

    if reward_weights is not None and reward_funcs is None:
        raise ValueError("rlhf.reward_weights requires rlhf.reward_funcs")

    if reward_funcs is not None:
        legacy_used = [
            name for name in reward_funcs if name in LEGACY_SUMMARY_REWARD_NAMES
        ]
        if legacy_used:
            mapping = ", ".join(
                f"{name}→{LEGACY_SUMMARY_REWARD_NAMES[name]}" for name in legacy_used
            )
            raise ValueError(
                "Legacy summary GRPO reward identifiers are unsupported; update to the "
                f"namespaced dot form ({mapping})."
            )
        if reward_weights is not None and len(reward_funcs) != len(reward_weights):
            raise ValueError(
                "rlhf.reward_weights length must match rlhf.reward_funcs length"
            )

    num_generations = _as_int(rlhf.get("num_generations"), "rlhf.num_generations")
    generation_batch_size = _as_int(
        rlhf.get("generation_batch_size"), "rlhf.generation_batch_size"
    )
    if num_generations and generation_batch_size:
        if generation_batch_size % num_generations != 0:
            raise ValueError(
                "rlhf.generation_batch_size must be divisible by rlhf.num_generations"
            )


__all__ = ["validate_grpo_config"]
