"""GRPO reward registration entrypoints."""

from .rewards.registry import register_summary_rewards


def register_grpo_rewards() -> None:
    """Register GRPO reward functions for summary post-training."""

    register_summary_rewards()


__all__ = ["register_grpo_rewards"]
