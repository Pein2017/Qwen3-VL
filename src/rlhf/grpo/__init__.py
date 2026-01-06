"""GRPO reward registration entrypoints."""

from .rewards.registry import register_dense_rewards, register_summary_rewards


def register_grpo_rewards() -> None:
    """Register GRPO reward functions for summary and dense post-training."""

    register_dense_rewards()
    register_summary_rewards()


__all__ = ["register_grpo_rewards"]
