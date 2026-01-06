"""Reward registry for summary GRPO."""

from __future__ import annotations

from typing import cast

from swift.plugin import orms as _orms_raw  # pyright: ignore[reportUnknownVariableType]
from swift.plugin.orm import ORM

from .names import DENSE_REWARD_NAMES, SUMMARY_REWARD_NAMES
from .dense.rewards import (
    DenseAttrWeightedRecallReward,
    DenseCategoryMeanF1Reward,
    DenseFormatReward,
    DenseLocalizationMeanFBetaReward,
    DenseLocalizationSoftRecallReward,
    DenseParseSchemaStrictReward,
)
from .summary.rewards import (
    SummaryCategoryF1Reward,
    SummaryCategoryRecallReward,
    SummaryContentEqReward,
    SummaryContentF1Reward,
    SummaryDatasetReward,
    SummaryFormatReward,
    SummaryGroupStatsPresenceReward,
    SummaryHeaderReward,
    SummaryNoDupKeysPenalty,
    SummaryNotesBBUReward,
    SummaryNotesPresenceReward,
    SummaryParsePenalty,
    SummaryStrictPenaltyReward,
    SummaryStructuredContentTverskyReward,
    SummaryTextBBUReward,
)

_orms_raw = cast(dict[str, object], _orms_raw)
_orms = cast(dict[str, type[ORM]], _orms_raw)

SUMMARY_REWARD_REGISTRY: dict[str, type[ORM]] = {
    "summary.format": SummaryFormatReward,
    "summary.header": SummaryHeaderReward,
    "summary.strict": SummaryStrictPenaltyReward,
    "summary.parse": SummaryParsePenalty,
    "summary.no_dup_keys": SummaryNoDupKeysPenalty,
    "summary.dataset": SummaryDatasetReward,
    "summary.category_recall": SummaryCategoryRecallReward,
    "summary.category_f1": SummaryCategoryF1Reward,
    "summary.content_eq": SummaryContentEqReward,
    "summary.content_f1": SummaryContentF1Reward,
    "summary.content_structured_tversky": SummaryStructuredContentTverskyReward,
    "summary.text_bbu": SummaryTextBBUReward,
    "summary.notes_bbu": SummaryNotesBBUReward,
    "summary.notes_presence": SummaryNotesPresenceReward,
    "summary.group_stats_presence": SummaryGroupStatsPresenceReward,
}

DENSE_REWARD_REGISTRY: dict[str, type[ORM]] = {
    "dense.format": DenseFormatReward,
    "dense.parse_schema_strict": DenseParseSchemaStrictReward,
    "dense.loc_mean_fbeta": DenseLocalizationMeanFBetaReward,
    "dense.loc_soft_recall": DenseLocalizationSoftRecallReward,
    "dense.cat_mean_f1": DenseCategoryMeanF1Reward,
    "dense.attr_weighted_recall": DenseAttrWeightedRecallReward,
}


def register_summary_rewards() -> None:
    """Register summary-mode GRPO rewards."""

    for name in SUMMARY_REWARD_NAMES:
        reward_cls = SUMMARY_REWARD_REGISTRY.get(name)
        if reward_cls is None:
            raise RuntimeError(f"Missing summary GRPO reward registration for {name!r}")
        _orms[name] = reward_cls


def register_dense_rewards() -> None:
    """Register dense-mode GRPO rewards."""

    for name in DENSE_REWARD_NAMES:
        reward_cls = DENSE_REWARD_REGISTRY.get(name)
        if reward_cls is None:
            raise RuntimeError(f"Missing dense GRPO reward registration for {name!r}")
        _orms[name] = reward_cls
