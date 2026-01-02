"""Reward registry for summary GRPO."""

from swift.plugin import orms as _orms

from .names import SUMMARY_REWARD_NAMES
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
    SummaryObjectsTotalLowerBoundReward,
    SummaryObjectsTotalReward,
    SummaryParsePenalty,
    SummaryStrictPenaltyReward,
    SummaryStructuredContentTverskyReward,
    SummaryTextBBUReward,
)

SUMMARY_REWARD_REGISTRY = {
    "summary.format": SummaryFormatReward,
    "summary.header": SummaryHeaderReward,
    "summary.strict": SummaryStrictPenaltyReward,
    "summary.parse": SummaryParsePenalty,
    "summary.no_dup_keys": SummaryNoDupKeysPenalty,
    "summary.dataset": SummaryDatasetReward,
    "summary.objects_total": SummaryObjectsTotalReward,
    "summary.objects_total_lb": SummaryObjectsTotalLowerBoundReward,
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


def register_summary_rewards() -> None:
    """Register summary-mode GRPO rewards."""

    for name in SUMMARY_REWARD_NAMES:
        reward_cls = SUMMARY_REWARD_REGISTRY.get(name)
        if reward_cls is None:
            raise RuntimeError(f"Missing summary GRPO reward registration for {name!r}")
        _orms[name] = reward_cls
