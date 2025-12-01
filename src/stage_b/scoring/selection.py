#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection policy for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ..config import ManualReviewConfig, SelectionConfig
from ..types import GroupTicket, SelectionResult, TrajectoryWithSignals


def _temperature_value(candidate: TrajectoryWithSignals) -> float:
    temperature = candidate.parsed.base.decode.temperature
    return temperature if temperature is not None else 1.0


def _sort_candidates(
    candidates: Sequence[TrajectoryWithSignals],
) -> List[TrajectoryWithSignals]:
    """Sort candidates by lower temperature then candidate index for stability."""

    def _sort_key(candidate: TrajectoryWithSignals) -> Tuple:
        decode_temp = candidate.parsed.base.decode.temperature or 0.0
        candidate_index = candidate.parsed.base.candidate_index
        return (decode_temp, candidate_index)

    return sorted(candidates, key=_sort_key)


def _vote_strength(candidates: Sequence[TrajectoryWithSignals], verdict: str) -> float:
    if not candidates:
        return 0.0
    total = len(candidates)
    agree = sum(1 for c in candidates if c.parsed.verdict == verdict)
    return agree / total if total > 0 else 0.0


def select_for_group(
    ticket: GroupTicket,
    candidates: Iterable[TrajectoryWithSignals],
    *,
    guidance_step: int,
    reflection_cycle: int,
    reflection_change: Optional[str],
    config: SelectionConfig,
    manual_review: ManualReviewConfig,
) -> SelectionResult:
    pool = [candidate for candidate in candidates if candidate.parsed.format_ok]
    warnings: List[str] = []
    if not pool:
        raise ValueError("No format_ok candidates for selection")

    ordered = _sort_candidates(pool)

    # Majority vote
    from collections import Counter

    verdicts = [c.parsed.verdict for c in pool if c.parsed.verdict is not None]
    if not verdicts:
        raise ValueError("No parsed verdicts available for selection")
    counts = Counter(verdicts)
    majority_verdict, majority_count = counts.most_common(1)[0]
    tied = [v for v, c in counts.items() if c == majority_count]
    if len(tied) > 1:
        # Fail-first tie-break: prefer 'fail' when tied, else lowest temperature then candidate order
        if "fail" in tied:
            ordered_tied = [c for c in ordered if c.parsed.verdict == "fail"]
        else:
            ordered_tied = [c for c in ordered if c.parsed.verdict in tied]
        chosen = ordered_tied[0]
    else:
        chosen = next(c for c in ordered if c.parsed.verdict == majority_verdict)

    parsed = chosen.parsed
    chosen_label_match = chosen.signals.label_match if chosen.signals else False
    final_verdict = parsed.verdict
    vote_strength = _vote_strength(pool, final_verdict)

    conflict_flag = bool(chosen.signals.conflict_flag) if chosen.signals else False
    needs_manual_review = False
    if chosen_label_match is False:
        conflict_flag = True
        warnings.append("label_mismatch")

    # Always route conflicts (label mismatch) to manual review; keep low-agreement logic.
    if conflict_flag:
        needs_manual_review = True
    if manual_review.enabled:
        if vote_strength is not None and vote_strength < manual_review.min_verdict_agreement:
            needs_manual_review = True
            warnings.append("low_agreement")

    return SelectionResult(
        group_id=parsed.base.group_id,
        mission=parsed.base.mission,
        verdict=final_verdict,
        reason=parsed.reason,
        vote_strength=vote_strength,
        label_match=chosen_label_match,
        selected_candidate=parsed.base.candidate_index,
        guidance_step=guidance_step,
        reflection_change=reflection_change,
        reflection_cycle=reflection_cycle,
        manual_review_recommended=needs_manual_review,
        eligible=True,
        ineligible_reason=None,
        warnings=tuple(warnings),
        conflict_flag=conflict_flag,
        needs_manual_review=needs_manual_review,
    )


__all__ = ["select_for_group"]
