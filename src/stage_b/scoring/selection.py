#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection policy for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

from ..config import ManualReviewConfig, SelectionConfig
from ..types import (
    ExperienceCandidate,
    GroupTicket,
    SelectionResult,
    TrajectoryWithSignals,
)


class _SummaryCritique(Protocol):
    summary: Optional[str]
    critique: Optional[str]


class SummarizerProtocol(Protocol):
    def summarize(
        self, ticket: GroupTicket, candidate: ExperienceCandidate
    ) -> _SummaryCritique: ...


def _confidence_value(candidate: TrajectoryWithSignals) -> float:
    confidence = candidate.signals.confidence
    return confidence if confidence is not None else -1.0


def _temperature_value(candidate: TrajectoryWithSignals) -> float:
    temperature = candidate.parsed.base.decode.temperature
    return temperature if temperature is not None else 1.0


def _sort_candidates(
    candidates: Sequence[TrajectoryWithSignals],
    config: SelectionConfig,
) -> List[TrajectoryWithSignals]:
    """Sort candidates by label_match, then tie-break by confidence or temperature.

    Note: semantic_advantage has been removed per training-free Stage-B design.
    Selection now relies on label_match + tie-breaking only. CriticEngine provides
    LLM-based evaluation for reflection context.
    """
    tie_breaker = (
        _confidence_value if config.tie_break == "confidence" else _temperature_value
    )

    def _sort_key(candidate: TrajectoryWithSignals) -> Tuple:
        signals = candidate.signals
        label_match_score = 0 if signals.label_match else 1
        tie_metric = -tie_breaker(candidate)
        decode_temp = candidate.parsed.base.decode.temperature or 0.0
        candidate_index = candidate.parsed.base.candidate_index
        return (label_match_score, tie_metric, decode_temp, candidate_index)

    ordered = sorted(candidates, key=_sort_key)
    # Note: "top_semantic" policy is deprecated but kept for backward compatibility
    # It now behaves the same as "top_label" since semantic_advantage is removed
    return ordered


def _should_recommend_manual_review(
    pool: Sequence[TrajectoryWithSignals],
    config: ManualReviewConfig,
) -> bool:
    """Heuristic gating for high-confidence, self-consistent label mismatches.

    If all candidates disagree with the ticket label but are highly self-consistent
    and confident about a shared verdict, we recommend deferring to manual review
    instead of forcing reflection to fit a potentially noisy label.
    """

    if not config.enabled:
        return False

    signals = [c.signals for c in pool if c.signals is not None]
    if not signals:
        return False

    any_true = any(s.label_match is True for s in signals)
    any_false = any(s.label_match is False for s in signals)
    # Only trigger when we have evidence that *all* known candidates disagree
    # with the label (no True, at least one False).
    if not any_false or any_true:
        return False

    # Compute majority verdict agreement across the pool.
    verdicts = [
        c.parsed.verdict
        for c in pool
        if c.parsed is not None and c.parsed.verdict is not None
    ]
    if not verdicts:
        return False

    from collections import Counter

    counts = Counter(verdicts)
    majority_verdict, majority_count = counts.most_common(1)[0]
    majority_fraction = majority_count / len(verdicts)
    if majority_fraction < config.min_verdict_agreement:
        return False

    majority_candidates = [
        c
        for c in pool
        if c.parsed is not None and c.parsed.verdict == majority_verdict
    ]
    majority_signals = [c.signals for c in majority_candidates if c.signals is not None]
    if not majority_signals:
        return False

    conf_values = [s.confidence for s in majority_signals if s.confidence is not None]
    if not conf_values:
        return False
    avg_conf = sum(conf_values) / len(conf_values)
    if avg_conf < config.min_confidence:
        return False

    sc_values = [
        s.self_consistency
        for s in majority_signals
        if s.self_consistency is not None
    ]
    if sc_values:
        avg_sc = sum(sc_values) / len(sc_values)
        if avg_sc < config.min_self_consistency:
            return False

    return True


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
    if not pool:
        raise ValueError("No valid trajectories available for selection")

    ordered = _sort_candidates(pool, config)

    chosen = ordered[0]
    parsed = chosen.parsed
    signals = chosen.signals

    if parsed.verdict is None:
        raise ValueError("Chosen trajectory is missing verdict")
    if parsed.reason is None or not parsed.reason.strip():
        raise ValueError("Chosen trajectory is missing rationale text")

    # Conservative override: rely on LLM-only critic signals when available
    final_verdict = parsed.verdict
    warnings: List[str] = []
    manual_review_recommended = _should_recommend_manual_review(pool, manual_review)
    if manual_review_recommended:
        warnings.append("manual_review: high_consistency_label_mismatch")
    eligible = True
    ineligible_reason: Optional[str] = None
    critic = getattr(chosen, "critic", None)
    if critic is not None:
        needs_recheck = getattr(critic, "needs_recheck", None)
        evidence_sufficiency = getattr(critic, "evidence_sufficiency", None)
        recommended_action = getattr(critic, "recommended_action", None)
        if needs_recheck is True:
            final_verdict = "fail"
            warnings.append("conservative_override: needs_recheck=true")
        elif evidence_sufficiency is False:
            final_verdict = "fail"
            warnings.append("conservative_override: evidence_sufficiency=false")
        elif recommended_action == "人工复核":
            final_verdict = "fail"
            warnings.append("conservative_override: recommended_action=人工复核")

    return SelectionResult(
        group_id=parsed.base.group_id,
        mission=parsed.base.mission,
        verdict=final_verdict,
        reason=parsed.reason,
        confidence=signals.confidence,
        label_match=signals.label_match,
        selected_candidate=parsed.base.candidate_index,
        guidance_step=guidance_step,
        reflection_change=reflection_change,
        reflection_cycle=reflection_cycle,
        manual_review_recommended=manual_review_recommended,
        eligible=eligible,
        ineligible_reason=ineligible_reason,
        warnings=tuple(warnings),
    )


__all__ = ["select_for_group"]
