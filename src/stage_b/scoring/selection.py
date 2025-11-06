#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection policy for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ..config import SelectionConfig
from ..reflection.summarizer import SampleSummarizer
from ..types import ExperienceCandidate, GroupTicket, SelectionResult, TrajectoryWithSignals


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
    tie_breaker = _confidence_value if config.tie_break == "confidence" else _temperature_value

    def _sort_key(candidate: TrajectoryWithSignals) -> Tuple:
        signals = candidate.signals
        label_match_score = 0 if signals.label_match else 1
        semantic = -signals.semantic_advantage
        tie_metric = -tie_breaker(candidate)
        decode_temp = candidate.parsed.base.decode.temperature or 0.0
        candidate_index = candidate.parsed.base.candidate_index
        return (label_match_score, semantic, tie_metric, decode_temp, candidate_index)

    ordered = sorted(candidates, key=_sort_key)
    if config.policy == "top_semantic":
        ordered = sorted(
            ordered,
            key=lambda c: (-c.signals.semantic_advantage, _sort_key(c)),
        )
    return ordered


def select_for_group(
    ticket: GroupTicket,
    candidates: Iterable[TrajectoryWithSignals],
    *,
    guidance_step: int,
    reflection_cycle: int,
    reflection_change: str | None,
    config: SelectionConfig,
    summarizer: Optional[SampleSummarizer] = None,
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

    # Generate summary and critique for the selected candidate
    summary: Optional[str] = None
    critique: Optional[str] = None
    if summarizer is not None:
        experience_candidate = ExperienceCandidate(
            candidate_index=parsed.base.candidate_index,
            verdict=parsed.verdict,
            reason=parsed.reason,
            confidence=signals.confidence,
            signals=signals,
        )
        summary_critique = summarizer.summarize(ticket, experience_candidate)
        summary = summary_critique.summary
        critique = summary_critique.critique

    return SelectionResult(
        group_id=parsed.base.group_id,
        mission=parsed.base.mission,
        verdict=parsed.verdict,
        reason=parsed.reason,
        confidence=signals.confidence,
        label_match=signals.label_match,
        semantic_advantage=signals.semantic_advantage,
        selected_candidate=parsed.base.candidate_index,
        guidance_step=guidance_step,
        reflection_change=reflection_change,
        reflection_cycle=reflection_cycle,
        summary=summary,
        critique=critique,
    )


__all__ = ["select_for_group"]

