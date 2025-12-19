#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection policy for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ..config import ManualReviewConfig, SelectionConfig
from ..signals import extract_mission_evidence
from ..types import GroupTicket, SelectionResult, TrajectoryWithSignals
from ..utils.chinese import normalize_spaces, to_simplified

_FORBIDDEN_THIRD_STATE_PHRASES: Tuple[str, ...] = (
    "需复核",
    "需人工复核",
    "need-review",
    "needreview",
    "证据不足",
    "待定",
    "通过但需复核",
    "通过但需人工复核",
)


def _contains_third_state(text: str) -> bool:
    normalized = to_simplified(text or "")
    normalized = normalize_spaces(normalized)
    return any(term in normalized for term in _FORBIDDEN_THIRD_STATE_PHRASES)


def _fallback_reason(verdict: str) -> str:
    if verdict == "pass":
        return "可从摘要/备注中给出支持通过的依据并覆盖任务要点，判通过。"
    return "可从摘要/备注中确认存在与任务要点相关的问题或无法确认关键点，判不通过。"


def _rewrite_reason_for_fail_first(hits) -> str:
    parts: List[str] = []
    for hit in hits:
        snippet = hit.for_reason()
        if not snippet:
            continue
        parts.append(f"{hit.image_key}:{snippet}")
        if len(parts) >= 3:
            break
    evidence = "；".join(parts) if parts else "命中明确负项"
    return f"{evidence}；总结: 存在与任务要点相关的明确负项，判不通过。"


def _temperature_value(candidate: TrajectoryWithSignals) -> float:
    if candidate.parsed is None:
        return 1.0
    temperature = candidate.parsed.base.decode.temperature
    return temperature if temperature is not None else 1.0


def _sort_candidates(
    candidates: Sequence[TrajectoryWithSignals],
) -> List[TrajectoryWithSignals]:
    """Sort candidates by lower temperature then candidate index for stability."""

    def _sort_key(candidate: TrajectoryWithSignals) -> Tuple:
        if candidate.parsed is None:
            return (0.0, 0)
        decode_temp = candidate.parsed.base.decode.temperature or 0.0
        candidate_index = candidate.parsed.base.candidate_index
        return (decode_temp, candidate_index)

    return sorted(candidates, key=_sort_key)


def _vote_strength(candidates: Sequence[TrajectoryWithSignals], verdict: str) -> float:
    if not candidates:
        return 0.0
    total = len(candidates)
    agree = sum(
        1 for c in candidates if c.parsed is not None and c.parsed.verdict == verdict
    )
    return agree / total if total > 0 else 0.0


def select_for_group(
    ticket: GroupTicket,
    candidates: Iterable[TrajectoryWithSignals],
    *,
    mission_g0: Optional[str],
    guidance_step: int,
    reflection_cycle: int,
    reflection_change: Optional[str],
    config: SelectionConfig,
    manual_review: ManualReviewConfig,
) -> SelectionResult:
    pool = [
        candidate
        for candidate in candidates
        if candidate.parsed is not None and candidate.parsed.format_ok
    ]
    warnings: List[str] = []
    if not pool:
        raise ValueError("No format_ok candidates for selection")

    ordered = _sort_candidates(pool)

    # Majority vote
    from collections import Counter

    verdicts = [
        c.parsed.verdict
        for c in pool
        if c.parsed is not None and c.parsed.verdict is not None
    ]
    if not verdicts:
        raise ValueError("No parsed verdicts available for selection")
    counts = Counter(verdicts)
    majority_verdict, majority_count = counts.most_common(1)[0]
    tied = [v for v, c in counts.items() if c == majority_count]
    verdict_set = set(verdicts)
    rollout_contradiction = "pass" in verdict_set and "fail" in verdict_set
    if len(tied) > 1:
        # Fail-first tie-break: prefer 'fail' when tied, else lowest temperature then candidate order
        warnings.append("tied_vote")
        if "fail" in tied:
            ordered_tied = [
                c
                for c in ordered
                if c.parsed is not None and c.parsed.verdict == "fail"
            ]
        else:
            ordered_tied = [
                c for c in ordered if c.parsed is not None and c.parsed.verdict in tied
            ]
        chosen = ordered_tied[0]
    else:
        chosen = next(
            c
            for c in ordered
            if c.parsed is not None and c.parsed.verdict == majority_verdict
        )

    parsed = chosen.parsed
    if parsed is None:
        raise ValueError("Selected candidate has no parsed trajectory")
    chosen_label_match = chosen.signals.label_match if chosen.signals else None
    final_verdict = parsed.verdict
    if final_verdict is None:
        raise ValueError("Selected candidate has no verdict")
    vote_strength = _vote_strength(pool, final_verdict)

    conflict_flag = bool(chosen.signals.conflict_flag) if chosen.signals else False
    needs_manual_review = False
    if chosen_label_match is False:
        conflict_flag = True
        warnings.append("label_mismatch")

    low_agreement_flag = (
        vote_strength is not None
        and vote_strength < manual_review.min_verdict_agreement
    )

    if low_agreement_flag:
        warnings.append("low_agreement")
        needs_manual_review = True

    # Rollout contradiction is an uncertainty signal even if selection matches GT.
    if rollout_contradiction:
        warnings.append("rollout_contradiction")
        needs_manual_review = True

    # Deterministic mission-scoped fail-first guardrail: override to fail if
    # any mission-relevant Stage-A summary contains explicit negative evidence.
    evidence = extract_mission_evidence(
        ticket.summaries.as_dict(),
        mission_g0=mission_g0,
    )
    if evidence.irrelevant_negative_hits:
        warnings.append("fail_first_irrelevant_hit")
        needs_manual_review = True
    if evidence.pending_signal_hits:
        warnings.append("pending_signal_hit")
        needs_manual_review = True
    if evidence.relevant_negative_hits:
        if final_verdict != "fail":
            warnings.append("fail_first_override")
        final_verdict = "fail"
        parsed_reason = parsed.reason or "" if parsed.reason is not None else ""
        rewritten = _rewrite_reason_for_fail_first(evidence.relevant_negative_hits)
        # If we overrode a pass verdict, always prefer rewritten reason.
        final_reason = rewritten if final_verdict == "fail" else parsed_reason
    else:
        final_reason = parsed.reason or "" if parsed.reason is not None else ""

    # Re-compute label alignment after deterministic overrides.
    final_label_match = final_verdict == ticket.label
    if final_label_match is False and "label_mismatch" not in warnings:
        warnings.append("label_mismatch")
    conflict_flag = not final_label_match

    # Ensure final reason is single-line simplified Chinese and does not contain third-state wording.
    final_reason = to_simplified(final_reason)
    final_reason = normalize_spaces(final_reason)
    final_reason = final_reason.replace("\n", " ").strip()
    if not final_reason or _contains_third_state(final_reason):
        final_reason = _fallback_reason(final_verdict)

    return SelectionResult(
        group_id=parsed.base.group_id,
        mission=parsed.base.mission,
        verdict=final_verdict,
        reason=final_reason,
        vote_strength=vote_strength,
        label_match=final_label_match,
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
