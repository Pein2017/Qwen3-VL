#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic signal extraction for Stage-B candidates."""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Union, cast

from .config import SignalsConfig
from .types import (
    DeterministicSignals,
    GroupTicket,
    ParsedTrajectory,
    TrajectoryWithSignals,
)
from .utils.verdict import normalize_verdict

# Removed uncertainty regex cues; rely on LLM reasoning instead.


def attach_signals(
    ticket: GroupTicket,
    candidates: Sequence[Union[ParsedTrajectory, TrajectoryWithSignals]],
    config: SignalsConfig,
) -> List[TrajectoryWithSignals]:
    """Attach deterministic signals to candidates (parsed or flat TrajectoryWithSignals)."""

    def _get_verdict(obj: Union[ParsedTrajectory, TrajectoryWithSignals]):
        if isinstance(obj, ParsedTrajectory):
            return obj.verdict
        return cast(TrajectoryWithSignals, obj).verdict

    verdict_counts = Counter(v for v in (_get_verdict(c) for c in candidates) if v is not None)
    total_considered = sum(verdict_counts.values())

    annotated: List[TrajectoryWithSignals] = []
    for candidate in candidates:
        if isinstance(candidate, ParsedTrajectory):
            verdict_value = candidate.verdict
        else:
            verdict_value = cast(TrajectoryWithSignals, candidate).verdict

        label_match = None
        if verdict_value is not None:
            normalized = normalize_verdict(verdict_value)
            label_match = normalized == ticket.label if normalized is not None else None

        if (
            config.enable_consistency
            and verdict_value is not None
            and total_considered > 0
        ):
            same_count = verdict_counts.get(verdict_value, 0)
            self_consistency = same_count / total_considered
        else:
            self_consistency = None

        needs_manual_review = False
        conflict_flag = label_match is False

        signals = DeterministicSignals(
            label_match=label_match,
            self_consistency=self_consistency,
            conflict_flag=conflict_flag,
            needs_manual_review=needs_manual_review,
        )

        if isinstance(candidate, ParsedTrajectory):
            annotated.append(TrajectoryWithSignals(parsed=candidate, signals=signals))
        else:
            flat = cast(TrajectoryWithSignals, candidate)
            annotated.append(
                TrajectoryWithSignals(
                    candidate_index=flat.candidate_index,
                    verdict=flat.verdict,
                    reason=flat.reason,
                    signals=signals,
                    warnings=flat.warnings,
                )
            )

    return annotated


__all__ = ["attach_signals"]
