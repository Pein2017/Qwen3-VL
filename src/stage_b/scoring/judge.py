#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic judge for Stage-B training-free pipeline."""

from __future__ import annotations

import logging
import math
import re
from typing import Iterable, Tuple

from ..types import (
    ChineseVerdict,
    GroupTicket,
    MissionGuidance,
    ParsedTrajectory,
)

logger = logging.getLogger(__name__)


_PUNCT_SPLIT = re.compile(r"[，,；;。：:]")
_FAIL_HINTS = ("缺", "未", "无", "不合格", "不符合", "松动", "破损", "遮挡", "无关")
_PASS_HINTS = ("符合", "完整", "按要求", "正确", "整齐", "有保护")


def _to_chinese_verdict(label: str) -> ChineseVerdict:
    return "通过" if label.lower() == "pass" else "不通过"


def _extract_keywords(text: str) -> Tuple[str, ...]:
    tokens = []
    for chunk in _PUNCT_SPLIT.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= 1:
            continue
        tokens.append(chunk)
    return tuple(tokens)


def _hit_ratio(keywords: Iterable[str], haystacks: Iterable[str]) -> float:
    haystack_text = "\n".join(haystacks)
    hits = 0
    total = 0
    for keyword in keywords:
        total += 1
        if keyword and keyword in haystack_text:
            hits += 1
    if total == 0:
        return 0.6  # neutral default
    return hits / total


def _reason_score(reason: str) -> float:
    if not reason:
        return 0.0
    length = len(reason)
    if length < 12:
        return 0.4
    if length > 80:
        return 1.0
    return min(1.0, 0.4 + (length - 12) / 68)


def _summary_confidence_penalty(confidence: str) -> float:
    if confidence == "high":
        return 0.0
    if confidence == "low":
        return 0.1
    return 0.2


def _keyword_diff(text: str) -> int:
    fail_hits = sum(text.count(term) for term in _FAIL_HINTS)
    pass_hits = sum(text.count(term) for term in _PASS_HINTS)
    return pass_hits - fail_hits


class DeterministicJudge:
    """Rule-based judge computing semantic advantage and escalation signals."""

    def evaluate(
        self,
        ticket: GroupTicket,
        guidance: MissionGuidance,
        trajectory: ParsedTrajectory,
    ):
        if (
            not trajectory.format_ok
            or trajectory.verdict is None
            or trajectory.reason is None
        ):
            logger.debug(
                "Skipping trajectory %s: format_ok=%s, verdict=%s, reason=%s (raw: %s)",
                trajectory.base.candidate_index,
                trajectory.format_ok,
                trajectory.verdict,
                trajectory.reason,
                trajectory.base.response_text[:80],
            )
            raise ValueError(
                f"Invalid trajectory: format_ok={trajectory.format_ok}, "
                f"verdict={trajectory.verdict}, reason={trajectory.reason}"
            )

        chinese_label = _to_chinese_verdict(ticket.label)
        label_match = trajectory.verdict == chinese_label

        stage_a_text = "\n".join(ticket.summaries.as_dict().values())
        evidence_sources = (trajectory.reason, stage_a_text, guidance.focus or "")

        focus_keywords = _extract_keywords(guidance.focus or "")
        focus_consistency = _hit_ratio(focus_keywords, evidence_sources)

        guidance_keywords = _extract_keywords(
            "\n".join(value for value in guidance.experiences.values())
        )
        heuristic_score = _hit_ratio(guidance_keywords, evidence_sources)

        rationale_quality = _reason_score(trajectory.reason)

        label_score = 1.0 if label_match else -1.0
        penalty = _summary_confidence_penalty("medium")
        delta_keywords = _keyword_diff(trajectory.reason)
        delta_score = math.tanh(delta_keywords / 3.0)

        aggregated = (
            label_score
            + focus_consistency
            + heuristic_score
            + rationale_quality
            + delta_score
        )

        semantic_advantage = aggregated - penalty

        label_contradiction = (not label_match) and (
            focus_consistency >= 0.6 or heuristic_score >= 0.6
        )
        needs_recheck = label_contradiction

        scores = {
            "label_match": label_match,
            "focus_consistency": float(round(focus_consistency, 3)),
            "heuristic_score": float(round(heuristic_score, 3)),
            "semantic_advantage": float(round(semantic_advantage, 3)),
            "label_contradiction": label_contradiction,
            "needs_recheck": needs_recheck,
            "rationale_quality": float(round(rationale_quality, 3)),
        }

        return {
            "parsed": trajectory,
            "scores": scores,
        }


__all__ = ["DeterministicJudge"]

