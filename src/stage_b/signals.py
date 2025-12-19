#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic signal extraction for Stage-B candidates."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple, Union, cast

from .config import SignalsConfig
from .types import (
    DeterministicSignals,
    GroupTicket,
    ParsedTrajectory,
    TrajectoryWithSignals,
)
from .utils.chinese import normalize_spaces, to_simplified
from .utils.verdict import normalize_verdict

FAIL_FIRST_NEGATIVE_TRIGGERS: Tuple[str, ...] = (
    "未按要求",
    "错误",
    "缺失",
    "松动",
    "损坏",
    "方向不正确",
    "反向",
    "不符合要求",
    "不合格",
    "不合理",
    "未安装",
    "未配备",
)

# Pattern-first: any "不符合要求/<issue>" counts as explicit negative evidence.
_NONCOMPLIANCE_PATTERN_RE = re.compile(r"不符合要求/[^\\s,，;；。]+")

# Minimal negation prefix exclusions to reduce false positives, e.g. "无错误/未发现错误".
_NEGATION_PREFIXES: Tuple[str, ...] = (
    "未发现明显",
    "未见明显",
    "未发现",
    "未见",
    "不存在",
    "没有",
    "无明显",
    "无",
)

PENDING_SIGNAL_PHRASES: Tuple[str, ...] = (
    "无法确认",
    "不能确认",
    "无法判断",
    "只显示部分",
    "模糊",
)

# Soft marker in Stage-A summaries; must NOT be treated as hard negative trigger.
REMARK_SOFT_MARKER = "需复核"

_REMARK_PREFIX_RE = re.compile(r"需复核[，,]\\s*备注[:：]")


def _normalize_for_match(text: str) -> str:
    simplified = to_simplified(text)
    simplified = normalize_spaces(simplified)
    return simplified.replace(" ", "")


def _strip_count_suffix(text: str) -> str:
    # Remove trailing "×<n>" count markers often used in Stage-A summaries.
    return re.sub(r"×\\d+$", "", text).strip()


def _extract_subject(clause: str) -> Optional[str]:
    cleaned = clause.strip()
    cleaned = re.sub(r"^(image\\d+|图片\\d+|图\\d+)[:：]\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\\d+[\\.、)\\]]\\s*", "", cleaned)
    if "/" not in cleaned:
        return None
    subject = cleaned.split("/", 1)[0].strip()
    return subject or None


def _extract_g0_keywords(g0: str) -> Tuple[str, ...]:
    """Extract coarse relevance keywords from G0 without introducing a noun glossary."""
    text = _normalize_for_match(g0)
    if not text:
        return ()

    # Replace common function words / connectors with separators.
    for phrase in (
        "至少需要检测到",
        "还需要检测到",
        "需要检测到",
        "根据情况判断是否需要",
        "根据情况判断是否",
        "若需要安装",
        "则判断是否",
        "符合要求",
        "按要求",
        "至少",
        "还需要",
        "需要",
        "检测到",
        "根据情况",
        "判断",
        "是否",
        "若",
        "则",
        "并",
        "且",
        "和",
        "与",
        "或",
        "还",
    ):
        text = text.replace(phrase, "|")
    for ch in ("，", ",", "。", "；", ";", "、"):
        text = text.replace(ch, "|")
    tokens = []
    for raw in text.split("|"):
        token = raw.strip()
        if len(token) < 2:
            continue
        if token in {"符合", "要求"}:
            continue
        tokens.append(token)
    # De-dup while keeping order
    seen = set()
    ordered = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
    # Expand common verb-prefix patterns (derived from G0 itself) to improve
    # relevance matching without hard-coding a global noun glossary.
    expanded: List[str] = []
    for t in ordered:
        expanded.append(t)
        if t.startswith("安装") and len(t) > 2:
            expanded.append(t[2:])
    # De-dup again while keeping order.
    seen2: set[str] = set()
    out: List[str] = []
    for t in expanded:
        if t in seen2:
            continue
        seen2.add(t)
        out.append(t)
    return tuple(out)


def _is_negated(text: str, *, hit_start: int) -> bool:
    for prefix in _NEGATION_PREFIXES:
        if hit_start - len(prefix) < 0:
            continue
        if text[hit_start - len(prefix) : hit_start] == prefix:
            return True
    return False


def _is_uncertain_context(text: str, *, hit_start: int) -> bool:
    """Return True if the hit is used in an uncertainty question/context (e.g., '是否缺失')."""
    window = text[max(0, hit_start - 6) : hit_start]
    return window.endswith(("是否", "能否", "是不是"))


def _is_mission_relevant(
    clause: str, *, subject: Optional[str], mission_g0: Optional[str]
) -> bool:
    if not mission_g0 or not str(mission_g0).strip():
        return False
    g0_norm = _normalize_for_match(mission_g0)
    clause_norm = _normalize_for_match(clause)
    keywords = _extract_g0_keywords(mission_g0)

    matched = [kw for kw in keywords if kw and kw in clause_norm]
    if subject:
        subject_norm = _normalize_for_match(subject)
        if subject_norm and subject_norm in g0_norm and subject_norm not in matched:
            matched.append(subject_norm)

    if not matched:
        return False

    # If G0 contains both a broad "设备" anchor and more specific items,
    # do not treat anchor-only matches as mission-relevant to avoid false
    # relevance when Stage-A encodes other topics under the device clause.
    anchors = [kw for kw in keywords if kw.endswith("设备")]
    if anchors:
        non_anchors = [kw for kw in keywords if kw not in anchors]
        if non_anchors and all(m in anchors for m in matched):
            return False

    return True


@dataclass(frozen=True)
class FailFirstHit:
    image_key: str
    trigger: str
    clause: str
    subject: Optional[str]
    mission_relevant: bool
    pattern_first: bool = False

    def for_reason(self) -> str:
        clause = _strip_count_suffix(self.clause)
        clause = _REMARK_PREFIX_RE.sub("备注:", clause)
        clause = clause.replace(REMARK_SOFT_MARKER, "")
        clause = normalize_spaces(clause).strip()
        return clause


@dataclass(frozen=True)
class MissionEvidence:
    relevant_negative_hits: Tuple[FailFirstHit, ...] = ()
    irrelevant_negative_hits: Tuple[FailFirstHit, ...] = ()
    pending_signal_hits: Tuple[str, ...] = ()


def extract_mission_evidence(
    stage_a_summaries: Mapping[str, str],
    *,
    mission_g0: Optional[str],
) -> MissionEvidence:
    """Extract mission-scoped negative evidence and pending signals from Stage-A summaries."""

    relevant: List[FailFirstHit] = []
    irrelevant: List[FailFirstHit] = []
    pending: List[str] = []

    for image_key, raw_text in stage_a_summaries.items():
        if not raw_text:
            continue
        text = to_simplified(raw_text)
        text = normalize_spaces(text)

        # Split Stage-A summaries into object-level entries:
        # - Stage-A uses Chinese comma "，" between entries
        # - English commas are used inside an entry for attributes and MUST NOT split.
        coarse = [c.strip() for c in re.split(r"[;；\\n]+", text) if c.strip()]
        if not coarse:
            coarse = [text.strip()] if text.strip() else []
        clauses: List[str] = []
        for part in coarse:
            sub = [seg.strip() for seg in part.split("，") if seg.strip()]
            clauses.extend(sub if sub else [part])

        for clause in clauses:
            clause_norm = _normalize_for_match(clause)
            subject = _extract_subject(clause)
            relevant_to_g0 = _is_mission_relevant(
                clause, subject=subject, mission_g0=mission_g0
            )

            # Pending signals (soft): track for reflection governance.
            for phrase in PENDING_SIGNAL_PHRASES:
                if phrase in clause_norm:
                    pending.append(f"{image_key}:{phrase}:{_strip_count_suffix(clause)}")
            has_soft_uncertainty = any(
                phrase in clause_norm for phrase in PENDING_SIGNAL_PHRASES
            ) or (REMARK_SOFT_MARKER in clause_norm)

            # Pattern-first negative evidence.
            for match in _NONCOMPLIANCE_PATTERN_RE.finditer(clause_norm):
                hit_start = match.start()
                if _is_negated(clause_norm, hit_start=hit_start) or _is_uncertain_context(
                    clause_norm, hit_start=hit_start
                ):
                    continue
                if has_soft_uncertainty:
                    # Treat negatives inside "只显示部分/无法判断/需复核" entries as soft signals;
                    # do not allow them to trigger deterministic fail-first overrides.
                    continue
                snippet = _strip_count_suffix(match.group(0))
                hit = FailFirstHit(
                    image_key=image_key,
                    trigger=snippet,
                    clause=_strip_count_suffix(clause),
                    subject=subject,
                    mission_relevant=relevant_to_g0,
                    pattern_first=True,
                )
                (relevant if relevant_to_g0 else irrelevant).append(hit)

            # Core trigger list (skip "不符合要求" if pattern-first already captured).
            has_pattern_first = "不符合要求/" in clause_norm
            for trigger in FAIL_FIRST_NEGATIVE_TRIGGERS:
                if trigger == "不符合要求" and has_pattern_first:
                    continue
                start = 0
                while True:
                    idx = clause_norm.find(trigger, start)
                    if idx == -1:
                        break
                    start = idx + len(trigger)
                    if _is_negated(clause_norm, hit_start=idx) or _is_uncertain_context(
                        clause_norm, hit_start=idx
                    ):
                        continue
                    if has_soft_uncertainty:
                        continue
                    hit = FailFirstHit(
                        image_key=image_key,
                        trigger=trigger,
                        clause=_strip_count_suffix(clause),
                        subject=subject,
                        mission_relevant=relevant_to_g0,
                        pattern_first=False,
                    )
                    (relevant if relevant_to_g0 else irrelevant).append(hit)

    # De-dup hits by (image_key, trigger, clause, mission_relevant)
    def _dedup(items: List[FailFirstHit]) -> Tuple[FailFirstHit, ...]:
        seen: set[tuple] = set()
        out: List[FailFirstHit] = []
        for h in items:
            key = (h.image_key, h.trigger, h.clause, h.mission_relevant, h.pattern_first)
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return tuple(out)

    pending_unique = tuple(dict.fromkeys(pending).keys())  # preserve order

    return MissionEvidence(
        relevant_negative_hits=_dedup(relevant),
        irrelevant_negative_hits=_dedup(irrelevant),
        pending_signal_hits=pending_unique,
    )


def attach_signals(
    ticket: GroupTicket,
    candidates: Sequence[Union[ParsedTrajectory, TrajectoryWithSignals]],
    config: SignalsConfig,
) -> List[TrajectoryWithSignals]:
    """Attach deterministic signals to candidates (parsed or flat TrajectoryWithSignals)."""

    def _get_verdict(obj: Union[ParsedTrajectory, TrajectoryWithSignals]):
        if isinstance(obj, ParsedTrajectory):
            return normalize_verdict(obj.verdict)
        return normalize_verdict(cast(TrajectoryWithSignals, obj).verdict)

    verdict_counts = Counter(
        v for v in (_get_verdict(c) for c in candidates) if v is not None
    )
    total_considered = sum(verdict_counts.values())

    annotated: List[TrajectoryWithSignals] = []
    for candidate in candidates:
        if isinstance(candidate, ParsedTrajectory):
            verdict_value = normalize_verdict(candidate.verdict)
            candidate_confidence = (
                candidate.confidence if config.store_confidence else None
            )
        else:
            flat = cast(TrajectoryWithSignals, candidate)
            verdict_value = normalize_verdict(flat.verdict)
            candidate_confidence = None
            if config.store_confidence and flat.parsed is not None:
                candidate_confidence = flat.parsed.confidence

        label_match = None
        if verdict_value is not None:
            label_match = verdict_value == ticket.label

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
            confidence=candidate_confidence,
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


__all__ = [
    "FAIL_FIRST_NEGATIVE_TRIGGERS",
    "PENDING_SIGNAL_PHRASES",
    "REMARK_SOFT_MARKER",
    "FailFirstHit",
    "MissionEvidence",
    "attach_signals",
    "extract_mission_evidence",
]
