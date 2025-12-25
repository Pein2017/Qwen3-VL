#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed value objects supporting the Stage-B rule-search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Mapping, Sequence
from typing import Literal, cast

ExperienceOperationKind = Literal["upsert", "update", "remove", "merge"]


@dataclass(frozen=True)
class ExperienceMetadata:
    """Provenance for a single guidance experience entry."""

    updated_at: datetime
    reflection_id: str
    sources: tuple[str, ...] = field(default_factory=tuple)
    rationale: str | None = None
    # Lifecycle management fields
    hit_count: int = 0
    miss_count: int = 0
    confidence: float = 1.0

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "updated_at": self.updated_at.isoformat(),
            "reflection_id": self.reflection_id,
            "sources": list(self.sources),
        }
        if self.rationale:
            payload["rationale"] = self.rationale
        # Lifecycle fields
        payload["hit_count"] = self.hit_count
        payload["miss_count"] = self.miss_count
        payload["confidence"] = self.confidence
        return payload

    @staticmethod
    def from_payload(payload: Mapping[str, object]) -> "ExperienceMetadata":
        updated_raw = payload.get("updated_at")
        if not isinstance(updated_raw, str):
            raise ValueError("experience metadata updated_at must be ISO string")
        updated_at = datetime.fromisoformat(updated_raw)

        reflection_raw = payload.get("reflection_id")
        if not isinstance(reflection_raw, str) or not reflection_raw.strip():
            raise ValueError(
                "experience metadata reflection_id must be non-empty string"
            )

        sources_raw = payload.get("sources", [])
        if isinstance(sources_raw, Sequence) and not isinstance(
            sources_raw, (str, bytes)
        ):
            sources = tuple(str(item) for item in sources_raw)
        else:
            sources = ()

        rationale_raw = payload.get("rationale")
        rationale = (
            str(rationale_raw).strip()
            if isinstance(rationale_raw, str) and rationale_raw.strip()
            else None
        )

        # Parse lifecycle fields with defaults for backward compatibility
        hit_count_raw = payload.get("hit_count", 0)
        hit_count = int(hit_count_raw) if isinstance(hit_count_raw, (int, float)) else 0

        miss_count_raw = payload.get("miss_count", 0)
        miss_count = (
            int(miss_count_raw) if isinstance(miss_count_raw, (int, float)) else 0
        )

        confidence_raw = payload.get("confidence", 1.0)
        confidence = (
            float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 1.0
        )

        return ExperienceMetadata(
            updated_at=updated_at,
            reflection_id=reflection_raw.strip(),
            sources=sources,
            rationale=rationale,
            hit_count=hit_count,
            miss_count=miss_count,
            confidence=confidence,
        )


@dataclass(frozen=True)
class ExperienceOperation:
    """Incremental edit for the mission guidance experience list."""

    op: ExperienceOperationKind
    key: str | None
    text: str | None
    rationale: str | None
    evidence: tuple[str, ...] = field(default_factory=tuple)
    merged_from: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HypothesisCandidate:
    """Candidate hypothesis proposed by the reflection ops pass."""

    text: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    falsifier: str | None = None
    dimension: str | None = None


GroupLabel = Literal["pass", "fail"]
ChineseVerdict = Literal["通过", "不通过"]


class ReflectionAction(str):
    """String subclass for reflection actions ('refine' or 'noop')."""

    pass


@dataclass(frozen=True)
class StageASummaries:
    """Normalized Stage-A per-image summaries."""

    per_image: Mapping[str, str]

    def as_dict(self) -> dict[str, str]:
        return dict(self.per_image)


@dataclass(frozen=True)
class LabelProvenance:
    """Describes how a group label was obtained."""

    source: str | None = None
    timestamp: datetime | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class GroupTicket:
    """Single Stage-B evaluation unit derived from Stage-A outputs."""

    group_id: str
    mission: str
    label: GroupLabel
    summaries: StageASummaries
    provenance: LabelProvenance | None = None
    uid: str | None = None

    @property
    def key(self) -> str:
        """Stable unique key for internal bookkeeping.

        Some datasets may contain duplicate `group_id` with different labels.
        Stage-B therefore treats `(group_id, label)` as the unique identity and
        uses `ticket_key = "{group_id}::{label}"` in artifacts and reflection.
        """
        return self.uid or f"{self.group_id}::{self.label}"


@dataclass(frozen=True)
class MissionGuidance:
    """Mission-level guidance entries with step metadata."""

    mission: str
    experiences: dict[str, str]
    step: int
    updated_at: datetime
    metadata: dict[str, ExperienceMetadata] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "step": self.step,
            "updated_at": self.updated_at.isoformat(),
            "experiences": self.experiences,
        }
        if self.metadata:
            payload["metadata"] = cast(
                object, {key: meta.to_payload() for key, meta in self.metadata.items()}
            )
        return payload


@dataclass(frozen=True)
class DecodeConfig:
    """Sampling configuration for a single rollout attempt."""

    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int | None = None
    stop: tuple[str, ...] = field(default_factory=tuple)
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int | None = None


@dataclass(frozen=True)
class Trajectory:
    """Raw model response captured during rollout."""

    group_id: str
    mission: str
    candidate_index: int
    decode: DecodeConfig
    response_text: str
    created_at: datetime


@dataclass(frozen=True)
class ParsedTrajectory:
    """Trajectory parsed into structured fields."""

    base: Trajectory
    verdict: GroupLabel | None
    reason: str | None
    format_ok: bool
    confidence: float | None = None


@dataclass(frozen=True)
class DeterministicSignals:
    """Minimal signals retained for compatibility (not used for selection)."""

    label_match: bool | None
    self_consistency: float | None
    conflict_flag: bool = False
    needs_manual_review: bool = False
    vote_strength: float | None = None
    low_agreement: bool = False
    confidence: float | None = None


@dataclass(frozen=True)
class ExperienceCandidate:
    """Candidate summary included in an experience bundle."""

    candidate_index: int
    verdict: GroupLabel | None
    reason: str | None
    signals: DeterministicSignals
    confidence: float | None = None
    # Critic insights (populated from CriticOutput when available)
    summary: str | None = None
    critique: str | None = None
    raw_text: str | None = None


@dataclass(frozen=True)
class ExperienceRecord:
    """Per-group record captured for reflection."""

    ticket: GroupTicket
    candidates: tuple[ExperienceCandidate, ...]
    winning_candidate: int | None
    guidance_step: int
    epoch_step: int | None = None
    global_step: int | None = None


@dataclass(frozen=True)
class ExperienceBundle:
    """Aggregated batch passed to the reflection LLM."""

    mission: str
    records: tuple[ExperienceRecord, ...]
    reflection_cycle: int
    guidance_step: int


@dataclass(frozen=True)
class ReflectionProposal:
    """Reflection LLM proposal for a guidance update."""

    action: ReflectionAction
    summary: str | None
    critique: str | None
    operations: tuple[ExperienceOperation, ...]
    evidence_group_ids: tuple[str, ...]
    hypotheses: tuple[HypothesisCandidate, ...] = field(default_factory=tuple)
    uncertainty_note: str | None = None
    # Ticket keys that the reflection model declares "no evidence / cannot explain"
    # even after seeing the GT label. These are candidates for stop-gradient review.
    no_evidence_group_ids: tuple[str, ...] = field(default_factory=tuple)
    text: str | None = None


@dataclass(frozen=True)
class ReflectionOutcome:
    """Result of applying (or not) a reflection proposal."""

    reflection_id: str
    mission: str
    proposal: ReflectionProposal
    applied: bool
    guidance_step_before: int
    guidance_step_after: int
    operations: tuple[ExperienceOperation, ...]
    eligible: bool
    applied_epoch: int | None = None
    ineligible_reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "ChineseVerdict",
    "DecodeConfig",
    "DeterministicSignals",
    "ExperienceMetadata",
    "ExperienceOperation",
    "ExperienceOperationKind",
    "HypothesisCandidate",
    "ExperienceBundle",
    "ExperienceCandidate",
    "ExperienceRecord",
    "GroupLabel",
    "GroupTicket",
    "LabelProvenance",
    "MissionGuidance",
    "ParsedTrajectory",
    "ReflectionAction",
    "ReflectionOutcome",
    "ReflectionProposal",
    "StageASummaries",
    "Trajectory",
]
