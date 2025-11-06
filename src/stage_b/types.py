#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed value objects supporting the Stage-B reflection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple
ExperienceOperationKind = Literal["upsert", "remove"]


@dataclass(frozen=True)
class ExperienceMetadata:
    """Provenance for a single guidance experience entry."""

    updated_at: datetime
    reflection_id: str
    sources: Tuple[str, ...] = field(default_factory=tuple)
    rationale: Optional[str] = None

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "updated_at": self.updated_at.isoformat(),
            "reflection_id": self.reflection_id,
            "sources": list(self.sources),
        }
        if self.rationale:
            payload["rationale"] = self.rationale
        return payload

    @staticmethod
    def from_payload(payload: Mapping[str, object]) -> "ExperienceMetadata":
        updated_raw = payload.get("updated_at")
        if not isinstance(updated_raw, str):
            raise ValueError("experience metadata updated_at must be ISO string")
        updated_at = datetime.fromisoformat(updated_raw)

        reflection_raw = payload.get("reflection_id")
        if not isinstance(reflection_raw, str) or not reflection_raw.strip():
            raise ValueError("experience metadata reflection_id must be non-empty string")

        sources_raw = payload.get("sources", [])
        if isinstance(sources_raw, Sequence) and not isinstance(sources_raw, (str, bytes)):
            sources = tuple(str(item) for item in sources_raw)
        else:
            sources = ()

        rationale_raw = payload.get("rationale")
        rationale = (
            str(rationale_raw).strip()
            if isinstance(rationale_raw, str) and rationale_raw.strip()
            else None
        )

        return ExperienceMetadata(
            updated_at=updated_at,
            reflection_id=reflection_raw.strip(),
            sources=sources,
            rationale=rationale,
        )


@dataclass(frozen=True)
class ExperienceOperation:
    """Incremental edit for the mission guidance experience list."""

    op: ExperienceOperationKind
    key: Optional[str]
    text: Optional[str]
    rationale: Optional[str]
    evidence: Tuple[str, ...] = field(default_factory=tuple)

GroupLabel = Literal["pass", "fail"]
ChineseVerdict = Literal["通过", "不通过"]
ReflectionAction = Literal["refine", "noop"]


@dataclass(frozen=True)
class StageASummaries:
    """Normalized Stage-A per-image summaries."""

    per_image: Mapping[str, str]

    def as_dict(self) -> Dict[str, str]:
        return dict(self.per_image)


@dataclass(frozen=True)
class LabelProvenance:
    """Describes how a group label was obtained."""

    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class GroupTicket:
    """Single Stage-B evaluation unit derived from Stage-A outputs."""

    group_id: str
    mission: str
    label: GroupLabel
    summaries: StageASummaries
    provenance: Optional[LabelProvenance] = None


@dataclass(frozen=True)
class MissionGuidance:
    """Mission-level guidance entries with step metadata."""

    mission: str
    focus: Optional[str]
    experiences: Dict[str, str]
    step: int
    updated_at: datetime
    metadata: Dict[str, ExperienceMetadata] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        payload = {
            "step": self.step,
            "updated_at": self.updated_at.isoformat(),
            "experiences": self.experiences,
        }
        if self.focus is not None:
            payload["focus"] = self.focus
        if self.metadata:
            payload["metadata"] = {
                key: meta.to_payload() for key, meta in self.metadata.items()
            }
        return payload


@dataclass(frozen=True)
class DecodeConfig:
    """Sampling configuration for a single rollout attempt."""

    temperature: float
    top_p: float
    max_new_tokens: int
    seed: Optional[int] = None
    stop: Tuple[str, ...] = field(default_factory=tuple)


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
    verdict: Optional[GroupLabel]
    reason: Optional[str]
    confidence: Optional[float]
    format_ok: bool


@dataclass(frozen=True)
class DeterministicSignals:
    """Deterministic sidecar metrics used for reflection context."""

    label_match: Optional[bool]
    self_consistency: Optional[float]
    candidate_agreement: Optional[bool]
    confidence: Optional[float]
    label_trust: Optional[float]
    semantic_advantage: float


@dataclass(frozen=True)
class TrajectoryWithSignals:
    """Parsed trajectory bundled with deterministic signals."""

    parsed: ParsedTrajectory
    signals: DeterministicSignals
    summary: Optional[str] = None
    critique: Optional[str] = None


@dataclass(frozen=True)
class SelectionResult:
    """Final verdict exported for a group ticket."""

    group_id: str
    mission: str
    verdict: GroupLabel
    reason: str
    confidence: Optional[float]
    label_match: Optional[bool]
    semantic_advantage: float
    selected_candidate: int
    guidance_step: int
    reflection_change: Optional[str]
    reflection_cycle: int
    summary: Optional[str] = None
    critique: Optional[str] = None


@dataclass(frozen=True)
class ExperienceCandidate:
    """Candidate summary included in an experience bundle."""

    candidate_index: int
    verdict: Optional[GroupLabel]
    reason: Optional[str]
    confidence: Optional[float]
    signals: DeterministicSignals


@dataclass(frozen=True)
class ExperienceRecord:
    """Per-group record captured for reflection."""

    ticket: GroupTicket
    candidates: Tuple[ExperienceCandidate, ...]
    winning_candidate: Optional[int]
    guidance_step: int


@dataclass(frozen=True)
class ExperienceBundle:
    """Aggregated batch passed to the reflection LLM."""

    mission: str
    records: Tuple[ExperienceRecord, ...]
    reflection_cycle: int
    guidance_step: int


@dataclass(frozen=True)
class ReflectionProposal:
    """Reflection LLM proposal for a guidance update."""

    action: ReflectionAction
    summary: Optional[str]
    critique: Optional[str]
    operations: Tuple[ExperienceOperation, ...]
    evidence_group_ids: Tuple[str, ...]
    uncertainty_note: Optional[str] = None
    text: Optional[str] = None


@dataclass(frozen=True)
class ReflectionOutcome:
    """Result of applying (or not) a reflection proposal."""

    reflection_id: str
    mission: str
    proposal: ReflectionProposal
    applied: bool
    pre_uplift: float
    post_uplift: float
    guidance_step_before: int
    guidance_step_after: int
    operations: Tuple[ExperienceOperation, ...]
    eligible: bool
    ineligible_reason: Optional[str] = None


__all__ = [
    "ChineseVerdict",
    "DecodeConfig",
    "DeterministicSignals",
    "ExperienceMetadata",
    "ExperienceOperation",
    "ExperienceOperationKind",
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
    "SelectionResult",
    "StageASummaries",
    "Trajectory",
    "TrajectoryWithSignals",
]
