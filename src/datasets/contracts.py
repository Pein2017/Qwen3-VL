"""Shared dataset contracts and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)


class MessageContent(TypedDict, total=False):
    type: str
    text: str
    image: Any


class MessageDict(TypedDict, total=False):
    role: str
    content: Sequence[MessageContent]


class ConversationRecord(TypedDict, total=False):
    messages: Sequence[MessageDict]
    metadata: Mapping[str, Any]


class GeometryDict(TypedDict, total=False):
    bbox: Sequence[float]
    polygon: Sequence[float]
    label: str
    score: float
    object_id: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class AugmentationTelemetry:
    kept_indices: Tuple[int, ...]
    coverages: Tuple[float, ...]
    allows_geometry_drops: bool
    width: Optional[int]
    height: Optional[int]
    padding_ratio: Optional[float]
    skip_reason: Optional[str]
    skip_counts: Mapping[str, int]


def validate_conversation_record(record: Mapping[str, Any]) -> ConversationRecord:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    messages = record.get("messages")
    if messages is None:
        # Raw dense-caption records (images/objects/summary) are accepted as-is.
        return cast(ConversationRecord, record)

    if not isinstance(messages, Sequence):
        raise ValueError("conversation record 'messages' must be a sequence")

    for index, turn in enumerate(messages):
        if not isinstance(turn, Mapping):
            raise ValueError(f"messages[{index}] must be a mapping")
        if "role" not in turn:
            raise ValueError(f"messages[{index}] missing 'role'")
        content = turn.get("content", [])
        if not isinstance(content, Sequence):
            raise ValueError(f"messages[{index}]['content'] must be a sequence")
    return cast(ConversationRecord, record)


def validate_geometry_sequence(
    geometries: Iterable[Mapping[str, Any]],
) -> Tuple[GeometryDict, ...]:
    validated: list[GeometryDict] = []
    for index, geom in enumerate(geometries):
        if not isinstance(geom, Mapping):
            raise ValueError(f"geometry[{index}] must be a mapping")
        bbox = geom.get("bbox")
        if bbox is not None and not isinstance(bbox, Sequence):
            raise ValueError(
                f"geometry[{index}]['bbox'] must be a sequence if provided"
            )
        polygon = geom.get("polygon")
        if polygon is not None and not isinstance(polygon, Sequence):
            raise ValueError(
                f"geometry[{index}]['polygon'] must be a sequence if provided"
            )
        validated.append(cast(GeometryDict, geom))
    return tuple(validated)


def clone_record(record: Mapping[str, Any]) -> MutableMapping[str, Any]:
    return cast(MutableMapping[str, Any], dict(record))
