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


class DatasetImage(TypedDict, total=False):
    type: str
    image: str


class DatasetObject(TypedDict, total=False):
    bbox_2d: Sequence[float]
    poly: Sequence[float]
    line: Sequence[float]
    desc: str
    ref: str
    label: str
    score: float
    object_id: str
    image_id: int | str
    attributes: Mapping[str, Any]
    poly_points: int
    line_points: int
    metadata: Mapping[str, Any]
    __src_geom_idx: int
    __aug_op: str


class ConversationRecord(TypedDict, total=False):
    messages: Sequence[MessageDict]
    metadata: Mapping[str, Any]
    summary: str
    width: float
    height: float
    images: Sequence[str | DatasetImage]
    objects: Sequence[DatasetObject]


class GeometryDict(TypedDict, total=False):
    bbox_2d: Sequence[float]
    poly: Sequence[float]
    line: Sequence[float]
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

    messages_seq = cast(Sequence[Mapping[str, object]], messages)
    for index, turn in enumerate(messages_seq):
        if not isinstance(turn, Mapping):
            raise ValueError(f"messages[{index}] must be a mapping")
        if "role" not in turn:
            raise ValueError(f"messages[{index}] missing 'role'")
        content_raw: object = turn.get("content", [])
        if not isinstance(content_raw, Sequence):
            raise ValueError(f"messages[{index}]['content'] must be a sequence")
    return cast(ConversationRecord, record)


def validate_geometry_sequence(
    geometries: Iterable[Mapping[str, object]],
) -> Tuple[GeometryDict, ...]:
    """Validate a sequence of geometry dictionaries.

    Expected geometry keys (exactly one per entry):
      - bbox_2d: [x1, y1, x2, y2]
      - poly:    flat [x0, y0, x1, y1, ...] (>= 3 points)
      - line:    flat [x0, y0, x1, y1, ...] (>= 2 points)

    Extra metadata keys are allowed (e.g., desc, __src_geom_idx, __aug_op).
    """
    validated: list[GeometryDict] = []
    for index, geom in enumerate(geometries):
        if not isinstance(geom, Mapping):
            raise TypeError(f"geometry[{index}] must be a mapping")

        bbox_raw = geom.get("bbox_2d")
        poly_raw = geom.get("poly")
        line_raw = geom.get("line")

        has_bbox = bbox_raw is not None
        has_poly = poly_raw is not None
        has_line = line_raw is not None
        num_geoms = int(has_bbox) + int(has_poly) + int(has_line)
        if num_geoms == 0:
            raise ValueError(
                f"geometry[{index}] must include exactly one of 'bbox_2d', 'poly', or 'line'"
            )
        if num_geoms != 1:
            raise ValueError(
                f"geometry[{index}] must include exactly one of 'bbox_2d', 'poly', or 'line'"
            )

        if has_bbox:
            bbox = bbox_raw
            if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
                raise TypeError(f"geometry[{index}].bbox_2d must be a numeric sequence")
            bbox_seq = cast(Sequence[object], bbox)
            if len(bbox_seq) != 4:
                raise ValueError(
                    f"geometry[{index}].bbox_2d must have length 4; got {len(bbox_seq)}"
                )
            for j, v in enumerate(bbox_seq):
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise TypeError(
                        f"geometry[{index}].bbox_2d[{j}] must be a number; got {type(v)!r}"
                    )

        if has_poly:
            poly = poly_raw
            if not isinstance(poly, Sequence) or isinstance(poly, (str, bytes)):
                raise TypeError(f"geometry[{index}].poly must be a numeric sequence")
            poly_seq = cast(Sequence[object], poly)
            if len(poly_seq) < 6 or len(poly_seq) % 2 != 0:
                raise ValueError(
                    f"geometry[{index}].poly must contain even-length >=6 coordinates; got {len(poly_seq)}"
                )
            for j, v in enumerate(poly_seq):
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise TypeError(
                        f"geometry[{index}].poly[{j}] must be a number; got {type(v)!r}"
                    )

        if has_line:
            line = line_raw
            if not isinstance(line, Sequence) or isinstance(line, (str, bytes)):
                raise TypeError(f"geometry[{index}].line must be a numeric sequence")
            line_seq = cast(Sequence[object], line)
            if len(line_seq) < 4 or len(line_seq) % 2 != 0:
                raise ValueError(
                    f"geometry[{index}].line must contain even-length >=4 coordinates; got {len(line_seq)}"
                )
            for j, v in enumerate(line_seq):
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise TypeError(
                        f"geometry[{index}].line[{j}] must be a number; got {type(v)!r}"
                    )

        validated.append(cast(GeometryDict, geom))
    return tuple(validated)


def clone_record(record: Mapping[str, Any]) -> MutableMapping[str, Any]:
    return cast(MutableMapping[str, Any], dict(record))
