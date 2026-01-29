"""Typed payloads for Stage-A JSONL records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, cast


class StageAGroupRecord(TypedDict):
    """Stage-A JSONL record for a single group."""

    group_id: str
    mission: str
    label: str
    images: list[str]
    per_image: dict[str, str]


def validate_stage_a_group_record(
    value: object, *, context: str = "stage_a"
) -> StageAGroupRecord:
    """Validate a Stage-A JSONL record at the boundary.

    This is a strict contract for Stage-A outputs (group-level JSONL rows).

    Raises:
        TypeError: for type mismatches
        ValueError: for missing keys / invalid values
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")

    value_map = cast(Mapping[str, object], value)

    group_id_raw = value_map.get("group_id")
    mission_raw = value_map.get("mission")
    label_raw = value_map.get("label")
    images_raw = value_map.get("images")
    per_image_raw = value_map.get("per_image")

    if group_id_raw is None:
        raise ValueError(f"{context}.group_id is required")
    if mission_raw is None:
        raise ValueError(f"{context}.mission is required")
    if label_raw is None:
        raise ValueError(f"{context}.label is required")
    if images_raw is None:
        raise ValueError(f"{context}.images is required")
    if per_image_raw is None:
        raise ValueError(f"{context}.per_image is required")

    if not isinstance(group_id_raw, str):
        raise TypeError(f"{context}.group_id must be a string")
    if not isinstance(mission_raw, str):
        raise TypeError(f"{context}.mission must be a string")
    if not isinstance(label_raw, str):
        raise TypeError(f"{context}.label must be a string")

    if not isinstance(images_raw, Sequence) or isinstance(images_raw, (str, bytes)):
        raise TypeError(f"{context}.images must be a list of strings")
    images_seq = cast(Sequence[object], images_raw)
    images: list[str] = []
    for idx, item in enumerate(images_seq):
        if not isinstance(item, str):
            raise TypeError(f"{context}.images[{idx}] must be a string")
        images.append(item)
    if len(images) < 1:
        raise ValueError(f"{context}.images must contain at least 1 item")

    if not isinstance(per_image_raw, Mapping):
        raise TypeError(f"{context}.per_image must be a mapping")
    per_image_map = cast(Mapping[object, object], per_image_raw)

    expected = {f"image_{i}" for i in range(1, len(images) + 1)}
    keys: set[str] = set()
    per_image: dict[str, str] = {}
    for raw_key, raw_value in per_image_map.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{context}.per_image keys must be strings")
        keys.add(raw_key)
        if not isinstance(raw_value, str):
            raise TypeError(f"{context}.per_image.{raw_key} must be a string")
        if not raw_value.strip():
            raise ValueError(f"{context}.per_image.{raw_key} must be non-empty")
        per_image[raw_key] = raw_value

    if keys != expected:
        raise ValueError(
            f"{context}.per_image keys must be exactly {sorted(expected)}; got {sorted(keys)}"
        )

    record: StageAGroupRecord = {
        "group_id": group_id_raw,
        "mission": mission_raw,
        "label": label_raw,
        "images": images,
        "per_image": per_image,
    }
    return cast(StageAGroupRecord, record)


__all__ = ["StageAGroupRecord", "validate_stage_a_group_record"]
