"""Shared utilities for dataset operations"""

import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import cast

from .contracts import ConversationRecord, DatasetObject, validate_conversation_record


def load_jsonl(
    jsonl_path: str, *, resolve_relative: bool = False
) -> list[ConversationRecord]:
    """Load records from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        resolve_relative: If True, resolves non-absolute image paths against the
            JSONL's parent directory and stores absolute paths in-memory.

    Returns:
        List of dictionaries, one per line
    """
    base_dir = Path(jsonl_path).resolve().parent
    records: list[ConversationRecord] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, MutableMapping):
                raise TypeError("JSONL record must be an object")
            if resolve_relative:
                images = record.get("images")
                if isinstance(images, list):
                    resolved: list[object] = []
                    for img in images:
                        if isinstance(img, str):
                            img_path = Path(img)
                            resolved_path = (
                                img_path
                                if img_path.is_absolute()
                                else (base_dir / img_path).resolve()
                            )
                            resolved.append(str(resolved_path))
                        else:
                            resolved.append(img)
                    record["images"] = resolved
            validated = validate_conversation_record(record)
            records.append(cast(ConversationRecord, validated))
    return records


def extract_object_points(obj: DatasetObject) -> tuple[str, list[float]]:
    """Extract geometry type and points from an object.

    Args:
    obj: Object dictionary containing geometry (bbox_2d, poly, or line)

    Returns:
        Tuple of (geometry_type, points_list)
    """
    if "quad" in obj:
        raise ValueError(
            "quad geometry is deprecated; replace with 'poly' (flat list of x,y)."
        )
    bbox = obj.get("bbox_2d")
    if bbox is not None:
        return "bbox_2d", [float(v) for v in bbox]
    poly = obj.get("poly")
    if poly is not None:
        return "poly", [float(v) for v in poly]
    line = obj.get("line")
    if line is not None:
        return "line", [float(v) for v in line]
    return "", []


def extract_geometry(obj: DatasetObject) -> dict[str, list[float]]:
    """Extract geometry dictionary from object.

    Useful for augmentation and processing pipelines.

    Args:
        obj: Object dictionary

    Returns:
        Dictionary with geometry key and points
    """
    geom: dict[str, list[float]] = {}
    bbox = obj.get("bbox_2d")
    if bbox is not None:
        geom["bbox_2d"] = [float(v) for v in bbox]
    poly = obj.get("poly")
    if poly is not None:
        geom["poly"] = [float(v) for v in poly]
    line = obj.get("line")
    if line is not None:
        geom["line"] = [float(v) for v in line]
    return geom


def is_same_record(record_a: ConversationRecord, record_b: ConversationRecord) -> bool:
    """Check if two records are the same (identity check).

    Args:
        record_a: First record
        record_b: Second record

    Returns:
        True if records are the same object
    """
    return record_a is record_b


def extract_assistant_text(messages: Sequence[Mapping[str, object]]) -> str | None:
    """Extract the assistant text payload from a message list.

    This is a best-effort helper used for debugging/telemetry (e.g., GRPO rollout dumps).
    It intentionally supports the mixed content schemas used by multimodal chat:
    - content as a raw string
    - content as a list of {type: ..., text: ...} items
    """

    for msg in reversed(messages):
        role = msg.get("role")
        if role != "assistant":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, Mapping):
            text = content.get("text")
            if isinstance(text, str):
                return text
            continue

        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    return text
            continue

    return None


__all__ = [
    "load_jsonl",
    "extract_object_points",
    "extract_geometry",
    "is_same_record",
    "extract_assistant_text",
]
