#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-B ingestion utilities for rule-search pipeline."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

from ..types import GroupLabel, GroupTicket, LabelProvenance, StageASummaries

logger = logging.getLogger(__name__)


_KEY_RE = re.compile(r"(\d+)$")


def _normalize_key(raw_key: str, fallback_index: int) -> Tuple[str, int]:
    match = _KEY_RE.search(raw_key)
    if match:
        try:
            idx = int(match.group(1))
        except ValueError:
            idx = fallback_index
    else:
        idx = fallback_index
    return f"image_{idx}", idx


def _maybe_unwrap_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, Mapping):
            values = [str(value) for value in parsed.values()]
            if values:
                return "，".join(values)
    return text


def _normalize_per_image(per_image: Mapping[str, object]) -> Dict[str, str]:
    normalized: Dict[int, str] = {}
    for fallback_index, (raw_key, raw_value) in enumerate(per_image.items(), start=1):
        _, index = _normalize_key(str(raw_key), fallback_index)
        text = _maybe_unwrap_json(str(raw_value))
        normalized[index] = text
    ordered = {f"image_{idx}": normalized[idx] for idx in sorted(normalized)}
    return ordered


def _parse_label(raw_label: str) -> GroupLabel:
    lowered = raw_label.strip().lower()
    if lowered not in {"pass", "fail"}:
        raise ValueError(f"Unsupported label value: {raw_label!r}")
    return cast(GroupLabel, lowered)


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def ingest_stage_a(
    stage_a_paths: Sequence[str | Path],
) -> Sequence[GroupTicket]:
    """Load Stage-A JSONL outputs into ticket objects for Stage-B."""
    records: list[GroupTicket] = []

    for path_like in stage_a_paths:
        path = Path(path_like)
        if not path.exists():
            logger.warning(f"Skipping missing Stage-A file: {path}")
            continue

        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload: MutableMapping[str, object] = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.error(
                        f"Invalid JSON at {path.name}:{line_number}: {exc}"
                    )
                    continue

                try:
                    mission = str(payload["mission"])
                    group_id = str(payload["group_id"])
                    label = _parse_label(str(payload["label"]))

                    per_image_raw = payload.get("per_image")
                    if not isinstance(per_image_raw, Mapping):
                        raise TypeError("per_image field must be an object")
                    normalized = _normalize_per_image(per_image_raw)
                    if not normalized:
                        raise ValueError("per_image summaries must be non-empty")

                    timestamp: Optional[datetime] = None
                    if isinstance(payload.get("label_timestamp"), str):
                        timestamp = _parse_timestamp(
                            str(payload.get("label_timestamp"))
                        )

                    provenance = LabelProvenance(
                        source=str(payload.get("label_source", "human")),
                        timestamp=timestamp,
                        metadata=None,
                    )

                    ticket = GroupTicket(
                        group_id=group_id,
                        mission=mission,
                        label=label,
                        summaries=StageASummaries(per_image=normalized),
                        provenance=provenance,
                        uid=f"{group_id}::{label}",
                    )
                    records.append(ticket)
                except Exception as exc:  # noqa: BLE001 - contextual logging
                    logger.error(
                        f"Failed to parse Stage-A record at {path.name}:{line_number} ({exc})"
                    )

    if not records:
        raise RuntimeError("No Stage-A records were ingested")

    return records


__all__ = ["ingest_stage_a"]
