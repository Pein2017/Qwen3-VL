"""Typed payloads for Stage-A JSONL records."""

from __future__ import annotations

from typing import TypedDict


class StageAGroupRecord(TypedDict, total=False):
    """Stage-A JSONL record for a single group."""

    group_id: str
    mission: str
    label: str
    images: list[str]
    per_image: dict[str, str]
