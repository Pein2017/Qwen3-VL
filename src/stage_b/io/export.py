#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export helpers for Stage-B reflection pipeline artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from ..types import SelectionResult, TrajectoryWithSignals
from ..utils.chinese import normalize_spaces, to_simplified

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def serialize_trajectory(
    item: TrajectoryWithSignals,
    *,
    reflection_cycle: int,
    guidance_step: int,
) -> Dict[str, object]:
    if item.parsed is None:
        raise ValueError("Cannot serialize trajectory with None parsed field")

    base = item.parsed.base
    decode = base.decode
    signals = item.signals

    # Normalize text and reason fields as safety net
    normalized_text = base.response_text
    if normalized_text:
        normalized_text = to_simplified(normalized_text)
        normalized_text = normalize_spaces(normalized_text)

    normalized_reason = item.parsed.reason
    if normalized_reason:
        normalized_reason = to_simplified(normalized_reason)
        normalized_reason = normalize_spaces(normalized_reason)

    return {
        "group_id": base.group_id,
        "mission": base.mission,
        "reflection_cycle": reflection_cycle,
        "guidance_step": guidance_step,
        "result": {
            "candidate_index": base.candidate_index,
            "decode": {
                "temperature": decode.temperature,
                "top_p": decode.top_p,
                "max_new_tokens": decode.max_new_tokens,
                "seed": decode.seed,
                "stop": list(decode.stop),
            },
            "text": normalized_text,
            "verdict": item.parsed.verdict,
            "reason": normalized_reason,
            "format_ok": item.parsed.format_ok,
            "created_at": base.created_at.isoformat(),
            "warnings": list(item.warnings),
            "label_match": signals.label_match if signals else False,
        },
    }


def serialize_selection(item: SelectionResult) -> Dict[str, object]:
    label_match = bool(item.label_match) if item.label_match is not None else False

    # Normalize reason field as safety net
    normalized_reason = item.reason
    if normalized_reason:
        normalized_reason = to_simplified(normalized_reason)
        normalized_reason = normalize_spaces(normalized_reason)

    return {
        "group_id": item.group_id,
        "mission": item.mission,
        "result": {
            "verdict": item.verdict,
            "reason": normalized_reason,
            "vote_strength": item.vote_strength,
            "label_match": label_match,
            "selected_candidate": item.selected_candidate,
            "guidance_step": item.guidance_step,
            "reflection_change": item.reflection_change,
            "reflection_cycle": item.reflection_cycle,
            "manual_review_recommended": item.manual_review_recommended,
            "eligible": item.eligible,
            "ineligible_reason": item.ineligible_reason,
            "warnings": list(item.warnings),
            "conflict_flag": item.conflict_flag,
            "needs_manual_review": item.needs_manual_review,
        },
    }


def export_selections(
    selections: Iterable[SelectionResult],
    *,
    jsonl_path: str | Path,
) -> Path:
    items: List[SelectionResult] = list(selections)
    if not items:
        raise ValueError("No selections to export")

    jsonl_target = Path(jsonl_path)
    _ensure_parent(jsonl_target)
    with jsonl_target.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(serialize_selection(item), ensure_ascii=False))
            fh.write("\n")

    logger.info(f"Exported {len(items)} selections to {jsonl_target}")
    return jsonl_target


def export_trajectories(
    trajectories: Iterable[TrajectoryWithSignals],
    *,
    path: str | Path,
    reflection_cycle: int,
    guidance_step: int,
) -> Path:
    target = Path(path)
    _ensure_parent(target)
    items = list(trajectories)
    with target.open("w", encoding="utf-8") as fh:
        for item in items:
            payload = serialize_trajectory(
                item,
                reflection_cycle=reflection_cycle,
                guidance_step=guidance_step,
            )
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")
    logger.info(f"Saved {len(items)} trajectories to {target}")
    return target


__all__ = [
    "export_trajectories",
    "export_selections",
    "serialize_selection",
    "serialize_trajectory",
]
