#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export helpers for Stage-B reflection pipeline artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from ..types import SelectionResult, TrajectoryWithSignals

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def serialize_trajectory(
    item: TrajectoryWithSignals,
    *,
    reflection_cycle: int,
    guidance_step: int,
) -> Dict[str, object]:
    base = item.parsed.base
    decode = base.decode
    signals = item.signals
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
            "text": base.response_text,
            "verdict": item.parsed.verdict,
            "reason": item.parsed.reason,
            "confidence": item.parsed.confidence,
            "format_ok": item.parsed.format_ok,
            "created_at": base.created_at.isoformat(),
            "signals": {
                "label_match": signals.label_match,
                "self_consistency": signals.self_consistency,
                "confidence": signals.confidence,
                "conflict_flag": signals.conflict_flag,
                "needs_manual_review": signals.needs_manual_review,
            },
            "critic": (
                {
                    "summary": item.critic.summary,
                    "critique": item.critic.critique,
                    "verdict": item.critic.verdict,
                    "needs_recheck": item.critic.needs_recheck,
                    "evidence_sufficiency": item.critic.evidence_sufficiency,
                    "recommended_action": item.critic.recommended_action,
                }
                if item.critic
                else None
            ),
            "warnings": list(item.warnings),
        },
    }


def serialize_selection(item: SelectionResult) -> Dict[str, object]:
    return {
        "group_id": item.group_id,
        "mission": item.mission,
        "result": {
            "verdict": item.verdict,
            "reason": item.reason,
            "confidence": item.confidence,
            "label_match": item.label_match,
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
