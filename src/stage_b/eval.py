#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holdout evaluation helpers for the Stage-B reflection pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence

from .config import SelectionConfig, SignalsConfig
from .io.guidance import GuidanceRepository
from .rollout import RolloutSampler
from .selection import select_for_group
from .signals import attach_signals
from .types import GroupTicket, MissionGuidance, SelectionResult

logger = logging.getLogger(__name__)


def _evaluate_selections(selections: Sequence[SelectionResult]) -> Dict[str, float]:
    total = len(selections)
    if total == 0:
        return {
            "label_match_rate": 0.0,
            "mean_semantic_advantage": 0.0,
            "sample_size": 0.0,
        }

    label_matches = sum(1 for item in selections if item.label_match is True)
    semantic_sum = sum(item.semantic_advantage for item in selections)

    return {
        "label_match_rate": label_matches / total,
        "mean_semantic_advantage": semantic_sum / total,
        "sample_size": float(total),
    }


def evaluate_holdout(
    tickets: Sequence[GroupTicket],
    *,
    sampler: RolloutSampler,
    guidance_repo: GuidanceRepository,
    signals_config: SignalsConfig,
    selection_config: SelectionConfig,
    chunk_size: int = 32,
    guidance_override: Optional[Mapping[str, MissionGuidance]] = None,
) -> Dict[str, float]:
    """Run holdout evaluation for a set of tickets using current guidance."""

    ticket_list = list(tickets)
    if not ticket_list:
        return {
            "label_match_rate": 0.0,
            "mean_semantic_advantage": 0.0,
            "sample_size": 0.0,
        }

    if guidance_override is not None:
        guidance_map = dict(guidance_override)
    else:
        guidance_map = guidance_repo.load()
    selections: list[SelectionResult] = []

    for start in range(0, len(ticket_list), chunk_size):
        batch = ticket_list[start : start + chunk_size]
        parsed_map = sampler.generate_for_batch(batch, guidance_map)

        for ticket in batch:
            candidates = parsed_map.get(ticket.group_id, [])
            if not candidates:
                logger.warning(
                    "Holdout evaluation missing candidates for group %s (mission %s)",
                    ticket.group_id,
                    ticket.mission,
                )
                continue

            scored = attach_signals(ticket, candidates, signals_config)
            guidance = guidance_map.get(ticket.mission)
            if guidance is None:
                raise KeyError(f"Missing guidance for mission {ticket.mission}")

            selection = select_for_group(
                scored,
                guidance_step=guidance.step,
                reflection_cycle=0,
                reflection_change=None,
                config=selection_config,
            )
            selections.append(selection)

    metrics = _evaluate_selections(selections)
    logger.debug(
        "Holdout evaluation: size=%d label_match_rate=%.4f semantic=%.4f",
        int(metrics["sample_size"]),
        metrics["label_match_rate"],
        metrics["mean_semantic_advantage"],
    )
    return metrics


__all__ = ["evaluate_holdout"]
