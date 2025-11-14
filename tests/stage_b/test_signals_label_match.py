#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as _dt

from src.stage_b.signals import attach_signals
from src.stage_b.config import SignalsConfig
from src.stage_b.types import (
    DecodeConfig,
    GroupTicket,
    ParsedTrajectory,
    StageASummaries,
    Trajectory,
)


def _mk_candidate(group_id: str, mission: str, verdict, confidence=0.9) -> ParsedTrajectory:
    base = Trajectory(
        group_id=group_id,
        mission=mission,
        candidate_index=0,
        decode=DecodeConfig(temperature=0.2, top_p=0.9, max_new_tokens=16),
        response_text="",
        created_at=_dt.datetime.now(),
    )
    return ParsedTrajectory(
        base=base,
        verdict=verdict,  # Allow Chinese verdict to exercise normalization path
        reason="测试",
        confidence=confidence,
        format_ok=True,
    )


def test_label_match_uses_normalize_verdict_pass_case():
    ticket = GroupTicket(
        group_id="g1",
        mission="m1",
        label="pass",
        summaries=StageASummaries(per_image={}),
    )
    candidates = [
        _mk_candidate("g1", "m1", verdict="通过"),
    ]
    cfg = SignalsConfig(store_confidence=True, enable_consistency=True, weights=None)

    annotated = attach_signals(ticket, candidates, cfg)

    assert annotated[0].signals.label_match is True


def test_label_match_uses_normalize_verdict_fail_case():
    ticket = GroupTicket(
        group_id="g2",
        mission="m1",
        label="pass",
        summaries=StageASummaries(per_image={}),
    )
    candidates = [
        _mk_candidate("g2", "m1", verdict="不通过"),
    ]
    cfg = SignalsConfig(store_confidence=False, enable_consistency=False, weights=None)

    annotated = attach_signals(ticket, candidates, cfg)

    assert annotated[0].signals.label_match is False

