#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone

from src.stage_b import build_messages as build_messages_from_pkg
from src.stage_b.sampling.prompts import build_messages as build_messages_canonical
from src.stage_b.types import GroupTicket, MissionGuidance, StageASummaries


def test_build_messages_reexport_and_canonical_equivalence():
    ticket = GroupTicket(
        group_id="QC-UNIFY-001",
        mission="挡风板安装检查",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "BBU设备存在"}),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        experiences={"G0": "若挡风板缺失则判定不通过"},
        step=1,
        updated_at=datetime.now(timezone.utc),
        metadata={},
    )

    # Ensure __init__ re-exports the canonical builder
    assert build_messages_from_pkg is build_messages_canonical

    m1 = build_messages_from_pkg(ticket, guidance)
    m2 = build_messages_canonical(ticket, guidance)

    assert isinstance(m1, list) and isinstance(m2, list)
    assert m1 == m2
    assert m1[0]["role"] == "system" and m1[1]["role"] == "user"
    assert m1[0]["content"] and m1[1]["content"]
