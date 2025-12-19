from __future__ import annotations

from datetime import datetime, timezone

from src.stage_b.config import ManualReviewConfig, SelectionConfig
from src.stage_b.scoring.selection import select_for_group
from src.stage_b.types import (
    DecodeConfig,
    DeterministicSignals,
    GroupTicket,
    ParsedTrajectory,
    StageASummaries,
    Trajectory,
    TrajectoryWithSignals,
)


def _mk_candidate(
    *,
    group_id: str,
    mission: str,
    candidate_index: int,
    verdict: str,
    reason: str,
) -> TrajectoryWithSignals:
    parsed = ParsedTrajectory(
        base=Trajectory(
            group_id=group_id,
            mission=mission,
            candidate_index=candidate_index,
            decode=DecodeConfig(temperature=1.0, top_p=0.95, max_new_tokens=64),
            response_text=f"Verdict: {verdict}\nReason: {reason}",
            created_at=datetime.now(timezone.utc),
        ),
        verdict=verdict,  # type: ignore[arg-type]
        reason=reason,
        format_ok=True,
    )
    return TrajectoryWithSignals(
        parsed=parsed,
        signals=DeterministicSignals(label_match=None, self_consistency=None),
    )


def test_select_sets_needs_manual_review_on_low_agreement_and_contradiction() -> None:
    ticket = GroupTicket(
        group_id="QC-001",
        mission="m",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "摘要"}),
    )

    candidates = [
        _mk_candidate(group_id=ticket.group_id, mission=ticket.mission, candidate_index=0, verdict="pass", reason="ok"),
        _mk_candidate(group_id=ticket.group_id, mission=ticket.mission, candidate_index=1, verdict="fail", reason="bad"),
    ]

    result = select_for_group(
        ticket,
        candidates,
        mission_g0=None,
        guidance_step=1,
        reflection_cycle=0,
        reflection_change=None,
        config=SelectionConfig(policy="majority_vote", tie_break="fail_first"),
        manual_review=ManualReviewConfig(min_verdict_agreement=0.8),
    )

    assert result.needs_manual_review is True


def test_select_sets_needs_manual_review_on_pending_signal_hit() -> None:
    ticket = GroupTicket(
        group_id="QC-002",
        mission="m",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "无法确认"}),
    )

    candidates = [
        _mk_candidate(group_id=ticket.group_id, mission=ticket.mission, candidate_index=0, verdict="pass", reason="ok"),
    ]

    result = select_for_group(
        ticket,
        candidates,
        mission_g0=None,
        guidance_step=1,
        reflection_cycle=0,
        reflection_change=None,
        config=SelectionConfig(policy="majority_vote", tie_break="fail_first"),
        manual_review=ManualReviewConfig(min_verdict_agreement=0.8),
    )

    assert result.needs_manual_review is True

