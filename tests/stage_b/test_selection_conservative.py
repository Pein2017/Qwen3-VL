from src.stage_b.scoring.selection import select_for_group
from src.stage_b.types import (
    DecodeConfig,
    DeterministicSignals,
    GroupTicket,
    StageASummaries,
    Trajectory,
    ParsedTrajectory,
    TrajectoryWithSignals,
    CriticOutput,
)
from src.stage_b.config import SelectionConfig, ManualReviewConfig
from datetime import datetime


def make_candidate(needs_recheck: bool) -> TrajectoryWithSignals:
    base = Trajectory(
        group_id="g1",
        mission="M",
        candidate_index=0,
        decode=DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=32),
        response_text="...",
        created_at=datetime.utcnow(),
    )
    parsed = ParsedTrajectory(base=base, verdict="pass", reason="ok", confidence=0.9, format_ok=True)
    signals = DeterministicSignals(label_match=True, self_consistency=None, confidence=0.9)
    critic = CriticOutput(summary="s", critique="c", needs_recheck=needs_recheck)
    return TrajectoryWithSignals(parsed=parsed, signals=signals, critic=critic)


def test_conservative_override_to_fail_on_needs_recheck():
    ticket = GroupTicket(group_id="g1", mission="M", label="pass", summaries=StageASummaries(per_image={}))
    selection = select_for_group(
        ticket,
        [make_candidate(needs_recheck=True)],
        guidance_step=1,
        reflection_cycle=0,
        reflection_change=None,
        config=SelectionConfig(policy="top_label", tie_break="confidence"),
        manual_review=ManualReviewConfig(),
    )
    assert selection.verdict == "fail"
    assert any("needs_recheck=true" in w for w in selection.warnings)


def test_label_fail_override_blocks_pass():
    base = Trajectory(
        group_id="g2",
        mission="M",
        candidate_index=0,
        decode=DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=32),
        response_text="Verdict: 通过",
        created_at=datetime.utcnow(),
    )
    parsed = ParsedTrajectory(
        base=base,
        verdict="pass",  # type: ignore[arg-type]
        reason="看起来合规",
        confidence=0.8,
        format_ok=True,
    )
    signals = DeterministicSignals(
        label_match=False,
        self_consistency=1.0,
        confidence=0.8,
        conflict_flag=True,
    )
    ticket = GroupTicket(group_id="g2", mission="M", label="fail", summaries=StageASummaries(per_image={}))
    selection = select_for_group(
        ticket,
        [TrajectoryWithSignals(parsed=parsed, signals=signals)],
        guidance_step=1,
        reflection_cycle=0,
        reflection_change=None,
        config=SelectionConfig(policy="top_label", tie_break="confidence"),
        manual_review=ManualReviewConfig(),
    )
    assert selection.verdict == "fail"
    assert selection.conflict_flag is True
    assert any("label_fail_override" in w for w in selection.warnings)
