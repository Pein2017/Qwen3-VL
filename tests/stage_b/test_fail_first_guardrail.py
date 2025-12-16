"""Unit tests for Stage-B mission-scoped fail-first guardrails and strict output protocol."""

from datetime import datetime, timezone

from src.stage_b.config import ManualReviewConfig, SelectionConfig
from src.stage_b.rollout import _parse_two_line_response
from src.stage_b.scoring.selection import select_for_group
from src.stage_b.signals import FAIL_FIRST_NEGATIVE_TRIGGERS, extract_mission_evidence
from src.stage_b.types import (
    DecodeConfig,
    DeterministicSignals,
    GroupTicket,
    ParsedTrajectory,
    StageASummaries,
    Trajectory,
    TrajectoryWithSignals,
)


def test_fail_first_triggers_contain_no_common_mission_nouns():
    nouns = (
        "BBU",
        "挡风板",
        "接地线",
        "光纤",
        "螺丝",
        "标签",
        "机柜",
        "地排",
    )
    for trigger in FAIL_FIRST_NEGATIVE_TRIGGERS:
        assert "/" not in trigger
        assert all(noun not in trigger for noun in nouns)


def test_fail_first_hit_and_negation_exclusion():
    stage_a = {
        "image_1": "光纤/缺失×1",
        "image_2": "光纤/无错误×1",
    }
    g0 = "至少需要检测到光纤且需要符合要求"
    evidence = extract_mission_evidence(stage_a, mission_g0=g0)

    assert any(h.trigger == "缺失" and h.image_key == "image_1" for h in evidence.relevant_negative_hits)
    # "无错误" should not trigger the "错误" fail-first phrase.
    assert not any(h.trigger == "错误" and h.image_key == "image_2" for h in evidence.relevant_negative_hits)


def test_pattern_first_noncompliance_triggers_any_issue():
    stage_a = {"image_1": "光纤/不符合要求/未来新增问题描述×1"}
    g0 = "至少需要检测到光纤且需要符合要求"
    evidence = extract_mission_evidence(stage_a, mission_g0=g0)

    assert any(
        h.pattern_first and "不符合要求/未来新增问题描述" in h.trigger
        for h in evidence.relevant_negative_hits
    )


def test_mission_scoped_relevance_allows_different_missions():
    stage_a = {"image_1": "光纤/缺失×1"}

    g0_relevant = "至少需要检测到光纤且需要符合要求"
    evidence_relevant = extract_mission_evidence(stage_a, mission_g0=g0_relevant)
    assert evidence_relevant.relevant_negative_hits

    g0_irrelevant = "至少需要检测到BBU设备且需要符合要求"
    evidence_irrelevant = extract_mission_evidence(stage_a, mission_g0=g0_irrelevant)
    assert not evidence_irrelevant.relevant_negative_hits
    assert evidence_irrelevant.irrelevant_negative_hits


def test_remark_soft_marker_not_hard_negative():
    stage_a = {"image_1": "光纤/需复核,备注:拍摄范围原因,不能确认是否缺失×1"}
    g0 = "至少需要检测到光纤且需要符合要求"
    evidence = extract_mission_evidence(stage_a, mission_g0=g0)
    assert not evidence.relevant_negative_hits


def test_selection_fail_first_override_rewrites_reason():
    ticket = GroupTicket(
        group_id="QC-FF-001",
        mission="BBU线缆布放要求",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "光纤/缺失×1"}),
    )

    decode = DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=64)
    base = Trajectory(
        group_id=ticket.group_id,
        mission=ticket.mission,
        candidate_index=0,
        decode=decode,
        response_text="Verdict: 通过\nReason: Image1: 关键要素齐全; 总结: 符合任务要点",
        created_at=datetime.now(timezone.utc),
    )
    parsed = ParsedTrajectory(
        base=base,
        verdict="pass",
        reason="Image1: 关键要素齐全; 总结: 符合任务要点",
        format_ok=True,
    )
    tws = TrajectoryWithSignals(
        parsed=parsed,
        signals=DeterministicSignals(label_match=True, self_consistency=None),
    )

    result = select_for_group(
        ticket,
        [tws],
        mission_g0="至少需要检测到光纤且需要符合要求",
        guidance_step=1,
        reflection_cycle=0,
        reflection_change=None,
        config=SelectionConfig(policy="top_label", tie_break="temperature"),
        manual_review=ManualReviewConfig(min_verdict_agreement=0.8),
    )

    assert result.verdict == "fail"
    assert "fail_first_override" in result.warnings
    assert "判不通过" in result.reason
    # No third-state wording should appear in final reason.
    assert "需复核" not in result.reason
    assert "need-review" not in result.reason.lower()


def test_parse_two_line_protocol_rejects_third_state_and_extra_lines():
    ok, verdict, reason = _parse_two_line_response("Verdict: 通过\nReason: 关键要点可确认且符合要求")
    assert ok is True
    assert verdict == "pass"
    assert reason

    bad_third_state = "Verdict: 不通过\nReason: 需复核,备注: 需要补拍"
    ok2, _, _ = _parse_two_line_response(bad_third_state)
    assert ok2 is False

    bad_extra = "Verdict: 不通过\nReason: 缺失关键要素\n补充: xxx"
    ok3, _, _ = _parse_two_line_response(bad_extra)
    assert ok3 is False
