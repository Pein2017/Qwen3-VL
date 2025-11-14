"""Unit tests for minimal signal extraction.

Tests verify that the minimal signal set (label_match, confidence, optional self_consistency)
works correctly per training-free Stage-B design.
"""

from datetime import datetime, timezone

import pytest

from src.stage_b.config import SignalsConfig
from src.stage_b.signals import attach_signals
from src.stage_b.types import (
    DecodeConfig,
    GroupTicket,
    ParsedTrajectory,
    StageASummaries,
    Trajectory,
)


def test_label_match_and_confidence_signals():
    """Test that label_match and confidence signals are computed correctly."""
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )

    # Test case 1: Label matches, high confidence
    candidate1 = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=0,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="不通过\n理由: 测试\n置信度: 0.9",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="不通过",  # type: ignore[arg-type]
        reason="测试",
        format_ok=True,
        confidence=0.9,
    )  # type: ignore[arg-type]
    config = SignalsConfig(
        store_confidence=True, enable_consistency=False,     )
    results = attach_signals(ticket, [candidate1], config)
    assert len(results) == 1
    assert results[0].signals.label_match is True  # "不通过" == "fail" (converted)
    assert results[0].signals.confidence == 0.9

    # Test case 2: Label doesn't match, high confidence
    candidate2 = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=1,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="通过\n理由: 测试\n置信度: 0.9",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="通过",  # type: ignore[arg-type]
        reason="测试",
        format_ok=True,
        confidence=0.9,
    )  # type: ignore[arg-type]
    results2 = attach_signals(ticket, [candidate2], config)
    assert len(results2) == 1
    assert results2[0].signals.label_match is False  # "通过" != "fail"
    assert results2[0].signals.confidence == 0.9

    # Test case 3: Label doesn't match, low confidence
    candidate3 = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=2,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="通过\n理由: 测试\n置信度: 0.2",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="通过",  # type: ignore[arg-type]
        reason="测试",
        format_ok=True,
        confidence=0.2,
    )  # type: ignore[arg-type]
    results3 = attach_signals(ticket, [candidate3], config)
    assert len(results3) == 1
    assert results3[0].signals.label_match is False
    assert results3[0].signals.confidence == 0.2


def test_signals_without_confidence():
    """Test signal extraction when confidence is not available."""
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )

    config = SignalsConfig(
        store_confidence=False, enable_consistency=False,     )

    # Test case 1: Label matches, no confidence
    candidate1 = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=0,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="不通过\n理由: 测试",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="不通过",  # type: ignore[arg-type]
        reason="测试",
        format_ok=True,
        confidence=None,
    )  # type: ignore[arg-type]
    results1 = attach_signals(ticket, [candidate1], config)
    assert len(results1) == 1
    assert results1[0].signals.label_match is True
    assert results1[0].signals.confidence is None

    # Test case 2: Label doesn't match, no confidence
    candidate2 = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=1,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="通过\n理由: 测试",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="通过",  # type: ignore[arg-type]
        reason="测试",
        format_ok=True,
        confidence=None,
    )  # type: ignore[arg-type]
    results2 = attach_signals(ticket, [candidate2], config)
    assert len(results2) == 1
    assert results2[0].signals.label_match is False
    assert results2[0].signals.confidence is None


def test_signals_without_label_match():
    """Test signal extraction when label_match cannot be determined."""
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )

    config = SignalsConfig(
        store_confidence=True, enable_consistency=False,     )

    # Test case: Verdict is None (cannot determine label_match)
    candidate = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=0,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="Invalid response",
            created_at=datetime.now(timezone.utc),
        ),
        verdict=None,
        reason=None,
        format_ok=False,
        confidence=0.8,
    )
    results = attach_signals(ticket, [candidate], config)
    assert len(results) == 1
    assert results[0].signals.label_match is None
    assert results[0].signals.confidence == 0.8


def test_attach_signals_mixed_language_verdicts():
    """Signals should treat Chinese and English verdict variants equivalently."""

    ticket = GroupTicket(
        group_id="QC-002",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )

    decode_cfg = DecodeConfig(temperature=0.2, top_p=0.95, max_new_tokens=128)
    timestamp = datetime.now(timezone.utc)

    candidates = [
        ParsedTrajectory(
            base=Trajectory(
                group_id="QC-002",
                mission="挡风板安装检查",
                candidate_index=0,
                decode=decode_cfg,
                response_text="Verdict: 不通过",
                created_at=timestamp,
            ),
            verdict="不通过",  # type: ignore[arg-type]
            reason="检测到挡风板缺失",
            confidence=0.8,
            format_ok=True,
        ),  # type: ignore[arg-type]
        ParsedTrajectory(
            base=Trajectory(
                group_id="QC-002",
                mission="挡风板安装检查",
                candidate_index=1,
                decode=decode_cfg,
                response_text="Verdict: fail",
                created_at=timestamp,
            ),
            verdict="fail",
            reason="English fail verdict",
            confidence=0.7,
            format_ok=True,
        ),
        ParsedTrajectory(
            base=Trajectory(
                group_id="QC-002",
                mission="挡风板安装检查",
                candidate_index=2,
                decode=decode_cfg,
                response_text="Verdict: 通过",
                created_at=timestamp,
            ),
            verdict="通过",  # type: ignore[arg-type]
            reason="挡风板安装方向正确",
            confidence=0.6,
            format_ok=True,
        ),  # type: ignore[arg-type]
    ]

    config = SignalsConfig(store_confidence=True, enable_consistency=True, )
    annotated = attach_signals(ticket, candidates, config)

    assert len(annotated) == 3

    # Two candidates predict fail variants (Chinese + English)
    first, second, third = annotated

    assert first.signals.label_match is True
    assert second.signals.label_match is True
    assert third.signals.label_match is False

    # Self-consistency uses normalized counts; both fail variants share the same value
    assert first.signals.self_consistency == pytest.approx(
        second.signals.self_consistency
    )
    assert third.signals.self_consistency == pytest.approx(1 / 3)
