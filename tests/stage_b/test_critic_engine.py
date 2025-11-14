#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for CriticEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stage_b.config import CriticConfig
from src.stage_b.critic import CriticEngine
from src.stage_b.types import (
    CriticOutput,
    DecodeConfig,
    DeterministicSignals,
    GroupTicket,
    StageASummaries,
    ParsedTrajectory,
    Trajectory,
    TrajectoryWithSignals,
)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    return MagicMock()


@pytest.fixture
def critic_config(tmp_path: Path):
    """Create a test CriticConfig."""
    prompt_path = tmp_path / "test_critic_prompt.txt"
    prompt_path.write_text(
        "任务: {mission}\n"
        "Stage-A 总结: {stage_a_summary}\n"
        "候选回答: {candidate_response}\n"
        "信号: {signals}\n\n"
        "请以严格的 JSON 格式输出评估结果，包含以下字段:\n"
        '{"summary": "...", "critique": "..."}\n'
    )
    return CriticConfig(
        enabled=True,
        prompt_path=prompt_path,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )


@pytest.fixture
def sample_ticket():
    """Create a sample GroupTicket for testing."""
    return GroupTicket(
        group_id="TEST-001",
        mission="测试任务",
        label="pass",
        summaries=StageASummaries(per_image={"img1.jpg": "图像正常"}),
    )




@pytest.fixture
def sample_candidate():
    """Create a sample TrajectoryWithSignals for testing."""
    decode_cfg = DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=128)
    base = Trajectory(
        group_id="TEST-001",
        mission="测试任务",
        candidate_index=0,
        decode=decode_cfg,
        response_text="判定: 通过\n理由: 部件完整",
        created_at=datetime.now(timezone.utc),
    )
    parsed = ParsedTrajectory(
        base=base,
        verdict="pass",
        reason="部件完整",
        confidence=0.85,
        format_ok=True,
    )
    signals = DeterministicSignals(
        label_match=True,
        self_consistency=0.8,
        confidence=0.85,
    )
    return TrajectoryWithSignals(parsed=parsed, signals=signals)


def test_critic_engine_initialization(mock_model, mock_tokenizer, mock_processor, critic_config):
    """Test CriticEngine initialization and template validation."""
    engine = CriticEngine(
        config=critic_config,
        model=mock_model,
        processor=mock_processor,
        tokenizer=mock_tokenizer,
    )
    assert engine.model == mock_model
    assert engine.tokenizer == mock_tokenizer
    assert engine.config == critic_config


def test_critic_engine_template_validation_missing_placeholders(
    mock_model, mock_tokenizer, mock_processor, tmp_path
):
    """Test that template validation fails when required placeholders are missing."""
    prompt_path = tmp_path / "invalid_prompt.txt"
    prompt_path.write_text("Invalid prompt without placeholders")

    config = CriticConfig(
        enabled=True,
        prompt_path=prompt_path,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )

    with pytest.raises(ValueError, match="missing required placeholders"):
        CriticEngine(model=mock_model, tokenizer=mock_tokenizer, processor=mock_processor, config=config)


def test_critic_engine_template_validation_missing_json_instruction(
    mock_model, mock_tokenizer, mock_processor, tmp_path
):
    """Test that template validation warns when JSON instruction is missing."""
    prompt_path = tmp_path / "no_json_prompt.txt"
    prompt_path.write_text(
        "任务: {mission}\n"
        "Stage-A 总结: {stage_a_summary}\n"
        "候选回答: {candidate_response}\n"
        "信号: {signals}\n"
    )

    config = CriticConfig(
        enabled=True,
        prompt_path=prompt_path,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )

    CriticEngine(model=mock_model, tokenizer=mock_tokenizer, processor=mock_processor, config=config)


@patch("src.stage_b.critic.engine.CriticEngine._generate_critic_response")
def test_critic_engine_evaluate_success(
    mock_generate,
    mock_model,
    mock_tokenizer,
    mock_processor,
    critic_config,
    sample_ticket,
    sample_candidate,
):
    """Test successful evaluation with valid JSON response."""
    # Mock LLM response
    mock_generate.return_value = (
        '{"summary": "候选判定正确", "critique": "理由充分", '
        '"root_cause": "检查完整", "issues": ["无"], '
        '"candidate_ops": [{"op": "keep"}], "uncertainty_note": null}'
    )

    engine = CriticEngine(
        config=critic_config,
        model=mock_model,
        processor=mock_processor,
        tokenizer=mock_tokenizer,
    )

    results = engine.evaluate(
        group_id=sample_ticket.group_id,
        mission=sample_ticket.mission,
        candidates=[sample_candidate.parsed],
        signals=[sample_candidate.signals],
        stage_a_summary="图像正常",
    )

    assert len(results) == 1
    assert results[0] is not None
    assert isinstance(results[0], CriticOutput)
    assert results[0].summary == "候选判定正确"
    assert results[0].critique == "理由充分"
    assert results[0].root_cause == "检查完整"


@patch("src.stage_b.critic.engine.CriticEngine._generate_critic_response")
def test_critic_engine_evaluate_invalid_json(
    mock_generate,
    mock_model,
    mock_tokenizer,
    mock_processor,
    critic_config,
    sample_ticket,
    sample_candidate,
):
    """Test evaluation with invalid JSON response returns None."""
    # Mock invalid JSON response
    mock_generate.return_value = "This is not valid JSON"

    engine = CriticEngine(
        config=critic_config,
        model=mock_model,
        processor=mock_processor,
        tokenizer=mock_tokenizer,
    )

    results = engine.evaluate(
        group_id=sample_ticket.group_id,
        mission=sample_ticket.mission,
        candidates=[sample_candidate.parsed],
        signals=[sample_candidate.signals],
        stage_a_summary="图像正常",
    )

    assert len(results) == 1
    assert results[0] is None


@patch("src.stage_b.critic.engine.CriticEngine._generate_critic_response")
def test_critic_engine_length_cap_enforcement(
    mock_generate,
    mock_model,
    mock_tokenizer,
    mock_processor,
    critic_config,
    sample_ticket,
    sample_candidate,
):
    """Test that summary and critique are truncated to max length."""
    # Mock response with very long summary and critique
    long_summary = "A" * 500
    long_critique = "B" * 500
    mock_generate.return_value = (
        f'{{"summary": "{long_summary}", "critique": "{long_critique}"}}'
    )

    engine = CriticEngine(
        config=critic_config,
        model=mock_model,
        processor=mock_processor,
        tokenizer=mock_tokenizer,
    )

    results = engine.evaluate(
        group_id=sample_ticket.group_id,
        mission=sample_ticket.mission,
        candidates=[sample_candidate.parsed],
        signals=[sample_candidate.signals],
        stage_a_summary="图像正常",
    )

    assert len(results) == 1
    assert results[0] is not None
    assert len(results[0].summary) <= critic_config.summary_max_chars
    assert len(results[0].critique) <= critic_config.critique_max_chars


def test_critic_prefilter_detects_contradictions(
    mock_model, mock_tokenizer, mock_processor, critic_config, sample_ticket
):
    """Test that prefilter correctly detects contradictions from mixed label_match values."""
    from src.stage_b.types import ParsedTrajectory, Trajectory, DecodeConfig, DeterministicSignals
    from datetime import datetime, timezone
    
    engine = CriticEngine(
        config=critic_config,
        model=mock_model,
        processor=mock_processor,
        tokenizer=mock_tokenizer,
    )
    
    # Create candidates with mixed label_match (contradiction)
    decode_cfg = DecodeConfig(temperature=0.7, top_p=0.9, max_new_tokens=128)
    candidates = [
        ParsedTrajectory(
            base=Trajectory(
                group_id="TEST-001",
                mission="测试任务",
                candidate_index=0,
                decode=decode_cfg,
                response_text="判定: 通过",
                created_at=datetime.now(timezone.utc),
            ),
            verdict="pass",
            reason="正常",
            confidence=0.8,
            format_ok=True,
        ),
        ParsedTrajectory(
            base=Trajectory(
                group_id="TEST-001",
                mission="测试任务",
                candidate_index=1,
                decode=decode_cfg,
                response_text="判定: 不通过",
                created_at=datetime.now(timezone.utc),
            ),
            verdict="fail",
            reason="异常",
            confidence=0.7,
            format_ok=True,
        ),
    ]
    
    signals = [
        DeterministicSignals(
            label_match=True,  # First candidate matches label
            self_consistency=None,
            confidence=0.8,
        ),
        DeterministicSignals(
            label_match=False,  # Second candidate doesn't match (contradiction)
            self_consistency=None,
            confidence=0.7,
        ),
    ]
    
    # Prefilter should detect contradiction and prioritize both candidates
    selected_indices = engine._prefilter_candidates(candidates, signals)
    
    # Both candidates should be selected due to contradiction
    assert len(selected_indices) == 2
    assert 0 in selected_indices
    assert 1 in selected_indices

