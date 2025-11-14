#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for prompt template validation in CriticEngine and ReflectionEngine."""

from pathlib import Path

import pytest

from src.stage_b.config import CriticConfig, ReflectionConfig
from src.stage_b.critic import CriticEngine
from src.stage_b.reflection.engine import ReflectionEngine


class MockModel:
    """Mock model for testing."""

    device = "cpu"


class MockTokenizer:
    """Mock tokenizer for testing."""

    pass


def test_template_validation_missing_allowed_ops_canonical_name(tmp_path):
    """Test that validation raises error when canonical template is missing allowed ops."""
    # Create a template file with canonical name but missing allowed ops
    prompt_file = tmp_path / "stage_b_reflection_prompt.txt"
    prompt_file.write_text(
        "这是一个反思提示模板，但是缺少允许的操作列表。\n"
        "每次最多提出 K 条 operations。\n",
        encoding="utf-8",
    )

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
        max_operations=3,
    )

    # Should raise ValueError for canonical template name
    with pytest.raises(ValueError, match="allowed ops 'upsert\\|remove\\|merge'"):
        ReflectionEngine(
            model=MockModel(),  # type: ignore[arg-type]
            tokenizer=MockTokenizer(),  # type: ignore[arg-type]
            config=config,
            guidance_repo=None,  # type: ignore[arg-type]
        )


def test_template_validation_missing_allowed_ops_custom_name(tmp_path, caplog):
    """Test that validation logs warning when custom template is missing allowed ops."""
    # Create a template file with custom name but missing allowed ops
    prompt_file = tmp_path / "custom_prompt.txt"
    prompt_file.write_text(
        "这是一个反思提示模板，但是缺少允许的操作列表。\n"
        "每次最多提出 K 条 operations。\n",
        encoding="utf-8",
    )

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
        max_operations=3,
    )

    # Should log warning for custom template name (not raise)
    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
        guidance_repo=None,  # type: ignore[arg-type]
    )

    assert engine is not None
    assert "allowed ops 'upsert|remove|merge'" in caplog.text


def test_template_validation_missing_k_budget_canonical_name(tmp_path):
    """Test that validation raises error when canonical template is missing K budget mention."""
    # Create a template file with canonical name but missing K mention
    prompt_file = tmp_path / "stage_b_reflection_prompt.txt"
    prompt_file.write_text(
        "允许的操作: op 字段为 upsert|remove|merge\n输出 JSON 格式，严格结构化。\n",
        encoding="utf-8",
    )

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
        max_operations=3,  # Budget is set, so K must be mentioned
    )

    # Should raise ValueError for canonical template name
    with pytest.raises(ValueError, match="budget symbol 'K'"):
        ReflectionEngine(
            model=MockModel(),  # type: ignore[arg-type]
            tokenizer=MockTokenizer(),  # type: ignore[arg-type]
            config=config,
            guidance_repo=None,  # type: ignore[arg-type]
        )


def test_template_validation_missing_k_budget_custom_name(tmp_path, caplog):
    """Test that validation logs warning when custom template is missing K budget mention."""
    # Create a template file with custom name but missing K mention
    prompt_file = tmp_path / "custom_prompt.txt"
    prompt_file.write_text(
        "允许的操作: op 字段为 upsert|remove|merge\n输出 JSON 格式，严格结构化。\n",
        encoding="utf-8",
    )

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
        max_operations=3,  # Budget is set, so K must be mentioned
    )

    # Should log warning for custom template name (not raise)
    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
        guidance_repo=None,  # type: ignore[arg-type]
    )

    assert engine is not None
    assert "budget symbol 'K'" in caplog.text


def test_template_validation_no_k_required_when_no_budget(tmp_path):
    """Test that K mention is not required when max_operations is None."""
    # Create a template file without K mention
    prompt_file = tmp_path / "stage_b_reflection_prompt.txt"
    prompt_file.write_text(
        "允许的操作: op 字段为 upsert|remove|merge\n输出 JSON 格式，严格结构化。\n",
        encoding="utf-8",
    )

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
        max_operations=None,  # No budget set, so K is not required
    )

    # Should not raise error
    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
        guidance_repo=None,  # type: ignore[arg-type]
    )

    assert engine is not None


def test_template_validation_valid_template(tmp_path):
    """Test that validation passes for a valid template."""
    # Create a valid template file
    prompt_file = tmp_path / "stage_b_reflection_prompt.txt"
    prompt_file.write_text(
        "允许的操作: op 字段为 upsert|remove|merge; 每次最多提出 K 条 operations。\n"
        "输出 JSON 格式，严格结构化。\n",
        encoding="utf-8",
    )

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
        max_operations=3,
    )

    # Should not raise error
    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
        guidance_repo=None,  # type: ignore[arg-type]
    )

    assert engine is not None
    assert engine.prompt_template is not None
    assert "upsert|remove|merge" in engine.prompt_template
    assert "K" in engine.prompt_template


# ============================================================================
# CriticEngine Template Validation Tests
# ============================================================================


def test_critic_template_validation_valid(tmp_path):
    """Test that a valid CriticEngine template passes validation."""
    prompt_file = tmp_path / "valid_critic_prompt.txt"
    prompt_file.write_text(
        "任务: {mission}\n"
        "Stage-A 总结: {stage_a_summary}\n"
        "候选回答: {candidate_response}\n"
        "信号: {signals}\n\n"
        "请以严格的 JSON 格式输出评估结果。\n",
        encoding="utf-8",
    )

    config = CriticConfig(
        enabled=True,
        prompt_path=prompt_file,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )

    # Should not raise
    engine = CriticEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
    )
    assert engine is not None


def test_critic_template_validation_missing_mission(tmp_path):
    """Test that missing {mission} placeholder raises error."""
    prompt_file = tmp_path / "no_mission_prompt.txt"
    prompt_file.write_text(
        "Stage-A 总结: {stage_a_summary}\n"
        "候选回答: {candidate_response}\n"
        "信号: {signals}\n",
        encoding="utf-8",
    )

    config = CriticConfig(
        enabled=True,
        prompt_path=prompt_file,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )

    with pytest.raises(ValueError, match="missing required placeholders.*mission"):
        CriticEngine(
            model=MockModel(),  # type: ignore[arg-type]
            tokenizer=MockTokenizer(),  # type: ignore[arg-type]
            config=config,
        )


def test_critic_template_validation_missing_stage_a_summary(tmp_path):
    """Test that missing {stage_a_summary} placeholder raises error."""
    prompt_file = tmp_path / "no_stage_a_prompt.txt"
    prompt_file.write_text(
        "任务: {mission}\n候选回答: {candidate_response}\n信号: {signals}\n",
        encoding="utf-8",
    )

    config = CriticConfig(
        enabled=True,
        prompt_path=prompt_file,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )

    with pytest.raises(
        ValueError, match="missing required placeholders.*stage_a_summary"
    ):
        CriticEngine(
            model=MockModel(),  # type: ignore[arg-type]
            tokenizer=MockTokenizer(),  # type: ignore[arg-type]
            config=config,
        )


def test_critic_template_validation_missing_json_warns(tmp_path):
    """Test that missing JSON instruction triggers warning."""
    prompt_file = tmp_path / "no_json_prompt.txt"
    prompt_file.write_text(
        "任务: {mission}\n"
        "Stage-A 总结: {stage_a_summary}\n"
        "候选回答: {candidate_response}\n"
        "信号: {signals}\n",
        encoding="utf-8",
    )

    config = CriticConfig(
        enabled=True,
        prompt_path=prompt_file,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        max_candidates=5,
        summary_max_chars=200,
        critique_max_chars=300,
    )

    with pytest.warns(UserWarning, match="should mention strict JSON output"):
        CriticEngine(
            model=MockModel(),  # type: ignore[arg-type]
            tokenizer=MockTokenizer(),  # type: ignore[arg-type]
            config=config,
        )
