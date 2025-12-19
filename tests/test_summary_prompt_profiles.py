import pytest

from src.prompts.summary_profiles import (
    DEFAULT_SUMMARY_PROFILE_RUNTIME,
    DEFAULT_SUMMARY_PROFILE_TRAIN,
    build_summary_system_prompt,
)


def test_summary_train_min_excludes_domain_block():
    prompt = build_summary_system_prompt(DEFAULT_SUMMARY_PROFILE_TRAIN)
    assert "领域提示" not in prompt
    assert "BBU领域提示" not in prompt


def test_summary_runtime_includes_bbu_domain_block():
    prompt = build_summary_system_prompt(DEFAULT_SUMMARY_PROFILE_RUNTIME, domain="bbu")
    assert "BBU领域提示" in prompt
    assert "核心类别" in prompt


def test_summary_runtime_missing_domain_raises():
    with pytest.raises(ValueError, match="domain must be provided"):
        build_summary_system_prompt(DEFAULT_SUMMARY_PROFILE_RUNTIME)
