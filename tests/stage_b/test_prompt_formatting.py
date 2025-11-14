"""Unit tests for prompt formatting with numbered experiences."""

from datetime import datetime, timezone

import pytest

from src.stage_b.sampling.prompts import (
    _render_guidance_snippets,
    build_messages,
    build_user_prompt,
)
from src.stage_b.types import GroupTicket, MissionGuidance, StageASummaries


def test_render_guidance_snippets():
    """Test formatting experiences dict as numbered experiences text block."""
    # Test case 1: Standard experiences dict
    experiences1 = {
        "G0": "若挡风板缺失则判定不通过",
        "G1": "摘要置信度低时请返回不通过并说明原因",
    }
    result1 = _render_guidance_snippets(experiences1)
    assert "[G0]. 若挡风板缺失则判定不通过" in result1
    assert "[G1]. 摘要置信度低时请返回不通过并说明原因" in result1
    assert result1.count("\n") == 1  # Two experiences, one newline

    # Test case 2: Single experience
    experiences2 = {
        "G0": "若挡风板缺失则判定不通过",
    }
    result2 = _render_guidance_snippets(experiences2)
    assert result2 == "[G0]. 若挡风板缺失则判定不通过"

    # Test case 3: Multiple experiences (verify sorted order)
    experiences3 = {
        "G2": "第三个经验",
        "G0": "第一个经验",
        "G1": "第二个经验",
    }
    result3 = _render_guidance_snippets(experiences3)
    lines = result3.split("\n")
    assert lines[0] == "[G0]. 第一个经验"
    assert lines[1] == "[G1]. 第二个经验"
    assert lines[2] == "[G2]. 第三个经验"

    # Test case 4: Empty experiences dict raises error
    experiences4 = {}
    with pytest.raises(ValueError, match="Experiences dict must be non-empty"):
        _render_guidance_snippets(experiences4)


def test_build_user_prompt_with_experiences():
    """Test building user prompt with experiences prepended."""
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"图片_1": "摘要内容"}),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        focus="挡风板安装检查任务要点",
        experiences={
            "G0": "若挡风板缺失则判定不通过",
            "G1": "摘要置信度低时请返回不通过并说明原因",
        },
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    prompt = build_user_prompt(ticket, guidance)

    # Verify experiences are prepended
    assert "[G0]. 若挡风板缺失则判定不通过" in prompt
    assert "[G1]. 摘要置信度低时请返回不通过并说明原因" in prompt
    assert "补充提示：" in prompt

    # Verify Stage-A summaries are included
    assert "1. 摘要内容" in prompt
    assert "任务: 挡风板安装检查" in prompt


def test_build_user_prompt_empty_experiences_raises_error():
    """Test that empty experiences dict causes abort."""
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"图片_1": "摘要内容"}),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        focus="挡风板安装检查任务要点",
        experiences={},  # Empty experiences
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValueError, match="Experiences dict must be non-empty"):
        build_user_prompt(ticket, guidance)


def test_build_messages_with_experiences():
    """Test building messages with experiences formatted correctly."""
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"图片_1": "摘要内容"}),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        focus="挡风板安装检查任务要点",
        experiences={
            "G0": "若挡风板缺失则判定不通过",
        },
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    messages = build_messages(ticket, guidance)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Verify experiences are in user prompt
    user_content = messages[1]["content"]
    assert "[G0]. 若挡风板缺失则判定不通过" in user_content
    assert "补充提示：" in user_content

    # Verify system prompt includes focus
    system_content = messages[0]["content"]
    assert "任务要点" in system_content or "挡风板安装检查任务要点" in system_content
