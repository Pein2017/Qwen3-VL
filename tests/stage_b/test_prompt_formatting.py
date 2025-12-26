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
    assert "1. 若挡风板缺失则判定不通过" in result1
    assert "2. 摘要置信度低时请返回不通过并说明原因" in result1
    assert result1.count("\n") == 1  # Two experiences, one newline

    # Test case 2: Single experience
    experiences2 = {
        "G0": "若挡风板缺失则判定不通过",
    }
    result2 = _render_guidance_snippets(experiences2)
    assert result2 == "1. 若挡风板缺失则判定不通过"

    # Test case 3: Multiple experiences (verify sorted order)
    experiences3 = {
        "G2": "第三个经验",
        "G0": "第一个经验",
        "G1": "第二个经验",
    }
    result3 = _render_guidance_snippets(experiences3)
    lines = result3.split("\n")
    assert lines[0] == "1. 第一个经验"
    assert lines[1] == "2. 第二个经验"
    assert lines[2] == "3. 第三个经验"

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
        experiences={
            "G0": "若挡风板缺失则判定不通过",
            "G1": "摘要置信度低时请返回不通过并说明原因",
        },
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    prompt = build_user_prompt(ticket, guidance)

    # Verify headline uses G0 and guidance block starts at G1
    assert "重点: 若挡风板缺失则判定不通过" in prompt
    assert "2. 摘要置信度低时请返回不通过并说明原因" in prompt
    assert "可学习规则：" in prompt
    assert "1. 若挡风板缺失则判定不通过" in prompt

    # Verify Stage-A summaries are included
    assert "1. 摘要内容" in prompt
    assert "任务: 挡风板安装检查" in prompt


def test_build_user_prompt_sanitizes_need_review_marker():
    """Stage-B user prompt must not leak '需复核' tokens from Stage-A summaries."""
    ticket = GroupTicket(
        group_id="QC-NEED-REVIEW-001",
        mission="挡风板安装检查",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(
            per_image={
                "image_1": "BBU设备/需复核,备注:无法判断品牌×1，标签/无法识别×1",
            }
        ),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        experiences={"G0": "至少需要检测到BBU设备并判断挡风板是否按要求"},
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    prompt = build_user_prompt(ticket, guidance)
    assert "需复核" not in prompt
    assert "备注" in prompt


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
        experiences={
            "G0": "若挡风板缺失则判定不通过",
        },
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    messages = build_messages(ticket, guidance, domain="bbu")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Verify G0 is used as headline and listed in mutable guidance block
    user_content = messages[1]["content"]
    assert "重点: 若挡风板缺失则判定不通过" in user_content
    assert "可学习规则：" in user_content
    assert "1. 若挡风板缺失则判定不通过" in user_content

    # Verify system prompt includes verdict contract
    system_content = messages[0]["content"]
    assert "Verdict: 通过" in system_content


def test_build_user_prompt_headline_and_list_start_at_g1():
    """G0 becomes headline, guidance list includes G0/G1/G2."""
    ticket = GroupTicket(
        group_id="QC-002",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"图片_1": "摘要内容"}),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        experiences={
            "G0": "若挡风板缺失则判定不通过",
            "G1": "当摘要无法确认挡风板是否存在时，判定不通过",
            "G2": "若安装方向错误则判定不通过",
        },
        step=2,
        updated_at=datetime.now(timezone.utc),
    )

    prompt = build_user_prompt(ticket, guidance)

    assert "重点: 若挡风板缺失则判定不通过" in prompt
    assert "2. 当摘要无法确认挡风板是否存在时，判定不通过" in prompt
    assert "3. 若安装方向错误则判定不通过" in prompt
    assert "1. 若挡风板缺失则判定不通过" in prompt


def test_build_user_prompt_json_summary_uses_objects_total():
    summary = (
        '{"dataset":"BBU","objects_total":3,'
        '"统计":[{"类别":"BBU设备","品牌":{"华为":1}}]}'
    )
    ticket = GroupTicket(
        group_id="QC-JSON-001",
        mission="挡风板安装检查",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": summary}),
    )

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        experiences={"G0": "至少需要检测到BBU设备并判断挡风板是否按要求"},
        step=1,
        updated_at=datetime.now(timezone.utc),
    )

    prompt = build_user_prompt(ticket, guidance)
    assert "Image1(obj=3)" in prompt
    assert "\"dataset\": \"BBU\"" in prompt
