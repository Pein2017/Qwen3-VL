"""Unit tests for simplified reflection action parsing."""

import json

import pytest

from src.stage_b.config import ReflectionConfig
from src.stage_b.reflection import ReflectionEngine
from src.stage_b.types import (
    ExperienceBundle,
    ExperienceRecord,
    GroupTicket,
    StageASummaries,
)


def test_parse_reflection_response_refine(tmp_path):
    """Test parsing 'refine' action from reflection response."""

    class MockModel:
        pass

    class MockTokenizer:
        pass

    # Create prompt template file
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Reflection prompt template", encoding="utf-8")

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        allow_uncertain=True,
    )

    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
        guidance_repo=None,  # type: ignore[arg-type]
        reflection_log=tmp_path / "reflection.log",
    )

    # Create a minimal bundle for testing
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )
    record = ExperienceRecord(
        ticket=ticket,
        candidates=tuple(),
        winning_candidate=None,
        guidance_step=1,
    )
    bundle = ExperienceBundle(
        mission="挡风板安装检查",
        records=(record,),
        reflection_cycle=0,
        guidance_step=1,
    )

    # Test case 1: Valid refine action with structured operations
    response1 = json.dumps(
        {
            "action": "refine",
            "summary": "候选之间存在标签冲突，需补充缺陷判定",
            "critique": "当前经验未覆盖挡板缺失的判定",
            "operations": [
                {
                    "op": "upsert",
                    "key": "G0",
                    "text": "[G0]. 如果摘要明确提到挡风板缺失，应判定为不通过并提示复检",
                    "rationale": "真实标签为不通过而经验仍放行",
                    "evidence": ["QC-001"],
                },
                {
                    "op": "remove",
                    "key": "G3",
                    "text": None,
                    "rationale": "旧条目与现有数据冲突",
                    "evidence": ["QC-001"],
                },
            ],
            "evidence_group_ids": ["QC-001", "QC-002"],
            "uncertainty_note": None,
        }
    )
    proposal1 = engine._parse_reflection_response(response1, bundle)
    assert proposal1.action == "refine"
    assert proposal1.summary == "候选之间存在标签冲突，需补充缺陷判定"
    assert proposal1.critique == "当前经验未覆盖挡板缺失的判定"
    assert len(proposal1.operations) == 2
    assert proposal1.operations[0].op == "upsert"
    assert proposal1.operations[0].key == "G0"
    assert "挡风板缺失" in (proposal1.operations[0].text or "")
    assert proposal1.operations[0].evidence == ("QC-001",)
    assert proposal1.operations[1].op == "remove"
    assert proposal1.operations[1].key == "G3"
    assert proposal1.evidence_group_ids == ("QC-001", "QC-002")
    assert proposal1.uncertainty_note is None

    # Test case 2: Noop action
    response2 = json.dumps(
        {
            "action": "noop",
            "summary": None,
            "critique": None,
            "operations": [],
            "evidence_group_ids": [],
        }
    )
    proposal2 = engine._parse_reflection_response(response2, bundle)
    assert proposal2.action == "noop"
    assert proposal2.summary is None
    assert proposal2.operations == tuple()
    assert proposal2.evidence_group_ids == tuple()
    assert proposal2.uncertainty_note is None

    # Test case 3: Invalid action (should default to noop)
    response3 = json.dumps(
        {
            "action": "add",  # Invalid - should be converted to noop
            "summary": "some",
            "critique": "text",
            "operations": [],
            "evidence_group_ids": ["QC-001"],
        }
    )
    proposal3 = engine._parse_reflection_response(response3, bundle)
    assert proposal3.action == "noop"
    assert proposal3.uncertainty_note is not None

    # Test case 4: Missing action (should default to noop)
    response4 = json.dumps(
        {
            "summary": "some",
            "critique": "text",
            "operations": [],
            "evidence_group_ids": ["QC-001"],
        }
    )
    proposal4 = engine._parse_reflection_response(response4, bundle)
    assert proposal4.action == "noop"

    # Test case 5: Response with markdown formatting
    response5 = (
        "Here is some text.\n```json\n"
        + json.dumps(
            {
                "action": "refine",
                "summary": "挡板缺失",
                "critique": "经验缺口",
                "operations": [
                    {
                        "op": "upsert",
                        "key": None,
                        "text": "[G1]. 若挡风板缺失则判定不通过",
                        "rationale": None,
                        "evidence": ["QC-001"],
                    }
                ],
                "evidence_group_ids": ["QC-001"],
            }
        )
        + "\n```\nMore text."
    )
    proposal5 = engine._parse_reflection_response(response5, bundle)
    assert proposal5.action == "refine"
    assert proposal5.operations[0].text == "[G1]. 若挡风板缺失则判定不通过"
    assert proposal5.evidence_group_ids == ("QC-001",)

    # Test case 6: Invalid JSON (should return noop)
    response6 = "This is not valid JSON {"
    with pytest.raises(ValueError):
        engine._parse_reflection_response(response6, bundle)
    assert engine._last_debug_info is not None
    assert engine._last_debug_info["parse_error"] == "No valid JSON with 'action' field found"
    assert engine._last_debug_info["json_candidates_count"] == 0

    # Test case 7: Empty evidence_group_ids (should use bundle group ids)
    response7 = json.dumps(
        {
            "action": "refine",
            "summary": None,
            "critique": None,
            "operations": [
                {
                    "op": "upsert",
                    "key": "G0",
                    "text": "[G0]. 若挡风板缺失则判定不通过",
                    "rationale": None,
                    "evidence": ["QC-001"],
                }
            ],
        }
    )
    proposal7 = engine._parse_reflection_response(response7, bundle)
    assert proposal7.action == "refine"
    assert proposal7.evidence_group_ids == ("QC-001",)  # Falls back to bundle records
