from __future__ import annotations

import pytest

from src.stage_a.types import validate_stage_a_group_record


def test_validate_stage_a_group_record_accepts_valid_payload() -> None:
    record: dict[str, object] = {
        "group_id": "QC-TEST-20250101-1",
        "mission": "some_mission",
        "label": "pass",
        "images": ["a.jpg", "b.jpg"],
        "per_image": {"image_1": "ok", "image_2": '{"统计": []}'},
    }
    validated = validate_stage_a_group_record(record, context="stage_a")
    assert validated["group_id"] == "QC-TEST-20250101-1"
    assert validated["images"] == ["a.jpg", "b.jpg"]


def test_validate_stage_a_group_record_rejects_non_mapping_per_image() -> None:
    record: dict[str, object] = {
        "group_id": "g",
        "mission": "m",
        "label": "pass",
        "images": ["a.jpg"],
        "per_image": 123,
    }
    with pytest.raises(TypeError):
        validate_stage_a_group_record(record, context="stage_a")


def test_validate_stage_a_group_record_enforces_image_alignment() -> None:
    record: dict[str, object] = {
        "group_id": "g",
        "mission": "m",
        "label": "pass",
        "images": ["a.jpg", "b.jpg", "c.jpg"],
        "per_image": {"image_1": "ok", "image_2": "ok"},
    }
    with pytest.raises(ValueError):
        validate_stage_a_group_record(record, context="stage_a")


def test_validate_stage_a_group_record_rejects_empty_summary_text() -> None:
    record: dict[str, object] = {
        "group_id": "g",
        "mission": "m",
        "label": "pass",
        "images": ["a.jpg"],
        "per_image": {"image_1": "   "},
    }
    with pytest.raises(ValueError):
        validate_stage_a_group_record(record, context="stage_a")
