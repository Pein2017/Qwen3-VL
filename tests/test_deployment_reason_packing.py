"""Unit tests for deployment.inference reason -> identify_result packing.

These tests are intentionally lightweight: they must not load models.
"""

from deployment.inference import (
    FAIL_TEXT,
    NEED_CHECK_MSG,
    PASS_TEXT,
    ParsedVerdict,
    _build_identify_result,
    _extract_subtasks_by_image,
)


def test_extract_subtasks_image_markers_and_summary() -> None:
    # When per-image tags are present, we prefer them and do not replicate the
    # trailing summary into every image slot.
    reason = "Image1: 缺螺丝; 松动; Image2: 正常; 总结: 整体不通过"
    per_image = _extract_subtasks_by_image(reason, num_images=2)
    assert per_image == [
        ["缺螺丝", "松动"],
        ["正常"],
    ]


def test_extract_subtasks_global_reason_replicates() -> None:
    reason = "整体通过; 证据充分。"
    per_image = _extract_subtasks_by_image(reason, num_images=3)
    assert per_image == [
        ["整体通过", "证据充分"],
        ["整体通过", "证据充分"],
        ["整体通过", "证据充分"],
    ]


def test_build_identify_result_pass_sets_status() -> None:
    verdict = ParsedVerdict(verdict="pass", reason="Image1: ok; Image2: ok")
    identify, total = _build_identify_result(verdict=verdict, num_images=2)
    assert total is True
    assert identify == [
        [["ok", PASS_TEXT, NEED_CHECK_MSG]],
        [["ok", PASS_TEXT, NEED_CHECK_MSG]],
    ]


def test_build_identify_result_fail_sets_status() -> None:
    verdict = ParsedVerdict(
        verdict="fail", reason="Image1: 缺失; Image2: 松动; 总结: 整体不通过"
    )
    identify, total = _build_identify_result(verdict=verdict, num_images=2)
    assert total is False
    assert identify == [
        [["缺失", FAIL_TEXT, NEED_CHECK_MSG]],
        [["松动", FAIL_TEXT, NEED_CHECK_MSG]],
    ]
