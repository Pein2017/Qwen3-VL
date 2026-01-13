from typing import cast

from src.datasets.builders.jsonlines import JSONLinesBuilder
from src.datasets.contracts import ConversationRecord, DatasetObject


def test_jsonlines_builder_sorts_tlbr() -> None:
    builder = JSONLinesBuilder(user_prompt="prompt", emit_norm="none")
    objects: list[DatasetObject] = [
        {"bbox_2d": [50, 50, 70, 70], "desc": "bottom"},  # lower
        {"poly": [10, 10, 20, 10, 20, 20, 10, 20], "desc": "top-left"},
        {"line": [30, 30, 80, 30], "desc": "middle-line"},
        {"bbox_2d": [5, 40, 15, 50], "desc": "mid-left"},
    ]
    record = cast(
        ConversationRecord,
        {"images": [], "objects": objects, "width": 100, "height": 100},
    )

    payload = builder.build(record)
    assistant = payload["assistant_payload"]
    ordered_desc = [assistant[key]["desc"] for key in sorted(assistant.keys())]
    assert ordered_desc == [
        "top-left",
        "middle-line",
        "mid-left",
        "bottom",
    ]


def test_jsonlines_builder_sorts_center_tlbr_when_enabled() -> None:
    builder = JSONLinesBuilder(
        user_prompt="prompt",
        emit_norm="none",
        object_ordering_policy="center_tlbr",
    )
    # Deliberately construct a case where legacy (reference-point) ordering differs
    # from AABB-center ordering.
    objects: list[DatasetObject] = [
        {"bbox_2d": [0, 0, 100, 10], "desc": "wide-top"},  # center_y=5
        {"bbox_2d": [60, 1, 70, 2], "desc": "small-upper"},  # center_y=1.5
    ]
    record = cast(
        ConversationRecord,
        {"images": [], "objects": objects, "width": 100, "height": 100},
    )

    payload = builder.build(record)
    assistant = payload["assistant_payload"]
    ordered_desc = [assistant[key]["desc"] for key in sorted(assistant.keys())]
    assert ordered_desc == ["small-upper", "wide-top"]
