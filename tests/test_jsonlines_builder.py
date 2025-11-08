import json

import pytest

from src.datasets.builders import BaseBuilder, JSONLinesBuilder


def _sample_dense_record():
    return {
        "images": ["img1.jpg"],
        "objects": [
            {
                "bbox_2d": [100, 200, 300, 400],
                "desc": "BBU设备/华为/显示完整",
            },
            {
                "line": [10, 10, 20, 20, 30, 40, 50, 60],
                "desc": "光纤/有保护",
            },
        ],
        "width": 800,
        "height": 600,
    }


def _sample_summary_record():
    record = _sample_dense_record()
    record["summary"] = "BBU设备×1，光纤×1"
    return record


def test_jsonlines_builder_dense_outputs_object_map():
    builder = JSONLinesBuilder(
        user_prompt="列出所有对象", emit_norm="norm1000", mode="dense"
    )
    record = _sample_dense_record()

    merged = builder.build_many([record])
    assistant_text = merged["messages"][1]["content"][0]["text"]
    payload = json.loads(assistant_text)

    assert '": ' not in assistant_text
    assert ", " not in assistant_text
    assert isinstance(payload, dict)
    assert "object_1" in payload
    assert "object_2" in payload
    # Ensure legacy wrapper is gone
    assert not any(key.startswith("图片_") for key in payload.keys())


def test_jsonlines_builder_normalises_desc_whitespace():
    builder = JSONLinesBuilder(
        user_prompt="列出所有对象", emit_norm="norm1000", mode="dense"
    )
    record = _sample_dense_record()
    record["objects"][0]["desc"] = "  BBU设备/华为/显示完整  "

    merged = builder.build_many([record])
    payload = merged["assistant_payload"]

    assert payload["object_1"]["desc"] == "BBU设备/华为/显示完整"


def test_jsonlines_builder_rejects_desc_with_internal_spaces():
    builder = JSONLinesBuilder(
        user_prompt="列出所有对象", emit_norm="norm1000", mode="dense"
    )
    record = _sample_dense_record()
    record["objects"][0]["desc"] = "BBU设备/华为, 显示完整"

    with pytest.raises(ValueError):
        builder.build_many([record])


def test_jsonlines_builder_summary_outputs_plain_string():
    builder = JSONLinesBuilder(
        user_prompt="总结对象", emit_norm="norm1000", mode="summary"
    )
    record = _sample_summary_record()

    merged = builder.build_many([record])
    assistant_text = merged["messages"][1]["content"][0]["text"]

    assert assistant_text == record["summary"]


def test_base_builder_rejects_multiple_records():
    class DummyBuilder(BaseBuilder):
        def build(self, record):
            return {"messages": []}

    builder = DummyBuilder()

    with pytest.raises(ValueError):
        builder.build_many([{}, {}])
