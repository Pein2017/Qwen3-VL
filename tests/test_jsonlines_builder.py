import json

import pytest

from src.datasets.builders import (
    BaseBuilder,
    JSONLinesBuilder,
    ToonRow,
    decode_toon_payload,
    decode_toon_block,
    encode_toon_block,
)


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
    builder = JSONLinesBuilder(user_prompt="列出所有对象", emit_norm="norm1000", mode="dense")
    record = _sample_dense_record()

    merged = builder.build_many([record])
    assistant_text = merged["messages"][1]["content"][0]["text"]
    payload = json.loads(assistant_text)

    assert isinstance(payload, dict)
    assert "object_1" in payload
    assert "object_2" in payload
    # Ensure legacy wrapper is gone
    assert not any(key.startswith("图片_") for key in payload.keys())


def test_jsonlines_builder_summary_outputs_plain_string():
    builder = JSONLinesBuilder(user_prompt="总结对象", emit_norm="norm1000", mode="summary")
    record = _sample_summary_record()

    merged = builder.build_many([record])
    assistant_text = merged["messages"][1]["content"][0]["text"]

    assert assistant_text == record["summary"]


def test_jsonlines_builder_toon_outputs_table():
    builder = JSONLinesBuilder(
        user_prompt="列出所有对象",
        emit_norm="norm1000",
        mode="dense",
        toon_mode=True,
    )
    record = _sample_dense_record()

    merged = builder.build_many([record])
    assistant_text = merged["messages"][1]["content"][0]["text"]

    assert assistant_text.startswith("objs[")
    assert "line_points" not in assistant_text

    decoded = decode_toon_payload(assistant_text)
    assert decoded == merged.get("assistant_payload")


def test_encode_toon_block_handles_quoting_and_tab_delimiter():
    rows = [
        ToonRow(type_id=0, desc="光纤,有保护", coords=(1, 2, 3, 4)),
        ToonRow(type_id=1, desc='含"引号"', coords=(10, 20, 30, 40, 50, 60, 70, 80)),
    ]

    block = encode_toon_block(rows)
    assert block.startswith("objs[2]{type,desc,xs}:")
    assert '"光纤,有保护"' in block
    assert '含\\"引号\\"' in block
    decoded = decode_toon_block(block)
    assert decoded == rows

    block_tab = encode_toon_block(rows, delimiter="\t")
    assert "[2\t]" in block_tab
    decoded_tab = decode_toon_block(block_tab)
    assert decoded_tab == rows


def test_base_builder_rejects_multiple_records():
    class DummyBuilder(BaseBuilder):
        def build(self, record):
            return {"messages": []}

    builder = DummyBuilder()

    with pytest.raises(ValueError):
        builder.build_many([{}, {}])

