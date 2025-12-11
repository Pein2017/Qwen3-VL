from src.datasets.builders.jsonlines import JSONLinesBuilder


def test_jsonlines_builder_sorts_tlbr():
    builder = JSONLinesBuilder(user_prompt="prompt", emit_norm="none")
    objects = [
        {"bbox_2d": [50, 50, 70, 70], "desc": "bottom"},  # lower
        {"poly": [10, 10, 20, 10, 20, 20, 10, 20], "desc": "top-left"},
        {"line": [30, 30, 80, 30], "desc": "middle-line"},
        {"bbox_2d": [5, 40, 15, 50], "desc": "mid-left"},
    ]
    record = {"images": [], "objects": objects, "width": 100, "height": 100}

    payload = builder.build(record)
    assistant = payload["assistant_payload"]
    ordered_desc = [assistant[key]["desc"] for key in sorted(assistant.keys())]
    assert ordered_desc == [
        "top-left",
        "middle-line",
        "mid-left",
        "bottom",
    ]
