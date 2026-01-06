from src.rlhf.grpo.rewards.dense.parsing import (
    JsonDuplicateKeyError,
    parse_dense_completion_strict,
    parse_dense_payload_mapping,
    parse_desc,
)


def test_parse_desc_normalizes_whitespace():
    desc = parse_desc("品牌 = 华为, 类别 = 设备 ,可见性=清晰")
    assert desc.get("品牌") == "华为"
    assert desc.get("类别") == "设备"
    assert desc.get("可见性") == "清晰"


def test_parse_dense_completion_strict_rejects_duplicate_keys():
    meta = {
        "_fusion_mode": "dense",
        "_fusion_template": "target_dense_bbu",
        "_fusion_source": "bbu_dense",
    }
    # Duplicate object_1 key must be rejected (strict JSON duplicate-key validation).
    text = (
        "<DOMAIN=BBU>, <TASK=DETECTION>\n"
        '{"object_1":{"desc":"类别=设备","bbox_2d":[0,0,10,10]},"object_1":{"desc":"类别=设备","bbox_2d":[0,0,10,10]}}'
    )

    try:
        parse_dense_completion_strict(text=text, meta=meta)
        assert False, "Expected JsonDuplicateKeyError"
    except JsonDuplicateKeyError:
        pass


def test_parse_dense_payload_mapping_rejects_poly_with_too_few_points():
    raw: dict[str, object] = {
        "object_1": {
            "desc": "类别=设备",
            "poly": [[0, 0], [10, 10]],
        }
    }
    try:
        parse_dense_payload_mapping(raw=raw, path="gt")
        assert False, "Expected ValueError for poly with <3 points"
    except ValueError:
        pass


def test_parse_dense_payload_mapping_rejects_line_points_mismatch():
    raw: dict[str, object] = {
        "object_1": {
            "desc": "类别=设备",
            "line_points": 3,
            "line": [[0, 0], [10, 10]],
        }
    }
    try:
        parse_dense_payload_mapping(raw=raw, path="gt")
        assert False, "Expected ValueError for line_points mismatch"
    except ValueError:
        pass
