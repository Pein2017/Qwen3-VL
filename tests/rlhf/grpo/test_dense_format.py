from src.rlhf.grpo.rewards.dense.parsing import check_dense_completion_format
from src.rlhf.grpo.rewards.dense.rewards import DenseFormatReward, DenseParseSchemaStrictReward


def _dense_meta_bbu() -> dict[str, object]:
    return {
        "_fusion_mode": "dense",
        "_fusion_template": "target_dense_bbu",
        "_fusion_source": "bbu_dense",
    }


def test_check_dense_completion_format_accepts_header_and_json_like_line():
    meta = _dense_meta_bbu()
    lines = ["<DOMAIN=BBU>, <TASK=DETECTION>", "{not valid json but braces}"]
    assert check_dense_completion_format(lines=lines, meta=meta) is True


def test_dense_format_fails_on_wrong_line_count():
    reward = DenseFormatReward()
    meta = _dense_meta_bbu()
    out = reward(["<DOMAIN=BBU>, <TASK=DETECTION>"], metadata=[meta])
    assert out == [0.0]


def test_dense_format_fails_on_header_mismatch():
    reward = DenseFormatReward()
    meta = _dense_meta_bbu()
    out = reward(["<DOMAIN=RRU>, <TASK=DETECTION>\n{}"], metadata=[meta])
    assert out == [0.0]


def test_dense_format_passes_on_correct_header_two_lines():
    reward = DenseFormatReward()
    meta = _dense_meta_bbu()
    out = reward(["<DOMAIN=BBU>, <TASK=DETECTION>\n{}"], metadata=[meta])
    assert out == [1.0]


def test_dense_parse_schema_strict_is_penalty_only():
    reward = DenseParseSchemaStrictReward()
    meta = _dense_meta_bbu()

    ok = "<DOMAIN=BBU>, <TASK=DETECTION>\n{}"
    bad = "<DOMAIN=BBU>, <TASK=DETECTION>\n{"

    assert reward([ok], metadata=[meta]) == [0.0]
    assert reward([bad], metadata=[meta]) == [-1.0]

