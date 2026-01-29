import json

from src.stage_a.inference import sanitize_summary_by_dataset as sanitize_inference
from src.stage_a.postprocess import sanitize_summary_by_dataset as sanitize_postprocess


def test_stage_a_inference_sanitization_accepts_stat_only_json():
    summary = '{"统计":[{"类别":"标签","文本":{"A":1}}]}'
    output = sanitize_inference(summary, "bbu")
    parsed = json.loads(output)
    assert "统计" in parsed
    assert "dataset" not in parsed


def test_stage_a_postprocess_sanitization_accepts_stat_only_json():
    summary = '{"统计":[{"类别":"标签","文本":{"A":1}}]}'
    output = sanitize_postprocess(summary, "rru")
    parsed = json.loads(output)
    assert "统计" in parsed
    assert "dataset" not in parsed


def test_stage_a_sanitization_repairs_stats_assignment_list_format():
    # Some one-shot generations drift to a DETECTION-like format:
    #   <DOMAIN=BBU>, <TASK=DETECTION>
    #   统计=[{...}]
    # This is not valid JSON, but it is unambiguous to repair into {"统计": [...]}.
    raw = '<DOMAIN=BBU>, <TASK=DETECTION>\\n统计=[{"类别":"标签","文本":{"A":1}}]'

    out_inf = sanitize_inference(raw, "bbu")
    out_post = sanitize_postprocess(raw, "bbu")

    for out in (out_inf, out_post):
        parsed = json.loads(out)
        assert isinstance(parsed.get("统计"), list)
        assert parsed["统计"][0]["类别"] == "标签"


def test_stage_a_sanitization_repairs_rru_group_stats_assignment():
    raw = (
        '统计=[{"类别":"站点距离","站点距离":{"13":1}}]\\n分组统计=[{"组":1,"统计":1}]'
    )
    out_inf = sanitize_inference(raw, "rru")
    out_post = sanitize_postprocess(raw, "rru")
    for out in (out_inf, out_post):
        parsed = json.loads(out)
        assert "统计" in parsed
        assert "分组统计" in parsed
