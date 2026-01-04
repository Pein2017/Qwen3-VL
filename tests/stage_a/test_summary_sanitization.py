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
