from src.datasets.preprocessors.summary_labels import SummaryLabelNormalizer


def test_summary_label_normalizer_passthrough_json():
    normalizer = SummaryLabelNormalizer()
    summary = '{"统计": [{"类别": "标签", "文本": {"ABC-01": 1}}]}'
    record = {"summary": summary}
    processed = normalizer(record)
    assert processed is not None
    assert processed["summary"] == summary
