import json
from itertools import islice
from pathlib import Path

import pytest

from src.datasets.preprocessors.summary_labels import SummaryLabelNormalizer


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bbu_full_768_poly-need_review" / "train.jsonl"


def _parse_label_counts(summary: str) -> tuple[int, int]:
    """Return (identifiable, unknown) counts from a summary string."""
    identifiable = 0
    unknown = 0
    for seg in summary.replace(",", "，").split("，"):
        seg = seg.strip()
        if not seg.startswith("标签/"):
            continue
        if "×" not in seg and "x" not in seg and "X" not in seg:
            continue
        # Split off count using both full-width and ascii separators.
        for marker in ("×", "x", "X"):
            if marker in seg:
                desc, count_str = seg.split(marker, 1)
                break
        try:
            count = int(count_str)
        except ValueError:
            continue
        desc_only = desc[len("标签/") :].strip()
        if "无法识别" in desc_only:
            unknown += count
        else:
            identifiable += count
    return identifiable, unknown


@pytest.mark.skipif(not DATA_PATH.is_file(), reason="BBU train.jsonl not available")
def test_summary_label_normalizer_collapses_labels():
    normalizer = SummaryLabelNormalizer()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in islice(f, 0, 64):  # small sample for runtime
            record = json.loads(line)
            if "summary" not in record:
                continue

            orig_id, orig_unknown = _parse_label_counts(record["summary"])
            processed = normalizer(record)
            assert processed is not None

            new_id, new_unknown = _parse_label_counts(processed["summary"])

            # Totals are preserved
            assert (orig_id + orig_unknown) == (new_id + new_unknown)
            assert new_unknown == orig_unknown

            if orig_id > 0:
                # All identifiable labels collapse into exactly one descriptor
                label_segments = [
                    seg.strip()
                    for seg in processed["summary"].replace(",", "，").split("，")
                    if seg.strip().startswith("标签/")
                ]
                merged = [seg for seg in label_segments if seg.startswith("标签/可以识别")]
                assert len(merged) == 1
                assert f"×{new_id}" in merged[0]
