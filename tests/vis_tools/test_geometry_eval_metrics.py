from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from vis_tools.geometry_eval_metrics import (
    DescLabels,
    EvalObject,
    EvalMode,
    NORM1000_GRID_SIZE,
    _as_eval_objects,
    _greedy_match,
    parse_desc_labels,
    region_iou,
    tube_iou_line,
)


def test_parse_desc_labels_key_value() -> None:
    labels = parse_desc_labels("类别=BBU安装螺丝,可见性=完整,符合性=符合")
    assert labels.phase == "BBU安装螺丝"
    assert labels.category == "BBU安装螺丝"


def test_parse_desc_labels_legacy_umbrella_phase() -> None:
    labels = parse_desc_labels("螺丝、光纤插头/BBU安装螺丝,完整,符合")
    assert labels.phase == "螺丝、光纤插头"
    assert labels.category == "BBU安装螺丝"


def test_region_iou_bbox_poly_cross_type_identity() -> None:
    bbox: EvalObject = {"type": "bbox_2d", "points": [0, 0, 10, 10], "desc": ""}
    poly: EvalObject = {
        "type": "poly",
        "points": [0, 0, 10, 0, 10, 10, 0, 10],
        "desc": "",
    }
    assert math.isclose(region_iou(bbox, poly), 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_tube_iou_line_identity() -> None:
    line: EvalObject = {"type": "line", "points": [100, 100, 900, 100], "desc": ""}
    assert math.isclose(
        tube_iou_line(line, line, tol=8.0), 1.0, rel_tol=0.0, abs_tol=1e-9
    )


def test_tube_iou_line_separated_is_zero() -> None:
    line_a: EvalObject = {"type": "line", "points": [100, 100, 900, 100], "desc": ""}
    line_b: EvalObject = {"type": "line", "points": [100, 300, 900, 300], "desc": ""}
    assert tube_iou_line(line_a, line_b, tol=8.0) == 0.0


def test_tube_iou_line_increases_with_tolerance() -> None:
    line_a: EvalObject = {"type": "line", "points": [100, 100, 900, 100], "desc": ""}
    line_b: EvalObject = {
        "type": "line",
        "points": [100, 110, 900, 110],
        "desc": "",
    }  # 10px apart
    iou_tol1 = tube_iou_line(line_a, line_b, tol=1.0)
    iou_tol8 = tube_iou_line(line_a, line_b, tol=8.0)
    assert iou_tol1 == 0.0
    assert iou_tol8 > iou_tol1


def test_matching_determinism_tie_break() -> None:
    overlap = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    gt_labels = [
        DescLabels(phase="x", category="x"),
        DescLabels(phase="x", category="x"),
    ]
    pred_labels = [
        DescLabels(phase="x", category="x"),
        DescLabels(phase="x", category="x"),
    ]

    matches = _greedy_match(
        overlap,
        threshold=0.5,
        mode=EvalMode.LOCALIZATION,
        gt_labels=gt_labels,
        pred_labels=pred_labels,
    )
    assert matches == [(0, 0), (1, 1)]


def test_eval_dump_requires_output_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from vis_tools import eval_dump

    jsonl_path = tmp_path / "gt_vs_pred.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["eval_dump.py", str(jsonl_path)])
    with pytest.raises(SystemExit) as excinfo:
        eval_dump.main()
    assert excinfo.value.code != 0

    err = capsys.readouterr().err
    assert "--output-json" in err


def test_eval_dump_rejects_multiple_inputs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from vis_tools import eval_dump

    jsonl_a = tmp_path / "a.jsonl"
    jsonl_b = tmp_path / "b.jsonl"
    out_json = tmp_path / "out.json"
    jsonl_a.write_text("{}", encoding="utf-8")
    jsonl_b.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_dump.py",
            str(jsonl_a),
            str(jsonl_b),
            "--output-json",
            str(out_json),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        eval_dump.main()
    assert excinfo.value.code != 0

    err = capsys.readouterr().err
    assert "only a single input" in err
    assert not out_json.exists()


def test_norm1000_clamping_and_grid_size() -> None:
    assert NORM1000_GRID_SIZE == 1000

    raw_bbox: list[dict[str, object]] = [
        {
            "type": "bbox_2d",
            "points": [-5, 0, 1000, 1001],
            "desc": "",
        }
    ]
    out = _as_eval_objects(raw_bbox)
    assert out[0]["points"] == [0.0, 0.0, 999.0, 999.0]

    raw_line: list[dict[str, object]] = [
        {
            "type": "line",
            "points": [0, 0, 1000, 1000],
            "desc": "",
        }
    ]
    out_line = _as_eval_objects(raw_line)
    assert out_line[0]["points"] == [0.0, 0.0, 999.0, 999.0]
