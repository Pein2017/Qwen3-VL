from src.rlhf.grpo.rewards.dense.matching import (
    DEFAULT_LINE_TOL,
    region_iou_mask,
    tube_iou_line,
)
from src.rlhf.grpo.rewards.dense.parsing import DenseGeometry
from vis_tools.geometry_eval_metrics import EvalObject
from vis_tools.geometry_eval_metrics import region_iou as eval_region_iou
from vis_tools.geometry_eval_metrics import tube_iou_line as eval_tube_iou_line


def test_region_iou_mask_identical_bbox_is_one():
    geom = DenseGeometry(type="bbox_2d", points=(0.0, 0.0, 10.0, 10.0))
    assert region_iou_mask(geom, geom) == 1.0


def test_region_iou_mask_disjoint_is_zero():
    a = DenseGeometry(type="bbox_2d", points=(0.0, 0.0, 10.0, 10.0))
    b = DenseGeometry(type="bbox_2d", points=(100.0, 100.0, 110.0, 110.0))
    assert region_iou_mask(a, b) == 0.0


def test_region_iou_mask_clamps_to_0_999():
    a = DenseGeometry(type="bbox_2d", points=(-10.0, -10.0, 1000.0, 1000.0))
    b = DenseGeometry(type="bbox_2d", points=(0.0, 0.0, 999.0, 999.0))
    assert region_iou_mask(a, b) == 1.0


def test_tube_iou_line_identical_is_one():
    geom = DenseGeometry(type="line", points=(0.0, 0.0, 999.0, 999.0))
    assert tube_iou_line(geom, geom, tol=DEFAULT_LINE_TOL) == 1.0


def test_tube_iou_line_no_overlap_is_zero():
    a = DenseGeometry(type="line", points=(0.0, 0.0, 0.0, 999.0))
    b = DenseGeometry(type="line", points=(999.0, 0.0, 999.0, 999.0))
    assert tube_iou_line(a, b, tol=DEFAULT_LINE_TOL) == 0.0


def test_offline_eval_region_iou_matches_reward_region_iou_mask():
    bbox = DenseGeometry(type="bbox_2d", points=(0.0, 0.0, 10.0, 10.0))
    poly = DenseGeometry(
        type="poly", points=(0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
    )
    expected = region_iou_mask(bbox, poly)

    obj_bbox: EvalObject = {
        "type": "bbox_2d",
        "points": [0.0, 0.0, 10.0, 10.0],
        "desc": "",
    }
    obj_poly: EvalObject = {
        "type": "poly",
        "points": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        "desc": "",
    }
    got = eval_region_iou(obj_bbox, obj_poly)
    assert abs(got - expected) < 1e-12


def test_offline_eval_tube_iou_matches_reward_tube_iou():
    line = DenseGeometry(type="line", points=(0.0, 0.0, 999.0, 999.0))
    expected = tube_iou_line(line, line, tol=DEFAULT_LINE_TOL)

    obj: EvalObject = {"type": "line", "points": [0.0, 0.0, 999.0, 999.0], "desc": ""}
    got = eval_tube_iou_line(obj, obj, tol=DEFAULT_LINE_TOL)
    assert abs(got - expected) < 1e-12
