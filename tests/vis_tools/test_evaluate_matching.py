from __future__ import annotations

import math

from vis_tools import evaluate as geom_eval


def test_match_geometries_bbox_to_poly_cross_type() -> None:
    gt: list[geom_eval.GeometryObject] = [{"type": "bbox_2d", "points": [0, 0, 10, 10]}]
    pred: list[geom_eval.GeometryObject] = [
        {"type": "poly", "points": [0, 0, 10, 0, 10, 10, 0, 10]}
    ]
    res = geom_eval.match_geometries(gt, pred, iou_threshold=0.5)
    assert res.num_gt == 1
    assert res.num_pred == 1
    assert res.num_matched == 1
    assert res.num_missing == 0
    assert res.matches == [(0, 0)]


def test_match_geometries_poly_to_bbox_cross_type() -> None:
    gt: list[geom_eval.GeometryObject] = [
        {"type": "poly", "points": [0, 0, 10, 0, 10, 10, 0, 10]}
    ]
    pred: list[geom_eval.GeometryObject] = [{"type": "bbox_2d", "points": [0, 0, 10, 10]}]
    res = geom_eval.match_geometries(gt, pred, iou_threshold=0.5)
    assert res.num_matched == 1
    assert res.num_missing == 0
    assert res.matches == [(0, 0)]


def test_match_geometries_same_type_bbox_unchanged() -> None:
    gt: list[geom_eval.GeometryObject] = [{"type": "bbox_2d", "points": [0, 0, 10, 10]}]
    pred: list[geom_eval.GeometryObject] = [{"type": "bbox_2d", "points": [0, 0, 10, 10]}]
    res = geom_eval.match_geometries(gt, pred, iou_threshold=0.5)
    assert res.num_matched == 1
    assert res.num_missing == 0


def test_match_geometries_different_families_do_not_match() -> None:
    gt: list[geom_eval.GeometryObject] = [{"type": "bbox_2d", "points": [0, 0, 10, 10]}]
    pred: list[geom_eval.GeometryObject] = [{"type": "line", "points": [0, 0, 10, 10]}]
    res = geom_eval.match_geometries(gt, pred, iou_threshold=0.01)
    assert res.num_matched == 0
    assert res.num_missing == 1


def test_compute_iou_bbox_poly_identity_is_one() -> None:
    gt: geom_eval.GeometryObject = {"type": "bbox_2d", "points": [0, 0, 10, 10]}
    pred: geom_eval.GeometryObject = {"type": "poly", "points": [0, 0, 10, 0, 10, 10, 0, 10]}
    assert math.isclose(geom_eval.compute_iou(gt, pred), 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_match_geometries_multi_includes_cross_type() -> None:
    gt: list[geom_eval.GeometryObject] = [{"type": "bbox_2d", "points": [0, 0, 10, 10]}]
    pred: list[geom_eval.GeometryObject] = [
        {"type": "poly", "points": [0, 0, 10, 0, 10, 10, 0, 10]}
    ]
    out = geom_eval.match_geometries_multi(gt, pred, iou_thresholds=(0.5, 0.9))
    assert out[0.5].num_matched == 1
    assert out[0.9].num_matched == 1
