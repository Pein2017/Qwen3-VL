"""Geometry matching for dense GRPO rewards (exact norm1000 raster rulers)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .parsing import DenseGeometry, DenseObject

# Keep consistent with existing offline evaluator (`vis_tools/geometry_eval_metrics.py`):
# We clamp all norm1000 coordinates into [0, 999] for rasterization on a 1000x1000 grid.
NORM1000_GRID_SIZE = 1000

DEFAULT_COCO_THRESHOLDS: tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)

DEFAULT_PRIMARY_THRESHOLD = 0.50
DEFAULT_LINE_TOL = 8.0


def _clamp_norm1000(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return min(float(NORM1000_GRID_SIZE - 1), max(0.0, v))


def _geom_family(gtype: str) -> Literal["region", "line", ""]:
    if gtype in ("bbox_2d", "poly"):
        return "region"
    if gtype == "line":
        return "line"
    return ""


def _aabb_from_points(points: Sequence[float]) -> tuple[float, float, float, float]:
    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _aabb_intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _to_region_poly(geom: DenseGeometry) -> list[tuple[float, float]]:
    pts = list(geom.points)
    if geom.type == "bbox_2d":
        if len(pts) != 4:
            return []
        x1, y1, x2, y2 = pts
        xa = min(x1, x2)
        xb = max(x1, x2)
        ya = min(y1, y2)
        yb = max(y1, y2)
        return [
            (_clamp_norm1000(xa), _clamp_norm1000(ya)),
            (_clamp_norm1000(xb), _clamp_norm1000(ya)),
            (_clamp_norm1000(xb), _clamp_norm1000(yb)),
            (_clamp_norm1000(xa), _clamp_norm1000(yb)),
        ]

    if geom.type != "poly":
        return []
    if len(pts) < 6 or len(pts) % 2 != 0:
        return []
    poly: list[tuple[float, float]] = []
    for i in range(0, len(pts), 2):
        poly.append((_clamp_norm1000(pts[i]), _clamp_norm1000(pts[i + 1])))
    return poly


def region_iou_mask(
    geom_gt: DenseGeometry,
    geom_pred: DenseGeometry,
    *,
    grid_size: int = NORM1000_GRID_SIZE,
) -> float:
    """Filled-shape IoU on a norm1000 grid by rasterizing both shapes."""

    poly_g = _to_region_poly(geom_gt)
    poly_p = _to_region_poly(geom_pred)
    if not poly_g or not poly_p:
        return 0.0

    aabb_g = _aabb_from_points([c for xy in poly_g for c in xy])
    aabb_p = _aabb_from_points([c for xy in poly_p for c in xy])
    if _aabb_intersection_area(aabb_g, aabb_p) <= 0.0:
        return 0.0

    width = int(grid_size)
    height = int(grid_size)
    if width <= 0 or height <= 0:
        return 0.0

    # Tight union window for speed; +1 on ceil to include boundary pixels.
    ux1 = max(0, int(math.floor(min(aabb_g[0], aabb_p[0]))))
    uy1 = max(0, int(math.floor(min(aabb_g[1], aabb_p[1]))))
    ux2 = min(width, int(math.ceil(max(aabb_g[2], aabb_p[2]))) + 1)
    uy2 = min(height, int(math.ceil(max(aabb_g[3], aabb_p[3]))) + 1)
    w = max(1, ux2 - ux1)
    h = max(1, uy2 - uy1)

    def _rasterize(poly: list[tuple[float, float]]) -> NDArray[np.bool_]:
        mask = Image.new("1", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([(x - ux1, y - uy1) for x, y in poly], fill=1)
        return np.asarray(mask, dtype=np.bool_)

    ma = _rasterize(poly_g)
    mb = _rasterize(poly_p)
    inter = int(np.count_nonzero(np.logical_and(ma, mb)))
    union = int(np.count_nonzero(np.logical_or(ma, mb)))
    return (inter / union) if union > 0 else 0.0


def _to_line_xy(points: Sequence[float]) -> list[tuple[float, float]]:
    if len(points) < 4 or len(points) % 2 != 0:
        return []
    return [
        (_clamp_norm1000(points[i]), _clamp_norm1000(points[i + 1]))
        for i in range(0, len(points), 2)
    ]


def tube_iou_line(
    geom_gt: DenseGeometry,
    geom_pred: DenseGeometry,
    *,
    tol: float,
    grid_size: int = NORM1000_GRID_SIZE,
) -> float:
    """Mask-wise TubeIoU for polyline geometries on the norm1000 grid."""

    if geom_gt.type != "line" or geom_pred.type != "line":
        return 0.0
    xy_g = _to_line_xy(geom_gt.points)
    xy_p = _to_line_xy(geom_pred.points)
    if len(xy_g) < 2 or len(xy_p) < 2:
        return 0.0

    width = int(grid_size)
    height = int(grid_size)
    if width <= 0 or height <= 0:
        return 0.0

    line_width = max(1, int(round(2.0 * float(tol))))

    aabb_g = _aabb_from_points([c for xy in xy_g for c in xy])
    aabb_p = _aabb_from_points([c for xy in xy_p for c in xy])
    pad = float(tol)
    aabb_gb = (aabb_g[0] - pad, aabb_g[1] - pad, aabb_g[2] + pad, aabb_g[3] + pad)
    aabb_pb = (aabb_p[0] - pad, aabb_p[1] - pad, aabb_p[2] + pad, aabb_p[3] + pad)
    if _aabb_intersection_area(aabb_gb, aabb_pb) <= 0.0:
        return 0.0

    # Tight union window for speed; +1 on ceil to include boundary pixels.
    ux1 = max(0, int(math.floor(min(aabb_gb[0], aabb_pb[0]))))
    uy1 = max(0, int(math.floor(min(aabb_gb[1], aabb_pb[1]))))
    ux2 = min(width, int(math.ceil(max(aabb_gb[2], aabb_pb[2]))) + 1)
    uy2 = min(height, int(math.ceil(max(aabb_gb[3], aabb_pb[3]))) + 1)
    w = max(1, ux2 - ux1)
    h = max(1, uy2 - uy1)

    def _rasterize(xy: list[tuple[float, float]]) -> NDArray[np.bool_]:
        mask = Image.new("1", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.line([(x - ux1, y - uy1) for x, y in xy], fill=1, width=line_width)
        return np.asarray(mask, dtype=np.bool_)

    ma = _rasterize(xy_g)
    mb = _rasterize(xy_p)
    inter = int(np.count_nonzero(np.logical_and(ma, mb)))
    union = int(np.count_nonzero(np.logical_or(ma, mb)))
    return (inter / union) if union > 0 else 0.0


def build_overlap_matrix(
    gt_objects: Sequence[DenseObject],
    pred_objects: Sequence[DenseObject],
    *,
    line_tol: float,
) -> list[list[float]]:
    num_gt = len(gt_objects)
    num_pred = len(pred_objects)
    matrix: list[list[float]] = [[0.0] * num_pred for _ in range(num_gt)]
    for gi, gt in enumerate(gt_objects):
        gfam = _geom_family(gt.geometry.type)
        if not gfam:
            continue
        for pi, pred in enumerate(pred_objects):
            pfam = _geom_family(pred.geometry.type)
            if pfam != gfam:
                continue
            if gfam == "region":
                matrix[gi][pi] = region_iou_mask(gt.geometry, pred.geometry)
            else:
                matrix[gi][pi] = tube_iou_line(gt.geometry, pred.geometry, tol=line_tol)
    return matrix


def greedy_match(
    overlap: Sequence[Sequence[float]],
    *,
    threshold: float,
    gt_objects: Sequence[DenseObject],
    pred_objects: Sequence[DenseObject],
    require_category_match: bool,
) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for gi, row in enumerate(overlap):
        for pi, score in enumerate(row):
            if score < threshold:
                continue
            if require_category_match:
                gcat = gt_objects[gi].category
                pcat = pred_objects[pi].category
                if not gcat or gcat != pcat:
                    continue
            pairs.append((float(score), gi, pi))

    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))
    return matches


@dataclass(frozen=True)
class Prf:
    precision: float
    recall: float


def prf(*, matched: int, gt_total: int, pred_total: int) -> Prf:
    recall = (matched / gt_total) if gt_total > 0 else 0.0
    precision = (matched / pred_total) if pred_total > 0 else 0.0
    return Prf(precision=precision, recall=recall)


def fbeta(*, matched: int, gt_total: int, pred_total: int, beta: float) -> float:
    if beta <= 0:
        beta = 1.0
    pr = prf(matched=matched, gt_total=gt_total, pred_total=pred_total)
    p = pr.precision
    r = pr.recall
    beta2 = beta * beta
    denom = beta2 * p + r
    if denom <= 0:
        return 0.0
    return (1.0 + beta2) * p * r / denom


__all__ = [
    "DEFAULT_COCO_THRESHOLDS",
    "DEFAULT_LINE_TOL",
    "DEFAULT_PRIMARY_THRESHOLD",
    "NORM1000_GRID_SIZE",
    "build_overlap_matrix",
    "fbeta",
    "greedy_match",
    "region_iou_mask",
    "tube_iou_line",
]
