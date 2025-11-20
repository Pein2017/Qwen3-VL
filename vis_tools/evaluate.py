from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from src.datasets.geometry import aabb_area, get_aabb, intersect_aabb


GeometryObject = Dict[str, Any]


@dataclass(frozen=True)
class MatchResult:
    """Result of geometry-based matching for one image/batch.

    The evaluation is intentionally geometry-only:
    - No parsing of `desc` or taxonomy
    - Matching is based on overlap (IoU) and geometry type only
    """

    num_gt: int
    num_pred: int
    num_matched: int
    num_missing: int
    # (gt_index, pred_index) pairs after 1-1 matching
    matches: List[Tuple[int, int]]
    # Indices into the GT list that have no matching prediction
    missing_gt_indices: List[int]


def _to_geom_dict(obj: GeometryObject) -> Dict[str, List[float]]:
    """Convert a vis_qwen3-style object to geometry dict used by src.datasets.geometry.

    Expected input schema per object:
    {"type": "bbox_2d"|"poly"|"line", "points": [x0, y0, ...]}
    Coordinates must already be in a single, consistent space (pixel or norm).
    """

    gtype = obj.get("type")
    pts = obj.get("points") or []
    if not isinstance(pts, (list, tuple)):
        pts = []
    pts_f = [float(x) for x in pts]

    if gtype == "bbox_2d":
        return {"bbox_2d": pts_f}
    if gtype == "poly":
        return {"poly": pts_f}
    if gtype == "line":
        return {"line": pts_f}

    raise ValueError(f"Unknown geometry type for evaluation: {gtype!r}")


def _pair_points(points: Sequence[float]) -> List[Tuple[float, float]]:
    assert len(points) % 2 == 0, f"points length must be even, got {len(points)}"
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


def _polygon_area(points: Sequence[float]) -> float:
    """Signed polygon area (positive for one orientation, negative for the other)."""

    pts = _pair_points(points)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _convex_clip(subject: Sequence[float], clip: Sequence[float]) -> List[float]:
    """Clip a convex subject polygon by a convex clip polygon (Sutherland–Hodgman).

    Both polygons are given as flat [x0,y0,x1,y1,...]. Returns the intersection
    polygon in the same format, or an empty list if there is no overlap.
    """

    subject_pts = _pair_points(subject)
    clip_pts = _pair_points(clip)
    if len(subject_pts) < 3 or len(clip_pts) < 3:
        return []

    # Orientation of the clip polygon (sign only)
    orient = 0.0
    for i in range(len(clip_pts)):
        x1, y1 = clip_pts[i]
        x2, y2 = clip_pts[(i + 1) % len(clip_pts)]
        orient += x1 * y2 - x2 * y1
    if orient == 0.0:
        return []

    def is_inside(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        # cross > 0 when p is to the left of ab in standard coords; multiply by
        # orient so the sign matches the interior side regardless of winding.
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return cross * orient >= 0.0

    def line_intersection(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> Tuple[float, float]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0.0:
            # Parallel or nearly parallel; fallback to second point
            return p2
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        return px, py

    output = subject_pts
    for i in range(len(clip_pts)):
        a = clip_pts[i]
        b = clip_pts[(i + 1) % len(clip_pts)]
        if not output:
            break
        input_pts = output
        output = []
        s = input_pts[-1]
        for e in input_pts:
            if is_inside(e, a, b):
                if not is_inside(s, a, b):
                    output.append(line_intersection(s, e, a, b))
                output.append(e)
            elif is_inside(s, a, b):
                output.append(line_intersection(s, e, a, b))
            s = e
    if len(output) < 3:
        return []
    flat: List[float] = []
    for x, y in output:
        flat.extend([x, y])
    return flat


def _segment_distance_sq(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Squared distance from point P to segment AB."""

    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0.0:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        dx = px - bx
        dy = py - by
        return dx * dx + dy * dy
    t = c1 / c2
    projx = ax + t * vx
    projy = ay + t * vy
    dx = px - projx
    dy = py - projy
    return dx * dx + dy * dy


def _build_segments(points: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    pts = _pair_points(points)
    if len(pts) < 2:
        return []
    segments: List[Tuple[float, float, float, float]] = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        segments.append((x1, y1, x2, y2))
    return segments


def _sample_polyline(points: Sequence[float], step: float = 2.0) -> List[Tuple[float, float]]:
    """Sample points along a polyline at roughly fixed step in pixels."""

    pts = _pair_points(points)
    if not pts:
        return []
    if len(pts) == 1:
        return pts
    samples: List[Tuple[float, float]] = [pts[0]]
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len <= 0.0:
            continue
        n = max(1, int(seg_len / step))
        for k in range(1, n + 1):
            t = k / n
            samples.append((x1 + t * dx, y1 + t * dy))
    return samples


def _polyline_coverage(
    points: Sequence[float],
    other_segments: List[Tuple[float, float, float, float]],
    tol_sq: float,
    step: float = 2.0,
) -> float:
    """Fraction of samples on `points` within sqrt(tol_sq) of `other_segments`."""

    samples = _sample_polyline(points, step=step)
    if not samples or not other_segments:
        return 0.0
    hit = 0
    for px, py in samples:
        min_d2 = min(
            _segment_distance_sq(px, py, ax, ay, bx, by)
            for (ax, ay, bx, by) in other_segments
        )
        if min_d2 <= tol_sq:
            hit += 1
    return hit / len(samples)


def iou_aabb(obj_gt: GeometryObject, obj_pred: GeometryObject) -> float:
    """Axis-aligned IoU between two geometry objects via their AABBs."""

    g_geom = _to_geom_dict(obj_gt)
    p_geom = _to_geom_dict(obj_pred)

    bb_g = get_aabb(g_geom)
    bb_p = get_aabb(p_geom)

    inter = intersect_aabb(bb_g, bb_p)
    inter_area = aabb_area(inter)
    if inter_area <= 0.0:
        return 0.0

    area_g = aabb_area(bb_g)
    area_p = aabb_area(bb_p)
    union = area_g + area_p - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def iou_bbox(obj_gt: GeometryObject, obj_pred: GeometryObject) -> float:
    """Exact IoU for axis-aligned bbox_2d geometries."""

    g_geom = _to_geom_dict(obj_gt)
    p_geom = _to_geom_dict(obj_pred)
    pts_g = g_geom.get("bbox_2d") or []
    pts_p = p_geom.get("bbox_2d") or []
    if len(pts_g) != 4 or len(pts_p) != 4:
        return iou_aabb(obj_gt, obj_pred)
    x1g, y1g, x2g, y2g = pts_g
    x1p, y1p, x2p, y2p = pts_p
    x1i = max(x1g, x1p)
    y1i = max(y1g, y1p)
    x2i = min(x2g, x2p)
    y2i = min(y2g, y2p)
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    inter_area = (x2i - x1i) * (y2i - y1i)
    area_g = max(0.0, (x2g - x1g) * (y2g - y1g))
    area_p = max(0.0, (x2p - x1p) * (y2p - y1p))
    union = area_g + area_p - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def iou_poly(obj_gt: GeometryObject, obj_pred: GeometryObject) -> float:
    """Polygon IoU for polygon geometries using convex polygon intersection."""

    g_geom = _to_geom_dict(obj_gt)
    p_geom = _to_geom_dict(obj_pred)
    pts_g = g_geom.get("poly") or []
    pts_p = p_geom.get("poly") or []
    if len(pts_g) < 6 or len(pts_p) < 6:
        return iou_aabb(obj_gt, obj_pred)

    area_g = abs(_polygon_area(pts_g))
    area_p = abs(_polygon_area(pts_p))
    if area_g <= 0.0 or area_p <= 0.0:
        return 0.0

    inter_poly = _convex_clip(pts_g, pts_p)
    if not inter_poly:
        return 0.0
    inter_area = abs(_polygon_area(inter_poly))
    if inter_area <= 0.0:
        return 0.0

    union = area_g + area_p - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def iou_line(
    obj_gt: GeometryObject,
    obj_pred: GeometryObject,
    *,
    tol: float = 3.0,
    step: float = 2.0,
) -> float:
    """Symmetric coverage-based F1-like score for polyline geometries.

    We treat each line as a continuous curve:
    - Sample points along each polyline at roughly fixed arc-length step.
    - For each sampled point, check whether it's within distance `tol` of the
      other line's segments (in the same coordinate space: pixels or norm1000).
    - cov_gt: fraction of GT samples covered by the prediction (recall-like).
    - cov_pred: fraction of prediction samples close to GT (precision-like).

    The final score is the harmonic mean of these coverages:
        score = 2 * cov_gt * cov_pred / (cov_gt + cov_pred)
    which down-weights cases where only one side overlaps well.
    """

    g_geom = _to_geom_dict(obj_gt)
    p_geom = _to_geom_dict(obj_pred)
    pts_g = g_geom.get("line") or []
    pts_p = p_geom.get("line") or []
    if len(pts_g) < 4 or len(pts_p) < 4:
        return iou_aabb(obj_gt, obj_pred)

    segs_g = _build_segments(pts_g)
    segs_p = _build_segments(pts_p)
    if not segs_g or not segs_p:
        return 0.0

    tol_sq = tol * tol
    cov_g = _polyline_coverage(pts_g, segs_p, tol_sq, step=step)
    cov_p = _polyline_coverage(pts_p, segs_g, tol_sq, step=step)
    if cov_g <= 0.0 and cov_p <= 0.0:
        return 0.0
    if cov_g <= 0.0 or cov_p <= 0.0:
        return 0.0
    return 2.0 * cov_g * cov_p / (cov_g + cov_p)


def compute_iou(obj_gt: GeometryObject, obj_pred: GeometryObject) -> float:
    """Dispatch IoU computation based on geometry type.

    bbox_2d  -> exact rectangle IoU
    poly     -> convex polygon IoU
    line     -> coverage-based IoU-like score
    """

    gtype = obj_gt.get("type")
    if gtype == "bbox_2d":
        return iou_bbox(obj_gt, obj_pred)
    if gtype == "poly":
        return iou_poly(obj_gt, obj_pred)
    if gtype == "line":
        return iou_line(obj_gt, obj_pred)
    # Fallback: AABB IoU for unknown types
    return iou_aabb(obj_gt, obj_pred)


def match_geometries(
    gt_objects: Sequence[GeometryObject],
    pred_objects: Sequence[GeometryObject],
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy 1-1 matching between GT and predictions using geometry-aware IoU.

    - Matches only objects with the same `type` (bbox_2d / poly / line).
    - Uses type-specific IoU metrics (bbox, polygon, line coverage).
    - Prediction scores are ignored; the best IoU matches are chosen first.

    This is designed to be pluggable into both visualization (vis_qwen3) and
    training eval loops: all it needs is two flat lists of geometry objects.
    """

    # Build candidate pairs (IoU >= threshold) with type constraint
    pairs: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_objects):
        gtype = gt.get("type")
        if gtype not in ("bbox_2d", "poly", "line"):
            continue
        for pi, pred in enumerate(pred_objects):
            if pred.get("type") != gtype:
                continue
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))

    # Greedy 1-1 assignment by descending IoU
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: List[Tuple[int, int]] = []

    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))

    num_gt = len(gt_objects)
    num_pred = len(pred_objects)
    missing_gt_indices = sorted(i for i in range(num_gt) if i not in matched_gt)
    num_missing = len(missing_gt_indices)

    return MatchResult(
        num_gt=num_gt,
        num_pred=num_pred,
        num_matched=len(matches),
        num_missing=num_missing,
        matches=matches,
        missing_gt_indices=missing_gt_indices,
    )


def _build_iou_matrix(
    gt_objects: Sequence[GeometryObject], pred_objects: Sequence[GeometryObject]
) -> Tuple[List[List[float]], List[str], List[str]]:
    """Precompute IoUs once so we can sweep multiple thresholds efficiently."""

    num_gt = len(gt_objects)
    num_pred = len(pred_objects)
    gt_types = [str(o.get("type", "")) for o in gt_objects]
    pred_types = [str(o.get("type", "")) for o in pred_objects]

    matrix: List[List[float]] = [[0.0] * num_pred for _ in range(num_gt)]
    for gi, gt in enumerate(gt_objects):
        gtype = gt_types[gi]
        if gtype not in ("bbox_2d", "poly", "line"):
            continue
        for pi, pred in enumerate(pred_objects):
            if pred_types[pi] != gtype:
                continue
            matrix[gi][pi] = compute_iou(gt, pred)
    return matrix, gt_types, pred_types


def _greedy_match_from_matrix(matrix: List[List[float]], threshold: float) -> List[Tuple[int, int]]:
    """Greedy 1-1 matching using a precomputed IoU matrix."""

    pairs: List[Tuple[float, int, int]] = []
    for gi, row in enumerate(matrix):
        for pi, iou in enumerate(row):
            if iou >= threshold:
                pairs.append((iou, gi, pi))

    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))
    return matches


def match_geometries_multi(
    gt_objects: Sequence[GeometryObject],
    pred_objects: Sequence[GeometryObject],
    *,
    iou_thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
) -> Dict[float, MatchResult]:
    """Match geometries across multiple IoU thresholds (COCO-like sweep).

    This reuses a shared IoU matrix so callers can compute precision/recall per
    threshold without recomputing IoUs each time. Prediction scores are still
    ignored; you get thresholded counts, not AP.
    """

    thresholds = sorted(set(float(t) for t in iou_thresholds))
    if not thresholds:
        return {}

    matrix, _, _ = _build_iou_matrix(gt_objects, pred_objects)
    num_gt = len(gt_objects)
    num_pred = len(pred_objects)

    out: Dict[float, MatchResult] = {}
    for thr in thresholds:
        matches = _greedy_match_from_matrix(matrix, thr)
        matched_gt = {gi for gi, _ in matches}
        missing_gt_indices = sorted(i for i in range(num_gt) if i not in matched_gt)
        out[thr] = MatchResult(
            num_gt=num_gt,
            num_pred=num_pred,
            num_matched=len(matches),
            num_missing=len(missing_gt_indices),
            matches=matches,
            missing_gt_indices=missing_gt_indices,
        )
    return out
