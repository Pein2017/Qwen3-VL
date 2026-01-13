#!/usr/bin/env python3
"""
Sorting utilities for objects in the data conversion pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Literal, Tuple, cast

ObjectOrderingPolicy = Literal["reference_tlbr", "center_tlbr"]


def normalize_object_ordering_policy(value: str | None) -> ObjectOrderingPolicy:
    """Normalize object ordering policy strings.

    This function intentionally accepts a small set of aliases to reduce user
    friction in shell scripts/configs while keeping the canonical values stable.
    """
    if value is None:
        return "center_tlbr"

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"reference_tlbr", "reference", "legacy", "first_xy"}:
        return "reference_tlbr"
    if normalized in {"center_tlbr", "center", "aabb_center"}:
        return "center_tlbr"

    raise ValueError(
        "object_ordering_policy must be one of {'reference_tlbr', 'center_tlbr'}; "
        f"got {value!r}"
    )


def _first_xy(obj: Dict[str, Any]) -> Tuple[float, float]:
    """
    Get sorting reference point according to prompt specification:
    - bbox_2d: top-left corner (x1, y1)
    - poly: first vertex (x1, y1)
    - line: leftmost point (min X, then min Y if tie)
    """
    if "bbox_2d" in obj:
        # Prompt: "使用左上角坐标 (x1, y1)"
        return float(obj["bbox_2d"][0]), float(obj["bbox_2d"][1])
    if "poly" in obj:
        # Prompt: "使用第一个顶点 (x1, y1)"
        return float(obj["poly"][0]), float(obj["poly"][1])
    if "line" in obj:
        # Prompt: "使用最左端点（X 坐标最小的点）作为排序参考；若多个点的 X 坐标相同，则取其中 Y 坐标最小的点"
        coords_raw = obj.get("line")
        if not isinstance(coords_raw, Sequence) or isinstance(coords_raw, (str, bytes)):
            return 0.0, 0.0
        coords = cast(Sequence[float], coords_raw)
        if len(coords) < 2:
            return 0.0, 0.0
        # Extract all points as (x, y) pairs
        points = [
            (float(coords[i]), float(coords[i + 1]))
            for i in range(0, len(coords), 2)
        ]
        # Find leftmost point (min X, then min Y if tie)
        leftmost = min(points, key=lambda p: (p[0], p[1]))
        return leftmost[0], leftmost[1]
    return 0, 0


def _aabb_xyxy(obj: Dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox_2d" in obj:
        x1, y1, x2, y2 = obj["bbox_2d"]
        fx1 = float(x1)
        fy1 = float(y1)
        fx2 = float(x2)
        fy2 = float(y2)
        return (min(fx1, fx2), min(fy1, fy2), max(fx1, fx2), max(fy1, fy2))

    if "poly" in obj:
        pts_raw = obj.get("poly")
        if not isinstance(pts_raw, Sequence) or isinstance(pts_raw, (str, bytes)):
            return (0.0, 0.0, 0.0, 0.0)
        pts = cast(Sequence[float], pts_raw)
        if len(pts) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [float(v) for v in pts[0::2]]
        ys = [float(v) for v in pts[1::2]]
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    if "line" in obj:
        pts_raw = obj.get("line")
        if not isinstance(pts_raw, Sequence) or isinstance(pts_raw, (str, bytes)):
            return (0.0, 0.0, 0.0, 0.0)
        pts = cast(Sequence[float], pts_raw)
        if len(pts) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [float(v) for v in pts[0::2]]
        ys = [float(v) for v in pts[1::2]]
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    return (0.0, 0.0, 0.0, 0.0)


def _aabb_center_xy(obj: Dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = _aabb_xyxy(obj)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _geom_rank(obj: Dict[str, Any]) -> int:
    if "bbox_2d" in obj:
        return 0
    if "poly" in obj:
        return 1
    if "line" in obj:
        return 2
    return 3


def sort_objects_tlbr(
    objects: List[Dict[str, Any]],
    *,
    policy: str | ObjectOrderingPolicy | None = None,
) -> List[Dict[str, Any]]:
    """
    Sort objects top-to-bottom, then left-to-right.

    Policies:
    - ``reference_tlbr`` (legacy): uses per-geometry reference points
      as described in dense prompts (bbox top-left, poly first vertex, line leftmost point).
    - ``center_tlbr`` (default): uses geometry center (AABB center) for ordering. Ties are
      broken deterministically by geometry type rank and the legacy reference point.
    """
    resolved = normalize_object_ordering_policy(None if policy is None else str(policy))

    if resolved == "reference_tlbr":
        # Preserve exact legacy behavior (including stable sort tie behavior).
        return sorted(objects, key=lambda o: (_first_xy(o)[1], _first_xy(o)[0]))

    def _center_key(o: Dict[str, Any]) -> tuple[float, float, int, float, float]:
        cx, cy = _aabb_center_xy(o)
        rx, ry = _first_xy(o)
        return (cy, cx, _geom_rank(o), ry, rx)

    return sorted(objects, key=_center_key)


__all__ = [
    "ObjectOrderingPolicy",
    "normalize_object_ordering_policy",
    "sort_objects_tlbr",
]
