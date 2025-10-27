from __future__ import annotations

import math
from typing import List, Sequence, Tuple
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union


def _pair_points(points: Sequence[float]) -> List[Tuple[float, float]]:
    assert len(points) % 2 == 0, f"points length must be even, got {len(points)}"
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


def points_to_xyxy(points: Sequence[float]) -> List[float]:
    """Compute [x1,y1,x2,y2] that encloses arbitrary points (bbox, quad, or line)."""
    pts = _pair_points(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def scale_points(points: Sequence[float], sx: float, sy: float) -> List[float]:
    """Scale x by sx and y by sy for all points in a flat [x0,y0,x1,y1,...] list."""
    out: List[float] = []
    for i, v in enumerate(points):
        if i % 2 == 0:
            out.append(float(v) * sx)
        else:
            out.append(float(v) * sy)
    return out


def normalize_points(points: Sequence[float], width: float, height: float, space: str) -> List[int]:
    """Normalize to integer grid (norm100 or norm1000) from pixel coords.

    space: "norm100" or "norm1000"
    """
    if space not in {"norm100", "norm1000"}:
        raise ValueError(f"space must be 'norm100' or 'norm1000', got {space}")
    denom_x = width if width else 1.0
    denom_y = height if height else 1.0
    scale = 100 if space == "norm100" else 1000
    out: List[int] = []
    for i, v in enumerate(points):
        if i % 2 == 0:
            out.append(int(round(float(v) / denom_x * scale)))
        else:
            out.append(int(round(float(v) / denom_y * scale)))
    return out


def _matmul_affine(M: List[List[float]], x: float, y: float) -> Tuple[float, float]:
    """Apply 3x3 affine to (x, y, 1)."""
    nx = M[0][0] * x + M[0][1] * y + M[0][2]
    ny = M[1][0] * x + M[1][1] * y + M[1][2]
    w = M[2][0] * x + M[2][1] * y + M[2][2]
    if w == 0:
        return nx, ny
    return nx / w, ny / w


def compose_affine(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Return A @ B for 3x3 matrices."""
    C = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(3))
    return C


def translate(tx: float, ty: float) -> List[List[float]]:
    return [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]


def scale_matrix(sx: float, sy: float) -> List[List[float]]:
    return [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]


def rotate_center(deg: float, cx: float, cy: float) -> List[List[float]]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    R = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T1 = translate(-cx, -cy)
    T2 = translate(cx, cy)
    return compose_affine(T2, compose_affine(R, T1))


def hflip_matrix(width: float) -> List[List[float]]:
    # x' = (width - 1) - x
    return [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def vflip_matrix(height: float) -> List[List[float]]:
    # y' = (height - 1) - y
    return [[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]]


def apply_affine(points: Sequence[float], M: List[List[float]]) -> List[float]:
    out: List[float] = []
    for x, y in _pair_points(points):
        nx, ny = _matmul_affine(M, x, y)
        out.extend([nx, ny])
    return out


def invert_affine(M: List[List[float]]) -> List[List[float]]:
    """Invert an affine 3x3 matrix with bottom row [0,0,1].

    For M = [[a,b,c],[d,e,f],[0,0,1]], returns M_inv such that M_inv @ M = I.
    """
    a, b, c = M[0]
    d, e, f = M[1]
    det = a * e - b * d
    if det == 0:
        # Singular; return identity as safe fallback
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    inv_a = e / det
    inv_b = -b / det
    inv_d = -d / det
    inv_e = a / det
    inv_c = -(inv_a * c + inv_b * f)
    inv_f = -(inv_d * c + inv_e * f)
    return [[inv_a, inv_b, inv_c], [inv_d, inv_e, inv_f], [0.0, 0.0, 1.0]]


def scale_center(sx: float, sy: float, cx: float, cy: float) -> List[List[float]]:
    """Scale about a center point (cx, cy)."""
    T1 = translate(-cx, -cy)
    S = scale_matrix(sx, sy)
    T2 = translate(cx, cy)
    return compose_affine(T2, compose_affine(S, T1))


def clamp_points(points: Sequence[float], width: float, height: float) -> List[int]:
    out: List[int] = []
    for i, v in enumerate(points):
        if i % 2 == 0:
            out.append(max(0, min(int(width) - 1, int(round(float(v))))))
        else:
            out.append(max(0, min(int(height) - 1, int(round(float(v))))))
    return out


def dedupe_consecutive_points(points: Sequence[int]) -> List[int]:
    if not points:
        return []
    out: List[int] = []
    prev: Tuple[int, int] | None = None
    for x, y in _pair_points(points):
        pt = (int(round(x)), int(round(y)))
        if prev is None or pt != prev:
            out.extend([pt[0], pt[1]])
            prev = pt
    return out


# Round points to nearest integer without clamping to bounds
def round_points(points: Sequence[float]) -> List[int]:
    out: List[int] = []
    for v in points:
        out.append(int(round(float(v))))
    return out


# --- Robust geometry helpers for augmentation ---

from typing import Callable


def _rect_bounds(width: float, height: float) -> Tuple[float, float, float, float]:
    # left, top, right, bottom in float space
    return 0.0, 0.0, float(max(0, int(round(width)) - 1)), float(max(0, int(round(height)) - 1))


def is_clockwise(points: Sequence[float]) -> bool:
    # Signed area < 0 means clockwise for screen coords (y down)
    pts = _pair_points(points)
    if len(pts) < 3:
        return True
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += (x2 - x1) * (y2 + y1)  # variant equivalent to shoelace sign
    return area > 0.0


def to_clockwise(points: Sequence[float]) -> List[float]:
    if is_clockwise(points):
        return [float(v) for v in points]
    pts = list(reversed(_pair_points(points)))
    out: List[float] = []
    for x, y in pts:
        out.extend([float(x), float(y)])
    return out


def _inside(px: float, py: float, edge: str, bounds: Tuple[float, float, float, float]) -> bool:
    l, t, r, b = bounds
    if edge == "left":
        return px >= l
    if edge == "right":
        return px <= r
    if edge == "top":
        return py >= t
    if edge == "bottom":
        return py <= b
    return True


def _intersect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    edge: str,
    bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    l, t, r, b = bounds
    dx = x2 - x1
    dy = y2 - y1
    if edge == "left":
        x = l
        y = y1 + dy * (l - x1) / (dx if dx != 0 else 1e-12)
        return x, y
    if edge == "right":
        x = r
        y = y1 + dy * (r - x1) / (dx if dx != 0 else 1e-12)
        return x, y
    if edge == "top":
        y = t
        x = x1 + dx * (t - y1) / (dy if dy != 0 else 1e-12)
        return x, y
    if edge == "bottom":
        y = b
        x = x1 + dx * (b - y1) / (dy if dy != 0 else 1e-12)
        return x, y
    return x2, y2


def sutherland_hodgman_clip(points: Sequence[float], width: float, height: float) -> List[float]:
    """Clip polygon to image rectangle [0..W-1]x[0..H-1] in float.

    Returns a flat list of points (may be empty). Assumes input polygon is simple/convex.
    """
    bounds = _rect_bounds(width, height)
    poly = _pair_points(points)
    if len(poly) < 3:
        return []
    for edge in ("left", "right", "top", "bottom"):
        if not poly:
            break
        output: List[Tuple[float, float]] = []
        sx, sy = poly[-1]
        for ex, ey in poly:
            if _inside(ex, ey, edge, bounds):
                if not _inside(sx, sy, edge, bounds):
                    ix, iy = _intersect(sx, sy, ex, ey, edge, bounds)
                    output.append((ix, iy))
                output.append((ex, ey))
            elif _inside(sx, sy, edge, bounds):
                ix, iy = _intersect(sx, sy, ex, ey, edge, bounds)
                output.append((ix, iy))
            sx, sy = ex, ey
        poly = output
    out: List[float] = []
    for x, y in poly:
        out.extend([x, y])
    return out


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _convex_hull(points: Sequence[float]) -> List[Tuple[float, float]]:
    pts = _pair_points(points)
    pts = sorted(set((float(x), float(y)) for x, y in pts))
    if len(pts) <= 1:
        return pts
    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _orientation(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _orientation(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


def _rotate_point(x: float, y: float, c: float, s: float) -> Tuple[float, float]:
    return c * x - s * y, s * x + c * y


def min_area_rect(points: Sequence[float]) -> List[float]:
    """Approximate minimum-area rectangle around polygon points.

    Uses rotating calipers on the convex hull; returns 4 vertices (clockwise).
    """
    hull = _convex_hull(points)
    if len(hull) == 0:
        return []
    if len(hull) == 1:
        x, y = hull[0]
        return [x, y, x, y, x, y, x, y]
    if len(hull) == 2:
        (x1, y1), (x2, y2) = hull
        return [x1, y1, x2, y2, x2, y2, x1, y1]
    best_area = float("inf")
    best_rect: List[float] = []
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            continue
        c, s = dx / length, dy / length
        xs: List[float] = []
        ys: List[float] = []
        for x, y in hull:
            rx, ry = _rotate_point(x, y, c, s)
            xs.append(rx)
            ys.append(ry)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            # corners in rotated frame
            rect_rot = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ]
            # rotate back
            inv_c, inv_s = c, -s
            pts: List[float] = []
            for rx, ry in rect_rot:
                x, y = _rotate_point(rx, ry, inv_c, inv_s)
                pts.extend([x, y])
            best_rect = pts
    return to_clockwise(best_rect)


def classify_affine_kind(M: List[List[float]], tol: float = 1e-6) -> str:
    """Classify affine as axis-aligned or general.

    Axis-aligned iff off-diagonals ~ 0 (no rotation/shear). Allows flips and scales.
    """
    a, b, _ = M[0]
    d, e, _ = M[1]
    if abs(b) <= tol and abs(d) <= tol:
        # a,e can be negative (flip) or positive (scale)
        return "axis_aligned"
    return "general"


def _cohen_sutherland_code(x: float, y: float, bounds: Tuple[float, float, float, float]) -> int:
    l, t, r, b = bounds
    code = 0
    if x < l:
        code |= 1  # left
    elif x > r:
        code |= 2  # right
    if y < t:
        code |= 4  # top
    elif y > b:
        code |= 8  # bottom
    return code


def _clip_segment_cs(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bounds: Tuple[float, float, float, float],
) -> Tuple[bool, float, float, float, float]:
    l, t, r, b = bounds
    c1 = _cohen_sutherland_code(x1, y1, bounds)
    c2 = _cohen_sutherland_code(x2, y2, bounds)
    accept = False
    while True:
        if not (c1 | c2):
            accept = True
            break
        if c1 & c2:
            break
        out_code = c1 or c2
        if out_code & 8:  # bottom
            x = x1 + (x2 - x1) * (b - y1) / ((y2 - y1) if y2 != y1 else 1e-12)
            y = b
        elif out_code & 4:  # top
            x = x1 + (x2 - x1) * (t - y1) / ((y2 - y1) if y2 != y1 else 1e-12)
            y = t
        elif out_code & 2:  # right
            y = y1 + (y2 - y1) * (r - x1) / ((x2 - x1) if x2 != x1 else 1e-12)
            x = r
        else:  # left
            y = y1 + (y2 - y1) * (l - x1) / ((x2 - x1) if x2 != x1 else 1e-12)
            x = l
        if out_code == c1:
            x1, y1 = x, y
            c1 = _cohen_sutherland_code(x1, y1, bounds)
        else:
            x2, y2 = x, y
            c2 = _cohen_sutherland_code(x2, y2, bounds)
    return accept, x1, y1, x2, y2


def clip_polyline_to_rect(points: Sequence[float], width: float, height: float) -> List[float]:
    bounds = _rect_bounds(width, height)
    pts = _pair_points(points)
    if len(pts) < 2:
        return []
    out: List[Tuple[float, float]] = []
    # Process each segment; stitch consecutive segments
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        ok, nx1, ny1, nx2, ny2 = _clip_segment_cs(x1, y1, x2, y2, bounds)
        if not ok:
            continue
        if not out:
            out.append((nx1, ny1))
        else:
            # avoid duplicate point
            if (abs(out[-1][0] - nx1) > 1e-9) or (abs(out[-1][1] - ny1) > 1e-9):
                out.append((nx1, ny1))
        out.append((nx2, ny2))
    flat: List[float] = []
    for x, y in out:
        flat.extend([x, y])
    return flat


# --- Typed geometry value objects and transform entrypoint ---


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_quad_points(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y1, self.x2, self.y2, self.x1, self.y2]

    def apply_affine(self, M: List[List[float]]) -> Union["BBox", "Quad"]:
        kind = classify_affine_kind(M)
        pts = apply_affine(self.to_quad_points(), M)
        if kind == "axis_aligned":
            xs = pts[0::2]
            ys = pts[1::2]
            return BBox(min(xs), min(ys), max(xs), max(ys))
        return Quad(tuple(pts))


@dataclass(frozen=True)
class Quad:
    points: Tuple[float, float, float, float, float, float, float, float]

    def apply_affine(self, M: List[List[float]]) -> "Quad":
        pts = apply_affine(self.points, M)
        return Quad(tuple(pts))


@dataclass(frozen=True)
class Polyline:
    points: Tuple[float, ...]

    def apply_affine(self, M: List[List[float]]) -> "Polyline":
        pts = apply_affine(self.points, M)
        return Polyline(tuple(pts))


def geometry_from_dict(g: Dict[str, Any]) -> Union[BBox, Quad, Polyline]:
    if "bbox_2d" in g:
        x1, y1, x2, y2 = map(float, g["bbox_2d"])
        return BBox(x1, y1, x2, y2)
    if "quad" in g:
        pts = tuple(float(v) for v in g["quad"])
        assert len(pts) == 8, f"quad must have 8 floats, got {len(pts)}"
        return Quad(pts)  # type: ignore[arg-type]
    if "line" in g:
        pts = tuple(float(v) for v in g["line"])
        assert len(pts) >= 4 and len(pts) % 2 == 0, "line must have >= 2 points"
        return Polyline(pts)
    raise ValueError("unknown geometry dict type; expected bbox_2d|quad|line")


def transform_geometry(
    g: Dict[str, Any],
    M: List[List[float]],
    *,
    width: int,
    height: int,
    allow_poly: bool = False,
) -> Dict[str, Any]:
    """
    Single entrypoint for geometry transform with promotion, ordering, and clipping/rounding.

    - BBox under general affine promotes to Quad.
    - Quad is transformed exactly; enforce clockwise order.
    - Polyline is transformed and clipped to rect; degenerate outputs dropped.
    - Rounding/clamping to integer pixel grid occurs at the end.
    """
    obj = geometry_from_dict(g)
    if isinstance(obj, BBox):
        res = obj.apply_affine(M)
        if isinstance(res, BBox):
            bb = clamp_points([res.x1, res.y1, res.x2, res.y2], width, height)
            return {"bbox_2d": bb}
        # Quad under general affine → clip polygon then normalize to quad
        t = list(res.points)
        clipped = sutherland_hodgman_clip(t, width, height)
        if len(clipped) // 2 >= 3:
            poly = to_clockwise(clipped)
            if len(poly) // 2 != 4:
                poly = min_area_rect(poly)
            q = clamp_points(poly, width, height)
            return {"quad": q}
        # fully outside: keep clamped transform to preserve geometry (degenerate possible)
        q = clamp_points(to_clockwise(t), width, height)
        return {"quad": q}
    if isinstance(obj, Quad):
        t = list(obj.apply_affine(M).points)
        clipped = sutherland_hodgman_clip(t, width, height)
        if len(clipped) // 2 >= 3:
            poly = to_clockwise(clipped)
            if len(poly) // 2 != 4:
                poly = min_area_rect(poly)
            q = clamp_points(poly, width, height)
            return {"quad": q}
        q = clamp_points(to_clockwise(t), width, height)
        return {"quad": q}
    # Polyline
    pl = obj.apply_affine(M)
    clipped = clip_polyline_to_rect(list(pl.points), width, height)
    l = clamp_points(clipped, width, height)
    l = dedupe_consecutive_points(l)
    if len(l) < 4:
        # preserve by collapsing to minimal 2-point line on clamped endpoints
        raw = clamp_points(list(pl.points), width, height)
        raw = dedupe_consecutive_points(raw)
        if len(raw) >= 4:
            return {"line": raw[:4]}
        # fallback to a point repeated
        if len(raw) >= 2:
            return {"line": [raw[0], raw[1], raw[0], raw[1]]}
        return {"line": [0, 0, 0, 0]}
    return {"line": l}


__all_typed__ = ["BBox", "Quad", "Polyline", "geometry_from_dict", "transform_geometry"]

