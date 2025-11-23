from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union


def _pair_points(points: Sequence[float]) -> List[Tuple[float, float]]:
    assert len(points) % 2 == 0, f"points length must be even, got {len(points)}"
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


def points_to_xyxy(points: Sequence[float]) -> List[float]:
    """
    Compute [x1,y1,x2,y2] axis-aligned bounding box that encloses arbitrary points.

    This is the canonical function for converting polygon or line points to bbox_2d format.
    Used throughout the codebase for poly-to-bbox conversion (e.g., in offline
    conversion steps that cap polygon complexity).

    Args:
        points: Flat list of coordinates [x0, y0, x1, y1, ..., xn, yn]

    Returns:
        [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right

    Examples:
        >>> points_to_xyxy([10, 20, 30, 40, 50, 60])
        [10.0, 20.0, 50.0, 60.0]
        >>> points_to_xyxy([100, 50, 150, 100, 120, 80])
        [100.0, 50.0, 150.0, 100.0]
    """
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


def normalize_points(
    points: Sequence[float], width: float, height: float, space: str
) -> List[int]:
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


def _rect_bounds(width: float, height: float) -> Tuple[float, float, float, float]:
    # left, top, right, bottom in float space
    return (
        0.0,
        0.0,
        float(max(0, int(round(width)) - 1)),
        float(max(0, int(round(height)) - 1)),
    )


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


def _inside(
    px: float, py: float, edge: str, bounds: Tuple[float, float, float, float]
) -> bool:
    left, t, r, b = bounds
    if edge == "left":
        return px >= left
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
    left, t, r, b = bounds
    dx = x2 - x1
    dy = y2 - y1
    if edge == "left":
        x = left
        y = y1 + dy * (left - x1) / (dx if dx != 0 else 1e-12)
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


def sutherland_hodgman_clip(
    points: Sequence[float], width: float, height: float
) -> List[float]:
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


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def simplify_polygon(
    points: Sequence[float], *, eps_collinear: float = 1e-6, eps_dup: float = 1e-9
) -> List[float]:
    """Remove duplicate and nearly-collinear consecutive vertices from a polygon.

    This keeps the polygon shape but drops redundant points that often appear
    after Sutherland–Hodgman clipping along axis-aligned boundaries.
    """
    pts = _pair_points(points)
    if len(pts) <= 2:
        return [v for p in pts for v in p]

    # Remove consecutive duplicates
    dedup: List[Tuple[float, float]] = []
    for x, y in pts:
        if not dedup or _dist2(dedup[-1][0], dedup[-1][1], x, y) > eps_dup:
            dedup.append((x, y))
    # Close polygon handling: if last equals first, drop last
    if (
        len(dedup) >= 2
        and _dist2(dedup[0][0], dedup[0][1], dedup[-1][0], dedup[-1][1]) <= eps_dup
    ):
        dedup.pop()

    if len(dedup) <= 2:
        return [v for p in dedup for v in p]

    # Remove nearly collinear interior points
    out: List[Tuple[float, float]] = []
    n = len(dedup)
    for i in range(n):
        x_prev, y_prev = dedup[(i - 1) % n]
        x_cur, y_cur = dedup[i]
        x_next, y_next = dedup[(i + 1) % n]
        # Cross product magnitude for collinearity
        cross = abs(
            (x_cur - x_prev) * (y_next - y_prev) - (y_cur - y_prev) * (x_next - x_prev)
        )
        if cross > eps_collinear:
            out.append((x_cur, y_cur))
    if len(out) < 3:
        out = dedup
    flat: List[float] = []
    for x, y in out:
        flat.extend([x, y])
    return flat


def _polygon_centroid(points: Sequence[float]) -> Tuple[float, float]:
    pts = _pair_points(points)
    if not pts:
        return 0.0, 0.0
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    return cx, cy


def choose_four_corners(points: Sequence[float]) -> List[float]:
    """Select four most salient corners from a convex polygon.

    Heuristic: rank vertices by corner strength (normalized cross product of
    adjacent edges), pick top-4, then order clockwise around centroid.
    Returns empty list if polygon has fewer than 3 vertices.
    """
    pts = _pair_points(points)
    n = len(pts)
    if n < 3:
        return []
    # Corner strength per vertex
    strengths: List[Tuple[float, int]] = []
    for i in range(n):
        x_prev, y_prev = pts[(i - 1) % n]
        x_cur, y_cur = pts[i]
        x_next, y_next = pts[(i + 1) % n]
        ax, ay = x_cur - x_prev, y_cur - y_prev
        bx, by = x_next - x_cur, y_next - y_cur
        cross = abs(ax * by - ay * bx)
        na = max((ax * ax + ay * ay) ** 0.5, 1e-12)
        nb = max((bx * bx + by * by) ** 0.5, 1e-12)
        strength = cross / (na * nb)
        strengths.append((strength, i))
    strengths.sort(reverse=True)
    chosen_idx = sorted([i for _, i in strengths[:4]])

    # If polygon has exactly 3 unique strong corners, duplicate the weakest edge endpoint
    if len(set(chosen_idx)) < 4 and n >= 4:
        # Add remaining indices by spacing
        for i in range(n):
            if i not in chosen_idx:
                chosen_idx.append(i)
            if len(chosen_idx) >= 4:
                break

    sel = [pts[i] for i in chosen_idx[:4]]
    if len(sel) < 4:
        return []
    # Order clockwise around centroid
    cx, cy = _polygon_centroid([v for p in sel for v in p])

    def angle(p: Tuple[float, float]) -> float:
        return math.atan2(p[1] - cy, p[0] - cx)

    sel.sort(key=angle)
    flat: List[float] = []
    for x, y in sel:
        flat.extend([x, y])
    return to_clockwise(flat)


def _orientation(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> float:
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


def _cohen_sutherland_code(
    x: float, y: float, bounds: Tuple[float, float, float, float]
) -> int:
    left, t, r, b = bounds
    code = 0
    if x < left:
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
    left, t, r, b = bounds
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
            y = y1 + (y2 - y1) * (left - x1) / ((x2 - x1) if x2 != x1 else 1e-12)
            x = left
        if out_code == c1:
            x1, y1 = x, y
            c1 = _cohen_sutherland_code(x1, y1, bounds)
        else:
            x2, y2 = x, y
            c2 = _cohen_sutherland_code(x2, y2, bounds)
    return accept, x1, y1, x2, y2


def clip_polyline_to_rect(
    points: Sequence[float], width: float, height: float
) -> List[float]:
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

    def to_poly_points(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y1, self.x2, self.y2, self.x1, self.y2]

    def apply_affine(self, M: List[List[float]]) -> Union["BBox", "Polygon"]:
        kind = classify_affine_kind(M)
        pts = apply_affine(self.to_poly_points(), M)
        if kind == "axis_aligned":
            xs = pts[0::2]
            ys = pts[1::2]
            return BBox(min(xs), min(ys), max(xs), max(ys))
        if len(pts) % 2 != 0:
            raise ValueError(
                f"Transformed polygon points must be even-length, got {len(pts)}"
            )
        return Polygon(tuple(pts))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Polygon:
    points: Tuple[float, ...]

    def apply_affine(self, M: List[List[float]]) -> "Polygon":
        pts = apply_affine(self.points, M)
        if len(pts) % 2 != 0:
            raise ValueError(f"Polygon must have even number of points, got {len(pts)}")
        return Polygon(tuple(pts))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Polyline:
    points: Tuple[float, ...]

    def apply_affine(self, M: List[List[float]]) -> "Polyline":
        pts = apply_affine(self.points, M)
        return Polyline(tuple(pts))


def geometry_from_dict(g: Dict[str, Any]) -> Union[BBox, Polygon, Polyline]:
    if "bbox_2d" in g:
        x1, y1, x2, y2 = map(float, g["bbox_2d"])
        return BBox(x1, y1, x2, y2)
    if "poly" in g:
        pts = tuple(float(v) for v in g["poly"])
        if len(pts) < 8 or len(pts) % 2 != 0:
            raise ValueError(
                f"poly must contain >=8 floats with even length, got {len(pts)}"
            )
        return Polygon(pts)  # type: ignore[arg-type]
    if "line" in g:
        pts = tuple(float(v) for v in g["line"])
        assert len(pts) >= 4 and len(pts) % 2 == 0, "line must have >= 2 points"
        return Polyline(pts)
    raise ValueError("unknown geometry dict type; expected bbox_2d|poly|line")


# ============================================================================
# Coverage and Cropping Utilities
# ============================================================================


def get_aabb(geom: Dict[str, Any]) -> List[float]:
    """
    Get axis-aligned bounding box [x1, y1, x2, y2] from any geometry type.

    For bbox_2d: returns the bbox directly
    For poly: computes min/max from polygon points using points_to_xyxy
    For line: computes min/max from line points using points_to_xyxy

    Returns:
        [x1, y1, x2, y2] where x1 <= x2 and y1 <= y2
    """
    if "bbox_2d" in geom:
        return list(map(float, geom["bbox_2d"]))
    elif "poly" in geom:
        return points_to_xyxy(geom["poly"])
    elif "line" in geom:
        return points_to_xyxy(geom["line"])
    else:
        raise ValueError(f"Unknown geometry type: {list(geom.keys())}")


def intersect_aabb(bbox_a: List[float], bbox_b: List[float]) -> List[float]:
    """
    Compute intersection of two axis-aligned bounding boxes.

    Args:
        bbox_a: [x1, y1, x2, y2]
        bbox_b: [x1, y1, x2, y2]

    Returns:
        [x1, y1, x2, y2] of intersection, or [0, 0, 0, 0] if no overlap
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    # No intersection
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 0.0, 0.0]

    return [x1, y1, x2, y2]


def aabb_area(bbox: List[float]) -> float:
    """
    Compute area of an axis-aligned bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        Area (>= 0.0)
    """
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return width * height


def _polygon_area(points: List[float]) -> float:
    if len(points) < 6:
        return 0.0
    area = 0.0
    pts = _pair_points(points)
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def compute_polygon_coverage(
    geom: Dict[str, Any],
    crop_bbox: List[float],
    *,
    fallback: Literal["bbox", "auto"] = "auto",
) -> float:
    """Compute coverage using polygon clipping when possible.

    Args:
        geom: geometry dict (poly or bbox_2d)
        crop_bbox: crop [x1, y1, x2, y2]
        fallback: if "bbox", returns AABB-based coverage when polygon coverage is zero.

    Returns:
        coverage ratio in [0, 1]
    """
    x1, y1, x2, y2 = crop_bbox
    crop_w = max(0.0, x2 - x1)
    crop_h = max(0.0, y2 - y1)
    if crop_w <= 0 or crop_h <= 0:
        return 0.0

    if "poly" in geom:
        pts = geom["poly"]
        total_area = _polygon_area(pts)
        if total_area <= 0.0:
            return 0.0
        translated: List[float] = []
        for i in range(0, len(pts), 2):
            translated.append(pts[i] - x1)
            translated.append(pts[i + 1] - y1)
        clipped = sutherland_hodgman_clip(translated, crop_w, crop_h)
        clipped = simplify_polygon(clipped)
        visible_area = _polygon_area(clipped)
        if visible_area <= 0.0:
            return 0.0
        coverage = visible_area / total_area
    elif "bbox_2d" in geom:
        gx1, gy1, gx2, gy2 = geom["bbox_2d"]
        total_area = max(0.0, (gx2 - gx1) * (gy2 - gy1))
        if total_area <= 0.0:
            return 0.0
        inter_x1 = max(gx1, x1)
        inter_y1 = max(gy1, y1)
        inter_x2 = min(gx2, x2)
        inter_y2 = min(gy2, y2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        visible_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        coverage = visible_area / total_area
    else:
        return compute_coverage(geom, crop_bbox)

    coverage = max(0.0, min(1.0, coverage))
    if coverage == 0.0 and fallback == "bbox":
        return compute_coverage(geom, crop_bbox)
    return coverage


def compute_coverage(geom: Dict[str, Any], crop_bbox: List[float]) -> float:
    """
    Compute fraction of geometry that falls inside crop region.

    Uses axis-aligned bounding box approximation for efficiency.
    For polys, this may overestimate the actual geometry area (since rotated
    polygons have larger AABBs), making the coverage estimate conservative.

    Args:
        geom: Geometry dict with bbox_2d, poly, or line field
        crop_bbox: Crop region [x1, y1, x2, y2]

    Returns:
        Coverage ratio in [0.0, 1.0]
        - 0.0: completely outside crop
        - 1.0: completely inside crop
        - (0.0, 1.0): partially inside crop

    Used for two purposes:
    1. Filtering: Drop objects if coverage < min_coverage (e.g., 0.3)
    2. Completeness tracking: Mark "只显示部分" if coverage < completeness_threshold (e.g., 0.95)
    """
    geom_bbox = get_aabb(geom)
    geom_area = aabb_area(geom_bbox)

    # Degenerate geometry (collapsed to line or point)
    if geom_area <= 0.0:
        # Check if any point is inside crop
        x1, y1, x2, y2 = crop_bbox
        gx1, gy1, gx2, gy2 = geom_bbox
        # If the degenerate point/line is inside crop, consider it 100% covered
        if x1 <= gx1 <= x2 and y1 <= gy1 <= y2:
            return 1.0
        return 0.0

    intersection = intersect_aabb(geom_bbox, crop_bbox)
    intersection_area = aabb_area(intersection)

    coverage = intersection_area / geom_area
    # Clamp to [0, 1] to handle floating point errors
    return max(0.0, min(1.0, coverage))


def translate_geometry(geom: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    """
    Translate geometry by offset (dx, dy).

    Used after cropping to shift geometries from image coordinates
    to crop-relative coordinates.

    Args:
        geom: Geometry dict with bbox_2d, poly, or line field
        dx: X offset (typically negative crop x)
        dy: Y offset (typically negative crop y)

    Returns:
        New geometry dict with translated coordinates
    """
    if "bbox_2d" in geom:
        x1, y1, x2, y2 = geom["bbox_2d"]
        return {"bbox_2d": [x1 + dx, y1 + dy, x2 + dx, y2 + dy]}
    elif "poly" in geom:
        pts = geom["poly"]
        translated = []
        for i in range(0, len(pts), 2):
            translated.append(pts[i] + dx)
            translated.append(pts[i + 1] + dy)
        return {"poly": translated}
    elif "line" in geom:
        pts = geom["line"]
        translated = []
        for i in range(0, len(pts), 2):
            translated.append(pts[i] + dx)
            translated.append(pts[i + 1] + dy)
        return {"line": translated}
    else:
        raise ValueError(f"Unknown geometry type: {list(geom.keys())}")


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

    - BBox under general affine promotes to Polygon.
    - Polygon is transformed exactly; enforce clockwise order.
    - Polyline is transformed and clipped to rect; degenerate outputs dropped.
    - Rounding/clamping to integer pixel grid occurs at the end.
    """
    obj = geometry_from_dict(g)
    if isinstance(obj, BBox):
        res = obj.apply_affine(M)
        if isinstance(res, BBox):
            bb = clamp_points([res.x1, res.y1, res.x2, res.y2], width, height)
            return {"bbox_2d": bb}
        # BBox promoted to Polygon under general affine (rotation/shear)
        t = list(res.points)
        # Check if polygon is fully inside image bounds - if so, skip clipping to preserve exact rotation
        eps = 0.5
        all_inside = all(
            -eps <= t[i] < width + eps and -eps <= t[i + 1] < height + eps
            for i in range(0, len(t), 2)
        )
        if all_inside:
            # Polygon fully inside - use rotated points directly, just round/clamp
            q = clamp_points(to_clockwise(t), width, height)
            return {"poly": q}
        # Polygon needs clipping - use Sutherland-Hodgman
        clipped = sutherland_hodgman_clip(t, width, height)
        if len(clipped) // 2 >= 3:
            poly = to_clockwise(clipped)
            if len(poly) // 2 != 4:
                poly = min_area_rect(poly)
            q = clamp_points(poly, width, height)
            return {"poly": q}
        # fully outside: keep clamped transform to preserve geometry (degenerate possible)
        q = clamp_points(to_clockwise(t), width, height)
        return {"poly": q}
    if isinstance(obj, Polygon):
        t = list(obj.apply_affine(M).points)
        # Check if polygon is fully inside image bounds - if so, skip clipping to preserve exact rotation
        eps = 0.5
        all_inside = all(
            -eps <= t[i] < width + eps and -eps <= t[i + 1] < height + eps
            for i in range(0, len(t), 2)
        )
        if all_inside:
            # Polygon fully inside - use rotated points directly, just round/clamp
            q = clamp_points(to_clockwise(t), width, height)
            return {"poly": q}
        # Polygon needs clipping - use Sutherland-Hodgman
        clipped = sutherland_hodgman_clip(t, width, height)
        if len(clipped) // 2 >= 3:
            poly = to_clockwise(clipped)
            if len(poly) // 2 != 4:
                poly = min_area_rect(poly)
            q = clamp_points(poly, width, height)
            return {"poly": q}
        q = clamp_points(to_clockwise(t), width, height)
        return {"poly": q}
    # Polyline
    pl = obj.apply_affine(M)
    clipped = clip_polyline_to_rect(list(pl.points), width, height)
    line_points = clamp_points(clipped, width, height)
    line_points = dedupe_consecutive_points(line_points)
    if len(line_points) < 4:
        # preserve by collapsing to minimal 2-point line on clamped endpoints
        raw = clamp_points(list(pl.points), width, height)
        raw = dedupe_consecutive_points(raw)
        if len(raw) >= 4:
            return {"line": raw[:4]}
        # fallback to a point repeated
        if len(raw) >= 2:
            return {"line": [raw[0], raw[1], raw[0], raw[1]]}
        return {"line": [0, 0, 0, 0]}
    return {"line": line_points}


__all_typed__ = [
    "BBox",
    "Polygon",
    "Polyline",
    "geometry_from_dict",
    "transform_geometry",
]
