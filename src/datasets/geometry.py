from __future__ import annotations

import math
from typing import List, Sequence, Tuple


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



