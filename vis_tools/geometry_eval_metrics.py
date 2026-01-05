from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


class EvalObject(TypedDict):
    type: str
    points: list[float]
    desc: str


GeometryObject = EvalObject

# We clamp all norm1000 coordinates into [0, 999] so TubeIoU can rasterize on a
# 1000x1000 grid without out-of-range coordinates.
NORM1000_GRID_SIZE = 1000
DEFAULT_PRIMARY_THRESHOLD = 0.50
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
DEFAULT_LINE_TOL = 8.0


class EvalMode(str, Enum):
    LOCALIZATION = "localization"
    PHASE = "phase"
    CATEGORY = "category"


@dataclass(frozen=True)
class DescLabels:
    phase: str
    category: str


def _find_kv_value(text: str, key: str) -> str:
    """Extract `key=value` from a comma-separated key=value string."""
    needle = f"{key}="
    idx = text.find(needle)
    if idx == -1:
        return ""
    start = idx + len(needle)
    end = text.find(",", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _clamp_norm1000(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return min(999.0, max(0.0, v))


@lru_cache(maxsize=1)
def _umbrella_phase_labels_from_mapping() -> frozenset[str]:
    """Return phase labels that encode fine category as a dedicated `type` field in legacy desc.

    This is derived from `data_conversion/hierarchical_attribute_mapping.json`.
    """
    mapping_path = (
        Path(__file__).resolve().parents[1]
        / "data_conversion"
        / "hierarchical_attribute_mapping.json"
    )
    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception:
        return frozenset()

    object_types = raw.get("object_types", {})
    if not isinstance(object_types, dict):
        return frozenset()

    labels: set[str] = set()
    for spec in cast(dict[str, Any], object_types).values():
        if not isinstance(spec, dict):
            continue
        spec_dict = cast(dict[str, Any], spec)
        chinese_label = spec_dict.get("chinese_label")
        attrs = spec_dict.get("attributes", [])
        if not isinstance(chinese_label, str) or not isinstance(attrs, list):
            continue
        has_type = False
        for a in cast(list[object], attrs):
            if not isinstance(a, dict):
                continue
            if cast(dict[str, Any], a).get("name") == "type":
                has_type = True
                break
        if has_type:
            labels.add(chinese_label.strip())
    return frozenset(x for x in labels if x)


def parse_desc_labels(desc: object) -> DescLabels:
    """Extract (phase/head, fine category) labels from `desc`.

    Supported formats:
    - key=value: uses `类别=...` as both labels.
    - legacy slash: uses prefix before first `/` as phase; derives fine category for umbrella phases.
    """
    if not isinstance(desc, str):
        return DescLabels(phase="", category="")

    text = desc.strip()
    if not text:
        return DescLabels(phase="", category="")

    cat = _find_kv_value(text, "类别")
    if cat:
        return DescLabels(phase=cat, category=cat)

    if "/" not in text:
        return DescLabels(phase=text, category=text)

    phase, rest = text.split("/", 1)
    phase = phase.strip()
    if not phase:
        return DescLabels(phase="", category="")

    if phase in _umbrella_phase_labels_from_mapping():
        level1 = rest.split("/", 1)[0]
        tokens = [t.strip() for t in level1.split(",") if t.strip()]
        if tokens:
            return DescLabels(phase=phase, category=tokens[0])

    return DescLabels(phase=phase, category=phase)


def _pair_points(points: Sequence[float]) -> list[tuple[float, float]]:
    if len(points) % 2 != 0:
        raise ValueError(f"points length must be even, got {len(points)}")
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


def _polygon_area(points: Sequence[float]) -> float:
    """Signed polygon area (positive for one winding, negative for the other)."""
    pts = _pair_points(points)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _ensure_polygon_order(points: Sequence[float]) -> list[float]:
    """Return points ordered cyclically around centroid (stable for convex quads)."""
    pts = _pair_points(points)
    if len(pts) < 3:
        return [float(p) for p in points]
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)

    def _angle(p: tuple[float, float]) -> float:
        return math.atan2(p[1] - cy, p[0] - cx)

    ordered = sorted(pts, key=_angle)
    flat: list[float] = []
    for x, y in ordered:
        flat.extend([x, y])
    return flat


def _aabb_from_points(points: Sequence[float]) -> tuple[float, float, float, float]:
    xs = [float(x) for x in points[::2]]
    ys = [float(y) for y in points[1::2]]
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def _aabb_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _convex_clip(subject: Sequence[float], clip: Sequence[float]) -> list[float]:
    """Clip a convex subject polygon by a convex clip polygon (Sutherland–Hodgman)."""
    subject_pts = _pair_points(subject)
    clip_pts = _pair_points(clip)
    if len(subject_pts) < 3 or len(clip_pts) < 3:
        return []

    orient = 0.0
    for i in range(len(clip_pts)):
        x1, y1 = clip_pts[i]
        x2, y2 = clip_pts[(i + 1) % len(clip_pts)]
        orient += x1 * y2 - x2 * y1
    if orient == 0.0:
        return []

    def _is_inside(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return cross * orient >= 0.0

    def _line_intersection(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> tuple[float, float]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0.0:
            return p2
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        return px, py

    output: list[tuple[float, float]] = subject_pts
    for i in range(len(clip_pts)):
        a = clip_pts[i]
        b = clip_pts[(i + 1) % len(clip_pts)]
        if not output:
            break
        input_pts = output
        output = []
        s = input_pts[-1]
        for e in input_pts:
            if _is_inside(e, a, b):
                if not _is_inside(s, a, b):
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif _is_inside(s, a, b):
                output.append(_line_intersection(s, e, a, b))
            s = e

    if len(output) < 3:
        return []
    flat: list[float] = []
    for x, y in output:
        flat.extend([x, y])
    return flat


def _to_region_poly(obj: GeometryObject) -> list[float]:
    gtype = obj["type"]
    pts_f = obj["points"]
    if gtype == "bbox_2d" and len(pts_f) == 4:
        x1, y1, x2, y2 = pts_f
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    if gtype == "poly" and len(pts_f) >= 6 and len(pts_f) % 2 == 0:
        return _ensure_polygon_order(pts_f)

    return []


def region_iou(obj_a: GeometryObject, obj_b: GeometryObject) -> float:
    """Filled-shape IoU between region-family objects (bbox_2d and poly), cross-type."""
    poly_a = _to_region_poly(obj_a)
    poly_b = _to_region_poly(obj_b)
    if not poly_a or not poly_b:
        return 0.0

    aabb_a = _aabb_from_points(poly_a)
    aabb_b = _aabb_from_points(poly_b)
    if _aabb_intersection_area(aabb_a, aabb_b) <= 0.0:
        return 0.0

    area_a = abs(_polygon_area(poly_a))
    area_b = abs(_polygon_area(poly_b))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_poly = _convex_clip(poly_a, poly_b)
    if not inter_poly:
        return 0.0
    inter_area = abs(_polygon_area(inter_poly))
    if inter_area <= 0.0:
        return 0.0

    union = area_a + area_b - inter_area
    return (inter_area / union) if union > 0.0 else 0.0


def _to_line_xy(points: Sequence[float]) -> list[tuple[float, float]]:
    if len(points) < 4 or len(points) % 2 != 0:
        return []
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]


def tube_iou_line(
    obj_gt: GeometryObject,
    obj_pred: GeometryObject,
    *,
    tol: float,
    grid_size: int = NORM1000_GRID_SIZE,
) -> float:
    """Mask-wise TubeIoU for polyline geometries on the norm1000 grid."""
    pts_g = obj_gt["points"]
    pts_p = obj_pred["points"]
    if obj_gt["type"] != "line" or obj_pred["type"] != "line":
        return 0.0

    xy_g = _to_line_xy(pts_g)
    xy_p = _to_line_xy(pts_p)
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


def _geom_family(gtype: object) -> str:
    if gtype in ("bbox_2d", "poly"):
        return "region"
    if gtype == "line":
        return "line"
    return ""


def _as_eval_objects(raw: object) -> list[EvalObject]:
    if not isinstance(raw, list):
        return []
    out: list[EvalObject] = []
    for o in cast(list[object], raw):
        if not isinstance(o, dict):
            continue
        o_dict = cast(dict[str, Any], o)
        gtype = o_dict.get("type")
        pts_obj = o_dict.get("points")
        if gtype not in ("bbox_2d", "poly", "line"):
            continue
        if not isinstance(pts_obj, (list, tuple)):
            continue
        pts_seq = cast(Sequence[object], pts_obj)
        if len(pts_seq) % 2 != 0:
            continue
        pts_f: list[float] = []
        bad = False
        for p in pts_seq:
            if isinstance(p, (int, float)):
                pts_f.append(_clamp_norm1000(float(p)))
            else:
                bad = True
                break
        if bad:
            continue

        desc = o_dict.get("desc", "")
        out.append(
            {
                "type": str(gtype),
                "points": pts_f,
                "desc": desc if isinstance(desc, str) else "",
            }
        )
    return out


def _build_overlap_matrix(
    gt_objects: Sequence[GeometryObject],
    pred_objects: Sequence[GeometryObject],
    *,
    line_tol: float,
) -> list[list[float]]:
    num_gt = len(gt_objects)
    num_pred = len(pred_objects)
    matrix: list[list[float]] = [[0.0] * num_pred for _ in range(num_gt)]
    for gi, gt in enumerate(gt_objects):
        gtype = gt["type"]
        gfam = _geom_family(gtype)
        if not gfam:
            continue
        for pi, pred in enumerate(pred_objects):
            pfam = _geom_family(pred["type"])
            if pfam != gfam:
                continue
            if gfam == "region":
                matrix[gi][pi] = region_iou(gt, pred)
            else:
                matrix[gi][pi] = tube_iou_line(gt, pred, tol=line_tol)
    return matrix


def _label_ok(
    mode: EvalMode,
    gt_labels: Sequence[DescLabels],
    pred_labels: Sequence[DescLabels],
    gi: int,
    pi: int,
) -> bool:
    if mode == EvalMode.LOCALIZATION:
        return True
    if mode == EvalMode.PHASE:
        return gt_labels[gi].phase != "" and gt_labels[gi].phase == pred_labels[pi].phase
    if mode == EvalMode.CATEGORY:
        return (
            gt_labels[gi].category != ""
            and gt_labels[gi].category == pred_labels[pi].category
        )
    return False


def _greedy_match(
    overlap: Sequence[Sequence[float]],
    *,
    threshold: float,
    mode: EvalMode,
    gt_labels: Sequence[DescLabels],
    pred_labels: Sequence[DescLabels],
) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for gi, row in enumerate(overlap):
        for pi, score in enumerate(row):
            if score < threshold:
                continue
            if not _label_ok(mode, gt_labels, pred_labels, gi, pi):
                continue
            pairs.append((float(score), gi, pi))

    # Deterministic tie-break: score desc, gt asc, pred asc.
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
class Prf1:
    precision: float
    recall: float
    f1: float


def _prf1(*, matched: int, gt_total: int, pred_total: int) -> Prf1:
    recall = (matched / gt_total) if gt_total > 0 else 0.0
    precision = (matched / pred_total) if pred_total > 0 else 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0
    return Prf1(precision=precision, recall=recall, f1=f1)


def _format_thr(t: float) -> str:
    return f"{float(t):.2f}"


def evaluate_dump_records(
    records: Iterable[Mapping[str, Any]],
    *,
    modes: Sequence[EvalMode] = (EvalMode.LOCALIZATION, EvalMode.PHASE, EvalMode.CATEGORY),
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    coco_thresholds: Sequence[float] = DEFAULT_COCO_THRESHOLDS,
    line_tol: float = DEFAULT_LINE_TOL,
    top_k_categories: int = 25,
) -> dict[str, Any]:
    thresholds = sorted({float(t) for t in coco_thresholds})
    primary = float(primary_threshold)
    if primary not in thresholds:
        thresholds.append(primary)
        thresholds = sorted(thresholds)

    supported_types = ("bbox_2d", "poly", "line")

    totals: dict[str, dict[str, dict[str, int]]] = {
        mode.value: {_format_thr(t): {"gt": 0, "pred": 0, "matched": 0} for t in thresholds}
        for mode in modes
    }
    by_type: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        mode.value: {
            _format_thr(t): {
                g: {"gt": 0, "pred": 0, "matched_gt": 0, "matched_pred": 0}
                for g in supported_types
            }
            for t in thresholds
        }
        for mode in modes
    }

    count_abs_err_sum = 0.0
    count_over = 0
    count_under = 0
    count_equal = 0
    num_images = 0

    # Category breakdown (computed at primary threshold only, category-aware mode).
    category_totals: dict[str, dict[str, int]] = {}
    primary_key = _format_thr(primary)

    for rec in records:
        gt_raw: object = rec.get("gt_norm1000") or rec.get("gt_norm") or []
        pred_raw: object = rec.get("pred_norm1000") or rec.get("pred") or []
        gt = _as_eval_objects(gt_raw)
        pred = _as_eval_objects(pred_raw)
        if not gt and not pred:
            continue

        num_images += 1

        gt_types = [o["type"] for o in gt]
        pred_types = [o["type"] for o in pred]

        gt_labels = [parse_desc_labels(o.get("desc", "")) for o in gt]
        pred_labels = [parse_desc_labels(o.get("desc", "")) for o in pred]

        overlap = _build_overlap_matrix(gt, pred, line_tol=line_tol)

        gt_count = len(gt)
        pred_count = len(pred)
        count_abs_err_sum += abs(pred_count - gt_count)
        if pred_count > gt_count:
            count_over += 1
        elif pred_count < gt_count:
            count_under += 1
        else:
            count_equal += 1

        for mode in modes:
            mode_key = mode.value
            for t in thresholds:
                thr_key = _format_thr(t)
                matches = _greedy_match(
                    overlap,
                    threshold=t,
                    mode=mode,
                    gt_labels=gt_labels,
                    pred_labels=pred_labels,
                )

                totals[mode_key][thr_key]["gt"] += len(gt)
                totals[mode_key][thr_key]["pred"] += len(pred)
                totals[mode_key][thr_key]["matched"] += len(matches)

                # Per-type totals: precision is pred-type based; recall is gt-type based.
                for g in supported_types:
                    by_type[mode_key][thr_key][g]["gt"] += sum(1 for x in gt_types if x == g)
                    by_type[mode_key][thr_key][g]["pred"] += sum(
                        1 for x in pred_types if x == g
                    )

                for gi, pi in matches:
                    gtype = gt_types[gi] if gi < len(gt_types) else ""
                    ptype = pred_types[pi] if pi < len(pred_types) else ""
                    if gtype in supported_types:
                        by_type[mode_key][thr_key][gtype]["matched_gt"] += 1
                    if ptype in supported_types:
                        by_type[mode_key][thr_key][ptype]["matched_pred"] += 1

                if mode == EvalMode.CATEGORY and thr_key == primary_key:
                    # Category breakdown uses category labels from desc parsing.
                    for labels in gt_labels:
                        if labels.category:
                            category_totals.setdefault(
                                labels.category, {"gt": 0, "pred": 0, "matched": 0}
                            )["gt"] += 1
                    for labels in pred_labels:
                        if labels.category:
                            category_totals.setdefault(
                                labels.category, {"gt": 0, "pred": 0, "matched": 0}
                            )["pred"] += 1
                    for gi, pi in matches:
                        gcat = gt_labels[gi].category
                        pcat = pred_labels[pi].category
                        if gcat and gcat == pcat:
                            category_totals.setdefault(
                                gcat, {"gt": 0, "pred": 0, "matched": 0}
                            )["matched"] += 1

    def _finalize_mode(mode_key: str) -> dict[str, Any]:
        out_by_thr: dict[str, Any] = {}
        out_by_type: dict[str, Any] = {}
        f1s: list[float] = []
        for thr_key in [_format_thr(t) for t in thresholds]:
            stats = totals[mode_key][thr_key]
            prf = _prf1(matched=stats["matched"], gt_total=stats["gt"], pred_total=stats["pred"])
            out_by_thr[thr_key] = {
                "gt": stats["gt"],
                "pred": stats["pred"],
                "matched": stats["matched"],
                "precision": prf.precision,
                "recall": prf.recall,
                "f1": prf.f1,
            }
            f1s.append(prf.f1)

            out_by_type[thr_key] = {}
            for g in supported_types:
                tstats = by_type[mode_key][thr_key][g]
                prf_t = _prf1(
                    matched=tstats["matched_gt"],
                    gt_total=tstats["gt"],
                    pred_total=tstats["pred"],
                )
                # Precision for a type is based on matched predictions of that type.
                prec = (
                    (tstats["matched_pred"] / tstats["pred"]) if tstats["pred"] > 0 else 0.0
                )
                denom = prec + prf_t.recall
                f1 = (2.0 * prec * prf_t.recall / denom) if denom > 0 else 0.0
                out_by_type[thr_key][g] = {
                    "gt": tstats["gt"],
                    "pred": tstats["pred"],
                    "matched_gt": tstats["matched_gt"],
                    "matched_pred": tstats["matched_pred"],
                    "precision": prec,
                    "recall": prf_t.recall,
                    "f1": f1,
                }

        mean_f1 = (sum(f1s) / len(f1s)) if f1s else 0.0
        return {
            "by_threshold": out_by_thr,
            "by_threshold_by_type": out_by_type,
            "mean_f1": mean_f1,
        }

    out_modes: dict[str, Any] = {m.value: _finalize_mode(m.value) for m in modes}

    # Top-k categories by GT frequency (deterministic ordering).
    cat_items = sorted(
        category_totals.items(),
        key=lambda kv: (-kv[1].get("gt", 0), kv[0]),
    )[: max(0, int(top_k_categories))]
    cat_out: dict[str, Any] = {}
    for cat, stats in cat_items:
        prf = _prf1(matched=stats["matched"], gt_total=stats["gt"], pred_total=stats["pred"])
        cat_out[cat] = {
            "gt": stats["gt"],
            "pred": stats["pred"],
            "matched": stats["matched"],
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
        }

    return {
        "images": num_images,
        "params": {
            "primary_threshold": primary,
            "coco_thresholds": [_format_thr(t) for t in thresholds],
            "line_tol": float(line_tol),
            "modes": [m.value for m in modes],
            "matching": {
                "algorithm": "greedy_1to1",
                "tie_break": ["score_desc", "gt_index_asc", "pred_index_asc"],
            },
        },
        "count_diagnostics": {
            "mae": (count_abs_err_sum / num_images) if num_images > 0 else 0.0,
            "over_rate": (count_over / num_images) if num_images > 0 else 0.0,
            "under_rate": (count_under / num_images) if num_images > 0 else 0.0,
            "equal_rate": (count_equal / num_images) if num_images > 0 else 0.0,
        },
        "modes": out_modes,
        "category_breakdown_primary": cat_out,
    }


def evaluate_jsonl(
    jsonl_path: str,
    *,
    modes: Sequence[EvalMode] = (EvalMode.LOCALIZATION, EvalMode.PHASE, EvalMode.CATEGORY),
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    coco_thresholds: Sequence[float] = DEFAULT_COCO_THRESHOLDS,
    line_tol: float = DEFAULT_LINE_TOL,
    top_k_categories: int = 25,
) -> dict[str, Any]:
    path = Path(jsonl_path)

    def _iter_records() -> Iterable[Mapping[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    yield cast(dict[str, Any], rec)

    summary = evaluate_dump_records(
        _iter_records(),
        modes=modes,
        primary_threshold=primary_threshold,
        coco_thresholds=coco_thresholds,
        line_tol=line_tol,
        top_k_categories=top_k_categories,
    )
    summary["path"] = str(path)
    return summary
