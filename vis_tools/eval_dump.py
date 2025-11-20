from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Sequence

from PIL import Image

from vis_tools import evaluate as geom_eval
from src.datasets.geometry import normalize_points


GeometryObject = Dict[str, Any]
DEFAULT_IOU_THRESHOLDS: Tuple[float, ...] = (
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
)


def _load_records(jsonl_path: str) -> Iterable[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _resolve_image_path(img_path: str, jsonl_dir: Path) -> Optional[Path]:
    """Best-effort resolution of image_path stored in the dump.

    We first try absolute/relative to the jsonl directory, then fall back to
    searching under data/*/ for a matching path.
    """

    p = Path(img_path)
    if p.is_absolute() and p.is_file():
        return p

    candidates = [
        jsonl_dir / img_path,
        Path(img_path),
    ]
    data_root = Path("data")
    if data_root.is_dir():
        for sub in data_root.iterdir():
            candidates.append(sub / img_path)

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _to_geom_list(objs: List[Dict[str, Any]]) -> List[GeometryObject]:
    out: List[GeometryObject] = []
    for o in objs:
        gtype = o.get("type")
        pts = o.get("points", [])
        if gtype not in ("bbox_2d", "poly", "line"):
            continue
        if not isinstance(pts, list) or len(pts) % 2 != 0:
            continue
        out.append(
            {
                "type": gtype,
                "points": [float(p) for p in pts],
                "desc": o.get("desc", ""),
            }
        )
    return out


def _normalize_points(points: Sequence[int | float], w: Optional[int], h: Optional[int]) -> List[float]:
    """Normalize pixel points to norm1000 when size is known; empty list otherwise."""

    if not w or not h or w <= 0 or h <= 0:
        return []
    try:
        norm = normalize_points(points, w, h, "norm1000")
        return [float(p) for p in norm]
    except Exception:
        return []


def _get_image_size_from_rec(
    rec: Dict[str, Any], img_path: Optional[str], jsonl_dir: Path
) -> Tuple[Optional[int], Optional[int]]:
    """Find image size from metadata first, then fallback to loading the image."""

    w = rec.get("width")
    h = rec.get("height")
    try:
        w_int = int(w) if w is not None else None
        h_int = int(h) if h is not None else None
    except Exception:
        w_int = h_int = None

    if w_int and h_int:
        return w_int, h_int

    if img_path:
        resolved = _resolve_image_path(str(img_path), jsonl_dir)
        if resolved is not None:
            try:
                with Image.open(str(resolved)) as img:
                    return img.size
            except Exception:
                return None, None
    return None, None


def _prepare_geometries(
    rec: Dict[str, Any],
    jsonl_dir: Path,
) -> Tuple[List[GeometryObject], List[GeometryObject], Optional[int], Optional[int]]:
    """Return (gt_norm1000, pred_norm1000, width, height) for one record."""

    img_path = rec.get("image_path")
    gt_raw = rec.get("gt", [])
    gt_norm_field = rec.get("gt_norm1000") or rec.get("gt_norm")
    pred_raw = rec.get("pred_norm1000") or rec.get("pred", [])

    img_w, img_h = _get_image_size_from_rec(rec, img_path, jsonl_dir)

    # If gt is already normalized, prefer it
    if isinstance(gt_norm_field, list) and gt_norm_field:
        gt_norm = _to_geom_list(gt_norm_field)
    else:
        gt_norm = []
        if img_w and img_h:
            for o in gt_raw:
                gtype = o.get("type")
                pts = o.get("points", [])
                if gtype not in ("bbox_2d", "poly", "line"):
                    continue
                if not isinstance(pts, list) or len(pts) % 2 != 0:
                    continue
                pts_norm = _normalize_points(pts, img_w, img_h)
                if not pts_norm:
                    continue
                gt_norm.append(
                    {"type": gtype, "points": pts_norm, "desc": o.get("desc", "")}
                )
        else:
            gt_norm = _to_geom_list(gt_raw)

    pred_norm = _to_geom_list(pred_raw)
    return gt_norm, pred_norm, img_w, img_h


def evaluate_jsonl(
    jsonl_path: str,
    *,
    iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> Dict[str, Any]:
    """Evaluate a single gt_vs_pred.jsonl file and return aggregate metrics.

    The JSONL schema is produced by vis_qwen3.py:
    - image_path: original (relative) image path
    - gt: list[{desc, type, points}] in *pixel* coordinates
    - pred: list[{desc, type, points}] in *norm1000* coordinates

    For evaluation, both GT and predictions are mapped into a shared
    norm1000 space so that IoU-based metrics are resolution-invariant.
    """

    path = Path(jsonl_path)
    jsonl_dir = path.resolve().parent

    primary_thr = float(iou_threshold)
    thresholds = sorted(set(float(t) for t in iou_thresholds)) or [primary_thr]
    if primary_thr not in thresholds:
        thresholds.append(primary_thr)
        thresholds = sorted(thresholds)

    total_images = 0
    totals_by_thr: Dict[float, Dict[str, float]] = {
        t: {"gt": 0, "pred": 0, "matched": 0, "missing": 0} for t in thresholds
    }
    per_type_by_thr: Dict[str, Dict[float, Dict[str, float]]] = {
        "bbox_2d": {
            t: {"gt": 0, "pred": 0, "matched": 0, "missing": 0} for t in thresholds
        },
        "poly": {
            t: {"gt": 0, "pred": 0, "matched": 0, "missing": 0} for t in thresholds
        },
        "line": {
            t: {"gt": 0, "pred": 0, "matched": 0, "missing": 0} for t in thresholds
        },
    }

    for rec in _load_records(jsonl_path):
        gt_norm, pred_norm, _, _ = _prepare_geometries(rec, jsonl_dir)

        if not gt_norm and not pred_norm:
            continue

        total_images += 1
        gt_types = [o.get("type", "") for o in gt_norm]
        pred_types = [o.get("type", "") for o in pred_norm]

        # Sweep thresholds once per image
        res_by_thr = geom_eval.match_geometries_multi(
            gt_norm, pred_norm, iou_thresholds=thresholds
        )

        for thr, res in res_by_thr.items():
            tot = totals_by_thr[thr]
            tot["gt"] += res.num_gt
            tot["pred"] += res.num_pred
            tot["matched"] += res.num_matched
            tot["missing"] += res.num_missing

            for gtype in ("bbox_2d", "poly", "line"):
                gt_count = sum(1 for t in gt_types if t == gtype)
                pred_count = sum(1 for t in pred_types if t == gtype)
                matched_count = sum(
                    1 for gi, _ in res.matches if gi < len(gt_types) and gt_types[gi] == gtype
                )
                stats = per_type_by_thr[gtype][thr]
                stats["gt"] += gt_count
                stats["pred"] += pred_count
                stats["matched"] += matched_count
                stats["missing"] += max(0, gt_count - matched_count)

    return {
        "path": str(path),
        "images": total_images,
        "thresholds": thresholds,
        "totals_by_threshold": totals_by_thr,
        "per_type_by_threshold": per_type_by_thr,
        "primary_threshold": primary_thr,
    }


def _print_report(summary: Dict[str, Any]) -> None:
    path = summary["path"]
    images = summary["images"]
    thresholds: List[float] = summary["thresholds"]
    totals_by_thr: Dict[float, Dict[str, float]] = summary["totals_by_threshold"]
    per_type_by_thr: Dict[str, Dict[float, Dict[str, float]]] = summary["per_type_by_threshold"]
    primary = float(summary.get("primary_threshold", thresholds[0]))

    def _pr(stats: Dict[str, float]) -> Tuple[float, float]:
        rec = stats["matched"] / stats["gt"] if stats["gt"] > 0 else 0.0
        prec = stats["matched"] / stats["pred"] if stats["pred"] > 0 else 0.0
        return rec, prec

    primary_stats = totals_by_thr.get(primary) or totals_by_thr.get(thresholds[0])
    primary_rec, primary_prec = _pr(primary_stats) if primary_stats else (0.0, 0.0)

    mean_rec = sum(_pr(totals_by_thr[t])[0] for t in thresholds) / len(thresholds)
    mean_prec = sum(_pr(totals_by_thr[t])[1] for t in thresholds) / len(thresholds)

    print(f"\n=== Evaluation for {path} ===")
    print(f"Images: {images}")
    print(
        f"Primary IoU={primary:.2f}: "
        f"Recall={primary_rec:.3f}, Precision={primary_prec:.3f} "
        f"(GT={primary_stats['gt']}, Pred={primary_stats['pred']}, "
        f"Matched={primary_stats['matched']}, Missing={primary_stats['missing']})"
    )
    print(
        f"Mean over {len(thresholds)} IoUs {thresholds[0]:.2f}:{thresholds[-1]:.2f}: "
        f"Recall={mean_rec:.3f}, Precision={mean_prec:.3f}"
    )

    for gtype in ("bbox_2d", "poly", "line"):
        stats_by_thr = per_type_by_thr[gtype]
        if all(stats_by_thr[t]["gt"] == 0 and stats_by_thr[t]["pred"] == 0 for t in thresholds):
            continue
        prim = stats_by_thr.get(primary) or stats_by_thr[thresholds[0]]
        prim_r, prim_p = _pr(prim)
        mean_r = sum(_pr(stats_by_thr[t])[0] for t in thresholds) / len(thresholds)
        mean_p = sum(_pr(stats_by_thr[t])[1] for t in thresholds) / len(thresholds)
        print(
            f"  [{gtype}] IoU={primary:.2f} R={prim_r:.3f} P={prim_p:.3f} "
            f"| mean R={mean_r:.3f} P={mean_p:.3f} "
            f"(GT={prim['gt']}, Pred={prim['pred']}, Matched={prim['matched']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate gt_vs_pred.jsonl dumps."
    )
    parser.add_argument("paths", nargs="+", help="One or more gt_vs_pred.jsonl files")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument(
        "--coco-thresholds",
        type=str,
        default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
        help="Comma-separated IoU thresholds for COCO-like sweep (default: 0.50:0.95 step 0.05)",
    )
    args = parser.parse_args()

    thresholds: List[float] = []
    for tok in args.coco_thresholds.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            thresholds.append(float(tok))
        except Exception:
            continue
    if not thresholds:
        thresholds = list(DEFAULT_IOU_THRESHOLDS)

    overall = {
        "images": 0,
        "totals_by_threshold": {
            t: {"gt": 0, "pred": 0, "matched": 0, "missing": 0} for t in thresholds
        },
    }

    for p in args.paths:
        summary = evaluate_jsonl(
            p,
            iou_threshold=args.iou_threshold,
            iou_thresholds=thresholds,
        )
        _print_report(summary)
        overall["images"] += summary["images"]
        for t in thresholds:
            src = summary["totals_by_threshold"][t]
            dst = overall["totals_by_threshold"][t]
            for k in ("gt", "pred", "matched", "missing"):
                dst[k] += src[k]

    if len(args.paths) > 1:
        print("\n=== Aggregate over all files ===")
        for t in thresholds:
            stats = overall["totals_by_threshold"][t]
            rec = stats["matched"] / stats["gt"] if stats["gt"] > 0 else 0.0
            prec = stats["matched"] / stats["pred"] if stats["pred"] > 0 else 0.0
            print(
                f"IoU={t:.2f}: Images={overall['images']}, "
                f"GT={stats['gt']}, Pred={stats['pred']}, "
                f"Matched={stats['matched']}, Missing={stats['missing']}, "
                f"Recall={rec:.3f}, Precision={prec:.3f}"
            )


if __name__ == "__main__":
    main()
