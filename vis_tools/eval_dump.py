from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

# Support both:
# - `python -m vis_tools.eval_dump` (repo/worktree root on sys.path), and
# - `python vis_tools/eval_dump.py` (script dir on sys.path; add parent).
try:
    from vis_tools.geometry_eval_metrics import EvalMode, evaluate_jsonl
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vis_tools.geometry_eval_metrics import EvalMode, evaluate_jsonl


def _load_thresholds(text: str) -> List[float]:
    out: List[float] = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out


def _load_modes(text: str) -> List[EvalMode]:
    out: List[EvalMode] = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(EvalMode(tok))
        except Exception:
            continue
    # Deterministic ordering.
    seen: set[EvalMode] = set()
    unique: List[EvalMode] = []
    for m in out:
        if m in seen:
            continue
        unique.append(m)
        seen.add(m)
    return unique


def _print_console_summary(summary: Dict[str, Any]) -> None:
    path = summary.get("path", "")
    images = int(summary.get("images", 0))
    params = cast(dict[str, Any], summary.get("params") or {})
    primary_raw = params.get("primary_threshold", 0.5)
    primary = float(primary_raw) if isinstance(primary_raw, (int, float)) else 0.5
    primary_key = f"{primary:.2f}"

    print(f"\n=== Geometry eval for {path} ===")
    print(f"Images: {images}")
    modes_raw = params.get("modes", [])
    if isinstance(modes_raw, list):
        mode_list = [str(x) for x in cast(list[object], modes_raw)]
    else:
        mode_list = []

    line_tol_raw = params.get("line_tol")
    line_tol = float(line_tol_raw) if isinstance(line_tol_raw, (int, float)) else None
    line_tol_str = f"{line_tol:g}" if line_tol is not None else "?"

    print(
        f"Primary IoU={primary_key}, line_tol={line_tol_str}, modes={','.join(mode_list)}"
    )

    modes = cast(dict[str, Any], summary.get("modes") or {})
    for mode_key in mode_list:
        mode_summary = cast(dict[str, Any], modes.get(mode_key) or {})
        by_thr = cast(dict[str, Any], mode_summary.get("by_threshold") or {})
        primary_stats = cast(dict[str, Any], by_thr.get(primary_key) or {})
        mean_f1_raw = mode_summary.get("mean_f1", 0.0)
        mean_f1 = float(mean_f1_raw) if isinstance(mean_f1_raw, (int, float)) else 0.0

        prec_raw = primary_stats.get("precision", 0.0)
        rec_raw = primary_stats.get("recall", 0.0)
        f1_raw = primary_stats.get("f1", 0.0)
        gt_raw = primary_stats.get("gt", 0)
        pred_raw = primary_stats.get("pred", 0)
        matched_raw = primary_stats.get("matched", 0)

        prec = float(prec_raw) if isinstance(prec_raw, (int, float)) else 0.0
        rec = float(rec_raw) if isinstance(rec_raw, (int, float)) else 0.0
        f1 = float(f1_raw) if isinstance(f1_raw, (int, float)) else 0.0
        gt = int(gt_raw) if isinstance(gt_raw, (int, float)) else 0
        pred = int(pred_raw) if isinstance(pred_raw, (int, float)) else 0
        matched = int(matched_raw) if isinstance(matched_raw, (int, float)) else 0
        print(
            f"  [{mode_key}] IoU={primary_key} "
            f"P={prec:.3f} "
            f"R={rec:.3f} "
            f"F1={f1:.3f} "
            f"| mean-F1={mean_f1:.3f} "
            f"(GT={gt}, Pred={pred}, Matched={matched})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a gt_vs_pred.jsonl dump (norm1000)."
    )
    parser.add_argument(
        "paths", nargs="+", help="Single gt_vs_pred.jsonl file (norm1000)"
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.50,
        help="Primary IoU threshold for console reporting (default: 0.50)",
    )
    parser.add_argument(
        "--coco-thresholds",
        type=str,
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated IoU thresholds for sweep (default: 0.50:0.95 step 0.05)",
    )
    parser.add_argument(
        "--line-tol",
        type=float,
        default=8.0,
        help="TubeIoU tolerance in norm1000 units (default: 8)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="localization,phase,category",
        help="Comma-separated modes: localization,phase,category (default: all)",
    )
    parser.add_argument(
        "--top-k-categories",
        type=int,
        default=25,
        help="Top-K categories to include in JSON breakdown (default: 25)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Required output JSON path for the machine-readable report",
    )
    args = parser.parse_args()

    if len(args.paths) != 1:
        print(
            f"ERROR: only a single input gt_vs_pred.jsonl is supported for now; got {len(args.paths)} paths.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    thresholds = _load_thresholds(args.coco_thresholds)
    if not thresholds:
        thresholds = [0.50 + 0.05 * i for i in range(10)]

    modes = _load_modes(args.modes)
    if not modes:
        modes = [EvalMode.LOCALIZATION, EvalMode.PHASE, EvalMode.CATEGORY]

    p = args.paths[0]
    summary = evaluate_jsonl(
        p,
        modes=modes,
        primary_threshold=float(args.primary_threshold),
        coco_thresholds=thresholds,
        line_tol=float(args.line_tol),
        top_k_categories=int(args.top_k_categories),
    )
    _print_console_summary(summary)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
