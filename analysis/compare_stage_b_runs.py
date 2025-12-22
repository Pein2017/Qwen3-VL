#!/usr/bin/env python3
"""
Compare two Stage-B rule-search run directories (same mission) and emit a Markdown report.

Designed for quick, dependency-free analysis of:
- rule_candidates.jsonl
- benchmarks.jsonl
- rule_search_hard_cases.jsonl
- rule_search_candidate_regressions.jsonl
- guidance.json + snapshots/
- distill_chatml.jsonl (optional)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                yield {"__parse_error__": str(e), "__line__": i, "__raw__": line}


def _safe_len(x: Any) -> int:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return 0


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{x*100:.{digits}f}%"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


@dataclass
class RunSummary:
    run_dir: Path
    mission: str
    guidance_step: Optional[int]
    snapshots: int
    candidates_total: int
    candidates_promoted: int
    candidates_rejected: int
    candidate_ops: Dict[str, int]
    best_rer: Optional[float]
    benchmarks_total: int
    last_benchmark: Dict[str, Any]
    hard_cases: int
    regressions: int
    distill_samples: int
    files_present: Dict[str, bool]


def _load_guidance_step(path: Path, mission: str) -> Optional[int]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict) and mission in obj and isinstance(obj[mission], dict):
        return obj[mission].get("step")
    if isinstance(obj, dict):
        return obj.get("step")
    return None


def _summarize_run(run_dir: Path, mission: str) -> RunSummary:
    files = {
        "rule_candidates.jsonl": (run_dir / "rule_candidates.jsonl").exists(),
        "benchmarks.jsonl": (run_dir / "benchmarks.jsonl").exists(),
        "rule_search_hard_cases.jsonl": (run_dir / "rule_search_hard_cases.jsonl").exists(),
        "rule_search_candidate_regressions.jsonl": (run_dir / "rule_search_candidate_regressions.jsonl").exists(),
        "guidance.json": (run_dir / "guidance.json").exists(),
        "snapshots/": (run_dir / "snapshots").exists(),
        "distill_chatml.jsonl": (run_dir / "distill_chatml.jsonl").exists(),
    }

    # candidates
    candidate_ops = Counter()
    promoted = 0
    rejected = 0
    total = 0
    best_rer: Optional[float] = None
    cand_path = run_dir / "rule_candidates.jsonl"
    if cand_path.exists():
        for rec in _read_jsonl(cand_path):
            total += 1
            decision = rec.get("decision")
            if decision == "promoted":
                promoted += 1
            elif decision == "rejected":
                rejected += 1
            op = rec.get("op") or rec.get("operation")
            if isinstance(op, str) and op:
                candidate_ops[op] += 1
            rer = rec.get("relative_error_reduction")
            if isinstance(rer, (int, float)):
                best_rer = rer if best_rer is None else max(best_rer, float(rer))

    # benchmarks
    last_benchmark: Dict[str, Any] = {}
    bench_path = run_dir / "benchmarks.jsonl"
    bench_total = 0
    last_iter = -1
    if bench_path.exists():
        for rec in _read_jsonl(bench_path):
            bench_total += 1
            it = rec.get("iteration")
            if isinstance(it, int) and it >= last_iter:
                last_iter = it
                last_benchmark = rec

    # hard cases / regressions
    hard_cases = 0
    hard_path = run_dir / "rule_search_hard_cases.jsonl"
    if hard_path.exists():
        for _ in _read_jsonl(hard_path):
            hard_cases += 1

    regressions = 0
    reg_path = run_dir / "rule_search_candidate_regressions.jsonl"
    if reg_path.exists():
        for rec in _read_jsonl(reg_path):
            reg_count = rec.get("regression_count")
            if isinstance(reg_count, int):
                regressions += reg_count
            else:
                regressions += _safe_len(rec.get("regressions"))

    # distill
    distill_samples = 0
    distill_path = run_dir / "distill_chatml.jsonl"
    if distill_path.exists():
        for _ in _read_jsonl(distill_path):
            distill_samples += 1

    guidance_step = _load_guidance_step(run_dir / "guidance.json", mission)
    snapshots_dir = run_dir / "snapshots"
    snapshots = len(list(snapshots_dir.glob("guidance-*.json"))) if snapshots_dir.exists() else 0

    return RunSummary(
        run_dir=run_dir,
        mission=mission,
        guidance_step=guidance_step,
        snapshots=snapshots,
        candidates_total=total,
        candidates_promoted=promoted,
        candidates_rejected=rejected,
        candidate_ops=dict(candidate_ops),
        best_rer=best_rer,
        benchmarks_total=bench_total,
        last_benchmark=last_benchmark,
        hard_cases=hard_cases,
        regressions=regressions,
        distill_samples=distill_samples,
        files_present=files,
    )


def _render_summary(a: RunSummary, b: RunSummary) -> str:
    lines: List[str] = []
    lines.append(f"# Stage-B rule-search run comparison")
    lines.append("")
    lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Runs")
    lines.append(f"- A: `{a.run_dir}`")
    lines.append(f"- B: `{b.run_dir}`")
    lines.append(f"- Mission: `{a.mission}`")
    lines.append("")

    rows = []
    rows.append(["guidance step", str(a.guidance_step), str(b.guidance_step)])
    rows.append(["snapshots", str(a.snapshots), str(b.snapshots)])
    rows.append(["candidates (total)", str(a.candidates_total), str(b.candidates_total)])
    rows.append(["candidates (promoted)", str(a.candidates_promoted), str(b.candidates_promoted)])
    rows.append(["candidates (rejected)", str(a.candidates_rejected), str(b.candidates_rejected)])
    rows.append(["best RER", str(a.best_rer), str(b.best_rer)])
    rows.append(["benchmarks rows", str(a.benchmarks_total), str(b.benchmarks_total)])
    rows.append(["hard cases", str(a.hard_cases), str(b.hard_cases)])
    rows.append(["regressions", str(a.regressions), str(b.regressions)])
    rows.append(["distill samples", str(a.distill_samples), str(b.distill_samples)])
    lines.append("## Summary")
    lines.append(_md_table(["metric", "A", "B"], rows))
    lines.append("")

    def _bench_row(run: RunSummary) -> List[str]:
        acc = run.last_benchmark.get("after_acc")
        base_acc = run.last_benchmark.get("base_acc")
        eval_acc = run.last_benchmark.get("eval_after_acc")
        rer = run.last_benchmark.get("relative_error_reduction")
        return [
            str(run.last_benchmark.get("iteration")),
            str(base_acc),
            str(acc),
            str(eval_acc),
            str(rer),
        ]

    lines.append("## Latest benchmark (if any)")
    lines.append(_md_table(["run", "iteration", "base_acc", "after_acc", "eval_after_acc", "RER"], [
        ["A"] + _bench_row(a),
        ["B"] + _bench_row(b),
    ]))
    lines.append("")

    def _ops(run: RunSummary) -> str:
        if not run.candidate_ops:
            return ""
        items = [f"{k}:{v}" for k, v in sorted(run.candidate_ops.items())]
        return ", ".join(items)

    lines.append("## Candidate ops breakdown")
    lines.append(_md_table(["run", "ops"], [["A", _ops(a)], ["B", _ops(b)]]))
    lines.append("")

    lines.append("## Files present")
    keys = sorted(set(a.files_present) | set(b.files_present))
    rows = [[k, "yes" if a.files_present.get(k) else "no", "yes" if b.files_present.get(k) else "no"] for k in keys]
    lines.append(_md_table(["file", "A", "B"], rows))
    return "\n".join(lines)


def _infer_mission(run_dir: Path) -> str:
    # Expected layout: output_post/stage_b/<mission>/<run_name>
    return run_dir.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path, help="Stage-B run directory A")
    parser.add_argument("run_b", type=Path, help="Stage-B run directory B")
    parser.add_argument("--mission", type=str, default=None, help="Mission name (defaults to run_dir parent)")
    parser.add_argument("--out", type=Path, default=None, help="Write report to file instead of stdout")
    args = parser.parse_args()

    if not args.run_a.exists() or not args.run_b.exists():
        raise SystemExit("Run directories must exist")

    mission = args.mission or _infer_mission(args.run_a)
    if args.mission is None and _infer_mission(args.run_b) != mission:
        raise SystemExit("Missions differ; pass --mission explicitly")

    a = _summarize_run(args.run_a, mission)
    b = _summarize_run(args.run_b, mission)
    report = _render_summary(a, b)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
