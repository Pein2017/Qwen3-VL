#!/usr/bin/env python3
"""
Compare two Stage-B run directories (same mission) and emit a Markdown report.

Designed for quick, dependency-free analysis of:
- trajectories.jsonl (rollouts)
- selections.jsonl
- reflection.jsonl + reflection_cache/
- guidance.json + snapshots/
- metrics.jsonl (and legacy metrics_epoch.jsonl)
- manual_review_queue.jsonl / need_review_queue.jsonl
- failure_malformed.jsonl / group_report.jsonl (optional)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _pct(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{x*100:.{digits}f}%"


def _quantiles(values: List[float], ps: List[float]) -> Dict[float, Optional[float]]:
    if not values:
        return {p: None for p in ps}
    xs = sorted(values)
    out: Dict[float, Optional[float]] = {}
    for p in ps:
        if p <= 0:
            out[p] = xs[0]
            continue
        if p >= 1:
            out[p] = xs[-1]
            continue
        k = (len(xs) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[p] = xs[int(k)]
        else:
            out[p] = xs[f] * (c - k) + xs[c] * (k - f)
    return out


def _trim(s: Any, n: int = 120) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _try_get_op_type(op: Any) -> str:
    if isinstance(op, str):
        return op
    if isinstance(op, dict):
        for k in ("op", "action", "type", "kind", "name"):
            v = op.get(k)
            if isinstance(v, str) and v:
                return v
        # Common schema: {"target":"G1","mode":"modify",...}
        for k in ("mode", "operation"):
            v = op.get(k)
            if isinstance(v, str) and v:
                return v
    return "unknown"


@dataclass
class RunSummary:
    run_dir: Path
    mission: str
    # gt labels (if any)
    gt_by_group: Dict[str, str]
    # per-epoch summaries
    traj_epoch: Dict[int, Dict[str, Any]]
    sel_epoch: Dict[int, Dict[str, Any]]
    # selections keyed for comparisons
    selection_by_epoch_group: Dict[Tuple[int, str], Dict[str, Any]]
    # reflection summaries
    reflection_epoch: Dict[int, Dict[str, Any]]
    # guidance info
    guidance: Dict[str, Any]
    snapshots: List[Dict[str, Any]]
    # queues / errors
    need_review: Dict[int, Counter]
    manual_review: Dict[int, Counter]
    malformed: Dict[int, Counter]
    # misc file presence
    files_present: Dict[str, bool]


def _extract_candidate(rec: Dict[str, Any]) -> Dict[str, Any]:
    result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
    decode = result.get("decode") if isinstance(result.get("decode"), dict) else {}
    gt_label = rec.get("gt_label")
    group_id = rec.get("group_id")
    ticket_key = rec.get("ticket_key")
    unit_key = ticket_key
    if not isinstance(unit_key, str) or not unit_key:
        if isinstance(group_id, str) and group_id and isinstance(gt_label, str) and gt_label:
            unit_key = f"{group_id}::{gt_label}"
        else:
            unit_key = group_id
    return {
        "epoch": rec.get("epoch"),
        "group_id": group_id,
        "ticket_key": ticket_key,
        "unit_key": unit_key,
        "mission": rec.get("mission"),
        "guidance_step": rec.get("guidance_step"),
        "reflection_cycle": rec.get("reflection_cycle"),
        "candidate_index": result.get("candidate_index"),
        "temperature": decode.get("temperature"),
        "top_p": decode.get("top_p"),
        "verdict": result.get("verdict"),
        "format_ok": result.get("format_ok"),
        "label_match": result.get("label_match"),
        "vote_strength": result.get("vote_strength"),
        "low_agreement": result.get("low_agreement"),
        "needs_manual_review": result.get("needs_manual_review"),
        "warnings": result.get("warnings") if isinstance(result.get("warnings"), list) else [],
        "reason": result.get("reason"),
        "text": result.get("text"),
        "gt_label": gt_label,
    }


def _extract_selection(rec: Dict[str, Any]) -> Dict[str, Any]:
    result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
    gt_label = rec.get("gt_label")
    group_id = rec.get("group_id")
    ticket_key = rec.get("ticket_key")
    unit_key = ticket_key
    if not isinstance(unit_key, str) or not unit_key:
        if isinstance(group_id, str) and group_id and isinstance(gt_label, str) and gt_label:
            unit_key = f"{group_id}::{gt_label}"
        else:
            unit_key = group_id
    return {
        "epoch": rec.get("epoch"),
        "group_id": group_id,
        "ticket_key": ticket_key,
        "unit_key": unit_key,
        "mission": rec.get("mission"),
        "verdict": (result.get("verdict") or rec.get("verdict")),
        "reason": (result.get("reason") or rec.get("reason")),
        "vote_strength": result.get("vote_strength"),
        "label_match": result.get("label_match"),
        "selected_candidate": result.get("selected_candidate"),
        "guidance_step": result.get("guidance_step"),
        "reflection_cycle": result.get("reflection_cycle"),
        "manual_review_recommended": result.get("manual_review_recommended"),
        "eligible": result.get("eligible"),
        "needs_manual_review": result.get("needs_manual_review"),
        "low_agreement": result.get("low_agreement"),
        "warnings": result.get("warnings") if isinstance(result.get("warnings"), list) else [],
        "gt_label": gt_label,
    }


def _load_guidance(path: Path, mission: str) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and mission in obj and isinstance(obj[mission], dict):
        return obj[mission]
    return obj if isinstance(obj, dict) else {}


def _load_snapshots(dir_path: Path, mission: str) -> List[Dict[str, Any]]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    out = []
    for p in sorted(dir_path.glob("guidance-*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            g = obj.get(mission) if isinstance(obj, dict) else None
            if isinstance(g, dict):
                out.append({"file": p.name, "step": g.get("step"), "updated_at": g.get("updated_at"), "experiences": g.get("experiences")})
            else:
                out.append({"file": p.name, "step": None, "updated_at": None, "experiences": None})
        except Exception as e:
            out.append({"file": p.name, "error": str(e)})
    return out


def _summarize_reflection_cache(dir_path: Path) -> Dict[str, Any]:
    if not dir_path.exists() or not dir_path.is_dir():
        return {"files": 0, "types": {}, "epochs": 0, "cycles": 0}
    types = Counter()
    epoch_cycles = set()
    for p in dir_path.glob("*.json"):
        name = p.name
        if "_summary.json" in name:
            types["summary"] += 1
        elif "_critique.json" in name:
            types["critique"] += 1
        elif "_plan.json" in name:
            types["plan"] += 1
        else:
            types["other"] += 1
        # epochX_cycleY_*.json
        parts = name.split("_")
        if parts and parts[0].startswith("epoch") and len(parts) >= 2 and parts[1].startswith("cycle"):
            try:
                epoch = int(parts[0].replace("epoch", ""))
                cycle = int(parts[1].replace("cycle", ""))
                epoch_cycles.add((epoch, cycle))
            except Exception:
                pass
    epochs = {e for e, _ in epoch_cycles}
    cycles = {c for _, c in epoch_cycles}
    return {"files": sum(types.values()), "types": dict(types), "epochs": len(epochs), "cycles": len(cycles), "epoch_cycles": sorted(epoch_cycles)}


def _summarize_run(run_dir: Path, mission: str) -> RunSummary:
    # file presence
    files = {
        "trajectories.jsonl": (run_dir / "trajectories.jsonl").exists(),
        "selections.jsonl": (run_dir / "selections.jsonl").exists(),
        "reflection.jsonl": (run_dir / "reflection.jsonl").exists(),
        "metrics.jsonl": (run_dir / "metrics.jsonl").exists(),
        "metrics_epoch.jsonl (legacy)": (run_dir / "metrics_epoch.jsonl").exists(),
        "manual_review_queue.jsonl": (run_dir / "manual_review_queue.jsonl").exists(),
        "need_review_queue.jsonl": (run_dir / "need_review_queue.jsonl").exists(),
        "failure_malformed.jsonl": (run_dir / "failure_malformed.jsonl").exists(),
        "group_report.jsonl": (run_dir / "group_report.jsonl").exists(),
        "snapshots/": (run_dir / "snapshots").exists(),
        "reflection_cache/": (run_dir / "reflection_cache").exists(),
        "guidance.json": (run_dir / "guidance.json").exists(),
    }

    gt_by_group: Dict[str, str] = {}
    selection_by_epoch_group: Dict[Tuple[int, str], Dict[str, Any]] = {}

    # trajectories
    traj_epoch: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "groups": set(),
            "units": set(),
            "temps": Counter(),
            "verdicts": Counter(),
            "format_ok": 0,
            "format_total": 0,
            "reasons_len": [],
            "candidates": 0,
            "by_group": defaultdict(list),
        }
    )
    traj_path = run_dir / "trajectories.jsonl"
    if traj_path.exists():
        for rec in _read_jsonl(traj_path):
            c = _extract_candidate(rec)
            epoch = c.get("epoch")
            group_id = c.get("group_id")
            if not isinstance(epoch, int) or not isinstance(group_id, str):
                continue
            if c.get("gt_label") and isinstance(c["gt_label"], str):
                gt_by_group.setdefault(group_id, c["gt_label"])
            s = traj_epoch[epoch]
            s["candidates"] += 1
            s["groups"].add(group_id)
            if isinstance(c.get("unit_key"), str) and c.get("unit_key"):
                s["units"].add(c["unit_key"])
            verdict = c.get("verdict")
            if isinstance(verdict, str) and verdict:
                s["verdicts"][verdict] += 1
            temp = c.get("temperature")
            if temp is not None:
                s["temps"][str(temp)] += 1
            fmt = c.get("format_ok")
            if isinstance(fmt, bool):
                s["format_total"] += 1
                if fmt:
                    s["format_ok"] += 1
            reason = c.get("reason") or ""
            if isinstance(reason, str):
                s["reasons_len"].append(float(len(reason)))
            s["by_group"][group_id].append(c)

    # post-process diversity per epoch
    for epoch, s in traj_epoch.items():
        mixed_verdict_groups = 0
        temp_sensitive_groups = 0
        per_group_unique_verdicts = []
        per_group_temps = []
        for gid, cs in s["by_group"].items():
            verdicts = [x.get("verdict") for x in cs if isinstance(x.get("verdict"), str)]
            temps = [x.get("temperature") for x in cs if x.get("temperature") is not None]
            uniq_verdicts = {v for v in verdicts if v}
            per_group_unique_verdicts.append(len(uniq_verdicts))
            per_group_temps.append(len({str(t) for t in temps}))
            if len(uniq_verdicts) > 1:
                mixed_verdict_groups += 1
            by_temp: Dict[str, set] = defaultdict(set)
            for x in cs:
                t = x.get("temperature")
                v = x.get("verdict")
                if t is None or not isinstance(v, str) or not v:
                    continue
                by_temp[str(t)].add(v)
            if len(by_temp) >= 2:
                sets = list(by_temp.values())
                if any(s0 != sets[0] for s0 in sets[1:]):
                    temp_sensitive_groups += 1
        s["mixed_verdict_groups"] = mixed_verdict_groups
        s["temp_sensitive_groups"] = temp_sensitive_groups
        s["avg_unique_verdicts_per_group"] = (sum(per_group_unique_verdicts) / len(per_group_unique_verdicts)) if per_group_unique_verdicts else None
        s["avg_temps_per_group"] = (sum(per_group_temps) / len(per_group_temps)) if per_group_temps else None

    # selections
    sel_epoch: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "groups": set(),
            "units": set(),
            "verdicts": Counter(),
            "vote_strength": [],
            "label_match": [],
            "selected_candidate": Counter(),
            "warnings": Counter(),
            "needs_manual_review": 0,
            "low_agreement": 0,
            "eligible": 0,
            "n": 0,
        }
    )
    sel_path = run_dir / "selections.jsonl"
    if sel_path.exists():
        for rec in _read_jsonl(sel_path):
            srec = _extract_selection(rec)
            epoch = srec.get("epoch")
            group_id = srec.get("group_id")
            if not isinstance(epoch, int) or not isinstance(group_id, str):
                continue
            if srec.get("gt_label") and isinstance(srec["gt_label"], str):
                gt_by_group.setdefault(group_id, srec["gt_label"])
            # Note: selections may contain duplicates for same (epoch, group_id) due to reflection cycles.
            # Keep the last-seen record for group-level analysis.
            selection_by_epoch_group[(epoch, group_id)] = srec

            s = sel_epoch[epoch]
            s["n"] += 1
            s["groups"].add(group_id)
            if isinstance(srec.get("unit_key"), str) and srec.get("unit_key"):
                s["units"].add(srec["unit_key"])
            verdict = srec.get("verdict")
            if isinstance(verdict, str) and verdict:
                s["verdicts"][verdict] += 1
            vs = srec.get("vote_strength")
            if isinstance(vs, (int, float)):
                s["vote_strength"].append(float(vs))
            lm = srec.get("label_match")
            if isinstance(lm, bool):
                s["label_match"].append(lm)
            sc = srec.get("selected_candidate")
            if isinstance(sc, int):
                s["selected_candidate"][str(sc)] += 1
            if srec.get("needs_manual_review") is True:
                s["needs_manual_review"] += 1
            if srec.get("low_agreement") is True:
                s["low_agreement"] += 1
            if srec.get("eligible") is True:
                s["eligible"] += 1
            for w in srec.get("warnings") or []:
                if isinstance(w, str) and w:
                    s["warnings"][w] += 1

    # metrics
    metrics_by_epoch: Dict[int, Dict[str, Any]] = {}
    metrics_path = run_dir / "metrics.jsonl"
    legacy_metrics_path = run_dir / "metrics_epoch.jsonl"
    if metrics_path.exists() or legacy_metrics_path.exists():
        src = metrics_path if metrics_path.exists() else legacy_metrics_path
        for rec in _read_jsonl(src):
            epoch = rec.get("epoch")
            if not isinstance(epoch, int):
                continue
            event = rec.get("event")
            # `metrics.jsonl` contains step-wise windows; use epoch_end rows for the epoch summary.
            # Legacy `metrics_epoch.jsonl` has no event field and is already epoch-wise.
            if src == metrics_path and event not in (None, "epoch_end"):
                continue
            metrics_by_epoch[epoch] = rec

    # reflection
    reflection_epoch: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "eligible": 0, "applied": 0, "ineligible_reason": Counter(), "proposal_action": Counter(), "op_types": Counter(), "warnings": Counter(), "critique_tags": Counter()})
    refl_cache_summary = _summarize_reflection_cache(run_dir / "reflection_cache")
    refl_path = run_dir / "reflection.jsonl"
    if refl_path.exists():
        for rec in _read_jsonl(refl_path):
            epoch = rec.get("epoch")
            refl = rec.get("reflection") if isinstance(rec.get("reflection"), dict) else None
            if not isinstance(epoch, int) or not isinstance(refl, dict):
                continue
            s = reflection_epoch[epoch]
            s["n"] += 1
            if refl.get("eligible") is True:
                s["eligible"] += 1
            if refl.get("applied") is True:
                s["applied"] += 1
            r = refl.get("ineligible_reason")
            if isinstance(r, str) and r:
                s["ineligible_reason"][r] += 1
            proposal = refl.get("proposal") if isinstance(refl.get("proposal"), dict) else {}
            action = proposal.get("action")
            if isinstance(action, str) and action:
                s["proposal_action"][action] += 1
            critique = proposal.get("critique")
            if isinstance(critique, str) and critique:
                s["critique_tags"][critique] += 1
            # Some errors are embedded in proposal.uncertainty_note
            unc = proposal.get("uncertainty_note")
            if isinstance(unc, str) and unc:
                s["critique_tags"][unc] += 1
            ops = proposal.get("operations")
            if isinstance(ops, list):
                for op in ops:
                    s["op_types"][_try_get_op_type(op)] += 1
            for w in refl.get("warnings") or []:
                if isinstance(w, str) and w:
                    s["warnings"][w] += 1
    # Attach cache stats
    for epoch, s in reflection_epoch.items():
        s["reflection_cache"] = refl_cache_summary

    # queues
    def _load_queue(path: Path) -> Dict[int, Counter]:
        out: Dict[int, Counter] = defaultdict(Counter)
        if not path.exists():
            return out
        for rec in _read_jsonl(path):
            epoch = rec.get("epoch")
            if not isinstance(epoch, int):
                epoch = -1
            tag = rec.get("tag") or rec.get("reason_tag") or rec.get("type")
            if isinstance(tag, str) and tag:
                out[epoch][tag] += 1
            else:
                out[epoch]["(no_tag)"] += 1
            gid = rec.get("group_id")
            gt = rec.get("gt_label")
            if isinstance(gid, str) and isinstance(gt, str) and gt:
                gt_by_group.setdefault(gid, gt)
        return out

    need_review = _load_queue(run_dir / "need_review_queue.jsonl")
    manual_review = _load_queue(run_dir / "manual_review_queue.jsonl")

    # malformed
    malformed: Dict[int, Counter] = defaultdict(Counter)
    mal_path = run_dir / "failure_malformed.jsonl"
    if mal_path.exists():
        for rec in _read_jsonl(mal_path):
            epoch = rec.get("epoch")
            if not isinstance(epoch, int):
                epoch = -1
            et = rec.get("error_type") or rec.get("error") or rec.get("kind") or rec.get("tag")
            if isinstance(et, str) and et:
                malformed[epoch][et] += 1
            else:
                malformed[epoch]["(unknown)"] += 1

    # group_report (optional) may contain canonical label mapping
    grp_path = run_dir / "group_report.jsonl"
    if grp_path.exists():
        for rec in _read_jsonl(grp_path):
            gid = rec.get("group_id")
            label = rec.get("label")
            if isinstance(gid, str) and gid and isinstance(label, str) and label:
                gt_by_group.setdefault(gid, label)

    guidance = _load_guidance(run_dir / "guidance.json", mission)
    snapshots = _load_snapshots(run_dir / "snapshots", mission)

    # store metrics under sel_epoch for convenience (optional)
    for epoch, m in metrics_by_epoch.items():
        sel_epoch.setdefault(epoch, {})
        sel_epoch[epoch]["metrics_epoch"] = m

    return RunSummary(
        run_dir=run_dir,
        mission=mission,
        gt_by_group=gt_by_group,
        traj_epoch=traj_epoch,
        sel_epoch=sel_epoch,
        selection_by_epoch_group=selection_by_epoch_group,
        reflection_epoch=reflection_epoch,
        guidance=guidance,
        snapshots=snapshots,
        need_review=need_review,
        manual_review=manual_review,
        malformed=malformed,
        files_present=files,
    )


def _pick_epochs(summary: RunSummary) -> Tuple[Optional[int], Optional[int]]:
    epochs = sorted(set(summary.sel_epoch.keys()) | set(summary.traj_epoch.keys()) | set(summary.reflection_epoch.keys()))
    if not epochs:
        return None, None
    return epochs[0], epochs[-1]


def _pick_report_epochs(summary: RunSummary) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Returns (first_epoch, final_epoch_for_report, max_epoch_seen).

    - first_epoch: min epoch observed in selections/trajectories/reflection
    - final_epoch_for_report: prefer max(metrics_epoch) if available; otherwise, pick epoch with max sel_records
    - max_epoch_seen: max epoch observed in selections/trajectories/reflection (may be partial)
    """
    epochs_seen = sorted(set(summary.sel_epoch.keys()) | set(summary.traj_epoch.keys()) | set(summary.reflection_epoch.keys()))
    if not epochs_seen:
        return None, None, None
    first = epochs_seen[0]
    max_seen = epochs_seen[-1]
    metrics = _collect_metrics(summary)
    if metrics:
        return first, max(metrics), max_seen
    # fallback: pick epoch with most selection records
    best_epoch = max(epochs_seen, key=lambda e: int(summary.sel_epoch.get(e, {}).get("n", 0) or 0))
    return first, best_epoch, max_seen


def _accuracy_for_epoch(summary: RunSummary, epoch: int) -> Tuple[Optional[float], int]:
    # Prefer label_match in selections (record-level); fallback to gt_label vs verdict if available.
    s = summary.sel_epoch.get(epoch)
    if not s:
        return None, 0
    lms = s.get("label_match") if isinstance(s.get("label_match"), list) else None
    if lms:
        return _pct(sum(1 for x in lms if x), len(lms)), len(lms)
    # fallback: compute from selection_by_epoch_group
    correct = 0
    total = 0
    for (e, gid), rec in summary.selection_by_epoch_group.items():
        if e != epoch:
            continue
        gt = summary.gt_by_group.get(gid) or rec.get("gt_label")
        v = rec.get("verdict")
        if isinstance(gt, str) and isinstance(v, str) and gt and v:
            total += 1
            if gt == v:
                correct += 1
    return (_pct(correct, total), total)


def _metrics_rows(metrics: Dict[int, Dict[str, Any]]) -> List[List[str]]:
    rows = []
    for epoch in sorted(metrics):
        m = metrics[epoch]
        emr = m.get("exclude_manual_review") or {}
        imr = m.get("include_manual_review") or {}
        rows.append(
            [
                str(epoch),
                f"{emr.get('acc', 'NA'):.4f}" if isinstance(emr.get("acc"), (int, float)) else "NA",
                str(emr.get("fn", "NA")),
                str(emr.get("fp", "NA")),
                str(emr.get("n", "NA")),
                f"{imr.get('acc', 'NA'):.4f}" if isinstance(imr.get("acc"), (int, float)) else "NA",
                str(imr.get("fn", "NA")),
                str(imr.get("fp", "NA")),
                str(imr.get("n", "NA")),
            ]
        )
    return rows


def _collect_metrics(summary: RunSummary) -> Dict[int, Dict[str, Any]]:
    out = {}
    for epoch, s in summary.sel_epoch.items():
        m = s.get("metrics_epoch")
        if isinstance(m, dict) and "epoch" in m:
            out[epoch] = m
    return out


def _compare_selected_examples(
    base: RunSummary,
    other: RunSummary,
    epoch_base: int,
    epoch_other: int,
    max_rows: int = 8,
) -> List[List[str]]:
    # Select groups present in both epochs and where we have gt label (from either).
    rows = []
    candidates = []
    for (e, gid), bsel in base.selection_by_epoch_group.items():
        if e != epoch_base:
            continue
        osel = other.selection_by_epoch_group.get((epoch_other, gid))
        if not osel:
            continue
        gt = base.gt_by_group.get(gid) or other.gt_by_group.get(gid) or bsel.get("gt_label") or osel.get("gt_label")
        if not isinstance(gt, str) or not gt:
            continue
        bv = bsel.get("verdict")
        ov = osel.get("verdict")
        if not (isinstance(bv, str) and isinstance(ov, str) and bv and ov):
            continue
        b_ok = (bsel.get("label_match") if isinstance(bsel.get("label_match"), bool) else (bv == gt))
        o_ok = (osel.get("label_match") if isinstance(osel.get("label_match"), bool) else (ov == gt))
        if b_ok == o_ok:
            continue
        candidates.append((gid, gt, bv, ov, bool(b_ok), bool(o_ok), bsel, osel))

    # Prioritize: other correct but base wrong, then base correct but other wrong.
    candidates.sort(key=lambda x: (x[4] and not x[5], x[5] and not x[4]), reverse=True)
    for gid, gt, bv, ov, b_ok, o_ok, bsel, osel in candidates[:max_rows]:
        rows.append(
            [
                gid,
                gt,
                f"{bv} ({'✓' if b_ok else '✗'})",
                _trim(bsel.get("reason"), 90),
                f"{ov} ({'✓' if o_ok else '✗'})",
                _trim(osel.get("reason"), 90),
            ]
        )
    return rows


def _render_report(
    model_a_name: str,
    a: RunSummary,
    model_b_name: str,
    b: RunSummary,
    out_path: Path,
) -> None:
    a_first, a_final, a_max_seen = _pick_report_epochs(a)
    b_first, b_final, b_max_seen = _pick_report_epochs(b)

    # Rollout/selection summary table (epoch 1 and final)
    def _epoch_row(summary: RunSummary, epoch: int) -> List[str]:
        t = summary.traj_epoch.get(epoch, {})
        s = summary.sel_epoch.get(epoch, {})
        traj_groups = len(t.get("groups", set()))
        n_candidates = int(t.get("candidates", 0) or 0)
        fmt_rate = _pct(t.get("format_ok", 0), t.get("format_total", 0))
        pass_rate = _pct(t.get("verdicts", {}).get("pass", 0), sum(t.get("verdicts", {}).values()) or 0) if isinstance(t.get("verdicts"), Counter) else None
        acc, acc_n = _accuracy_for_epoch(summary, epoch)
        sel_records = int(s.get("n", 0) or 0)
        sel_groups = len(s.get("groups", set()))
        sel_units = len(s.get("units", set()))
        sel_verdicts = s.get("verdicts", Counter())
        sel_pass_rate = None
        if isinstance(sel_verdicts, Counter):
            sel_pass_rate = _pct(sel_verdicts.get("pass", 0), sum(sel_verdicts.values()) or 0)
        vs = s.get("vote_strength", [])
        qs = _quantiles(vs, [0.1, 0.5, 0.9]) if isinstance(vs, list) else {0.1: None, 0.5: None, 0.9: None}
        return [
            str(epoch),
            str(traj_groups if traj_groups else sel_groups),
            str(n_candidates),
            _fmt_pct(fmt_rate),
            _fmt_pct(pass_rate),
            str(sel_records),
            str(sel_groups),
            str(sel_units),
            _fmt_pct(sel_pass_rate),
            f"{acc:.4f}" if isinstance(acc, float) else "NA",
            str(acc_n),
            f"{qs[0.1]:.3f}" if isinstance(qs[0.1], float) else "NA",
            f"{qs[0.5]:.3f}" if isinstance(qs[0.5], float) else "NA",
            f"{qs[0.9]:.3f}" if isinstance(qs[0.9], float) else "NA",
        ]

    def _epoch_div_row(summary: RunSummary, epoch: int) -> List[str]:
        t = summary.traj_epoch.get(epoch, {})
        n_groups = len(t.get("groups", set()))
        mixed = t.get("mixed_verdict_groups", 0)
        temp_s = t.get("temp_sensitive_groups", 0)
        return [
            str(epoch),
            str(n_groups),
            f"{mixed} ({_fmt_pct(_pct(mixed, n_groups))})" if n_groups else "NA",
            f"{temp_s} ({_fmt_pct(_pct(temp_s, n_groups))})" if n_groups else "NA",
            f"{t.get('avg_unique_verdicts_per_group'):.2f}" if isinstance(t.get("avg_unique_verdicts_per_group"), (int, float)) else "NA",
            f"{t.get('avg_temps_per_group'):.2f}" if isinstance(t.get("avg_temps_per_group"), (int, float)) else "NA",
        ]

    lines: List[str] = []
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Stage-B 对比报告：{a.mission}（{model_a_name} vs {model_b_name}）\n")
    lines.append(f"- 生成时间：`{now}`")
    lines.append(f"- 32B 输出目录：`{a.run_dir}`")
    lines.append(f"- 8B 输出目录：`{b.run_dir}`\n")

    # Executive summary (based on last available metrics if present)
    a_metrics = _collect_metrics(a)
    b_metrics = _collect_metrics(b)
    a_metrics_last_epoch = max(a_metrics) if a_metrics else None
    b_metrics_last_epoch = max(b_metrics) if b_metrics else None
    a_last_m = a_metrics.get(a_metrics_last_epoch) if a_metrics_last_epoch is not None else None
    b_last_m = b_metrics.get(b_metrics_last_epoch) if b_metrics_last_epoch is not None else None
    a_inc = (a_last_m or {}).get("include_manual_review", {}).get("acc")
    b_inc = (b_last_m or {}).get("include_manual_review", {}).get("acc")
    a_exc = (a_last_m or {}).get("exclude_manual_review", {}).get("acc")
    b_exc = (b_last_m or {}).get("exclude_manual_review", {}).get("acc")

    lines.append("## 1) 执行摘要\n")
    if isinstance(a_inc, (int, float)) and isinstance(b_inc, (int, float)):
        lines.append(
            f"- 在该 mission 的**metrics 最末 epoch**（{model_a_name}: {a_metrics_last_epoch} / {model_b_name}: {b_metrics_last_epoch}）上，`include_manual_review.acc`：{model_a_name}={a_inc:.4f}，{model_b_name}={b_inc:.4f}。"
        )
    if isinstance(a_exc, (int, float)) and isinstance(b_exc, (int, float)):
        lines.append(
            f"- `exclude_manual_review.acc`：{model_a_name}={a_exc:.4f}，{model_b_name}={b_exc:.4f}（注意 n 可能不同，见下表）。"
        )
    if a_first is not None and b_first is not None:
        a_acc1, a_n1 = _accuracy_for_epoch(a, a_first)
        b_acc1, b_n1 = _accuracy_for_epoch(b, b_first)
        if isinstance(a_acc1, float) and isinstance(b_acc1, float):
            lines.append(f"- epoch=1 的 `label_match`（record-level）匹配率：{model_a_name}={a_acc1:.4f} (n={a_n1})，{model_b_name}={b_acc1:.4f} (n={b_n1})。")

    # Reflection signal headline
    def _refl_head(summary: RunSummary) -> str:
        total = sum(s.get("n", 0) for s in summary.reflection_epoch.values())
        eligible = sum(s.get("eligible", 0) for s in summary.reflection_epoch.values())
        applied = sum(s.get("applied", 0) for s in summary.reflection_epoch.values())
        # Most common ineligible / critique tag
        inel = Counter()
        crit = Counter()
        for s in summary.reflection_epoch.values():
            inel.update(s.get("ineligible_reason", Counter()))
            crit.update(s.get("critique_tags", Counter()))
        inel_top = inel.most_common(1)[0][0] if inel else "NA"
        crit_top = crit.most_common(1)[0][0] if crit else "NA"
        return f"reflection 条目 {total}（eligible {eligible} / applied {applied}），常见 ineligible_reason=`{inel_top}`，常见 critique=`{crit_top}`"

    if a.files_present.get("reflection.jsonl") or b.files_present.get("reflection.jsonl"):
        lines.append(f"- Reflection 概览：{model_a_name}={_refl_head(a)}；{model_b_name}={_refl_head(b)}。")

    lines.append("\n> 说明：两次运行的 epoch 数量不同（本报告同时对齐对比 epoch=1 与各自最终 epoch）。\n")
    if a_max_seen is not None and a_final is not None and a_max_seen != a_final:
        lines.append(f"> 说明：{model_a_name} 观测到的最大 epoch={a_max_seen}，但用于对比的 final epoch={a_final}（优先取 metrics 最末或记录数最多的 epoch；更大的 epoch 可能是部分写出）。\n")
    if b_max_seen is not None and b_final is not None and b_max_seen != b_final:
        lines.append(f"> 说明：{model_b_name} 观测到的最大 epoch={b_max_seen}，但用于对比的 final epoch={b_final}。\n")

    # Detailed comparison
    lines.append("## 2) 详细对比\n")

    # Rollouts + selections
    lines.append("### 2.1 Rollouts / Selections 质量对比\n")
    headers = [
        "Epoch",
        "traj_groups",
        "traj_candidates",
        "format_ok",
        "cand.pass%",
        "sel_records",
        "sel_groups",
        "sel_units",
        "sel.pass%",
        "label_match",
        "match_n",
        "vote_p10",
        "vote_p50",
        "vote_p90",
    ]
    rows = []
    if a_first is not None:
        rows.append([model_a_name] + _epoch_row(a, a_first))
    if a_final is not None and a_final != a_first:
        rows.append([model_a_name] + _epoch_row(a, a_final))
    if b_first is not None:
        rows.append([model_b_name] + _epoch_row(b, b_first))
    if b_final is not None and b_final != b_first:
        rows.append([model_b_name] + _epoch_row(b, b_final))
    lines.append(_md_table(["Model"] + headers, rows))
    lines.append("")
    lines.append("- `label_match` 为 `selections.jsonl` 的 record-level 指标；同一 `group_id` 可能因 reflection cycle 等出现多条记录。")
    lines.append("")

    # Diversity
    lines.append("**候选多样性（verdict 在组内/温度间是否变化）**\n")
    headers2 = ["Epoch", "Groups", "mixed_verdict_groups", "temp_sensitive_groups", "avg_unique_verdicts", "avg_temps"]
    rows2 = []
    if a_first is not None:
        rows2.append([model_a_name] + _epoch_div_row(a, a_first))
    if a_final is not None and a_final != a_first:
        rows2.append([model_a_name] + _epoch_div_row(a, a_final))
    if b_first is not None:
        rows2.append([model_b_name] + _epoch_div_row(b, b_first))
    if b_final is not None and b_final != b_first:
        rows2.append([model_b_name] + _epoch_div_row(b, b_final))
    lines.append(_md_table(["Model"] + headers2, rows2))
    lines.append("")

    # Reflection + guidance
    lines.append("### 2.2 Reflection / Guidance 过程对比\n")
    rh = ["Epoch", "n_reflection", "eligible", "applied", "top_ineligible", "top_action", "top_ops", "cache_files(summary/critique/plan)"]
    rrows = []

    def _refl_row(summary: RunSummary, epoch: int) -> List[str]:
        s = summary.reflection_epoch.get(epoch, {})
        inel = s.get("ineligible_reason", Counter())
        act = s.get("proposal_action", Counter())
        ops = s.get("op_types", Counter())
        top_inel = inel.most_common(1)[0][0] if inel else "NA"
        top_act = act.most_common(1)[0][0] if act else "NA"
        top_ops = ", ".join([f"{k}:{v}" for k, v in ops.most_common(3)]) if ops else "NA"
        cache = s.get("reflection_cache") or {}
        t = cache.get("types") or {}
        cache_fmt = f"{cache.get('files',0)} ({t.get('summary',0)}/{t.get('critique',0)}/{t.get('plan',0)})"
        return [str(epoch), str(s.get("n", 0)), str(s.get("eligible", 0)), str(s.get("applied", 0)), f"`{top_inel}`", f"`{top_act}`", f"`{top_ops}`", cache_fmt]

    if a_first is not None and a.reflection_epoch:
        rrows.append([model_a_name] + _refl_row(a, a_first))
    if a_final is not None and a_final != a_first and a.reflection_epoch:
        rrows.append([model_a_name] + _refl_row(a, a_final))
    if b_first is not None and b.reflection_epoch:
        rrows.append([model_b_name] + _refl_row(b, b_first))
    if b_final is not None and b_final != b_first and b.reflection_epoch:
        rrows.append([model_b_name] + _refl_row(b, b_final))

    if rrows:
        lines.append(_md_table(["Model"] + rh, rrows))
        lines.append("")
    else:
        lines.append("- 未发现 `reflection.jsonl`，跳过该项。\n")

    def _guidance_brief(summary: RunSummary) -> str:
        g = summary.guidance or {}
        step = g.get("step")
        updated = g.get("updated_at")
        exp = g.get("experiences")
        if isinstance(exp, dict):
            exp_keys = sorted(exp.keys())
            exp_n = len(exp_keys)
            exp_keys_s = ", ".join(exp_keys[:8]) + ("…" if exp_n > 8 else "")
        else:
            exp_n = _safe_len(exp)
            exp_keys_s = "NA"
        snaps = len(summary.snapshots)
        return f"step={step}, updated_at={updated}, experiences={exp_n} ({exp_keys_s}), snapshots={snaps}"

    lines.append(f"- `guidance.json`：{model_a_name}：{_guidance_brief(a)}；{model_b_name}：{_guidance_brief(b)}。")
    if b.snapshots:
        steps = [x.get("step") for x in b.snapshots if isinstance(x, dict)]
        steps_s = ", ".join([str(x) for x in steps if x is not None][:10])
        lines.append(f"- `{model_b_name}` snapshots（按文件名排序）step 序列（截断）：{steps_s}{'…' if len(steps)>10 else ''}。")
    lines.append("")

    # Metrics
    lines.append("### 2.3 性能指标对比（metrics.jsonl）\n")
    if a_metrics or b_metrics:
        if a_metrics:
            lines.append(f"**{model_a_name}**")
            lines.append(_md_table(["epoch", "exc.acc", "exc.fn", "exc.fp", "exc.n", "inc.acc", "inc.fn", "inc.fp", "inc.n"], _metrics_rows(a_metrics)))
            lines.append("")
        if b_metrics:
            lines.append(f"**{model_b_name}**")
            lines.append(_md_table(["epoch", "exc.acc", "exc.fn", "exc.fp", "exc.n", "inc.acc", "inc.fn", "inc.fp", "inc.n"], _metrics_rows(b_metrics)))
            lines.append("")
    else:
        lines.append("- 未发现 `metrics.jsonl`（或 legacy `metrics_epoch.jsonl`）。\n")

    # Manual/need review
    lines.append("### 2.4 Manual Review / Need Review 队列对比\n")

    def _queue_summary(name: str, q: Dict[int, Counter]) -> str:
        total = sum(sum(c.values()) for c in q.values())
        top = Counter()
        for c in q.values():
            top.update(c)
        top_s = ", ".join([f"`{k}`:{v}" for k, v in top.most_common(6)]) if top else "NA"
        return f"{name}: {total}（top tags: {top_s}）"

    lines.append(f"- {model_a_name}：{_queue_summary('need_review_queue', a.need_review)}；{_queue_summary('manual_review_queue', a.manual_review)}。")
    lines.append(f"- {model_b_name}：{_queue_summary('need_review_queue', b.need_review)}；{_queue_summary('manual_review_queue', b.manual_review)}。")
    lines.append("")

    # Malformed
    lines.append("### 2.5 错误模式（failure_malformed.jsonl 等）\n")
    if a.malformed or b.malformed:
        def _mal_summary(m: Dict[int, Counter]) -> str:
            total = sum(sum(c.values()) for c in m.values())
            top = Counter()
            for c in m.values():
                top.update(c)
            top_s = ", ".join([f"`{k}`:{v}" for k, v in top.most_common(6)]) if top else "NA"
            return f"{total}（top: {top_s}）"
        lines.append(f"- {model_a_name} malformed：{_mal_summary(a.malformed)}。")
        lines.append(f"- {model_b_name} malformed：{_mal_summary(b.malformed)}。")
    else:
        lines.append("- 两次运行均未产生 malformed 记录，或文件缺失。")
    lines.append("")

    # Examples where one is correct and other is wrong
    lines.append("## 3) 关键差异样例（各自内部：label_match=false）\n")

    def _mismatch_rows(summary: RunSummary, epoch: int, max_rows: int = 8) -> List[List[str]]:
        items = []
        for (e, gid), rec in summary.selection_by_epoch_group.items():
            if e != epoch:
                continue
            lm = rec.get("label_match")
            if lm is False:
                gt = rec.get("gt_label") or summary.gt_by_group.get(gid)
                key = rec.get("ticket_key") or rec.get("unit_key") or gid
                items.append((key, gid, gt, rec.get("verdict"), rec.get("vote_strength"), rec.get("warnings"), rec.get("reason")))
        # Prefer higher vote_strength (confident but wrong)
        items.sort(key=lambda x: (x[4] if isinstance(x[4], (int, float)) else -1), reverse=True)
        rows = []
        for key, gid, gt, v, vs, w, reason in items[:max_rows]:
            rows.append(
                [
                    str(key),
                    str(gid),
                    str(gt) if gt is not None else "NA",
                    str(v) if v is not None else "NA",
                    f"{vs:.3f}" if isinstance(vs, (int, float)) else "NA",
                    _trim(w, 60),
                    _trim(reason, 90),
                ]
            )
        return rows

    if a_first is not None:
        a_m1 = _mismatch_rows(a, a_first, 8)
        lines.append(f"**{model_a_name} / epoch={a_first}**\n")
        lines.append(_md_table(["ticket_or_unit", "group_id", "gt_label", "verdict", "vote", "warnings", "reason"], a_m1 or [["NA"] * 7]))
        lines.append("")
    if b_first is not None:
        b_m1 = _mismatch_rows(b, b_first, 8)
        lines.append(f"**{model_b_name} / epoch={b_first}**\n")
        lines.append(_md_table(["ticket_or_unit", "group_id", "gt_label", "verdict", "vote", "warnings", "reason"], b_m1 or [["NA"] * 7]))
        lines.append("")

    if a_final is not None and a_first is not None and a_final != a_first:
        a_ml = _mismatch_rows(a, a_final, 8)
        lines.append(f"**{model_a_name} / epoch={a_final}**\n")
        lines.append(_md_table(["ticket_or_unit", "group_id", "gt_label", "verdict", "vote", "warnings", "reason"], a_ml or [["NA"] * 7]))
        lines.append("")
    if b_final is not None and b_first is not None and b_final != b_first:
        b_ml = _mismatch_rows(b, b_final, 8)
        lines.append(f"**{model_b_name} / epoch={b_final}**\n")
        lines.append(_md_table(["ticket_or_unit", "group_id", "gt_label", "verdict", "vote", "warnings", "reason"], b_ml or [["NA"] * 7]))
        lines.append("")

    # Key findings and recommendations (data-driven but short)
    lines.append("## 4) 关键发现\n")
    findings = []
    if isinstance(a_inc, (int, float)) and isinstance(b_inc, (int, float)):
        if b_inc > a_inc + 1e-6:
            findings.append(f"- 该 mission 上，{model_b_name} 最终 `include_manual_review.acc` 明显高于 {model_a_name}（{b_inc:.4f} vs {a_inc:.4f}）。")
        elif a_inc > b_inc + 1e-6:
            findings.append(f"- 该 mission 上，{model_a_name} 最终 `include_manual_review.acc` 高于 {model_b_name}（{a_inc:.4f} vs {b_inc:.4f}）。")
        else:
            findings.append(f"- 该 mission 上，两者最终 `include_manual_review.acc` 接近（{a_inc:.4f} vs {b_inc:.4f}）。")
    # Reflection quality signal: generation errors / applied ops
    def _has_generation_error(summary: RunSummary) -> bool:
        for s in summary.reflection_epoch.values():
            c = s.get("critique_tags", Counter())
            if any("generation_error" in k for k in c.keys()):
                return True
        return False

    if _has_generation_error(a):
        findings.append(f"- {model_a_name} reflection 存在 `generation_error`（反思响应未解析出有效 JSON），可能导致 guidance 更新停滞。")
    if _has_generation_error(b):
        findings.append(f"- {model_b_name} reflection 存在 `generation_error`，需检查反思 prompt/解析稳定性。")
    # Guidance delta
    a_step = (a.guidance or {}).get("step")
    b_step = (b.guidance or {}).get("step")
    if isinstance(a_step, int) and isinstance(b_step, int) and a_step != b_step:
        findings.append(f"- 最终 `guidance.step` 不同：{model_a_name}={a_step}，{model_b_name}={b_step}（结合 snapshots 可进一步追踪演化路径）。")
    # Queue tradeoff
    a_need = sum(sum(c.values()) for c in a.need_review.values())
    b_need = sum(sum(c.values()) for c in b.need_review.values())
    a_man = sum(sum(c.values()) for c in a.manual_review.values())
    b_man = sum(sum(c.values()) for c in b.manual_review.values())
    findings.append(f"- 队列规模差异：need_review {model_a_name}={a_need} vs {model_b_name}={b_need}；manual_review {model_a_name}={a_man} vs {model_b_name}={b_man}。")
    if findings:
        lines.extend(findings)
    else:
        lines.append("- 当前产物不足以给出稳定结论（缺少 metrics / gt_label / reflection 等关键信息）。")
    lines.append("")

    lines.append("## 5) 建议\n")
    recs = []
    if isinstance(a_inc, (int, float)) and isinstance(b_inc, (int, float)):
        if b_inc >= a_inc:
            recs.append(f"- 就该 mission 的这次对比结果而言，{model_b_name} **可以作为替代候选**（至少在 `include_manual_review` 指标上不弱于 {model_a_name}）。")
        else:
            recs.append(f"- 就该 mission 的这次对比结果而言，{model_b_name} 暂不建议直接替代 {model_a_name}（最终 `include_manual_review` 指标落后）。")
    recs.append("- 替代决策建议至少覆盖多个 missions/数据切片（不同品牌/遮挡/只显示部分/无关图比例），避免单 mission 偏差。")
    recs.append("- 若继续用 Stage-B training-free 迭代：优先修复/降低 reflection 的解析失败（例如 `generation_error`），否则 guidance 学习几乎不可用。")
    recs.append("- 对齐对比时建议固定相同 epoch 数/相同 reflection 策略（或显式关闭 reflection），否则“最终表现”会混入训练回合数差异。")
    lines.extend(recs)
    lines.append("")

    lines.append("## 6) 附录：文件存在性与规模\n")
    fp_rows = []
    keys = sorted(set(a.files_present.keys()) | set(b.files_present.keys()))
    for k in keys:
        fp_rows.append([k, "Y" if a.files_present.get(k) else "N", "Y" if b.files_present.get(k) else "N"])
    lines.append(_md_table(["artifact", model_a_name, model_b_name], fp_rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-32b", required=True, help="32B run mission output dir (contains trajectories.jsonl etc.)")
    ap.add_argument("--run-8b", required=True, help="8B run mission output dir (contains trajectories.jsonl etc.)")
    ap.add_argument("--mission", required=True, help="mission name, e.g. 挡风板安装检查")
    ap.add_argument("--out", required=True, help="output markdown path")
    ap.add_argument("--name-32b", default="qwen3-32B", help="label for 32B model")
    ap.add_argument("--name-8b", default="qwen3-vl-8B", help="label for 8B model")
    args = ap.parse_args()

    run_32b = Path(args.run_32b)
    run_8b = Path(args.run_8b)
    if not run_32b.exists():
        raise SystemExit(f"run-32b not found: {run_32b}")
    if not run_8b.exists():
        raise SystemExit(f"run-8b not found: {run_8b}")

    s32 = _summarize_run(run_32b, args.mission)
    s8 = _summarize_run(run_8b, args.mission)
    _render_report(args.name_32b, s32, args.name_8b, s8, Path(args.out))


if __name__ == "__main__":
    main()
