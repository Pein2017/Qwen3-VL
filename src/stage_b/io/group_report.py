#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bundle Stage-B outputs into per-group reports.

Utility used by both the runner (auto-generation) and CLI script.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from ..utils.chinese import normalize_spaces, to_simplified


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stage_a_labels(paths: Optional[Iterable[Path]]) -> Dict[Tuple[str, str], str]:
    labels: Dict[Tuple[str, str], str] = {}
    if not paths:
        return labels
    for path in paths:
        if not path or not path.exists():
            continue
        for item in _load_jsonl(path):
            gid = item.get("group_id")
            mission = item.get("mission")
            label = item.get("label")
            if gid and mission and label:
                ticket_key = f"{gid}::{label}"
                labels[(str(mission), ticket_key)] = str(label)
    return labels


def _ticket_key_from_payload(payload: Mapping[str, object]) -> Optional[str]:
    raw_key = payload.get("ticket_key")
    if isinstance(raw_key, str) and raw_key.strip():
        return raw_key.strip()

    gid = payload.get("group_id")
    if gid is None:
        return None
    gid_str = str(gid).strip()
    if not gid_str:
        return None

    # Prefer explicit ground-truth label fields when present.
    raw_label = payload.get("gt_label")
    if raw_label is None:
        raw_label = payload.get("label")
    if isinstance(raw_label, str) and raw_label.strip():
        return f"{gid_str}::{raw_label.strip().lower()}"

    # Backward-compat: fall back to group_id only when label is missing.
    return gid_str


def _index_by_ticket_key(items: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        key = _ticket_key_from_payload(item)
        if key:
            grouped[key].append(item)
    return grouped


def _reflection_ops(path: Path) -> Dict[str, List[dict]]:
    ops_by_group: Dict[str, List[dict]] = defaultdict(list)
    for entry in _load_jsonl(path):
        refl = entry.get("reflection", {})
        refl_id = refl.get("reflection_id")
        proposal = refl.get("proposal", {})
        operations = proposal.get("operations") or []
        evidence = proposal.get("evidence_group_ids") or []
        for op in operations:
            record = {
                "reflection_id": refl_id,
                "op": op.get("op"),
                "text": op.get("text"),
                "rationale": op.get("rationale"),
                "evidence": op.get("evidence") or evidence,
            }
            for gid in record["evidence"]:
                ops_by_group[gid].append(record)
    return ops_by_group


def _normalize_reason(reason: Optional[str]) -> Optional[str]:
    """Normalize reason text: convert to simplified Chinese and normalize spaces."""
    if not reason:
        return reason
    normalized = to_simplified(reason)
    normalized = normalize_spaces(normalized)
    return normalized


def build_group_report(
    run_dir: Path,
    stage_a_paths: Optional[Iterable[Path]] = None,
    *,
    output_path: Optional[Path] = None,
) -> Path:
    """Create consolidated per-group report JSONL.

    By default this writes `group_report.jsonl` under `run_dir`. Callers may
    override `output_path` to write snapshots with distinct filenames.
    """

    selections = _load_jsonl(run_dir / "selections.jsonl")
    trajectories = _load_jsonl(run_dir / "trajectories.jsonl")
    need_review = _index_by_ticket_key(_load_jsonl(run_dir / "need_review_queue.jsonl"))
    reflection = _reflection_ops(run_dir / "reflection.jsonl")
    labels = _stage_a_labels(stage_a_paths)

    sel_map: Dict[str, dict] = {}
    for item in selections:
        ticket_key = _ticket_key_from_payload(item)
        if ticket_key:
            sel_map[ticket_key] = item
    cand_map = defaultdict(list)
    for item in trajectories:
        ticket_key = _ticket_key_from_payload(item)
        if ticket_key:
            reason = item.get("reason") or item.get("result", {}).get("reason")
            cand_map[ticket_key].append(
                {
                    "mission": item.get("mission"),
                    "candidate_index": item.get("candidate_index") or item.get("result", {}).get("candidate_index"),
                    "verdict": item.get("verdict") or item.get("result", {}).get("verdict"),
                    "reason": _normalize_reason(reason),
                    "decode": (item.get("decode") or item.get("result", {}).get("decode")),
                    "format_ok": item.get("format_ok", True),
                    "warnings": item.get("warnings", []),
                    "label_match": item.get("label_match"),
                }
            )

    group_ids = (
        set(sel_map.keys())
        | set(cand_map.keys())
        | set(need_review.keys())
    )

    out_path = output_path if output_path is not None else (run_dir / "group_report.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for ticket_key in sorted(group_ids):
            base_gid = ticket_key.split("::", 1)[0]
            sel = sel_map.get(ticket_key)
            mission = sel.get("mission") if sel else None
            if mission is None:
                for pool in (need_review.get(ticket_key), cand_map.get(ticket_key)):
                    if pool:
                        mission = pool[0].get("mission")
                        break
            selection_result = sel.get("result") if sel else None
            
            # Normalize reason in selection_result
            if selection_result and "reason" in selection_result:
                selection_result = dict(selection_result)  # Make a copy
                selection_result["reason"] = _normalize_reason(selection_result.get("reason"))
            
            # Normalize fields in need-review entries
            normalized_need_review = []
            for review_entry in need_review.get(ticket_key, []):
                normalized_entry = dict(review_entry)
                if "reason" in normalized_entry:
                    normalized_entry["reason"] = _normalize_reason(normalized_entry.get("reason"))
                normalized_need_review.append(normalized_entry)
            
            # Normalize text in reflection entries
            normalized_reflection = []
            refl_entries = reflection.get(ticket_key, [])
            if not refl_entries and base_gid != ticket_key:
                # Backward-compat: older runs keyed reflection evidence by bare group_id.
                refl_entries = reflection.get(base_gid, [])
            for refl_entry in refl_entries:
                normalized_entry = dict(refl_entry)
                if "text" in normalized_entry:
                    normalized_entry["text"] = _normalize_reason(normalized_entry.get("text"))
                normalized_reflection.append(normalized_entry)

            record = {
                "ticket_key": ticket_key,
                "group_id": base_gid,
                "mission": mission,
                "label": (
                    labels.get((str(mission), ticket_key))
                    if mission
                    else (ticket_key.split("::", 1)[1] if "::" in ticket_key else None)
                ),
                "selection": selection_result,
                "candidates": cand_map.get(ticket_key, []),
                "need_review_queue": normalized_need_review,
                "reflection": normalized_reflection,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


__all__ = ["build_group_report"]
