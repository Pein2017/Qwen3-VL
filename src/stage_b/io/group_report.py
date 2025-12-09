#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bundle Stage-B outputs into per-group reports.

Utility used by both the runner (auto-generation) and CLI script.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..utils.chinese import normalize_spaces, to_simplified


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stage_a_labels(paths: Optional[Iterable[Path]]) -> Dict[tuple, str]:
    labels: Dict[tuple, str] = {}
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
                labels[(mission, gid)] = label
    return labels


def _index_by_group(items: Iterable[dict], key: str = "group_id") -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        gid = item.get(key)
        if gid:
            grouped[gid].append(item)
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


def build_group_report(run_dir: Path, stage_a_paths: Optional[Iterable[Path]] = None) -> Path:
    """Create consolidated per-group report JSONL under run_dir."""

    selections = _load_jsonl(run_dir / "selections.jsonl")
    trajectories = _load_jsonl(run_dir / "trajectories.jsonl")
    manual = _index_by_group(_load_jsonl(run_dir / "manual_review_queue.jsonl"))
    noise = _index_by_group(_load_jsonl(run_dir / "label_or_stageA_noise.jsonl"))
    reflection = _reflection_ops(run_dir / "reflection.jsonl")
    labels = _stage_a_labels(stage_a_paths)

    sel_map = {item["group_id"]: item for item in selections if item.get("group_id")}
    cand_map = defaultdict(list)
    for item in trajectories:
        gid = item.get("group_id")
        if gid:
            reason = item.get("reason") or item.get("result", {}).get("reason")
            cand_map[gid].append(
                {
                    "candidate_index": item.get("candidate_index") or item.get("result", {}).get("candidate_index"),
                    "verdict": item.get("verdict") or item.get("result", {}).get("verdict"),
                    "reason": _normalize_reason(reason),
                    "decode": (item.get("decode") or item.get("result", {}).get("decode")),
                    "format_ok": item.get("format_ok", True),
                    "warnings": item.get("warnings", []),
                    "label_match": item.get("label_match"),
                }
            )

    group_ids = set(sel_map.keys()) | set(cand_map.keys()) | set(manual.keys()) | set(noise.keys()) | set(reflection.keys())

    out_path = run_dir / "group_report.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for gid in sorted(group_ids):
            sel = sel_map.get(gid)
            mission = sel.get("mission") if sel else None
            selection_result = sel.get("result") if sel else None
            
            # Normalize reason in selection_result
            if selection_result and "reason" in selection_result:
                selection_result = dict(selection_result)  # Make a copy
                selection_result["reason"] = _normalize_reason(selection_result.get("reason"))
            
            # Normalize reasons in noise entries
            normalized_noise = []
            for noise_entry in noise.get(gid, []):
                normalized_entry = dict(noise_entry)
                if "why" in normalized_entry:
                    normalized_entry["why"] = _normalize_reason(normalized_entry.get("why"))
                normalized_noise.append(normalized_entry)
            
            # Normalize text in reflection entries
            normalized_reflection = []
            for refl_entry in reflection.get(gid, []):
                normalized_entry = dict(refl_entry)
                if "text" in normalized_entry:
                    normalized_entry["text"] = _normalize_reason(normalized_entry.get("text"))
                normalized_reflection.append(normalized_entry)

            record = {
                "group_id": gid,
                "mission": mission,
                "label": labels.get((mission, gid)),
                "selection": selection_result,
                "candidates": cand_map.get(gid, []),
                "manual_review": manual.get(gid, []),
                "label_or_stageA_noise": normalized_noise,
                "reflection": normalized_reflection,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


__all__ = ["build_group_report"]
