#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline audit: surface groups that repeatedly trigger conflict_flag but never
receive applied reflection operations. Useful to spot suspected label/summary
noise or missing guidance gaps.

Usage:
  python scripts/stage_b_conflict_audit.py --run-dir output/<run_name> [--mission <mission>]

Assumptions:
  - Selections are stored at {run_dir}/{mission}/selections.jsonl
  - Reflection log is stored at {run_dir}/{mission}/reflection.jsonl
  - selection lines include result.conflict_flag (added by Stage-B exports)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def _load_conflicts(selections_path: Path) -> Dict[str, Dict[str, object]]:
    conflicts: Dict[str, Dict[str, object]] = {}
    counts: Dict[str, int] = defaultdict(int)

    with selections_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            result = data.get("result", {})
            group_id = data.get("group_id") or result.get("group_id")
            conflict_flag = result.get("conflict_flag", False)
            if not conflict_flag or group_id is None:
                continue
            counts[group_id] += 1
            # Keep the latest snapshot for reporting
            conflicts[group_id] = {
                "reason": result.get("reason"),
                "warnings": result.get("warnings", []),
                "uncertainty_notes": result.get("uncertainty_notes", []),
                "epochs": conflicts.get(group_id, {}).get("epochs", set())
                if group_id in conflicts
                else set(),
            }
            epoch = data.get("epoch")
            if epoch is not None:
                conflicts[group_id]["epochs"].add(epoch)

    # Flatten epochs sets for output
    for gid, meta in conflicts.items():
        epochs = meta.get("epochs")
        meta["epochs"] = sorted(epochs) if isinstance(epochs, set) else epochs
        meta["count"] = counts.get(gid, 0)
    return conflicts


def _load_applied_groups(reflection_path: Path) -> Tuple[set, set]:
    """
    Returns:
      applied_groups: groups cited by applied operations
      attempted_groups: groups seen in any proposal (applied or not)
    """
    applied_groups: set = set()
    attempted_groups: set = set()
    with reflection_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            refl = data.get("reflection", data)
            proposal = refl.get("proposal", {})
            evidence = proposal.get("evidence_group_ids", []) or []
            for gid in evidence:
                attempted_groups.add(gid)
                if refl.get("applied"):
                    applied_groups.add(gid)
            # Also look at per-op evidence for completeness
            for op in proposal.get("operations", []) or []:
                for gid in op.get("evidence", []) or []:
                    attempted_groups.add(gid)
                    if refl.get("applied"):
                        applied_groups.add(gid)
    return applied_groups, attempted_groups


def audit_mission(mission_dir: Path) -> None:
    selections = mission_dir / "selections.jsonl"
    reflection = mission_dir / "reflection.jsonl"
    if not selections.exists():
        print(f"[skip] {mission_dir}: selections.jsonl not found")
        return
    if not reflection.exists():
        print(f"[skip] {mission_dir}: reflection.jsonl not found")
        return

    conflicts = _load_conflicts(selections)
    applied_groups, attempted_groups = _load_applied_groups(reflection)

    unresolved = []
    for gid, meta in conflicts.items():
        if gid in applied_groups:
            continue
        unresolved.append(
            (
                gid,
                meta.get("count", 0),
                meta.get("epochs", []),
                meta.get("reason"),
                meta.get("uncertainty_notes", []),
                gid in attempted_groups,
            )
        )

    if not unresolved:
        print(f"[ok] {mission_dir.name}: no unresolved conflict_flag groups")
        return

    print(f"[warn] {mission_dir.name}: unresolved conflict_flag groups (count, epochs, attempted_reflection?)")
    for gid, count, epochs, reason, notes, attempted in sorted(
        unresolved, key=lambda x: (-x[1], x[0])
    ):
        epoch_str = ",".join(map(str, epochs)) if epochs else "-"
        note_str = "|".join(notes) if notes else ""
        attempted_flag = "yes" if attempted else "no"
        print(f"  {gid}: count={count} epochs={epoch_str} attempted_reflection={attempted_flag}")
        if reason:
            print(f"    reason: {reason}")
        if note_str:
            print(f"    notes : {note_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit unresolved conflict_flag groups")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run directory (e.g., output/debug or output/11-27)",
    )
    parser.add_argument(
        "--mission",
        help="Optional mission name; default scans all missions under run-dir",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    missions = [Path(args.mission)] if args.mission else sorted(run_dir.iterdir())
    for mission_dir in missions:
        if not mission_dir.is_dir():
            continue
        audit_mission(mission_dir)


if __name__ == "__main__":
    main()
