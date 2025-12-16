#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess Stage-B outputs into a human review checklist.

This script reads Stage-B artifacts under {output.root}/{output.run_name}/{mission}/
and generates a markdown checklist for all tickets in need-review queue.

It is intended to be called automatically at the end of scripts/stage_b.sh.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.stage_b.config import load_stage_b_config


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _index_stage_a(stage_a_paths: Iterable[Path]) -> Dict[Tuple[str, str], dict]:
    """Map (mission, group_id) -> Stage-A record."""
    stage_a: Dict[Tuple[str, str], dict] = {}
    for path in stage_a_paths:
        if not path.exists():
            continue
        for item in _load_jsonl(path):
            mission = item.get("mission")
            group_id = item.get("group_id")
            if not mission or not group_id:
                continue
            stage_a[(str(mission), str(group_id))] = item
    return stage_a


def _pick_latest_need_review(entries: List[dict]) -> dict:
    def _key(entry: dict) -> Tuple[int, int, str]:
        epoch = entry.get("epoch")
        cycle = entry.get("reflection_cycle")
        epoch_val = int(epoch) if isinstance(epoch, int) or str(epoch).isdigit() else 0
        cycle_val = int(cycle) if isinstance(cycle, int) or str(cycle).isdigit() else 0
        rid = str(entry.get("reflection_id") or "")
        return (epoch_val, cycle_val, rid)

    return max(entries, key=_key)


def _format_markdown_list(items: Iterable[str]) -> str:
    return "\n".join([f"- {item}" for item in items])


def _render_group_markdown(
    *,
    group_id: str,
    mission: str,
    latest: Mapping[str, Any],
    history: List[Mapping[str, Any]],
    stage_a_record: Optional[Mapping[str, Any]],
) -> str:
    gt_label = latest.get("gt_label")
    chosen_verdict = latest.get("chosen_verdict")
    tag = latest.get("tag")
    reason = latest.get("reason")
    evidence_summary = latest.get("evidence_summary") or {}

    epoch_hist = []
    for item in sorted(history, key=lambda x: (int(x.get("epoch") or 0), int(x.get("reflection_cycle") or 0))):
        epoch_hist.append(
            f"epoch={item.get('epoch')} cycle={item.get('reflection_cycle')} tag={item.get('tag')} verdict={item.get('chosen_verdict')}"
        )

    images = []
    per_image = {}
    label = None
    if stage_a_record:
        label = stage_a_record.get("label")
        images = list(stage_a_record.get("images") or [])
        per_image = dict(stage_a_record.get("per_image") or {})

    parts: List[str] = [f"## {group_id}"]
    parts.append(f"- mission: {mission}")
    if label is not None:
        parts.append(f"- stage_a_label: {label}")
    parts.append(f"- gt_label(stage_b_input): {gt_label}")
    parts.append(f"- chosen_verdict: {chosen_verdict}")
    parts.append(f"- tag(latest): {tag}")
    parts.append("- need_review_history:")
    parts.append(_format_markdown_list(epoch_hist) if epoch_hist else "- (empty)")
    parts.append("- stage_b_reason(latest):")
    parts.append(f"  - {reason}" if reason else "  - (empty)")

    rel_neg = evidence_summary.get("relevant_negative_hits") or []
    pending = evidence_summary.get("pending_signal_hits") or []
    parts.append("- evidence_summary:")
    parts.append("  - relevant_negative_hits:")
    parts.append(
        _format_markdown_list([str(x) for x in rel_neg]).replace("- ", "    - ")
        if rel_neg
        else "    - (none)"
    )
    parts.append("  - pending_signal_hits:")
    parts.append(
        _format_markdown_list([str(x) for x in pending]).replace("- ", "    - ")
        if pending
        else "    - (none)"
    )

    parts.append("- images:")
    parts.append(
        _format_markdown_list([str(x) for x in images]).replace("- ", "  - ")
        if images
        else "  - (missing)"
    )

    parts.append("- stage_a_per_image:")
    if per_image:
        for key in sorted(per_image.keys()):
            text = str(per_image.get(key) or "")
            parts.append(f"  - {key}: {text}")
    else:
        parts.append("  - (missing)")

    parts.append("")  # trailing newline
    return "\n".join(parts)


def build_review_checklist_for_mission(
    mission_dir: Path,
    *,
    stage_a_index: Mapping[Tuple[str, str], dict],
) -> Optional[Path]:
    need_review_path = mission_dir / "need_review_queue.jsonl"
    if not need_review_path.exists():
        return None

    items = _load_jsonl(need_review_path)
    if not items:
        return None

    # Group by group_id, keep all history and pick a latest representative row.
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        gid = str(item.get("group_id") or "")
        mission = str(item.get("mission") or "")
        if not gid or not mission:
            continue
        grouped[gid].append(item)

    now = datetime.now(timezone.utc).isoformat()
    out_md = mission_dir / "review_checklist.md"
    out_jsonl = mission_dir / "review_checklist.jsonl"

    tag_counter = Counter()
    for gid, history in grouped.items():
        latest = _pick_latest_need_review(history)
        tag_counter[str(latest.get("tag") or "unknown")] += 1

    # Write JSONL (one line per group, latest + history + stage-a snapshot).
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for gid in sorted(grouped.keys()):
            history = grouped[gid]
            latest = _pick_latest_need_review(history)
            mission = str(latest.get("mission") or "")
            stage_a_record = stage_a_index.get((mission, gid))
            payload = {
                "group_id": gid,
                "mission": mission,
                "latest": latest,
                "history": history,
                "stage_a_record": stage_a_record,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # Write Markdown for humans.
    md_lines: List[str] = []
    md_lines.append("# Stage-B Need-Review 复核清单")
    md_lines.append("")
    md_lines.append(f"- generated_at: {now}")
    md_lines.append(f"- mission_dir: {mission_dir}")
    md_lines.append(f"- need_review_unique_groups: {len(grouped)}")
    md_lines.append("- tags(latest):")
    for tag, cnt in sorted(tag_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        md_lines.append(f"  - {tag}: {cnt}")
    md_lines.append("")
    md_lines.append("## 目录（group_id）")
    md_lines.extend([f"- {gid}" for gid in sorted(grouped.keys())])
    md_lines.append("")

    for gid in sorted(grouped.keys()):
        history = grouped[gid]
        latest = _pick_latest_need_review(history)
        mission = str(latest.get("mission") or "")
        stage_a_record = stage_a_index.get((mission, gid))
        md_lines.append(
            _render_group_markdown(
                group_id=gid,
                mission=mission,
                latest=latest,
                history=history,
                stage_a_record=stage_a_record,
            )
        )

    out_md.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    return out_md


def _discover_mission_dirs(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    dirs = []
    for child in run_dir.iterdir():
        if child.is_dir():
            dirs.append(child)
    return sorted(dirs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage-B YAML config path")
    args = parser.parse_args()

    config = load_stage_b_config(Path(args.config))
    run_dir = config.output.root / config.output.run_name
    mission_dirs = _discover_mission_dirs(run_dir)
    if not mission_dirs:
        print(f"[postprocess] No mission directories found under {run_dir}")
        return 0

    stage_a_index = _index_stage_a(config.stage_a_paths)
    wrote_any = False
    for mission_dir in mission_dirs:
        out_md = build_review_checklist_for_mission(
            mission_dir, stage_a_index=stage_a_index
        )
        if out_md is None:
            continue
        wrote_any = True
        print(f"[postprocess] Review checklist written: {out_md}")
        print(f"[postprocess] JSONL snapshot written: {mission_dir / 'review_checklist.jsonl'}")

    if not wrote_any:
        print(f"[postprocess] No need_review_queue.jsonl found under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
