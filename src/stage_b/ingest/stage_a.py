#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-B ingestion utilities for rule-search pipeline."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

from ..types import GroupLabel, GroupTicket, LabelProvenance, StageASummaries

logger = logging.getLogger(__name__)


_KEY_RE = re.compile(r"(\d+)$")
_DOMAIN_HEADER_RE = re.compile(r"^<DOMAIN=.*>\s*,\s*<TASK=.*>$")


def _is_rru_mission(mission: str) -> bool:
    # Mission names are Chinese strings like "RRU安装检查". Use a conservative prefix check.
    return mission.strip().startswith("RRU")


def _format_summary_json(obj: Mapping[str, object]) -> str:
    """Format summary JSON mapping to a single-line JSON string.

    Keep the contract's canonical top-level keys first; preserve other keys after.
    """
    ordered: dict[str, object] = {}
    for key in ("统计", "备注", "分组统计"):
        if key in obj:
            ordered[key] = obj[key]
    for key, value in obj.items():
        if key not in ordered:
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def _strip_domain_header_lines(text: str) -> str:
    """Drop Stage-A optional header lines like `<DOMAIN=...>, <TASK=...>`."""
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    kept = [line for line in lines if not _DOMAIN_HEADER_RE.match(line.strip())]
    return "\n".join(kept).strip()


def _strip_station_distance_from_bbu_summary(text: str) -> str:
    """For BBU tasks, remove '站点距离' evidence from Stage-A summaries if present.

    Rationale: station distance is only meaningful for RRU missions; keeping it in BBU prompts
    is noisy and can confuse the verdict model.

    This is intentionally best-effort:
    - If we can parse a summary JSON payload (with '统计'), we remove entries with 类别=站点距离.
    - If parsing fails, we conservatively drop lines that mention 站点距离 (keeping the rest).
    """
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("无关图片"):
        return "无关图片"

    cleaned = _strip_domain_header_lines(stripped)
    if not cleaned:
        return ""

    # Try parse JSON object directly.
    candidate = cleaned.strip()
    obj: object | None = None
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            obj = json.loads(candidate)
        except Exception:
            obj = None
    else:
        # Fallback: extract the first {...} block.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            block = candidate[start : end + 1].strip()
            if block.startswith("{") and block.endswith("}"):
                try:
                    obj = json.loads(block)
                except Exception:
                    obj = None

    if isinstance(obj, dict):
        obj_map = cast(dict[str, object], obj)
        stats_any = obj_map.get("统计")
        if isinstance(stats_any, list):
            stats = cast(list[object], stats_any)
            filtered_stats: list[object] = []
            for entry in stats:
                if isinstance(entry, dict):
                    entry_map = cast(dict[str, object], entry)
                    if entry_map.get("类别") == "站点距离":
                        continue
                    filtered_stats.append(entry_map)
                    continue
                filtered_stats.append(entry)
            obj_map["统计"] = filtered_stats

            # Also drop any direct "站点距离" key in 分组统计 if present (rare; defensive).
            grouped_any = obj_map.get("分组统计")
            if isinstance(grouped_any, dict) and "站点距离" in grouped_any:
                grouped = cast(dict[str, object], grouped_any)
                grouped.pop("站点距离", None)

            return _format_summary_json(obj_map).replace("站点距离", "")

    # Unparsable (often due to malformed JSON). Best-effort remove station-distance fragments
    # WITHOUT dropping the whole summary line (it may contain useful BBU evidence).

    def _drop_braced_object_containing(src: str, needle: str) -> str:
        """Remove the nearest {...} object that contains `needle` (best-effort, brace-matching)."""

        def _find_matching_brace(s: str, start_idx: int) -> int | None:
            depth = 0
            in_str = False
            escaped = False
            for i in range(start_idx, len(s)):
                ch = s[i]
                if in_str:
                    if escaped:
                        escaped = False
                        continue
                    if ch == "\\":
                        escaped = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                    continue
                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i
                    continue
            return None

        out = src
        while True:
            hit = out.find(needle)
            if hit == -1:
                return out
            start = out.rfind("{", 0, hit)
            if start == -1:
                return out
            end = _find_matching_brace(out, start)
            if end is None:
                # Truncated JSON: drop from start to end-of-string.
                end = len(out) - 1
            left = out[:start].rstrip()
            right = out[end + 1 :].lstrip()
            # Eat a single optional comma around the removed object.
            if right.startswith(","):
                right = right[1:].lstrip()
            elif left.endswith(","):
                left = left[:-1].rstrip()
            out = f"{left}{right}"

    def _drop_keyed_object(src: str, key: str) -> str:
        """Remove `"key": {...}` fragments (best-effort), keeping surrounding JSON-ish text."""

        out = src
        needle = f'"{key}"'
        while True:
            hit = out.find(needle)
            if hit == -1:
                return out
            # Find key start (including optional leading comma).
            start = hit
            # Remove a leading comma if present.
            j = start - 1
            while j >= 0 and out[j].isspace():
                j -= 1
            if j >= 0 and out[j] == ",":
                start = j
            # Find the colon, then the value start.
            colon = out.find(":", hit + len(needle))
            if colon == -1:
                return out
            k = colon + 1
            while k < len(out) and out[k].isspace():
                k += 1
            if k >= len(out):
                return out
            if out[k] == "{":
                # Brace-match value object.
                end = None
                depth = 0
                in_str = False
                escaped = False
                for i in range(k, len(out)):
                    ch = out[i]
                    if in_str:
                        if escaped:
                            escaped = False
                            continue
                        if ch == "\\":
                            escaped = True
                            continue
                        if ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == "{":
                        depth += 1
                        continue
                    if ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                        continue
                if end is None:
                    end = len(out) - 1
                left = out[:start].rstrip()
                right = out[end + 1 :].lstrip()
                if right.startswith(","):
                    right = right[1:].lstrip()
                out = f"{left}{right}"
                continue
            # Non-object value: drop until next comma.
            end = out.find(",", k)
            if end == -1:
                end = len(out)
            left = out[:start].rstrip()
            right = out[end:].lstrip()
            if right.startswith(","):
                right = right[1:].lstrip()
            out = f"{left}{right}"

    raw = cleaned
    # 1) Remove any summary entry object whose category is station distance.
    raw = _drop_braced_object_containing(raw, '"类别": "站点距离"')
    # 2) Remove station-distance key blocks in grouped stats (if present).
    raw = _drop_keyed_object(raw, "站点距离")
    return raw.replace("站点距离", "").strip()


def _normalize_key(raw_key: str, fallback_index: int) -> Tuple[str, int]:
    match = _KEY_RE.search(raw_key)
    if match:
        try:
            idx = int(match.group(1))
        except ValueError:
            idx = fallback_index
    else:
        idx = fallback_index
    return f"image_{idx}", idx


def _maybe_unwrap_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, str):
            return parsed
        # Preserve JSON objects for downstream prompting/parsing.
        # Stage-B prompt construction expects the Stage-A summary JSON payload
        # (containing keys like "统计") to remain intact.
        return stripped
    return text


def _normalize_per_image(per_image: Mapping[str, object]) -> Dict[str, str]:
    normalized: Dict[int, str] = {}
    for fallback_index, (raw_key, raw_value) in enumerate(per_image.items(), start=1):
        _, index = _normalize_key(str(raw_key), fallback_index)
        text = _maybe_unwrap_json(str(raw_value))
        normalized[index] = text
    ordered = {f"image_{idx}": normalized[idx] for idx in sorted(normalized)}
    return ordered


def _parse_label(raw_label: str) -> GroupLabel:
    lowered = raw_label.strip().lower()
    if lowered not in {"pass", "fail"}:
        raise ValueError(f"Unsupported label value: {raw_label!r}")
    return cast(GroupLabel, lowered)


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def ingest_stage_a(
    stage_a_paths: Sequence[str | Path],
) -> Sequence[GroupTicket]:
    """Load Stage-A JSONL outputs into ticket objects for Stage-B."""
    records: list[GroupTicket] = []

    for path_like in stage_a_paths:
        path = Path(path_like)
        if not path.exists():
            logger.warning(f"Skipping missing Stage-A file: {path}")
            continue

        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload: MutableMapping[str, object] = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.error(f"Invalid JSON at {path.name}:{line_number}: {exc}")
                    continue

                try:
                    mission = str(payload["mission"])
                    group_id = str(payload["group_id"])
                    label = _parse_label(str(payload["label"]))

                    per_image_raw = payload.get("per_image")
                    if not isinstance(per_image_raw, Mapping):
                        raise TypeError("per_image field must be an object")
                    per_image_raw = cast(Mapping[str, object], per_image_raw)
                    normalized = _normalize_per_image(per_image_raw)
                    if not normalized:
                        raise ValueError("per_image summaries must be non-empty")

                    # Stage-B postprocess: for BBU tasks, drop station-distance evidence if present.
                    # RRU missions rely on station distance for cross-image pairing; keep it intact there.
                    if not _is_rru_mission(mission):
                        normalized = {
                            k: _strip_station_distance_from_bbu_summary(v)
                            for k, v in normalized.items()
                        }

                    timestamp: Optional[datetime] = None
                    if isinstance(payload.get("label_timestamp"), str):
                        timestamp = _parse_timestamp(
                            str(payload.get("label_timestamp"))
                        )

                    provenance = LabelProvenance(
                        source=str(payload.get("label_source", "human")),
                        timestamp=timestamp,
                        metadata=None,
                    )

                    ticket = GroupTicket(
                        group_id=group_id,
                        mission=mission,
                        label=label,
                        summaries=StageASummaries(per_image=normalized),
                        provenance=provenance,
                        uid=f"{group_id}::{label}",
                    )
                    records.append(ticket)
                except Exception as exc:  # noqa: BLE001 - contextual logging
                    logger.error(
                        f"Failed to parse Stage-A record at {path.name}:{line_number} ({exc})"
                    )

    if not records:
        raise RuntimeError("No Stage-A records were ingested")

    return records


__all__ = ["ingest_stage_a"]
