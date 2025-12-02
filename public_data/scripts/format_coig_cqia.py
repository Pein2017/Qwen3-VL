#!/usr/bin/env python3
"""
Normalize COIG-CQIA chat JSONL for fusion training.

Fixes:
- Ensure metadata is a dict and injects a `source` tag.
- Strips message-level `loss` fields used upstream but not needed for SFT.
- Leaves all other fields intact.

Usage:
  python public_data/scripts/format_coig_cqia.py \
    --input public_data/coig_cqia/coig_cqia_merged.jsonl \
    --output public_data/coig_cqia/coig_cqia_merged_formatted.jsonl

Pass --inplace to rewrite the input file.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence


def _clean_messages(messages: Any) -> List[Mapping[str, Any]] | None:
    if not isinstance(messages, Sequence):
        return None

    cleaned: List[Mapping[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        m: MutableMapping[str, Any] = dict(msg)
        m.pop("loss", None)  # remove unused loss marker

        content = m.get("content")
        # If content is a list of strings, join; otherwise keep as-is
        if isinstance(content, list) and all(isinstance(c, str) for c in content):
            m["content"] = "".join(content)
        cleaned.append(m)
    return cleaned if cleaned else None


def _normalize_record(record: Mapping[str, Any], *, source_name: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = dict(record)

    # Metadata normalization
    meta = rec.get("metadata")
    if not isinstance(meta, Mapping):
        meta = {}
    meta = dict(meta)
    meta.setdefault("source", source_name)
    rec["metadata"] = meta

    # Messages cleanup
    messages = _clean_messages(rec.get("messages"))
    if messages is not None:
        rec["messages"] = messages

    return rec


def format_file(input_path: Path, output_path: Path, *, source_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=str(output_path.parent)
    ) as tmp:
        tmp_path = Path(tmp.name)
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec = _normalize_record(rec, source_name=source_name)
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format COIG-CQIA JSONL for chat fusion training."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path (default: <input> rewritten when --inplace is set)",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="coig_cqia",
        help="Value to inject into metadata.source (default: coig_cqia)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite the input file in place (mutually exclusive with --output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inplace and args.output is not None:
        raise SystemExit("Use either --inplace or --output, not both.")
    output_path = args.output or args.input
    format_file(args.input, output_path, source_name=args.source_name)
    print(f"Formatted {args.input} -> {output_path}")


if __name__ == "__main__":
    main()
