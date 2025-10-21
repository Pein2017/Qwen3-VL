#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from data_conversion.summary_builder import build_summary_from_objects


def read_jsonl(path: Path):
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				yield json.loads(line)
			except json.JSONDecodeError as e:
				print(f"JSON decode error: {e}", file=sys.stderr)
				continue


def main() -> int:
	parser = argparse.ArgumentParser(description="Test summary generation from JSONL samples")
	parser.add_argument(
		"--jsonl",
		type=str,
		default="data/bbu_full_768/val.jsonl",
		help="Path to JSONL file (default: data/bbu_full_768/val.jsonl)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=0,
		help="Limit number of samples to process (0 = no limit)",
	)
	parser.add_argument(
		"--only-mismatch",
		action="store_true",
		help="Only print samples where generated summary != saved summary (when saved present)",
	)
	parser.add_argument(
		"--no-compare",
		action="store_true",
		help="Do not compare with saved 'summary' in samples; just print generated",
	)
	args = parser.parse_args()

	jsonl_path = Path(args.jsonl)
	if not jsonl_path.exists():
		print(f"❌ File not found: {jsonl_path}", file=sys.stderr)
		return 1

	processed = 0
	mismatches = 0
	for idx, sample in enumerate(read_jsonl(jsonl_path), start=1):
		images: List[str] = sample.get("images", []) or []
		image_id = images[0] if images else f"sample_{idx}"
		objects: List[Dict[str, Any]] = sample.get("objects", []) or []
		if not objects:
			continue

		generated = build_summary_from_objects(objects)
		saved = sample.get("summary")

		if args.no_compare or saved is None:
			print(f"[{idx}] {image_id}\n  summary:   {generated}")
		else:
			match = (generated == saved)
			if args.only_mismatch and match:
				pass
			else:
				print(
					f"[{idx}] {image_id}\n  generated: {generated}\n  saved:     {saved}\n  match:     {match}"
				)
				if not match:
					mismatches += 1

		processed += 1
		if args.limit and processed >= args.limit:
			break

	print(f"\n✅ Done. Processed: {processed}, mismatches: {mismatches}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
