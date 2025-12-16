#!/usr/bin/env python3
"""
Validate dense-caption JSONL records against `docs/data/DATA_JSONL_CONTRACT.md`.

This is intentionally model-free and fast: it catches schema/geometry errors
before long training runs.

Usage:
  conda run -n ms python scripts/validate_dense_jsonl_contract.py --jsonl demo/data/train_tiny.jsonl
  conda run -n ms python scripts/validate_dense_jsonl_contract.py --jsonl data/bbu_full_768_poly/train.jsonl --limit 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value  # not NaN


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_seq(value: Any, field: str) -> Sequence[Any]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{field} must be a list")
    return value  # type: ignore[return-value]


def _as_map(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value  # type: ignore[return-value]


def _validate_geometry(obj: Mapping[str, Any], *, record_idx: int, obj_idx: int) -> None:
    geom_keys = [key for key in ("bbox_2d", "poly", "line") if key in obj]
    _require(
        len(geom_keys) == 1,
        f"record[{record_idx}].objects[{obj_idx}] must contain exactly one geometry key among bbox_2d/poly/line",
    )
    key = geom_keys[0]
    seq = _as_seq(obj.get(key), f"record[{record_idx}].objects[{obj_idx}].{key}")
    _require(len(seq) > 0, f"record[{record_idx}].objects[{obj_idx}].{key} must be non-empty")
    _require(all(_is_number(v) for v in seq), f"record[{record_idx}].objects[{obj_idx}].{key} must be numeric")

    if key == "bbox_2d":
        _require(len(seq) == 4, f"record[{record_idx}].objects[{obj_idx}].bbox_2d must have 4 values")
    else:
        _require(len(seq) % 2 == 0, f"record[{record_idx}].objects[{obj_idx}].{key} must have even length")
        if key == "poly":
            _require(len(seq) >= 6, f"record[{record_idx}].objects[{obj_idx}].poly must have >= 6 values")
            poly_points = obj.get("poly_points")
            if poly_points is not None:
                _require(int(poly_points) == len(seq) // 2, f"record[{record_idx}].objects[{obj_idx}].poly_points must equal len(poly)/2")
        if key == "line":
            _require(len(seq) >= 4, f"record[{record_idx}].objects[{obj_idx}].line must have >= 4 values")
            line_points = obj.get("line_points")
            if line_points is not None:
                _require(int(line_points) == len(seq) // 2, f"record[{record_idx}].objects[{obj_idx}].line_points must equal len(line)/2")

    # Reject legacy / forbidden keys that cause subtle bugs.
    _require("groups" not in obj, f"record[{record_idx}].objects[{obj_idx}] must not include legacy 'groups' key")


def validate_record(record: Mapping[str, Any], *, record_idx: int) -> None:
    images = _as_seq(record.get("images"), f"record[{record_idx}].images")
    _require(len(images) > 0, f"record[{record_idx}].images must be non-empty")
    for img_idx, img in enumerate(images):
        _require(isinstance(img, str) and img.strip(), f"record[{record_idx}].images[{img_idx}] must be a non-empty string")

    objects = _as_seq(record.get("objects"), f"record[{record_idx}].objects")
    _require(len(objects) > 0, f"record[{record_idx}].objects must be non-empty")
    for obj_idx, obj in enumerate(objects):
        obj_map = _as_map(obj, f"record[{record_idx}].objects[{obj_idx}]")
        desc = obj_map.get("desc")
        _require(isinstance(desc, str) and desc.strip(), f"record[{record_idx}].objects[{obj_idx}].desc must be a non-empty string")
        _validate_geometry(obj_map, record_idx=record_idx, obj_idx=obj_idx)

    width = record.get("width")
    height = record.get("height")
    _require(isinstance(width, int) and width > 0, f"record[{record_idx}].width must be a positive int")
    _require(isinstance(height, int) and height > 0, f"record[{record_idx}].height must be a positive int")

    summary = record.get("summary")
    if summary is not None:
        _require(isinstance(summary, str) and summary.strip(), f"record[{record_idx}].summary must be a non-empty string when present")


def load_jsonl(path: Path, *, limit: int) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if limit and idx > limit:
                return
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            _require(isinstance(parsed, Mapping), f"line {idx} must be a JSON object")
            yield parsed  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dense-caption JSONL contract")
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to JSONL file")
    parser.add_argument("--limit", type=int, default=100, help="Max records to validate (0 = all)")
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise FileNotFoundError(f"JSONL not found: {args.jsonl}")

    count = 0
    for idx, record in enumerate(load_jsonl(args.jsonl, limit=args.limit), start=0):
        validate_record(record, record_idx=idx)
        count += 1
    if count == 0:
        raise RuntimeError("No records validated (empty file?)")
    print(f"[OK] Validated {count} record(s): {args.jsonl}")


if __name__ == "__main__":
    main()

