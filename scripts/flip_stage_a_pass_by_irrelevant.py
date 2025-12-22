#!/usr/bin/env python3
"""Flip stage_a pass->fail when useful images < threshold after removing irrelevant ones.

Rule:
- Only consider records with label == "pass".
- Count images whose per_image text contains the irrelevant token (default: "无关图片").
- If irrelevant_count >= 1 and (total_images - irrelevant_count) < min_useful, mark as fail.

Also optionally move group folders from 审核通过 to 审核不通过.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flip stage_a pass->fail for orders with too few useful images.")
    parser.add_argument(
        "--stage-a",
        required=True,
        help="Path to *_stage_a.jsonl",
    )
    parser.add_argument(
        "--group-root",
        default="group_data/bbu_scene_2.0_order",
        help="Root directory that contains mission folders.")
    parser.add_argument(
        "--pass-dir",
        default="审核通过",
        help="Pass directory name under each mission folder.")
    parser.add_argument(
        "--fail-dir",
        default="审核不通过",
        help="Fail directory name under each mission folder.")
    parser.add_argument(
        "--irrelevant-token",
        default="无关图片",
        help="Substring that marks an image as irrelevant.")
    parser.add_argument(
        "--min-useful",
        type=int,
        default=3,
        help="Minimum number of useful images required to keep pass.")
    parser.add_argument(
        "--allow-no-irrelevant",
        action="store_true",
        help=(
            "Also flip when useful < min_useful even if irrelevant_count == 0. "
            "Default: require at least one irrelevant image."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply filesystem moves and write in-place if --in-place is set.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSONL (requires --apply).")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When using --in-place, also save a .bak copy of the original.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (ignored if --in-place).")
    return parser.parse_args()


def _is_irrelevant(text: str, token: str) -> bool:
    return token in (text or "")


def _count_irrelevant(per_image: Dict[str, str], token: str) -> int:
    return sum(1 for v in per_image.values() if _is_irrelevant(v, token))


def _should_flip(
    label: str,
    total_images: int,
    irrelevant_count: int,
    min_useful: int,
    allow_no_irrelevant: bool,
) -> bool:
    if label != "pass":
        return False
    if not allow_no_irrelevant and irrelevant_count == 0:
        return False
    useful = total_images - irrelevant_count
    return useful < min_useful


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def _move_group_dir(
    group_root: Path,
    mission: str,
    group_id: str,
    pass_dir: str,
    fail_dir: str,
    apply: bool,
) -> Tuple[bool, str]:
    mission_dir = group_root / mission
    src = mission_dir / pass_dir / group_id
    dst = mission_dir / fail_dir / group_id

    if not src.exists():
        return False, f"missing: {src}"
    if dst.exists():
        return False, f"exists: {dst}"

    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True, f"moved: {src} -> {dst}"
    return True, f"dry-run: {src} -> {dst}"


def main() -> None:
    args = parse_args()
    stage_a_path = Path(args.stage_a)
    group_root = Path(args.group_root)

    if args.in_place and not args.apply:
        raise SystemExit("--in-place requires --apply")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = stage_a_path.with_name(stage_a_path.stem + ".flipped" + stage_a_path.suffix)

    flipped_records: List[dict] = []
    move_logs: List[str] = []
    flip_count = 0

    for rec in _load_jsonl(stage_a_path):
        per_image = dict(rec.get("per_image") or {})
        images = list(rec.get("images") or [])
        mission = rec.get("mission")
        group_id = rec.get("group_id")
        label = rec.get("label")

        irrelevant_count = _count_irrelevant(per_image, args.irrelevant_token)
        total_images = len(images)

        if _should_flip(
            label=label,
            total_images=total_images,
            irrelevant_count=irrelevant_count,
            min_useful=args.min_useful,
            allow_no_irrelevant=args.allow_no_irrelevant,
        ):
            rec["label"] = "fail"
            flip_count += 1
            if mission and group_id:
                ok, msg = _move_group_dir(
                    group_root=group_root,
                    mission=mission,
                    group_id=group_id,
                    pass_dir=args.pass_dir,
                    fail_dir=args.fail_dir,
                    apply=args.apply,
                )
                if ok or msg:
                    move_logs.append(msg)
        flipped_records.append(rec)

    if args.in_place:
        if args.backup:
            backup_path = stage_a_path.with_suffix(stage_a_path.suffix + ".bak")
            shutil.copy2(stage_a_path, backup_path)
        _write_jsonl(stage_a_path, flipped_records)
    else:
        _write_jsonl(output_path, flipped_records)

    print(f"input: {stage_a_path}")
    if args.in_place:
        print("output: (in-place)")
    else:
        print(f"output: {output_path}")
    print(f"flipped_pass_to_fail: {flip_count}")
    if move_logs:
        print("moves:")
        for msg in move_logs:
            print(f"- {msg}")


if __name__ == "__main__":
    main()
