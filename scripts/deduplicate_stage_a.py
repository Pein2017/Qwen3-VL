#!/usr/bin/env python3
"""
Deduplicate items within per_image fields in Stage-A JSONL output files.

This script processes JSONL files from stage_a.sh and removes duplicate items
within each per_image[image_i] string, aggregating their counts.

Example:
    python scripts/deduplicate_stage_a.py output_post/stage_a/挡风板安装检查_stage_a.jsonl
    python scripts/deduplicate_stage_a.py output_post/stage_a/*.jsonl --in-place
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def parse_item(item_str: str) -> Tuple[str, int]:
    """
    Parse an item string to extract the base item and count.

    Args:
        item_str: Item string like "标签/无法识别×3" or "螺丝、光纤插头/BBU安装螺丝,符合要求×2"

    Returns:
        Tuple of (base_item, count). If no count found, count defaults to 1.
    """
    item_str = item_str.strip()
    if not item_str:
        return "", 0

    # Match pattern: ...×N at the end
    match = re.search(r"×(\d+)$", item_str)
    if match:
        count = int(match.group(1))
        base_item = item_str[: match.start()].rstrip()
    else:
        # No count found, default to 1
        count = 1
        base_item = item_str

    return base_item, count


def deduplicate_image_string(image_str: str) -> Tuple[str, Dict[str, int]]:
    """
    Deduplicate items in an image string and return the cleaned string and statistics.

    Args:
        image_str: String containing items separated by "，"

    Returns:
        Tuple of (deduplicated_string, stats_dict) where stats_dict contains
        counts of duplicates found per item type.
    """
    if not image_str or not image_str.strip():
        return image_str, {}

    # Split by "，" (Chinese comma)
    items = [item.strip() for item in image_str.split("，") if item.strip()]

    # Aggregate items by base (without count)
    item_counts: Dict[str, int] = defaultdict(int)
    stats: Dict[str, int] = defaultdict(int)

    for item in items:
        base_item, count = parse_item(item)
        if base_item:
            if item_counts[base_item] > 0:
                # This is a duplicate
                stats[base_item] += 1
            item_counts[base_item] += count

    # Reconstruct the string
    deduplicated_items = []
    for base_item, total_count in sorted(item_counts.items()):
        if total_count > 0:
            deduplicated_items.append(f"{base_item}×{total_count}")

    deduplicated_str = "，".join(deduplicated_items)

    return deduplicated_str, dict(stats)


def process_jsonl_file(
    input_path: Path, in_place: bool = False, verbose: bool = False
) -> Dict:
    """
    Process a JSONL file and deduplicate per_image fields.

    Returns:
        Dictionary with statistics about the processing.
    """
    stats = {
        "total_groups": 0,
        "total_images": 0,
        "images_with_duplicates": 0,
        "total_duplicate_items": 0,
        "duplicate_items_by_type": defaultdict(int),
        "groups_modified": 0,
    }

    output_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                output_lines.append(line)
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Failed to parse line {line_num} in {input_path}: {e}",
                    file=sys.stderr,
                )
                output_lines.append(line)
                continue

            stats["total_groups"] += 1
            group_modified = False

            if "per_image" in data and isinstance(data["per_image"], dict):
                for image_key, image_str in data["per_image"].items():
                    stats["total_images"] += 1

                    if isinstance(image_str, str):
                        deduplicated_str, item_stats = deduplicate_image_string(
                            image_str
                        )

                        if deduplicated_str != image_str:
                            # Duplicates were found and removed
                            stats["images_with_duplicates"] += 1
                            group_modified = True

                            # Aggregate statistics
                            for item_type, dup_count in item_stats.items():
                                stats["duplicate_items_by_type"][item_type] += dup_count
                                stats["total_duplicate_items"] += dup_count

                            if verbose:
                                print(
                                    f"  {image_key}: {len(item_stats)} duplicate types found",
                                    file=sys.stderr,
                                )

                            data["per_image"][image_key] = deduplicated_str

                if group_modified:
                    stats["groups_modified"] += 1

            output_lines.append(json.dumps(data, ensure_ascii=False))

    # Write output
    output_path = (
        input_path if in_place else input_path.with_suffix(".deduplicated.jsonl")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    if not in_place:
        print(f"Output written to: {output_path}", file=sys.stderr)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate items in Stage-A JSONL output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSONL file(s) to process (supports glob patterns)",
    )
    parser.add_argument(
        "--in-place",
        "-i",
        action="store_true",
        help="Modify files in place (default: create .deduplicated.jsonl files)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary statistics, do not write output files",
    )

    args = parser.parse_args()

    # Expand glob patterns
    all_files = []
    for pattern in args.files:
        if "*" in str(pattern) or "?" in str(pattern):
            all_files.extend(Path(".").glob(str(pattern)))
        else:
            all_files.append(pattern)

    # Remove duplicates and sort
    all_files = sorted(set(all_files))

    if not all_files:
        print("Error: No files found to process", file=sys.stderr)
        sys.exit(1)

    # Process each file
    total_stats = {
        "files_processed": 0,
        "total_groups": 0,
        "total_images": 0,
        "images_with_duplicates": 0,
        "total_duplicate_items": 0,
        "groups_modified": 0,
        "duplicate_items_by_type": defaultdict(int),
    }

    for file_path in all_files:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        if not file_path.is_file():
            print(f"Warning: Not a file: {file_path}", file=sys.stderr)
            continue

        print(f"Processing: {file_path}", file=sys.stderr)

        if not args.summary_only:
            stats = process_jsonl_file(
                file_path, in_place=args.in_place, verbose=args.verbose
            )
        else:
            # For summary-only, we still need to process to get stats
            # but we'll use a temp output
            stats = process_jsonl_file(file_path, in_place=False, verbose=args.verbose)
            # Remove the temp file
            temp_output = file_path.with_suffix(".deduplicated.jsonl")
            if temp_output.exists():
                temp_output.unlink()

        # Aggregate statistics
        total_stats["files_processed"] += 1
        total_stats["total_groups"] += stats["total_groups"]
        total_stats["total_images"] += stats["total_images"]
        total_stats["images_with_duplicates"] += stats["images_with_duplicates"]
        total_stats["total_duplicate_items"] += stats["total_duplicate_items"]
        total_stats["groups_modified"] += stats["groups_modified"]

        for item_type, count in stats["duplicate_items_by_type"].items():
            total_stats["duplicate_items_by_type"][item_type] += count

    # Print summary statistics
    print("\n" + "=" * 60, file=sys.stderr)
    print("DEDUPLICATION SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Files processed:        {total_stats['files_processed']}", file=sys.stderr)
    print(f"Total groups:           {total_stats['total_groups']}", file=sys.stderr)
    print(f"Total images:           {total_stats['total_images']}", file=sys.stderr)
    print(f"Groups modified:        {total_stats['groups_modified']}", file=sys.stderr)
    print(
        f"Images with duplicates: {total_stats['images_with_duplicates']}",
        file=sys.stderr,
    )
    print(
        f"Total duplicate items:  {total_stats['total_duplicate_items']}",
        file=sys.stderr,
    )

    if total_stats["duplicate_items_by_type"]:
        print("\nTop duplicate item types:", file=sys.stderr)
        sorted_items = sorted(
            total_stats["duplicate_items_by_type"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for item_type, count in sorted_items[:20]:  # Top 20
            print(f"  {item_type}: {count} duplicates", file=sys.stderr)
        if len(sorted_items) > 20:
            print(f"  ... and {len(sorted_items) - 20} more types", file=sys.stderr)

    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
