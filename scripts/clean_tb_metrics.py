#!/usr/bin/env python3
"""Clean TensorBoard event files by removing token_count and entropy metrics.

This script reads TensorBoard event files, filters out metrics containing
'_token_count' or '_entropy' in their tag names, and writes cleaned events
back to the same files (with backup).
"""

import argparse
import shutil
import struct
from pathlib import Path
from typing import List, Set, Tuple

try:
    from tensorboard.backend.event_processing import event_file_loader
    from tensorboard.compat.proto import event_pb2
except ImportError:
    try:
        import tensorflow as tf

        event_pb2 = tf.compat.v1
        event_file_loader = None
    except ImportError:
        raise ImportError(
            "Neither tensorboard nor tensorflow is available. "
            "Please install one of them."
        )


def should_keep_metric(tag: str, exclude_patterns: Set[str]) -> bool:
    """Check if a metric tag should be kept."""
    tag_lower = tag.lower()
    for pattern in exclude_patterns:
        if pattern in tag_lower:
            return False
    return True


def clean_event_file(
    event_file_path: Path,
    exclude_patterns: Set[str] = None,
    backup: bool = True,
) -> int:
    """Clean a single TensorBoard event file.

    Args:
        event_file_path: Path to the event file
        exclude_patterns: Set of patterns to exclude (default: ['_token_count', '_entropy'])
        backup: Whether to create a backup before modifying

    Returns:
        Number of events removed
    """
    if exclude_patterns is None:
        exclude_patterns = {"entropy", "token_count", "_count"}

    if not event_file_path.exists():
        print(f"Warning: {event_file_path} does not exist, skipping")
        return 0

    # Create backup
    if backup:
        backup_path = event_file_path.with_suffix(event_file_path.suffix + ".bak")
        if backup_path.exists():
            print(f"Backup already exists: {backup_path}, skipping backup")
        else:
            shutil.copy2(event_file_path, backup_path)
            print(f"Created backup: {backup_path}")

    # Read all events - we'll store both the raw record and parsed event
    events_to_keep: List[Tuple[bytes, event_pb2.Event]] = []
    events_removed = 0

    try:
        # Read raw file format to preserve exact structure
        with open(event_file_path, "rb") as f:
            while True:
                # Read length header (8 bytes) and its CRC32 (4 bytes)
                header_with_crc = f.read(12)
                if len(header_with_crc) < 12:
                    break

                event_len = struct.unpack("<Q", header_with_crc[:8])[0]
                if event_len == 0:
                    break

                # Read event data
                event_data = f.read(event_len)
                if len(event_data) < event_len:
                    break

                # Read CRC32 for event data
                event_crc = f.read(4)
                if len(event_crc) < 4:
                    break

                # Parse event to check if we should keep it
                event = event_pb2.Event()
                try:
                    event.ParseFromString(event_data)
                except Exception:
                    # Keep corrupted events as-is (might be important)
                    events_to_keep.append(
                        (header_with_crc + event_data + event_crc, None)
                    )
                    continue

                # Always keep non-summary events (file_version, graph_def, etc.)
                if not event.summary:
                    events_to_keep.append(
                        (header_with_crc + event_data + event_crc, event)
                    )
                    continue

                # Check if this is a scalar summary with unwanted metrics
                should_keep = True
                for value in event.summary.value:
                    if value.tag:
                        if not should_keep_metric(value.tag, exclude_patterns):
                            should_keep = False
                            events_removed += 1
                            break

                if should_keep:
                    events_to_keep.append(
                        (header_with_crc + event_data + event_crc, event)
                    )

    except Exception as e:
        print(f"Error reading {event_file_path}: {e}")
        import traceback

        traceback.print_exc()
        if backup:
            print(f"Restoring from backup: {backup_path}")
            shutil.copy2(backup_path, event_file_path)
        raise

    # Write cleaned events back - preserve original format
    try:
        with open(event_file_path, "wb") as f:
            for record, _ in events_to_keep:
                f.write(record)
    except Exception as e:
        print(f"Error writing {event_file_path}: {e}")
        import traceback

        traceback.print_exc()
        if backup:
            print(f"Restoring from backup: {backup_path}")
            shutil.copy2(backup_path, event_file_path)
        raise

    return events_removed


def clean_directory(
    directory: Path,
    exclude_patterns: Set[str] = None,
    backup: bool = True,
) -> dict:
    """Clean all event files in a directory.

    Returns:
        Dictionary mapping file paths to number of events removed
    """
    results = {}
    # Find event files but exclude backup files
    all_event_files = list(directory.rglob("events.out.tfevents.*"))
    event_files = [f for f in all_event_files if not str(f).endswith(".bak")]

    if not event_files:
        print(f"No event files found in {directory}")
        return results

    print(f"Found {len(event_files)} event file(s) in {directory}")

    for event_file in event_files:
        print(f"\nProcessing: {event_file}")
        try:
            removed = clean_event_file(event_file, exclude_patterns, backup)
            results[str(event_file)] = removed
            print(f"  Removed {removed} event(s) with unwanted metrics")
        except Exception as e:
            print(f"  Error: {e}")
            results[str(event_file)] = -1

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Clean TensorBoard event files by removing token_count and entropy metrics"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories containing TensorBoard event files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["entropy", "token_count", "_count"],
        help="Patterns to exclude (default: entropy token_count _count)",
    )

    args = parser.parse_args()

    exclude_patterns = set(args.exclude)
    backup = not args.no_backup

    all_results = {}
    for dir_path in args.directories:
        directory = Path(dir_path)
        if not directory.exists():
            print(f"Warning: {directory} does not exist, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Cleaning directory: {directory}")
        print(f"{'=' * 60}")
        results = clean_directory(directory, exclude_patterns, backup)
        all_results.update(results)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    total_removed = sum(v for v in all_results.values() if v > 0)
    total_files = len([v for v in all_results.values() if v >= 0])
    errors = len([v for v in all_results.values() if v < 0])

    print(f"Total files processed: {total_files}")
    print(f"Total events removed: {total_removed}")
    if errors > 0:
        print(f"Files with errors: {errors}")


if __name__ == "__main__":
    main()
