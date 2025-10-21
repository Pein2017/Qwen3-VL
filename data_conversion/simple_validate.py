#!/usr/bin/env python3
"""
Simple JSONL Validation for Clean Semantic Format

Validates basic structure of clean semantic JSONL files:
- Valid JSON format
- Required fields: images, objects
- Proper object structure: bbox_2d, desc
"""

import argparse
import json
import sys
from pathlib import Path


# Configure UTF-8 encoding for stdout/stderr if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

try:
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass


def validate_sample(sample: dict, line_num: int) -> bool:
    """Validate a single sample structure."""
    # Check required fields
    if "images" not in sample:
        print(f"   ❌ Line {line_num}: Missing 'images' field")
        return False

    if "objects" not in sample:
        print(f"   ❌ Line {line_num}: Missing 'objects' field")
        return False

    # Validate images
    images = sample["images"]
    if not isinstance(images, list) or len(images) == 0:
        print(f"   ❌ Line {line_num}: 'images' must be non-empty list")
        return False

    # Validate objects
    objects = sample["objects"]
    if not isinstance(objects, list):
        print(f"   ❌ Line {line_num}: 'objects' must be a list")
        return False

    # Validate each object
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            print(f"   ❌ Line {line_num}: Object {i} must be a dict")
            return False

        if "bbox_2d" not in obj or "desc" not in obj:
            print(f"   ❌ Line {line_num}: Object {i} missing 'bbox_2d' or 'desc'")
            return False

        # Validate bbox_2d format
        bbox_2d = obj["bbox_2d"]
        if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
            print(f"   ❌ Line {line_num}: Object {i} 'bbox_2d' must be [x1,y1,x2,y2]")
            return False

        # Validate description
        desc = obj["desc"]
        if not isinstance(desc, str) or not desc.strip():
            print(f"   ❌ Line {line_num}: Object {i} 'desc' must be non-empty string")
            return False

    return True


def validate_jsonl_file(filepath: str) -> bool:
    """Validate a JSONL file."""
    try:
        valid_count = 0
        total_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                total_count += 1

                try:
                    sample = json.loads(line)
                    if validate_sample(sample, line_num):
                        valid_count += 1
                except json.JSONDecodeError as e:
                    print(f"   ❌ Line {line_num}: Invalid JSON - {e}")

        print(f"   ✅ {filepath}: {valid_count}/{total_count} valid samples")
        return valid_count == total_count

    except Exception as e:
        print(f"   ❌ {filepath}: Error reading file - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate clean semantic JSONL files")
    parser.add_argument("files", nargs="+", help="JSONL files to validate")

    args = parser.parse_args()

    all_valid = True

    print("🔍 Validating clean semantic JSONL files...")
    print("=" * 50)

    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"   ❌ File not found: {filepath}")
            all_valid = False
            continue

        print(f"Validating {filepath}...")
        if not validate_jsonl_file(filepath):
            all_valid = False

    print("=" * 50)
    if all_valid:
        print("✅ All files are valid!")
        sys.exit(0)
    else:
        print("❌ Some files have validation errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
