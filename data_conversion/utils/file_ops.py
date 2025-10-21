#!/usr/bin/env python3
"""
File Operations Utilities

Handles all file operations including path management, JSON loading,
image discovery, and directory operations.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


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

logger = logging.getLogger(__name__)


class FileOperations:
    """Centralized file operations for the data conversion pipeline."""

    @staticmethod
    def find_json_files(directory: Path) -> List[Path]:
        """Find all JSON files in a directory."""
        json_files = list(directory.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory}")

        logger.info(f"Found {len(json_files)} JSON files in {directory}")
        return sorted(json_files)

    @staticmethod
    def find_image_file(json_path: Path) -> Path:
        """Find corresponding image file for a JSON file."""
        for ext in [".jpeg", ".jpg"]:
            image_path = json_path.with_suffix(ext)
            if image_path.is_file():
                return image_path

        raise FileNotFoundError(f"No image file found for {json_path}")

    @staticmethod
    def load_json_data(json_path: Path) -> Dict:
        """Load and validate JSON data structure."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Basic structure validation
            info = data.get("info", {})
            if "height" not in info or "width" not in info:
                raise ValueError(f"Missing height/width in info section: {json_path}")

            # Check for annotation data
            has_data_list = "dataList" in data and isinstance(data["dataList"], list)
            has_mark_result = "markResult" in data and isinstance(
                data.get("markResult", {}).get("features"), list
            )

            if not (has_data_list or has_mark_result):
                raise ValueError(f"No valid annotation data found: {json_path}")

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            raise

    @staticmethod
    def save_json_data(
        data: Dict, json_path: Path, indent: Optional[int] = None
    ) -> None:
        """Save JSON data to file."""
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        logger.debug(f"Saved JSON to {json_path}")

    @staticmethod
    def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions (width, height)."""
        try:
            with Image.open(image_path) as img:
                if img is None:
                    raise ValueError(f"Failed to open image: {image_path}")
                return img.size
        except Exception as e:
            logger.error(f"Error getting dimensions for {image_path}: {e}")
            raise

    @staticmethod
    def validate_image_dimensions(
        image_path: Path, expected_width: int, expected_height: int
    ) -> Tuple[int, int]:
        """Validate image dimensions match expected values."""
        actual_width, actual_height = FileOperations.get_image_dimensions(image_path)

        if expected_width != actual_width or expected_height != actual_height:
            raise ValueError(
                f"Dimension mismatch for {image_path.name}: "
                f"Expected {expected_width}x{expected_height} "
                f"but got {actual_width}x{actual_height}"
            )

        return actual_width, actual_height

    @staticmethod
    def copy_file(src: Path, dst: Path, preserve_structure: bool = True) -> None:
        """Copy file with optional structure preservation."""
        if preserve_structure:
            dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists():
            shutil.copy2(src, dst)
            logger.debug(f"Copied {src.name} to {dst}")

    @staticmethod
    def write_jsonl(samples: List[Dict], output_path: Path) -> None:
        """Write samples to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Written {len(samples)} samples to {output_path}")

    @staticmethod
    def load_label_hierarchy(hierarchy_path: Path) -> Dict[str, List[str]]:
        """Load label hierarchy from JSON file."""
        try:
            with open(hierarchy_path, "r", encoding="utf-8") as f:
                hierarchy = json.load(f)

            # Validate structure
            if not isinstance(hierarchy, dict):
                raise ValueError(f"Invalid hierarchy format in {hierarchy_path}")

            # Ensure all values are lists
            for key, value in hierarchy.items():
                if not isinstance(value, list):
                    hierarchy[key] = []

            logger.info(f"Loaded label hierarchy with {len(hierarchy)} categories")
            return hierarchy

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {hierarchy_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading hierarchy {hierarchy_path}: {e}")
            raise

    @staticmethod
    def load_token_map(token_map_path: Path) -> Dict[str, str]:
        """Load token mapping from JSON file."""
        try:
            with open(token_map_path, "r", encoding="utf-8") as f:
                token_map = json.load(f)

            # Validate structure
            if not isinstance(token_map, dict):
                raise ValueError(f"Invalid token map format in {token_map_path}")

            logger.info(f"Loaded token map with {len(token_map)} mappings")
            return token_map

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {token_map_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading token map {token_map_path}: {e}")
            raise

    @staticmethod
    def clean_directory(
        directory: Path, keep_patterns: Optional[List[str]] = None
    ) -> None:
        """Clean directory keeping only specified patterns."""
        if not directory.exists():
            return

        keep_patterns = keep_patterns or []

        for item in directory.iterdir():
            should_keep = any(item.match(pattern) for pattern in keep_patterns)
            if not should_keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                logger.debug(f"Removed {item}")

    @staticmethod
    def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Path:
        """Create backup of a file."""
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        return backup_path

    @staticmethod
    def calculate_relative_path(file_path: Path, base_dir: Path) -> str:
        """Calculate relative path from base directory."""
        try:
            return str(file_path.relative_to(base_dir))
        except ValueError:
            # If relative path calculation fails, return absolute path
            return str(file_path)
