"""
Base converter interface and configuration for dataset conversions.

All converters must implement BaseConverter to ensure consistent API.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConversionConfig:
    """
    Immutable configuration for dataset conversion.

    All paths must be absolute. No silent defaults for core parameters.
    """

    # Required fields
    input_path: str  # Path to input annotations (e.g., COCO JSON)
    output_path: str  # Path to output JSONL file
    image_root: str  # Root directory containing images

    # Optional fields
    split: str = "train"  # Dataset split name
    max_samples: Optional[int] = None  # Limit number of samples (for testing)
    min_box_area: float = 1.0  # Minimum bbox area in pixels
    min_box_dimension: float = 1.0  # Minimum bbox width/height
    clip_boxes: bool = True  # Clip boxes to image boundaries
    skip_crowd: bool = True  # Skip crowd annotations (iscrowd=1)
    relative_image_paths: bool = True  # Make image paths relative to output JSONL

    def __post_init__(self):
        """Validate configuration at construction time."""
        # Validate required paths are absolute
        for field_name in ["input_path", "output_path", "image_root"]:
            path = getattr(self, field_name)
            if not os.path.isabs(path):
                raise ValueError(f"{field_name} must be absolute path, got: {path}")

        # Validate input exists
        if not os.path.exists(self.input_path):
            raise ValueError(f"input_path does not exist: {self.input_path}")

        if not os.path.exists(self.image_root):
            raise ValueError(f"image_root does not exist: {self.image_root}")

        # Validate numeric constraints
        if self.min_box_area < 0:
            raise ValueError(f"min_box_area must be >= 0, got {self.min_box_area}")

        if self.min_box_dimension < 0:
            raise ValueError(
                f"min_box_dimension must be >= 0, got {self.min_box_dimension}"
            )

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be > 0, got {self.max_samples}")


class BaseConverter(ABC):
    """
    Abstract base class for dataset converters.

    Converts detection datasets to Qwen3-VL JSONL format:
    {
      "images": ["path/to/img.jpg"],
      "objects": [
        {"bbox_2d": [x1, y1, x2, y2], "desc": "category_name"}
      ],
      "width": 640,
      "height": 480,
      "summary": "optional description"
    }

    Subclasses must implement:
    - load_annotations(): Load dataset-specific annotation format
    - convert_sample(): Convert one sample to JSONL format
    """

    def __init__(self, config: ConversionConfig):
        """
        Initialize converter with validated configuration.

        Args:
            config: ConversionConfig instance (will be validated)
        """
        self.config = config
        self.stats: Dict[str, Any] = {
            "total_images": 0,
            "total_objects": 0,
            "skipped_images": 0,
            "skipped_objects": 0,
            "categories": {},
        }

    @abstractmethod
    def load_annotations(self) -> Dict[str, Any]:
        """
        Load annotations from input file.

        Returns:
            Dataset-specific annotation structure

        Raises:
            FileNotFoundError: If annotation file not found
            ValueError: If annotation format is invalid
        """
        pass

    @abstractmethod
    def convert_sample(self, sample_data: Any) -> Optional[Dict[str, Any]]:
        """
        Convert one sample to Qwen3-VL JSONL format.

        Args:
            sample_data: Dataset-specific sample data

        Returns:
            Dict in JSONL format, or None if sample should be skipped

        Format:
            {
                "images": [str],  # List with one image path
                "objects": [      # List of detected objects
                    {
                        "bbox_2d": [x1, y1, x2, y2],  # Pixel coordinates
                        "desc": str                    # Category name
                    }
                ],
                "width": int,
                "height": int,
                "summary": str  # Optional
            }
        """
        pass

    def convert(self) -> None:
        """
        Execute conversion pipeline.

        Steps:
        1. Load annotations
        2. Convert each sample
        3. Write to JSONL
        4. Save statistics

        Raises:
            Exception: Any error during conversion (fail-fast)
        """
        print(f"[Convert] Loading annotations from: {self.config.input_path}")
        annotations = self.load_annotations()

        print("[Convert] Converting samples...")
        samples = self._convert_all_samples(annotations)

        print(f"[Convert] Writing to: {self.config.output_path}")
        self._write_jsonl(samples)

        print("[Convert] Saving statistics...")
        self._save_stats()

        print("[Convert] ✓ Complete")
        self._print_summary()

    def _convert_all_samples(self, annotations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert all samples with progress tracking."""
        samples = []
        total = self._get_total_samples(annotations)

        for i, sample_data in enumerate(self._iterate_samples(annotations)):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{total}")

            if self.config.max_samples and i >= self.config.max_samples:
                print(f"  Reached max_samples limit: {self.config.max_samples}")
                break

            try:
                converted = self.convert_sample(sample_data)
                if converted is not None:
                    samples.append(converted)
                    self.stats["total_images"] += 1
                    self.stats["total_objects"] += len(converted.get("objects", []))
                else:
                    self.stats["skipped_images"] += 1
            except Exception as e:
                print(f"  ! Error converting sample {i}: {e}")
                self.stats["skipped_images"] += 1

        return samples

    def _write_jsonl(self, samples: List[Dict[str, Any]]) -> None:
        """Write samples to JSONL file."""
        output_dir = os.path.dirname(self.config.output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(self.config.output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _save_stats(self) -> None:
        """Save conversion statistics to JSON."""
        stats_path = self.config.output_path.replace(".jsonl", "_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def _print_summary(self) -> None:
        """Print conversion summary."""
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"  Output: {self.config.output_path}")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Total objects: {self.stats['total_objects']}")
        print(f"  Skipped images: {self.stats['skipped_images']}")
        print(f"  Skipped objects: {self.stats['skipped_objects']}")
        print(
            f"  Avg objects/image: {self.stats['total_objects'] / max(self.stats['total_images'], 1):.2f}"
        )
        print("=" * 60 + "\n")

    @abstractmethod
    def _get_total_samples(self, annotations: Dict[str, Any]) -> int:
        """Get total number of samples for progress tracking."""
        pass

    @abstractmethod
    def _iterate_samples(self, annotations: Dict[str, Any]):
        """Iterate over samples in dataset-specific way."""
        pass

    def _make_relative_path(self, image_path: str) -> str:
        """
        Make image path relative to output JSONL location.

        Args:
            image_path: Absolute image path

        Returns:
            Relative path from JSONL to image
        """
        if not self.config.relative_image_paths:
            return image_path

        output_dir = os.path.dirname(os.path.abspath(self.config.output_path))
        return os.path.relpath(image_path, output_dir)
