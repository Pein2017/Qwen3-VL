"""
LVIS dataset converter to Qwen3-VL JSONL format.

LVIS v1.0: 1203 categories, long-tail distribution (Frequent/Common/Rare)
Based on COCO 2017 images with more detailed annotations.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import BaseConverter, ConversionConfig
from .geometry import (
    clip_bbox_to_image,
    coco_bbox_to_xyxy,
    compute_bbox_area,
    validate_bbox_bounds,
)


class LVISConverter(BaseConverter):
    """
    Convert LVIS annotations (COCO-format JSON) to Qwen3-VL JSONL.

    LVIS annotation structure:
    {
      "images": [{"id": int, "coco_url": str, "width": int, "height": int, "file_name": str}],
      "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x,y,w,h],
                       "segmentation": [[x1,y1,...]], ...}],
      "categories": [{"id": int, "name": str, "frequency": str}]
    }

    Output JSONL format:
    {
      "images": ["path/to/image.jpg"],
      "objects": [
        {"bbox_2d": [x1,y1,x2,y2], "desc": "category"},
        {"poly": [x1,y1,...,xn,yn], "poly_points": n, "desc": "category"}  # if use_polygon=True
      ],
      "width": int,
      "height": int
    }

    Geometry interpretation:
    - bbox_2d: 2-point implicit rectangle [x1,y1,x2,y2]
    - poly: N-point closed polygon [x1,y1,...,xn,yn] where N >= 3
      If poly_max_points is set and N > poly_max_points, converts to bbox_2d instead

    Modes:
    - use_polygon=False (default): Only convert bounding boxes
    - use_polygon=True: Convert segmentation polygons (all N-point polygons → poly)
      If poly_max_points is set, polygons exceeding the cap are converted to bbox_2d
    """

    def __init__(
        self,
        config: ConversionConfig,
        use_polygon: bool = False,
        poly_max_points: Optional[int] = None,
    ):
        """
        Args:
            config: Conversion configuration
            use_polygon: If True, convert segmentation polygons to poly (arbitrary N points)
            poly_max_points: If set, polygons with more than this many points will be
                           converted to bbox_2d instead. If None, no cap is applied.
        """
        super().__init__(config)
        self.use_polygon = use_polygon
        self.poly_max_points = poly_max_points
        self.category_map: Dict[int, Dict[str, str]] = {}  # cat_id -> {name, frequency}
        self.image_map: Dict[int, Dict[str, Any]] = {}  # img_id -> image_info
        self.annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)

        # Track polygon statistics
        if use_polygon:
            self.stats["poly_converted"] = 0
            self.stats["polygon_skipped"] = 0
            if poly_max_points is not None:
                self.stats["poly_to_bbox_capped"] = 0

    def load_annotations(self) -> Dict[str, Any]:
        """
        Load LVIS annotations from JSON file.

        Returns:
            Dict with 'images', 'annotations', 'categories' keys

        Raises:
            FileNotFoundError: If annotation file not found
            ValueError: If JSON is malformed or missing required keys
        """
        print(f"  Loading LVIS JSON: {self.config.input_path}")

        try:
            with open(self.config.input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        # Validate required keys
        required_keys = ["images", "annotations", "categories"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(
                f"LVIS JSON missing required keys: {missing}. "
                f"Found keys: {list(data.keys())}"
            )

        print(f"    Images: {len(data['images'])}")
        print(f"    Annotations: {len(data['annotations'])}")
        print(f"    Categories: {len(data['categories'])}")

        # Build lookup structures
        self._build_category_map(data["categories"])
        self._build_image_map(data["images"])
        self._group_annotations_by_image(data["annotations"])

        return data

    def _build_category_map(self, categories: List[Dict]) -> None:
        """Build category ID to name/frequency mapping."""
        for cat in categories:
            cat_id = cat["id"]
            self.category_map[cat_id] = {
                "name": cat["name"],
                "frequency": cat.get("frequency", "unknown"),  # frequent/common/rare
            }

            # Track category statistics
            freq = cat.get("frequency", "unknown")
            if freq not in self.stats["categories"]:
                self.stats["categories"][freq] = 0
            self.stats["categories"][freq] += 1

        print("    Category distribution:")
        for freq, count in sorted(self.stats["categories"].items()):
            print(f"      {freq}: {count} categories")
        mode_str = (
            "enabled (N-point polygons → poly)"
            if self.use_polygon
            else "disabled (bbox only)"
        )
        print(f"    Polygon mode: {mode_str}")

    def _build_image_map(self, images: List[Dict]) -> None:
        """Build image ID to metadata mapping."""
        for img in images:
            # Try to get file_name from various sources
            file_name = img.get("coco_file_name") or img.get("file_name")

            # If still None, try to extract from coco_url
            if not file_name:
                coco_url = img.get("coco_url", "")
                if coco_url:
                    # Extract path from URL: http://images.cocodataset.org/train2017/000000391895.jpg
                    # -> train2017/000000391895.jpg
                    # The URL format is: http://images.cocodataset.org/{split}/{filename}
                    url_parts = coco_url.split("/")
                    if len(url_parts) >= 2:
                        # Get the last two parts: split directory and filename
                        file_name = "/".join(url_parts[-2:])

            self.image_map[img["id"]] = {
                "file_name": file_name,
                "width": img["width"],
                "height": img["height"],
                "coco_url": img.get("coco_url", ""),
            }

    def _group_annotations_by_image(self, annotations: List[Dict]) -> None:
        """Group annotations by image_id for efficient lookup."""
        for ann in annotations:
            self.annotations_by_image[ann["image_id"]].append(ann)

        print(f"    Images with annotations: {len(self.annotations_by_image)}")

    def convert_sample(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Convert one image + annotations to JSONL format.

        Args:
            image_id: LVIS image ID

        Returns:
            JSONL dict or None if image should be skipped
        """
        if image_id not in self.image_map:
            return None

        image_info = self.image_map[image_id]
        annotations = self.annotations_by_image.get(image_id, [])

        # Skip images without annotations
        if not annotations:
            return None

        # Build image path
        file_name = image_info.get("file_name")
        if not file_name or not isinstance(file_name, str):
            # Skip images without valid file_name
            print(
                f"  ! Image {image_id} has no valid file_name (got: {file_name}), skipping"
            )
            self.stats["skipped_images"] += 1
            return None

        image_path = os.path.join(self.config.image_root, file_name)

        # Verify image exists
        if not os.path.exists(image_path):
            print(f"  ! Image not found: {image_path}")
            return None

        # Convert annotations
        objects = []
        for ann in annotations:
            objs = self._convert_annotation(ann, image_info)
            if objs:
                objects.extend(objs if isinstance(objs, list) else [objs])

        # Skip if no valid objects
        if not objects:
            return None

        # Make path relative if configured
        if self.config.relative_image_paths:
            image_path = self._make_relative_path(image_path)

        return {
            "images": [image_path],
            "objects": objects,
            "width": image_info["width"],
            "height": image_info["height"],
        }

    def _convert_annotation(
        self, ann: Dict[str, Any], image_info: Dict[str, Any]
    ) -> Optional[list]:
        """
        Convert one LVIS annotation to object dict(s).

        Args:
            ann: LVIS annotation dict
            image_info: Image metadata

        Returns:
            List of objects: [{"bbox_2d": [...], "desc": str}] or
                            [{"quad": [...], "desc": str}] if use_polygon=True
            None if invalid
        """
        # Skip crowd annotations if configured
        if self.config.skip_crowd and ann.get("iscrowd", 0) == 1:
            self.stats["skipped_objects"] += 1
            return None

        # Get category name
        cat_id = ann["category_id"]
        if cat_id not in self.category_map:
            print(f"  ! Unknown category_id: {cat_id}")
            self.stats["skipped_objects"] += 1
            return None

        category_name = self.category_map[cat_id]["name"]

        # If polygon mode enabled, try to extract polygons from segmentation
        if self.use_polygon:
            poly_objs = self._extract_polygons(ann, category_name, image_info)
            if poly_objs:
                return poly_objs
            # If no valid polygons, fall through to bbox

        # Convert bbox from COCO [x,y,w,h] to [x1,y1,x2,y2]
        try:
            bbox_coco = ann["bbox"]
            bbox_xyxy = coco_bbox_to_xyxy(bbox_coco)
        except ValueError as e:
            print(f"  ! Invalid bbox: {e}")
            self.stats["skipped_objects"] += 1
            return None

        # Clip to image bounds if configured
        if self.config.clip_boxes:
            try:
                bbox_xyxy = clip_bbox_to_image(
                    bbox_xyxy, image_info["width"], image_info["height"]
                )
            except ValueError:
                # Box becomes degenerate after clipping
                self.stats["skipped_objects"] += 1
                return None
        else:
            # Just validate bounds
            if not validate_bbox_bounds(
                bbox_xyxy, image_info["width"], image_info["height"], allow_partial=True
            ):
                self.stats["skipped_objects"] += 1
                return None

        # Check minimum area/dimension
        area = compute_bbox_area(bbox_xyxy)
        if area < self.config.min_box_area:
            self.stats["skipped_objects"] += 1
            return None

        x1, y1, x2, y2 = bbox_xyxy
        width = x2 - x1
        height = y2 - y1
        if (
            width < self.config.min_box_dimension
            or height < self.config.min_box_dimension
        ):
            self.stats["skipped_objects"] += 1
            return None

        return [{"bbox_2d": bbox_xyxy, "desc": category_name}]

    def _extract_polygons(
        self, ann: Dict[str, Any], category_name: str, image_info: Dict[str, Any]
    ) -> Optional[list]:
        """
        Extract polygon geometries from LVIS segmentation.

        LVIS segmentation format: [[x1,y1,x2,y2,...], [part2...]]
        Converts all polygons to poly format (arbitrary N points).
        If poly_max_points is set and a polygon exceeds it, converts to bbox_2d.

        Returns:
            List of polygon objects (as 'poly') or bbox objects (as 'bbox_2d') or None
        """
        segmentation = ann.get("segmentation")

        if not segmentation or not isinstance(segmentation, list):
            return None

        objects = []

        for part in segmentation:
            if not isinstance(part, list):
                continue

            # Need at least 3 points (6 values) for a valid polygon
            num_coords = len(part)
            if num_coords < 6:  # Less than 3 points
                self.stats["polygon_skipped"] += 1
                continue

            num_points = num_coords // 2
            if num_coords % 2 != 0:
                # Odd number of coords, invalid
                self.stats["polygon_skipped"] += 1
                continue

            # Validate coordinates are within reasonable bounds
            try:
                coords = [float(c) for c in part]

                # Basic validation: all coords should be reasonable
                xs = coords[0::2]
                ys = coords[1::2]

                if (
                    min(xs) < -10
                    or max(xs) > image_info["width"] + 10
                    or min(ys) < -10
                    or max(ys) > image_info["height"] + 10
                ):
                    # Allow small margin but skip if way outside
                    self.stats["polygon_skipped"] += 1
                    continue

                # Check if polygon exceeds point cap - convert to bbox_2d if so
                if (
                    self.poly_max_points is not None
                    and num_points > self.poly_max_points
                ):
                    # Compute bounding box from polygon coordinates
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox_xyxy = [x_min, y_min, x_max, y_max]

                    # Clip to image bounds if configured
                    if self.config.clip_boxes:
                        try:
                            bbox_xyxy = clip_bbox_to_image(
                                bbox_xyxy, image_info["width"], image_info["height"]
                            )
                        except ValueError:
                            # Box becomes degenerate after clipping
                            self.stats["polygon_skipped"] += 1
                            continue
                    else:
                        # Just validate bounds
                        if not validate_bbox_bounds(
                            bbox_xyxy,
                            image_info["width"],
                            image_info["height"],
                            allow_partial=True,
                        ):
                            self.stats["polygon_skipped"] += 1
                            continue

                    # Check minimum area/dimension
                    area = compute_bbox_area(bbox_xyxy)
                    if area < self.config.min_box_area:
                        self.stats["polygon_skipped"] += 1
                        continue

                    x1, y1, x2, y2 = bbox_xyxy
                    width = x2 - x1
                    height = y2 - y1
                    if (
                        width < self.config.min_box_dimension
                        or height < self.config.min_box_dimension
                    ):
                        self.stats["polygon_skipped"] += 1
                        continue

                    # Add as bbox_2d
                    objects.append({"bbox_2d": bbox_xyxy, "desc": category_name})
                    self.stats["poly_to_bbox_capped"] = (
                        self.stats.get("poly_to_bbox_capped", 0) + 1
                    )
                else:
                    # Success: add as 'poly' (generic polygon)
                    objects.append(
                        {
                            "poly": coords,
                            "poly_points": num_points,  # Track number of points
                            "desc": category_name,
                        }
                    )
                    self.stats["poly_converted"] = (
                        self.stats.get("poly_converted", 0) + 1
                    )

            except (ValueError, TypeError):
                self.stats["polygon_skipped"] += 1
                continue

        return objects if objects else None

    def _get_total_samples(self, annotations: Dict[str, Any]) -> int:
        """Get total number of images with annotations."""
        return len(self.annotations_by_image)

    def _iterate_samples(self, annotations: Dict[str, Any]):
        """Iterate over image IDs that have annotations."""
        for image_id in self.annotations_by_image.keys():
            yield image_id


def convert_lvis_to_jsonl(
    annotation_path: str,
    image_root: str,
    output_path: str,
    *,
    split: str = "train",
    max_samples: Optional[int] = None,
    use_polygon: bool = False,
    poly_max_points: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Convenience function to convert LVIS dataset.

    Args:
        annotation_path: Path to LVIS JSON (e.g., lvis_v1_train.json)
        image_root: Root directory containing COCO images
        output_path: Output JSONL path
        split: Dataset split name
        max_samples: Limit samples (for testing)
        use_polygon: Convert segmentation to poly (arbitrary N-point polygons)
        poly_max_points: If set, polygons with more than this many points will be
                        converted to bbox_2d instead
        **kwargs: Additional ConversionConfig parameters

    Example:
        >>> convert_lvis_to_jsonl(
        ...     annotation_path="./lvis/raw/annotations/lvis_v1_train.json",
        ...     image_root="./lvis/raw/images/train2017",
        ...     output_path="./lvis/processed/train.jsonl",
        ...     split="train"
        ... )
    """
    config = ConversionConfig(
        input_path=os.path.abspath(annotation_path),
        output_path=os.path.abspath(output_path),
        image_root=os.path.abspath(image_root),
        split=split,
        max_samples=max_samples,
        **kwargs,
    )

    converter = LVISConverter(
        config, use_polygon=use_polygon, poly_max_points=poly_max_points
    )
    converter.convert()
