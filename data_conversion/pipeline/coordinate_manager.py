#!/usr/bin/env python3
"""
Centralized Coordinate Management System

This module provides a unified approach to handling all coordinate transformations
including EXIF orientation, dimension rescaling, and smart resize operations.

**Three-Stage Transformation Pipeline:**

All coordinate transformations follow a strict three-stage pipeline:

1. **EXIF Orientation Compensation**
   - Handles images with EXIF orientation tags (rotations/flips)
   - Transforms coordinates to match the visually-oriented image
   - Example: Portrait photo stored as landscape with rotation tag

2. **Dimension Mismatch Rescaling**
   - Handles cases where JSON metadata dimensions differ from actual image dimensions
   - Applies proportional scaling to align coordinates with actual image
   - Common when images are pre-processed but annotations are not updated

3. **Smart Resize Scaling**
   - Resizes image to fit within max_pixels constraint while maintaining aspect ratio
   - Scales coordinates proportionally to match resized image
   - Ensures dimensions are multiples of image_factor (e.g., 32 for model requirements)

**API Usage Guide:**

For complete pipeline (recommended):
- `transform_geometry_complete()` - Applies all three stages, works with any geometry type
- `transform_bbox_complete()` - Legacy bbox-only version (use transform_geometry_complete instead)

For individual stages (advanced use only):
- `apply_exif_orientation_to_geometry()` - Stage 1 only
- `apply_dimension_rescaling_to_geometry()` - Stage 2 only
- `apply_smart_resize_to_geometry()` - Stage 3 only

**Supported Geometry Types:**

- `bbox_2d`: [x1, y1, x2, y2] - Simple bounding box
- `poly`: [x1, y1, x2, y2, ..., xn, yn] - Polygon with arbitrary points
- `line`: [x1, y1, x2, y2, ..., xn, yn] - Multi-point line
- GeoJSON-style: {"type": "...", "coordinates": [...]} - Complex geometries

**Validation:**

- `validate_geometry_bounds()` - Check if all coordinates are within image bounds
- `validate_bbox_bounds()` - Legacy bbox-only validation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from data_conversion.utils.exif_utils import get_exif_transform

logger = logging.getLogger(__name__)


class CoordinateManager:
    """
    Centralized coordinate transformation and geometry processing manager.

    This class provides static methods for transforming coordinates through a
    three-stage pipeline (EXIF → dimension rescaling → smart resize).

    **Quick Start:**

    For most use cases, use `transform_geometry_complete()`:

    ```python
    bbox, transformed_geom, final_w, final_h = CoordinateManager.transform_geometry_complete(
        geometry_input=obj["poly"],  # or obj["bbox_2d"], obj["line"]
        image_path=Path("image.jpg"),
        json_width=1920,
        json_height=1080,
        smart_resize_factor=32,
        max_pixels=786432,
        enable_smart_resize=True
    )
    ```

    **Architecture:**

    All transformations are stateless and deterministic. The class uses static methods
    to ensure no hidden state and clear data flow. Each transformation stage is
    independent and can be tested in isolation.

    See module docstring for detailed pipeline documentation.
    """

    @staticmethod
    def get_exif_transform_matrix(
        image_path: Path,
    ) -> Tuple[bool, int, int, int, int, int]:
        """
        Analyze EXIF orientation and return transformation info.

        Returns:
            (has_orientation, original_width, original_height, oriented_width, oriented_height, orientation_tag)
        """
        (
            is_transformed,
            original_width,
            original_height,
            new_width,
            new_height,
            orientation,
        ) = get_exif_transform(image_path)
        return (
            is_transformed,
            original_width,
            original_height,
            new_width,
            new_height,
            orientation,
        )

    @staticmethod
    def _safe_int(value: Optional[Union[int, float, str]]) -> Optional[int]:
        """Best-effort conversion to int, returns None when conversion fails."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _dimensions_close(
        a: Optional[int], b: Optional[int], tolerance: int = 2
    ) -> bool:
        """
        Helper to compare dimensions with small tolerance (accounts for rounding).
        """
        if a is None or b is None:
            return False
        try:
            return abs(int(a) - int(b)) <= tolerance
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _ratio_close(
        width_a: Optional[int],
        height_a: Optional[int],
        width_b: Optional[int],
        height_b: Optional[int],
        tolerance: float = 1e-3,
    ) -> bool:
        """
        Compare aspect ratios with a configurable tolerance.
        """
        try:
            if not width_a or not height_a or not width_b or not height_b:
                return False
            ratio_a = float(width_a) / float(height_a)
            ratio_b = float(width_b) / float(height_b)
            return abs(ratio_a - ratio_b) <= tolerance
        except (TypeError, ZeroDivisionError, ValueError):
            return False

    @staticmethod
    def _dimension_difference(
        width_a: Optional[int],
        height_a: Optional[int],
        width_b: Optional[int],
        height_b: Optional[int],
    ) -> float:
        """
        Normalized difference between two dimension pairs (lower is closer).
        """
        if width_a is None or height_a is None or width_b is None or height_b is None:
            return float("inf")
        return abs(float(width_a) - float(width_b)) / max(float(width_b), 1.0) + abs(
            float(height_a) - float(height_b)
        ) / max(float(height_b), 1.0)

    @staticmethod
    def _ratio_difference(
        width_a: Optional[int],
        height_a: Optional[int],
        width_b: Optional[int],
        height_b: Optional[int],
    ) -> float:
        """
        Absolute difference between aspect ratios (lower is closer).
        """
        if (
            width_a is None
            or height_a is None
            or width_b is None
            or height_b is None
            or width_a == 0
            or height_a == 0
            or width_b == 0
            or height_b == 0
        ):
            return float("inf")
        return abs(
            (float(width_a) / float(height_a)) - (float(width_b) / float(height_b))
        )

    @classmethod
    def _should_apply_exif_orientation(
        cls,
        orientation: Optional[int],
        json_width: Optional[int],
        json_height: Optional[int],
        original_width: int,
        original_height: int,
        oriented_width: int,
        oriented_height: int,
    ) -> bool:
        """
        Decide whether EXIF compensation must be applied to coordinates.
        """
        if orientation in (None, 0, 1):
            return False

        matches_oriented = (
            cls._dimensions_close(json_width, oriented_width)
            and cls._dimensions_close(json_height, oriented_height)
        ) or cls._ratio_close(json_width, json_height, oriented_width, oriented_height)

        matches_raw = (
            cls._dimensions_close(json_width, original_width)
            and cls._dimensions_close(json_height, original_height)
        ) or cls._ratio_close(json_width, json_height, original_width, original_height)

        if matches_oriented and not matches_raw:
            return False

        if matches_raw and not matches_oriented:
            return True

        if matches_oriented and matches_raw:
            # Same dimensions/aspect ratio in both spaces (e.g., square images or pure flips).
            logger.debug(
                "JSON dimensions match both raw and EXIF-aligned sizes; skipping EXIF compensation to avoid duplicates."
            )
            return False

        if json_width is None or json_height is None:
            logger.warning(
                "Missing JSON width/height; defaulting to apply EXIF compensation for orientation=%s",
                orientation,
            )
            return True

        oriented_score = cls._ratio_difference(
            json_width, json_height, oriented_width, oriented_height
        ) + cls._dimension_difference(
            json_width, json_height, oriented_width, oriented_height
        )
        raw_score = cls._ratio_difference(
            json_width, json_height, original_width, original_height
        ) + cls._dimension_difference(
            json_width, json_height, original_width, original_height
        )

        apply = raw_score < oriented_score
        logger.debug(
            "Ambiguous EXIF decision (orientation=%s, json=%sx%s, raw=%sx%s, oriented=%sx%s) "
            "-- choosing %s compensation (raw_score=%.4f, oriented_score=%.4f)",
            orientation,
            json_width,
            json_height,
            original_width,
            original_height,
            oriented_width,
            oriented_height,
            "to apply" if apply else "to skip",
            raw_score,
            oriented_score,
        )
        return apply

    @staticmethod
    def _transform_point_by_orientation(
        x: float,
        y: float,
        width: int,
        height: int,
        orientation: Optional[int],
    ) -> Tuple[float, float]:
        """
        Transform a single (x, y) pair using EXIF orientation rules.
        """
        if orientation in (None, 0, 1):
            return x, y

        if orientation == 2:
            return float(width) - x, y
        if orientation == 3:
            return float(width) - x, float(height) - y
        if orientation == 4:
            return x, float(height) - y
        if orientation == 5:
            return y, x
        if orientation == 6:
            return float(height) - y, x
        if orientation == 7:
            return float(height) - y, float(width) - x
        if orientation == 8:
            return y, float(width) - x

        # Unknown orientation - no-op for safety
        return x, y

    @staticmethod
    def apply_exif_orientation_to_bbox(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
        exif_orientation: Optional[int] = None,
    ) -> List[float]:
        """
        Transform bbox coordinates to account for EXIF orientation changes.

        Args:
            bbox: [x1, y1, x2, y2] in original coordinate system
            original_width, original_height: Image dimensions before EXIF transform
            new_width, new_height: Image dimensions after EXIF transform
            exif_orientation: EXIF orientation value (optional, can detect from dimensions)

        Returns:
            Transformed bbox coordinates
        """
        if exif_orientation in (None, 0, 1):
            return bbox

        x1, y1, x2, y2 = bbox
        points = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]

        transformed_points = [
            CoordinateManager._transform_point_by_orientation(
                px, py, original_width, original_height, exif_orientation
            )
            for px, py in points
        ]

        xs = [p[0] for p in transformed_points]
        ys = [p[1] for p in transformed_points]

        final_x1 = max(0.0, min(xs))
        final_y1 = max(0.0, min(ys))
        final_x2 = min(float(new_width), max(xs))
        final_y2 = min(float(new_height), max(ys))

        logger.debug(
            "EXIF bbox transform (orientation=%s): "
            "[%.1f,%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f,%.1f]",
            exif_orientation,
            x1,
            y1,
            x2,
            y2,
            final_x1,
            final_y1,
            final_x2,
            final_y2,
        )

        return [final_x1, final_y1, final_x2, final_y2]

    @staticmethod
    def apply_dimension_rescaling(
        bbox: List[float],
        json_width: int,
        json_height: int,
        actual_width: int,
        actual_height: int,
    ) -> List[float]:
        """
        Rescale bbox coordinates when JSON dimensions differ from actual image dimensions.

        This typically happens when:
        1. EXIF orientation was applied to image but not to JSON coordinates
        2. Image was preprocessed but JSON coordinates weren't updated

        Note: Returns float coordinates to preserve precision for subsequent transformations.
        Final integer conversion happens in smart_resize_scaling.
        """
        if json_width == actual_width and json_height == actual_height:
            return bbox  # No rescaling needed

        scale_x = actual_width / json_width
        scale_y = actual_height / json_height

        x1, y1, x2, y2 = bbox

        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y

        # Clamp to image bounds (keep as float for precision)
        new_x1 = max(0.0, min(new_x1, float(actual_width)))
        new_y1 = max(0.0, min(new_y1, float(actual_height)))
        new_x2 = max(0.0, min(new_x2, float(actual_width)))
        new_y2 = max(0.0, min(new_y2, float(actual_height)))

        logger.debug(
            f"Dimension rescale: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"({json_width}x{json_height}) -> [{new_x1:.1f},{new_y1:.1f},{new_x2:.1f},{new_y2:.1f}] "
            f"({actual_width}x{actual_height})"
        )

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def apply_smart_resize_scaling(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> List[int]:
        """Scale bbox coordinates for smart resize operation.

        This function assumes input coordinates are already valid and inside the
        [0, original_width/height] range. In the trusted-input pipeline we avoid
        silently clamping or "fixing" coordinates here and only apply the
        deterministic scaling implied by the resize.
        """
        if original_width == new_width and original_height == new_height:
            # Convert to integers even if no scaling needed
            x1, y1, x2, y2 = bbox
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        x1, y1, x2, y2 = bbox

        # Apply scaling and round to integers (no extra clamping/correction)
        new_x1 = int(round(x1 * scale_x))
        new_y1 = int(round(y1 * scale_y))
        new_x2 = int(round(x2 * scale_x))
        new_y2 = int(round(y2 * scale_y))

        # Fail fast if invariants are broken; this should never happen for
        # pre-validated input and a correct resize pipeline.
        assert 0 <= new_x1 <= new_x2 <= new_width, (
            f"Invalid bbox after smart-resize scaling on x-axis: "
            f"[{new_x1}, {new_x2}] for width={new_width} (orig bbox={bbox}, "
            f"orig_size={original_width}x{original_height}, new_size={new_width}x{new_height})"
        )
        assert 0 <= new_y1 <= new_y2 <= new_height, (
            f"Invalid bbox after smart-resize scaling on y-axis: "
            f"[{new_y1}, {new_y2}] for height={new_height} (orig bbox={bbox}, "
            f"orig_size={original_width}x{original_height}, new_size={new_width}x{new_height})"
        )

        logger.debug(
            f"Smart resize scale: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"({original_width}x{original_height}) -> [{new_x1},{new_y1},{new_x2},{new_y2}] "
            f"({new_width}x{new_height})"
        )

        return [new_x1, new_y1, new_x2, new_y2]

    @classmethod
    def transform_bbox_complete(
        cls,
        bbox: List[float],
        image_path: Path,
        json_width: int,
        json_height: int,
        smart_resize_factor: int,
        max_pixels: int,
        enable_smart_resize: bool = True,
    ) -> Tuple[List[float], int, int]:
        """
        Apply complete bbox transformation pipeline.

        Pipeline:
        1. Apply EXIF orientation compensation
        2. Apply dimension mismatch rescaling
        3. Apply smart resize scaling (if enabled)

        Returns:
            (transformed_bbox, final_width, final_height)
        """
        # Step 1: Get EXIF transformation info
        (
            has_exif_orientation,
            orig_w,
            orig_h,
            oriented_w,
            oriented_h,
            orientation,
        ) = CoordinateManager.get_exif_transform_matrix(image_path)

        json_w = CoordinateManager._safe_int(json_width)
        json_h = CoordinateManager._safe_int(json_height)

        current_bbox = bbox
        coord_width = json_w if json_w else orig_w
        coord_height = json_h if json_h else orig_h

        # Step 2: Apply EXIF orientation compensation only when JSON coordinates
        # are still in the pre-orientation space.
        apply_exif = CoordinateManager._should_apply_exif_orientation(
            orientation,
            json_w,
            json_h,
            orig_w,
            orig_h,
            oriented_w,
            oriented_h,
        )
        if apply_exif:
            current_bbox = CoordinateManager.apply_exif_orientation_to_bbox(
                current_bbox,
                coord_width,
                coord_height,
                oriented_w,
                oriented_h,
                orientation,
            )
            coord_width, coord_height = oriented_w, oriented_h
            logger.debug(
                "Applied EXIF orientation compensation: json=%sx%s, raw=%sx%s -> oriented=%sx%s (orientation=%s)",
                json_w,
                json_h,
                orig_w,
                orig_h,
                oriented_w,
                oriented_h,
                orientation,
            )
        elif has_exif_orientation:
            logger.debug(
                "Detected EXIF orientation=%s but JSON coordinates already aligned; skipping compensation.",
                orientation,
            )

        # Step 3: Apply dimension mismatch rescaling if annotation dimensions still
        # differ from actual oriented image dimensions (e.g., pre-resized images).
        if coord_width != oriented_w or coord_height != oriented_h:
            source_w, source_h = coord_width, coord_height
            current_bbox = CoordinateManager.apply_dimension_rescaling(
                current_bbox, coord_width, coord_height, oriented_w, oriented_h
            )
            coord_width, coord_height = oriented_w, oriented_h
            logger.debug(
                "Applied dimension rescaling: %sx%s -> %sx%s",
                source_w,
                source_h,
                oriented_w,
                oriented_h,
            )

        current_width, current_height = coord_width, coord_height

        # Step 4: Apply smart resize scaling if enabled
        if enable_smart_resize:
            # Use the proper smart_resize function from vision_process.py
            from data_conversion.pipeline.vision_process import (
                MIN_PIXELS,
                smart_resize,
            )

            resize_h, resize_w = smart_resize(
                height=current_height,
                width=current_width,
                factor=smart_resize_factor,
                min_pixels=MIN_PIXELS,
                max_pixels=max_pixels,
            )

            if resize_w != current_width or resize_h != current_height:
                current_bbox = CoordinateManager.apply_smart_resize_scaling(
                    current_bbox, current_width, current_height, resize_w, resize_h
                )
                current_width, current_height = resize_w, resize_h
                logger.debug(
                    f"Applied smart resize: {current_width}x{current_height} (within max_pixels={max_pixels})"
                )

        return [float(x) for x in current_bbox], current_width, current_height

    @staticmethod
    def validate_bbox_bounds(bbox: List[float], width: int, height: int) -> bool:
        """
        Validate that bbox coordinates are within image bounds.

        Args:
            bbox: [x1, y1, x2, y2] (can be int or float)
            width, height: Image dimensions

        Returns:
            True if bbox is valid, False otherwise
        """
        x1, y1, x2, y2 = bbox

        # Check coordinate order
        if x1 >= x2 or y1 >= y2:
            return False

        # Check bounds (allow small floating point tolerance)
        tolerance = 0.1
        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > width + tolerance
            or y2 > height + tolerance
        ):
            return False

        return True

    @classmethod
    def process_sample_coordinates(
        cls,
        sample_data: Dict,
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
        smart_resize_factor: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Tuple[Dict, int, int]:
        """
        Process all object coordinates in a sample.

        Args:
            sample_data: Sample with 'objects' list containing 'bbox_2d' fields
            image_path: Path to the image file
            json_width, json_height: Dimensions from JSON metadata
            enable_smart_resize: Whether to apply smart resize

        Returns:
            (updated_sample_data, final_width, final_height)
        """
        if "objects" not in sample_data or not sample_data["objects"]:
            # No objects to process, just get final dimensions
            if enable_smart_resize:
                from data_conversion.pipeline.vision_process import (
                    MIN_PIXELS,
                    smart_resize,
                )

                _, _, _, final_w, final_h, _ = (
                    CoordinateManager.get_exif_transform_matrix(image_path)
                )
                if smart_resize_factor is None or max_pixels is None:
                    raise ValueError(
                        "process_sample_coordinates requires smart_resize_factor and max_pixels "
                        "when enable_smart_resize is True. Use transform_geometry_complete with "
                        "config values instead."
                    )
                resize_h, resize_w = smart_resize(
                    height=final_h,
                    width=final_w,
                    factor=smart_resize_factor,
                    min_pixels=MIN_PIXELS,
                    max_pixels=max_pixels,
                )
                return sample_data, resize_w, resize_h
            else:
                _, _, _, final_w, final_h, _ = (
                    CoordinateManager.get_exif_transform_matrix(image_path)
                )
                return sample_data, final_w, final_h

        # Process first object to get final dimensions
        if smart_resize_factor is None or max_pixels is None:
            raise ValueError(
                "process_sample_coordinates requires smart_resize_factor and max_pixels. "
                "Use transform_geometry_complete with config values instead."
            )
        first_bbox = sample_data["objects"][0]["bbox_2d"]
        _, final_width, final_height = CoordinateManager.transform_bbox_complete(
            first_bbox,
            image_path,
            json_width,
            json_height,
            enable_smart_resize=enable_smart_resize,
            smart_resize_factor=smart_resize_factor,
            max_pixels=max_pixels,
        )

        # Process all objects
        updated_objects = []
        for obj in sample_data["objects"]:
            if "bbox_2d" not in obj:
                updated_objects.append(obj)
                continue

            transformed_bbox, _, _ = CoordinateManager.transform_bbox_complete(
                obj["bbox_2d"],
                image_path,
                json_width,
                json_height,
                smart_resize_factor=smart_resize_factor,
                max_pixels=max_pixels,
                enable_smart_resize=enable_smart_resize,
            )

            # Validate transformed coordinates
            if CoordinateManager.validate_bbox_bounds(
                transformed_bbox, final_width, final_height
            ):
                updated_obj = obj.copy()
                updated_obj["bbox_2d"] = transformed_bbox

                # Apply coordinate normalization after transformation
                normalized_obj = CoordinateManager.normalize_object_coordinates(
                    updated_obj, final_width, final_height
                )
                updated_objects.append(normalized_obj)
            else:
                logger.warning(
                    f"Dropping invalid bbox after transformation: {transformed_bbox}"
                )

        updated_sample = sample_data.copy()
        updated_sample["objects"] = updated_objects

        return updated_sample, final_width, final_height

    @staticmethod
    def round_coordinates_to_int(coords: Union[List[float], List[int]]) -> List[int]:
        """
        Round all coordinates to integers for model compatibility.

        Args:
            coords: List of coordinate values (float or int)

        Returns:
            List of integer coordinates
        """
        return [int(round(float(coord))) for coord in coords]

    @staticmethod
    def apply_exif_orientation_to_geometry(
        geometry_input: Union[List, Dict],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
        exif_orientation: Optional[int] = None,
    ) -> Union[List, Dict]:
        """
        Apply EXIF orientation transformation to any geometry type.

        **Stage 1 of the transformation pipeline.**

        This method handles images with EXIF orientation tags (rotations/flips).
        It transforms coordinates to match the visually-oriented image.

        **When to use:**
        - Only use directly if you need fine-grained control over the pipeline
        - For most cases, use transform_geometry_complete() instead

        **Args:**
            geometry_input: Geometry data (bbox list, poly, line, GeoJSON geometry, etc.)
            original_width, original_height: Image dimensions before EXIF transform
            new_width, new_height: Image dimensions after EXIF transform
            exif_orientation: EXIF orientation value (1-8, or None for no transform)

        **Returns:**
            Transformed geometry maintaining original structure

        **Example:**
            ```python
            # Image has EXIF orientation 6 (90° CW rotation)
            # Original: 1920x1080, After rotation: 1080x1920
            transformed = CoordinateManager.apply_exif_orientation_to_geometry(
                geometry_input=[100, 100, 200, 200],  # bbox
                original_width=1920,
                original_height=1080,
                new_width=1080,
                new_height=1920,
                exif_orientation=6
            )
            ```
        """
        if exif_orientation in (None, 0, 1):
            return geometry_input  # No transformation needed

        # For simple bbox, use existing method
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            return CoordinateManager.apply_exif_orientation_to_bbox(
                geometry_input,
                original_width,
                original_height,
                new_width,
                new_height,
                exif_orientation=exif_orientation,
            )

        # For complex geometries, transform all coordinate points
        if isinstance(geometry_input, dict):
            all_points = CoordinateManager.get_all_coordinate_points(geometry_input)

            if not all_points:
                return geometry_input

            # Transform each point using the EXIF transformation logic
            transformed_points = []
            for x, y in all_points:
                new_x, new_y = CoordinateManager._transform_point_by_orientation(
                    x, y, original_width, original_height, exif_orientation
                )

                # Clamp to bounds
                new_x = max(0.0, min(new_x, float(new_width)))
                new_y = max(0.0, min(new_y, float(new_height)))
                transformed_points.append((new_x, new_y))

            # Reconstruct geometry with transformed points
            return CoordinateManager._reconstruct_geometry_from_points(
                geometry_input, transformed_points
            )

        return geometry_input

    @staticmethod
    def apply_dimension_rescaling_to_geometry(
        geometry_input: Union[List, Dict],
        json_width: int,
        json_height: int,
        actual_width: int,
        actual_height: int,
    ) -> Union[List, Dict]:
        """
        Apply dimension rescaling to any geometry type.

        **Stage 2 of the transformation pipeline.**

        This method handles cases where JSON metadata dimensions differ from
        actual image dimensions. It applies proportional scaling to align
        coordinates with the actual image.

        **When to use:**
        - Only use directly if you need fine-grained control over the pipeline
        - For most cases, use transform_geometry_complete() instead

        **Common scenarios:**
        - Images were resized but annotations were not updated
        - Annotation tool used different image resolution than actual file
        - Pre-processing pipeline changed dimensions

        **Args:**
            geometry_input: Geometry data (bbox list, poly, line, GeoJSON, etc.)
            json_width, json_height: Dimensions from JSON metadata
            actual_width, actual_height: Actual image dimensions

        **Returns:**
            Rescaled geometry maintaining original structure

        **Example:**
            ```python
            # JSON says 1920x1080, but actual image is 960x540 (downscaled)
            rescaled = CoordinateManager.apply_dimension_rescaling_to_geometry(
                geometry_input=[100, 100, 200, 200],  # bbox in JSON coordinates
                json_width=1920,
                json_height=1080,
                actual_width=960,
                actual_height=540
            )
            # Result: [50, 50, 100, 100] (scaled by 0.5x)
            ```
        """
        if json_width == actual_width and json_height == actual_height:
            return geometry_input  # No rescaling needed

        scale_x = actual_width / json_width
        scale_y = actual_height / json_height

        return CoordinateManager.scale_all_coordinates(geometry_input, scale_x, scale_y)

    @staticmethod
    def apply_smart_resize_to_geometry(
        geometry_input: Union[List, Dict],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> Union[List, Dict]:
        """
        Apply smart resize scaling to any geometry type.

        **Stage 3 of the transformation pipeline.**

        This method resizes images to fit within a max_pixels constraint while
        maintaining aspect ratio and ensuring dimensions are multiples of a
        factor (e.g., 32 for model requirements).

        **When to use:**
        - Only use directly if you need fine-grained control over the pipeline
        - For most cases, use transform_geometry_complete() instead

        **Purpose:**
        - Reduce memory usage by limiting total pixels
        - Ensure dimensions meet model requirements (e.g., multiples of 32)
        - Maintain aspect ratio to avoid distortion

        **Args:**
            geometry_input: Geometry data (bbox list, poly, line, GeoJSON, etc.)
            original_width, original_height: Original image dimensions
            new_width, new_height: Target dimensions after smart resize

        **Returns:**
            Scaled geometry maintaining original structure

        **Example:**
            ```python
            # Resize from 1920x1080 to 896x512 (within 786432 pixels, factor=32)
            resized = CoordinateManager.apply_smart_resize_to_geometry(
                geometry_input=[100, 100, 200, 200],  # bbox
                original_width=1920,
                original_height=1080,
                new_width=896,
                new_height=512
            )
            # Result: [46.67, 47.41, 93.33, 94.81] (scaled proportionally)
            ```

        **Note:**
            The new_width and new_height should be calculated by smart_resize()
            function from vision_process module, which ensures proper constraints.
        """
        if original_width == new_width and original_height == new_height:
            return geometry_input  # No scaling needed

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        return CoordinateManager.scale_all_coordinates(geometry_input, scale_x, scale_y)

    @classmethod
    def transform_geometry_complete(
        cls,
        geometry_input: Union[List, Dict],
        image_path: Path,
        json_width: int,
        json_height: int,
        smart_resize_factor: int,
        max_pixels: int,
        enable_smart_resize: bool = True,
    ) -> Tuple[List[float], Union[List, Dict], int, int]:
        """
        Apply complete geometry transformation pipeline for any geometry type.

        **This is the primary entry point for coordinate transformation.**
        Use this method for all coordinate transformations unless you have a specific
        reason to use individual transformation stages.

        This is the unified replacement for transform_bbox_complete that works
        with both simple bbox and complex geometries (poly, line, GeoJSON).

        **Three-Stage Pipeline:**

        1. **EXIF Orientation Compensation**
           - Reads EXIF orientation tag from image
           - Transforms coordinates if image has rotation/flip metadata
           - Skipped if JSON dimensions already match oriented dimensions

        2. **Dimension Mismatch Rescaling**
           - Compares JSON metadata dimensions with actual image dimensions
           - Applies proportional scaling if dimensions differ
           - Common when images are pre-processed but annotations are not updated

        3. **Smart Resize Scaling** (optional, controlled by enable_smart_resize)
           - Resizes to fit within max_pixels constraint
           - Maintains aspect ratio
           - Ensures dimensions are multiples of smart_resize_factor
           - Scales coordinates proportionally

        **Args:**
            geometry_input: Any geometry type:
                - bbox_2d: [x1, y1, x2, y2]
                - poly: [x1, y1, x2, y2, ..., xn, yn]
                - line: [x1, y1, x2, y2, ..., xn, yn]
                - GeoJSON: {"type": "...", "coordinates": [...]}
            image_path: Path to image file (for EXIF reading)
            json_width, json_height: Dimensions from JSON metadata
            smart_resize_factor: Factor for dimension alignment (e.g., 32 for model requirements)
            max_pixels: Maximum total pixels after resize (e.g., 786432 for 768*32*32)
            enable_smart_resize: Whether to apply smart resize (default: True)

        **Returns:**
            Tuple of (bbox_for_compatibility, transformed_geometry, final_width, final_height):
            - bbox_for_compatibility: [x1, y1, x2, y2] bounding box (for legacy code)
            - transformed_geometry: Transformed geometry in original format
            - final_width: Final image width after all transformations
            - final_height: Final image height after all transformations

        **Example:**
            ```python
            # Transform a polygon
            bbox, poly, w, h = CoordinateManager.transform_geometry_complete(
                geometry_input=[100, 100, 200, 100, 200, 200, 100, 200],  # poly
                image_path=Path("image.jpg"),
                json_width=1920,
                json_height=1080,
                smart_resize_factor=32,
                max_pixels=786432,
                enable_smart_resize=True
            )
            # bbox: [100, 100, 200, 200] (bounding box for compatibility)
            # poly: [transformed coordinates...]
            # w, h: final dimensions after all transformations
            ```

        **Notes:**
            - All transformations are deterministic and stateless
            - Coordinates are clamped to image bounds after each stage
            - Invalid geometries (out of bounds) are logged but not rejected here
            - Use validate_geometry_bounds() to check validity after transformation
        """
        (
            has_exif_orientation,
            orig_w,
            orig_h,
            oriented_w,
            oriented_h,
            orientation,
        ) = CoordinateManager.get_exif_transform_matrix(image_path)

        json_w = CoordinateManager._safe_int(json_width)
        json_h = CoordinateManager._safe_int(json_height)

        coord_width = json_w if json_w else orig_w
        coord_height = json_h if json_h else orig_h
        current_geometry = geometry_input

        apply_exif = CoordinateManager._should_apply_exif_orientation(
            orientation,
            json_w,
            json_h,
            orig_w,
            orig_h,
            oriented_w,
            oriented_h,
        )
        if apply_exif:
            current_geometry = CoordinateManager.apply_exif_orientation_to_geometry(
                current_geometry,
                coord_width,
                coord_height,
                oriented_w,
                oriented_h,
                exif_orientation=orientation,
            )
            coord_width, coord_height = oriented_w, oriented_h
            logger.debug(
                "Applied EXIF orientation for geometry: json=%sx%s, raw=%sx%s -> oriented=%sx%s (orientation=%s)",
                json_w,
                json_h,
                orig_w,
                orig_h,
                oriented_w,
                oriented_h,
                orientation,
            )
        elif has_exif_orientation:
            logger.debug(
                "EXIF orientation=%s detected for geometry but JSON already aligned; skipping compensation.",
                orientation,
            )

        if coord_width != oriented_w or coord_height != oriented_h:
            src_w, src_h = coord_width, coord_height
            current_geometry = CoordinateManager.apply_dimension_rescaling_to_geometry(
                current_geometry, coord_width, coord_height, oriented_w, oriented_h
            )
            coord_width, coord_height = oriented_w, oriented_h
            logger.debug(
                "Rescaled geometry dimensions: %sx%s -> %sx%s",
                src_w,
                src_h,
                oriented_w,
                oriented_h,
            )

        # Step 4: Apply smart resize scaling if enabled
        if enable_smart_resize:
            from data_conversion.pipeline.vision_process import (
                MIN_PIXELS,
                smart_resize,
            )

            resize_h, resize_w = smart_resize(
                height=coord_height,
                width=coord_width,
                factor=smart_resize_factor,
                min_pixels=MIN_PIXELS,
                max_pixels=max_pixels,
            )

            if resize_w != coord_width or resize_h != coord_height:
                current_geometry = CoordinateManager.apply_smart_resize_to_geometry(
                    current_geometry, coord_width, coord_height, resize_w, resize_h
                )
                coord_width, coord_height = resize_w, resize_h
                logger.debug(
                    f"Applied smart resize: {coord_width}x{coord_height} (within max_pixels={max_pixels})"
                )

        # Extract bbox for compatibility
        final_bbox = CoordinateManager.extract_bbox_from_geometry(current_geometry)

        return final_bbox, current_geometry, coord_width, coord_height

    @staticmethod
    def _reconstruct_geometry_from_points(
        original_geometry: Dict[str, Any],
        transformed_points: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Reconstruct geometry object with transformed coordinate points.

        This maintains the original structure while updating coordinate values.
        """
        if not original_geometry or not transformed_points:
            return original_geometry

        geometry_type = original_geometry.get("type", "")
        result_geometry = original_geometry.copy()

        # Convert points back to original coordinate structure
        if geometry_type in ["ExtentPolygon", "LineString"]:
            # Direct 2D array format
            result_geometry["coordinates"] = [[x, y] for x, y in transformed_points]

        elif geometry_type in ["Quad", "Polygon"]:
            # 3D array format with rings
            if len(transformed_points) > 0:
                result_geometry["coordinates"] = [
                    [[x, y] for x, y in transformed_points]
                ]

        else:
            # Generic fallback - try to preserve original structure
            logger.warning(f"Reconstructing unknown geometry type: {geometry_type}")
            result_geometry["coordinates"] = [[x, y] for x, y in transformed_points]

        return result_geometry

    # ============================================================================
    # GEOMETRY PROCESSING METHODS (merged from GeometryProcessor)
    # ============================================================================

    @staticmethod
    def extract_bbox_from_geometry(
        geometry_input: Union[List[float], Dict[str, Any]],
    ) -> List[float]:
        """
        Extract bounding box from any geometry type.

        Args:
            geometry_input: Geometry data (bbox list, GeoJSON geometry object, etc.)

        Returns:
            [x1, y1, x2, y2] bounding box coordinates (floats)
        """
        if not geometry_input:
            return [0.0, 0.0, 0.0, 0.0]

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            return [float(x) for x in geometry_input]

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            geometry_type = geometry_input.get("type", "")
            coordinates = geometry_input.get("coordinates", [])

            if not coordinates:
                logger.warning(f"Empty coordinates in geometry type: {geometry_type}")
                return [0.0, 0.0, 0.0, 0.0]

            # Extract all coordinate points
            all_points = CoordinateManager.get_all_coordinate_points(geometry_input)

            if not all_points:
                logger.warning(f"No valid coordinate points found in {geometry_type}")
                return [0.0, 0.0, 0.0, 0.0]

            # Calculate bounding box from all points
            x_coords = [point[0] for point in all_points]
            y_coords = [point[1] for point in all_points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            bbox = [
                float(round(x_min)),
                float(round(y_min)),
                float(round(x_max)),
                float(round(y_max)),
            ]
            logger.debug(
                f"Extracted bbox {bbox} from {geometry_type} with {len(all_points)} points"
            )
            return bbox

        logger.warning(f"Unknown geometry input type: {type(geometry_input)}")
        return [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def get_all_coordinate_points(
        geometry_input: Union[List[float], Dict[str, Any]],
    ) -> List[Tuple[float, float]]:
        """
        Extract all coordinate points from any geometry type.

        Args:
            geometry_input: Geometry data

        Returns:
            List of (x, y) coordinate tuples
        """
        if not geometry_input:
            return []

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            x1, y1, x2, y2 = geometry_input
            return [(float(x1), float(y1)), (float(x2), float(y2))]

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            coordinates = geometry_input.get("coordinates", [])
            geometry_type = geometry_input.get("type", "")

            return CoordinateManager._extract_points_from_coordinates(
                coordinates, geometry_type
            )

        return []

    @staticmethod
    def _extract_points_from_coordinates(
        coordinates: List[Any], geometry_type: str
    ) -> List[Tuple[float, float]]:
        """
        Extract coordinate points from GeoJSON coordinates array.

        Handles various coordinate structures:
        - ExtentPolygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x1,y1]]
        - Quad/Polygon: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x1,y1]]]
        - LineString: [[x1,y1], [x2,y2], [x3,y3], ...]
        """
        points = []

        try:
            if geometry_type in ["ExtentPolygon", "LineString"]:
                # Direct 2D array: [[x1,y1], [x2,y2], ...]
                for coord in coordinates:
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        points.append((float(coord[0]), float(coord[1])))

            elif geometry_type in ["Quad", "Polygon"]:
                # 3D array with lineType: [[[x1,y1], [x2,y2], ...]] (supports legacy "Square" type)
                if coordinates and isinstance(coordinates[0], list):
                    for ring in coordinates:
                        for coord in ring:
                            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                points.append((float(coord[0]), float(coord[1])))

            else:
                # Generic fallback - flatten all coordinate data
                points = CoordinateManager._flatten_coordinates_recursive(coordinates)

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Error extracting points from {geometry_type}: {e}")

        return points

    @staticmethod
    def _flatten_coordinates_recursive(coords: Any) -> List[Tuple[float, float]]:
        """Recursively flatten coordinate structures to extract all points."""
        points = []

        if isinstance(coords, (list, tuple)):
            # Check if this is a coordinate pair [x, y]
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                points.append((float(coords[0]), float(coords[1])))
            else:
                # Recurse into nested structures
                for item in coords:
                    points.extend(
                        CoordinateManager._flatten_coordinates_recursive(item)
                    )

        return points

    @staticmethod
    def scale_all_coordinates(
        geometry_input: Union[List[float], Dict[str, Any]],
        scale_x: float,
        scale_y: float,
    ) -> Union[List[float], Dict[str, Any]]:
        """
        Scale all coordinate points in any geometry type.

        Args:
            geometry_input: Original geometry data
            scale_x: Horizontal scaling factor
            scale_y: Vertical scaling factor

        Returns:
            Scaled geometry maintaining original structure
        """
        if not geometry_input:
            return geometry_input

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            x1, y1, x2, y2 = geometry_input
            return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            scaled_geometry = geometry_input.copy()
            coordinates = geometry_input.get("coordinates", [])
            geometry_type = geometry_input.get("type", "")

            if coordinates:
                scaled_coords = CoordinateManager._scale_coordinates_recursive(
                    coordinates, scale_x, scale_y
                )
                scaled_geometry["coordinates"] = scaled_coords

                logger.debug(
                    f"Scaled {geometry_type} coordinates by ({scale_x:.3f}, {scale_y:.3f})"
                )

            return scaled_geometry

        return geometry_input

    @staticmethod
    def _scale_coordinates_recursive(
        coords: Any, scale_x: float, scale_y: float
    ) -> Any:
        """Recursively scale coordinate structures."""
        if isinstance(coords, (list, tuple)):
            # Check if this is a coordinate pair [x, y]
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                return [coords[0] * scale_x, coords[1] * scale_y]
            else:
                # Recurse into nested structures
                return [
                    CoordinateManager._scale_coordinates_recursive(
                        item, scale_x, scale_y
                    )
                    for item in coords
                ]

        return coords

    @staticmethod
    def validate_geometry_bounds(
        geometry_input: Union[List[float], Dict[str, Any]],
        width: int,
        height: int,
        tolerance: float = 0.1,
    ) -> bool:
        """
        Validate that all coordinate points are within image bounds.

        **Use this after transformation to ensure coordinates are valid.**

        This method checks that all coordinate points (from any geometry type)
        are within the image bounds [0, width] x [0, height].

        **Args:**
            geometry_input: Geometry data to validate (bbox, poly, line, GeoJSON)
            width, height: Image dimensions
            tolerance: Floating point tolerance for bounds checking (default: 0.1)
                      Allows small floating-point errors near boundaries

        **Returns:**
            True if all coordinates are valid, False otherwise

        **Example:**
            ```python
            # After transformation, validate coordinates
            bbox, geom, w, h = CoordinateManager.transform_geometry_complete(...)

            if not CoordinateManager.validate_geometry_bounds(geom, w, h):
                logger.warning("Invalid geometry detected - out of bounds")
                # Handle invalid geometry (skip, clamp, or reject)
            ```

        **Note:**
            - Returns False if geometry is empty or has no coordinate points
            - Tolerance allows for small floating-point rounding errors
            - Does not modify the geometry - only validates it
        """
        all_points = CoordinateManager.get_all_coordinate_points(geometry_input)

        if not all_points:
            return False

        for x, y in all_points:
            if (
                x < -tolerance
                or y < -tolerance
                or x > width + tolerance
                or y > height + tolerance
            ):
                return False

        return True

    @staticmethod
    def extract_hierarchical_geometry(
        geometry_input: Union[List, Dict], preferred_format: str = "auto"
    ) -> Dict[str, Union[List[float], Dict, Any]]:
        """
        Extract geometry in hierarchical learning format with multiple annotation types.

        Args:
            geometry_input: Geometry data
            preferred_format: Preferred output format ("bbox_2d", "poly", "line", or "auto")

        Returns:
            Dictionary with annotation format and coordinates
        """
        if not geometry_input:
            return {"bbox_2d": [0.0, 0.0, 0.0, 0.0]}

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            return {"bbox_2d": [float(x) for x in geometry_input]}

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            geometry_type = geometry_input.get("type", "")
            coordinates = geometry_input.get("coordinates", [])

            if not coordinates:
                logger.warning(f"Empty coordinates in geometry type: {geometry_type}")
                return {"bbox_2d": [0.0, 0.0, 0.0, 0.0]}

            # Extract bbox for all types (always available)
            bbox = CoordinateManager.extract_bbox_from_geometry(geometry_input)
            result: Dict[str, Union[List[float], Dict, Any]] = {"bbox_2d": bbox}

            # Determine appropriate annotation format based on geometry type and preference
            if geometry_type == "LineString" and (preferred_format in ["line", "auto"]):
                # Line annotation: flatten coordinates for line format
                line_coords = CoordinateManager._extract_line_coordinates(
                    geometry_input
                )
                if line_coords:
                    result["line"] = line_coords
                    if preferred_format == "line":
                        # Remove bbox_2d if line is preferred format
                        result = {"line": line_coords, "bbox_2d": bbox}

            elif geometry_type in ["Quad", "Square", "Polygon"] and (
                preferred_format in ["poly", "auto"]
            ):
                # Poly annotation: extract polygon coordinates from Quad/Polygon/Square GeoJSON geometry types
                poly_coords = CoordinateManager.extract_poly_coordinates(geometry_input)
                if poly_coords:
                    result["poly"] = poly_coords
                    if preferred_format == "poly":
                        # Remove bbox_2d if poly is preferred format
                        result = {"poly": poly_coords, "bbox_2d": bbox}

            # Always preserve full geometry for reference
            result["geometry"] = geometry_input
            return result

        raise ValueError(
            f"Unknown geometry input type: {type(geometry_input)}. "
            f"Expected dict with geometry information. "
            f"Received: {geometry_input}"
        )

    @staticmethod
    def _extract_line_coordinates(geometry: Dict) -> List[float]:
        """Extract flattened line coordinates from LineString geometry."""
        coordinates = geometry.get("coordinates", [])
        line_coords = []

        for coord in coordinates:
            if isinstance(coord, list) and len(coord) >= 2:
                line_coords.extend(
                    [int(round(float(coord[0]))), int(round(float(coord[1])))]
                )

        return line_coords

    @staticmethod
    def extract_poly_coordinates(geometry: Dict) -> List[float]:
        """Extract polygon coordinates from Quad/Polygon GeoJSON geometry (source format).

        Applies canonical ordering starting from top-left vertex as specified in the prompt:
        "poly 排序参考点：使用第一个顶点 (x1, y1) 作为该对象的排序位置"
        Sorting rule: "首先按 Y 坐标（纵向）从小到大排列（图像上方优先），Y 坐标相同时按 X 坐标（横向）从小到大排列（图像左方优先）"

        Detects and rejects degenerate polygons (triangles stored as quads with duplicate vertices).
        """
        coordinates = geometry.get("coordinates", [])

        # Handle nested coordinate structure
        if coordinates and isinstance(coordinates[0], list):
            points = (
                coordinates[0] if isinstance(coordinates[0][0], list) else coordinates
            )
        else:
            points = coordinates

        # Extract all points for poly format (currently supports 4-point quads, but extensible)
        raw_coords = []
        for _, point in enumerate(points):  # Take all points (currently 4 for quads)
            if isinstance(point, list) and len(point) >= 2:
                raw_coords.extend(
                    [int(round(float(point[0]))), int(round(float(point[1])))]
                )

        # Require at least 4 points (8 coordinates) to form a polygon
        if len(raw_coords) >= 8 and len(raw_coords) % 2 == 0:
            # Strip any closing point if present
            if len(raw_coords) >= 4:
                first_x, first_y = raw_coords[0], raw_coords[1]
                last_x, last_y = raw_coords[-2], raw_coords[-1]
                # If already closed, remove the duplicate closing point
                if first_x == last_x and first_y == last_y:
                    raw_coords = raw_coords[:-2]

            # Check for duplicate vertices (degenerate quad: triangle stored as quad with duplicate vertex)
            points_list = [
                (raw_coords[i], raw_coords[i + 1]) for i in range(0, len(raw_coords), 2)
            ]
            unique_points = list(set(points_list))

            if len(unique_points) < len(points_list):
                logger.warning(
                    f"Degenerate polygon detected: {len(points_list)} points but only {len(unique_points)} unique points. "
                    f"This is likely a triangle stored as a quad with duplicate vertex. Deduplicating vertices."
                )
                # Remove duplicate points and reconstruct raw_coords
                raw_coords = [coord for point in unique_points for coord in point]
                points_list = unique_points

            # After removing closing point and deduplication, validate we have at least 4 points
            # If not, this is an invalid polygon (too few vertices)
            if len(points_list) < 4:
                logger.warning(
                    f"Invalid polygon: has less than 4 unique points "
                    f"({len(points_list)} points). Rejecting degenerate polygon."
                )
                return []

            # Apply canonical ordering to ensure clockwise traversal starting from top-left
            ordered_points = CoordinateManager.canonical_poly_ordering(points_list)
            poly_coords = [float(coord) for point in ordered_points for coord in point]
            return poly_coords
        else:
            logger.warning(
                f"Invalid poly coordinates: expected at least 8 values (even number), got {len(raw_coords)}"
            )
            return []

    # Backwards compatibility alias (prefer extract_poly_coordinates)
    _extract_poly_coordinates = extract_poly_coordinates

    # =========================================================================
    # COORDINATE NORMALIZATION METHODS
    # =========================================================================

    @staticmethod
    def _close_polygon(
        poly_coords: Union[List[float], List[int]],
    ) -> Union[List[float], List[int]]:
        """
        Close a polygon by adding the first point at the end if not already closed.

        Args:
            poly_coords: Polygon coordinates [x1, y1, x2, y2, ...] (int or float)

        Returns:
            Closed polygon coordinates [x1, y1, x2, y2, ..., x1, y1] (same type as input)
        """
        if len(poly_coords) < 4 or len(poly_coords) % 2 != 0:
            return poly_coords

        # Check if already closed (first point == last point)
        if len(poly_coords) >= 4:
            first_x, first_y = poly_coords[0], poly_coords[1]
            last_x, last_y = poly_coords[-2], poly_coords[-1]
            if first_x == last_x and first_y == last_y:
                return poly_coords  # Already closed

        # Close by appending first point
        return poly_coords + [poly_coords[0], poly_coords[1]]

    @staticmethod
    def normalize_object_coordinates(
        obj: Dict[str, Any], width: int, height: int
    ) -> Dict[str, Any]:
        """
        Normalize coordinates within an object preserving native geometry format.

        Args:
            obj: Object with geometry (bbox_2d, line, or poly) and description
            width: Image width for bounds checking
            height: Image height for bounds checking

        Returns:
            Object with normalized coordinates in the same geometry format
        """
        normalized_obj = obj.copy()

        if "bbox_2d" in obj:
            normalized_obj["bbox_2d"] = CoordinateManager.normalize_bbox_coordinates(
                obj["bbox_2d"], width, height
            )
        elif "line" in obj:
            normalized_obj["line"] = CoordinateManager.normalize_line_coordinates(
                obj["line"], width, height
            )
        elif "poly" in obj:
            poly_norm = CoordinateManager.normalize_poly_coordinates(
                obj["poly"], width, height
            )
            if len(poly_norm) >= 8 and len(poly_norm) % 2 == 0:
                pts = [
                    (float(poly_norm[i]), float(poly_norm[i + 1]))
                    for i in range(0, len(poly_norm), 2)
                ]
                ordered = CoordinateManager.canonical_poly_ordering(pts)
                poly_norm = [int(coord) for p in ordered for coord in p]
            normalized_obj["poly"] = poly_norm
        else:
            logger.warning(f"Object missing geometry type: {obj}")

        return normalized_obj

    @staticmethod
    def normalize_bbox_coordinates(
        bbox_coords: List[float], width: int, height: int
    ) -> List[int]:
        """
        Normalize bounding box coordinates ensuring proper ordering.

        Args:
            bbox_coords: [x1, y1, x2, y2] coordinates
            width: Image width for bounds checking
            height: Image height for bounds checking

        Returns:
            Normalized coordinates as [x1, y1, x2, y2] with x1 < x2, y1 < y2
        """
        if len(bbox_coords) != 4:
            logger.warning(
                f"Invalid bbox coordinates: expected 4 values, got {len(bbox_coords)}"
            )
            return [0, 0, 1, 1]

        x1, y1, x2, y2 = bbox_coords

        # Ensure proper ordering: x1 < x2, y1 < y2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Clamp to image bounds
        x_min = max(0, min(x_min, width - 1))
        x_max = max(0, min(x_max, width - 1))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(0, min(y_max, height - 1))

        # Handle degenerate cases
        if x_min == x_max:
            if x_max < width - 1:
                x_max += 1
            else:
                x_min = max(0, x_min - 1)

        if y_min == y_max:
            if y_max < height - 1:
                y_max += 1
            else:
                y_min = max(0, y_min - 1)

        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    @staticmethod
    def normalize_line_coordinates(
        line_coords: List[float], width: int, height: int
    ) -> List[int]:
        """
        Normalize line coordinates with canonical ordering and degenerate handling.

        Args:
            line_coords: [x1, y1, x2, y2, ...] coordinate pairs
            width: Image width for bounds checking
            height: Image height for bounds checking

        Returns:
            Normalized coordinates with canonical ordering and padding for degenerate cases
        """
        if len(line_coords) < 4 or len(line_coords) % 2 != 0:
            logger.warning(
                f"Invalid line coordinates: expected even number of coordinates (>=4), got {len(line_coords)}"
            )
            return [0, 0, 1, 0]

        # Extract points
        points = [
            (line_coords[i], line_coords[i + 1]) for i in range(0, len(line_coords), 2)
        ]

        # Apply canonical ordering
        ordered_points = CoordinateManager._canonical_line_ordering(points)

        # Clamp to image bounds
        clamped_points = []
        for x, y in ordered_points:
            x_clamped = max(0, min(int(x), width - 1))
            y_clamped = max(0, min(int(y), height - 1))
            clamped_points.append((x_clamped, y_clamped))

        # Handle degenerate cases (horizontal/vertical lines)
        if len(clamped_points) == 2:  # Simple line
            clamped_points = CoordinateManager._handle_degenerate_line(
                clamped_points, width, height
            )

        # Flatten back to coordinate list
        normalized_coords = []
        for x, y in clamped_points:
            normalized_coords.extend([x, y])

        return normalized_coords

    @staticmethod
    def normalize_poly_coordinates(
        poly_coords: List[float], width: int, height: int
    ) -> List[int]:
        """
        Normalize polygon coordinates with canonical vertex ordering.

        Args:
            poly_coords: [x1, y1, x2, y2, x3, y3, x4, y4, ...] polygon vertices
            width: Image width for bounds checking
            height: Image height for bounds checking

        Returns:
            Normalized coordinates with canonical vertex ordering
        """
        if len(poly_coords) < 8 or len(poly_coords) % 2 != 0:
            logger.warning(
                f"Invalid poly coordinates: expected at least 8 values (even number), got {len(poly_coords)}"
            )
            return [0, 0, 1, 0, 1, 1, 0, 1]

        # Clamp all coordinates to image bounds
        clamped_coords: List[int] = []
        for i in range(0, len(poly_coords), 2):
            x = max(0, min(poly_coords[i], width - 1))
            y = max(0, min(poly_coords[i + 1], height - 1))
            clamped_coords.extend([int(x), int(y)])

        # Convert to points and canonicalize whenever we have at least 4 vertices
        points = [
            (float(clamped_coords[i]), float(clamped_coords[i + 1]))
            for i in range(0, len(clamped_coords), 2)
        ]
        try:
            ordered_points = CoordinateManager.canonical_poly_ordering(points)
        except ValueError as exc:
            logger.warning("Failed to canonicalize polygon coordinates: %s", exc)
            ordered_points = points

        flattened = [int(coord) for point in ordered_points for coord in point]
        return flattened

    @staticmethod
    def _canonical_line_ordering(
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Establish canonical ordering for line points.

        For 2-point lines: order by x-coordinate first, then y-coordinate for consistency.
        For multi-point lines: preserve the original path structure and choose a canonical
        direction such that the first point is the leftmost endpoint (lowest x),
        tie-breaking by y (lowest y). If needed, reverse the entire sequence.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            Points in canonical order with consistent direction
        """
        if len(points) == 2:
            # Simple line: order lexicographically (x first, then y)
            p1, p2 = points
            if p1[0] < p2[0] or (p1[0] == p2[0] and p1[1] <= p2[1]):
                return [p1, p2]
            else:
                return [p2, p1]
        else:
            # Multi-point polyline: determine orientation by endpoints (leftmost-first)
            return CoordinateManager._normalize_polyline_direction(points)

    @staticmethod
    def _normalize_polyline_direction(
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Normalize multi-point line direction to establish canonical ordering.

        Strategy:
        1. Preserve path structure (do not reorder intermediate points)
        2. Establish consistent direction using endpoints only
        3. Start from the leftmost endpoint (lowest x), tie-breaking by y (lowest y)
        4. If the current sequence does not start from the leftmost endpoint,
           reverse the entire sequence

        Args:
            points: List of (x, y) coordinate tuples representing the path

        Returns:
            Points with canonical direction established (leftmost endpoint first)
        """
        if len(points) < 2:
            return points

        start = points[0]
        end = points[-1]

        # Compare endpoints by (x, y): leftmost-first, tie-break by y
        if (start[0], start[1]) <= (end[0], end[1]):
            return points
        else:
            return list(reversed(points))

    @staticmethod
    def _handle_degenerate_line(
        points: List[Tuple[int, int]], width: int, height: int
    ) -> List[Tuple[int, int]]:
        """
        Handle degenerate lines by adding minimal padding.

        Args:
            points: List of 2 points representing a line
            width: Image width
            height: Image height

        Returns:
            Points with padding applied to prevent degenerate bounding boxes
        """
        if len(points) != 2:
            return points

        (x1, y1), (x2, y2) = points

        # Check for horizontal line (y1 == y2)
        if y1 == y2:
            # Add vertical padding
            if y1 > 0:
                y1 -= 1
            if y2 < height - 1:
                y2 += 1
            else:
                # If we can't expand down, expand up
                if y1 > 0:
                    y1 -= 1

        # Check for vertical line (x1 == x2)
        elif x1 == x2:
            # Add horizontal padding
            if x1 > 0:
                x1 -= 1
            if x2 < width - 1:
                x2 += 1
            else:
                # If we can't expand right, expand left
                if x1 > 0:
                    x1 -= 1

        return [(x1, y1), (x2, y2)]

    @staticmethod
    def canonical_poly_ordering(
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Unified polygon ordering (quads and multi-vertex):
        - drop duplicated closing point
        - sort all vertices clockwise around centroid
        - rotate so the top-most then left-most vertex is first
        """
        import math

        if len(points) < 4:
            raise ValueError(f"Polygon must have at least 4 points: {points}")

        cleaned_points: List[Tuple[float, float]] = [
            (float(p[0]), float(p[1])) for p in points
        ]

        if (
            len(cleaned_points) > 1
            and cleaned_points[0][0] == cleaned_points[-1][0]
            and cleaned_points[0][1] == cleaned_points[-1][1]
        ):
            cleaned_points = cleaned_points[:-1]

        if len(cleaned_points) < 4:
            raise ValueError(
                "Polygon must retain at least 4 vertices after closing-point removal."
            )

        cx = sum(p[0] for p in cleaned_points) / len(cleaned_points)
        cy = sum(p[1] for p in cleaned_points) / len(cleaned_points)

        def angle_key(p: Tuple[float, float]) -> Tuple[float, float, float]:
            angle = math.atan2(p[1] - cy, p[0] - cx)  # CCW
            normalized = (angle + 2 * math.pi) % (2 * math.pi)
            return (-normalized, p[1], p[0])

        ordered = sorted(cleaned_points, key=angle_key)

        top_left_idx = min(
            range(len(ordered)),
            key=lambda i: (ordered[i][1], ordered[i][0]),
        )
        ordered = ordered[top_left_idx:] + ordered[:top_left_idx]

        return ordered

    # Backwards compatibility alias (prefer canonical_poly_ordering)
    _canonical_poly_ordering = canonical_poly_ordering


# Backward-compatibility re-exports for external callers.
# These keep older imports like `from data_conversion.coordinate_manager import FormatConverter` working
# while the actual implementations live in dedicated modules.
from data_conversion.pipeline.format_converter import (  # noqa: E402
    FormatConverter as _FormatConverter,
)
from data_conversion.pipeline.validation_manager import (  # noqa: E402
    DataValidator as _DataValidator,
)
from data_conversion.pipeline.validation_manager import (  # noqa: E402
    StructureValidator as _StructureValidator,
)

FormatConverter = _FormatConverter
DataValidator = _DataValidator
StructureValidator = _StructureValidator
