"""
Geometry utilities for coordinate transformations.

Pure functions for bbox/polygon conversions and validations.
"""

from typing import List, Tuple


def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
    """
    Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2].

    Args:
        bbox: COCO format [x, y, w, h] where (x, y) is top-left corner

    Returns:
        [x1, y1, x2, y2] format where (x1, y1) is top-left, (x2, y2) is bottom-right

    Raises:
        ValueError: If bbox has invalid dimensions
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}: {bbox}")

    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox dimensions: width={w}, height={h}")

    return [x, y, x + w, y + h]


def validate_bbox_bounds(
    bbox: List[float], img_width: int, img_height: int, *, allow_partial: bool = True
) -> bool:
    """
    Validate bbox is within image bounds.

    Args:
        bbox: [x1, y1, x2, y2] format
        img_width: Image width in pixels
        img_height: Image height in pixels
        allow_partial: If True, allow boxes partially outside image

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If bbox format is invalid
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}")

    x1, y1, x2, y2 = bbox

    # Check bbox is well-formed
    if x2 <= x1 or y2 <= y1:
        return False

    if allow_partial:
        # At least some part of box must be in image
        if x2 <= 0 or x1 >= img_width:
            return False
        if y2 <= 0 or y1 >= img_height:
            return False
        return True
    else:
        # Entire box must be in image
        if x1 < 0 or y1 < 0:
            return False
        if x2 > img_width or y2 > img_height:
            return False
        return True


def clip_bbox_to_image(
    bbox: List[float], img_width: int, img_height: int
) -> List[float]:
    """
    Clip bbox to image boundaries.

    Args:
        bbox: [x1, y1, x2, y2] format
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Clipped bbox [x1, y1, x2, y2]

    Raises:
        ValueError: If bbox format is invalid or becomes degenerate after clipping
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}")

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Bbox becomes degenerate after clipping: "
            f"[{x1}, {y1}, {x2}, {y2}] in image {img_width}x{img_height}"
        )

    return [x1, y1, x2, y2]


def compute_bbox_area(bbox: List[float]) -> float:
    """
    Compute area of bbox.

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Area in square pixels

    Raises:
        ValueError: If bbox format is invalid
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}")

    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def filter_small_boxes(
    bboxes: List[List[float]], min_area: float = 1.0, min_dimension: float = 1.0
) -> Tuple[List[List[float]], List[int]]:
    """
    Filter out boxes that are too small.

    Args:
        bboxes: List of bboxes in [x1, y1, x2, y2] format
        min_area: Minimum area in square pixels
        min_dimension: Minimum width or height in pixels

    Returns:
        Tuple of (filtered_bboxes, kept_indices)
    """
    filtered = []
    kept_indices = []

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area >= min_area and width >= min_dimension and height >= min_dimension:
            filtered.append(bbox)
            kept_indices.append(i)

    return filtered, kept_indices
