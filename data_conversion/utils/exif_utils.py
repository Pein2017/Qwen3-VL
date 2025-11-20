#!/usr/bin/env python3
"""
EXIF utilities for unified image handling.

Provides:
- apply_exif_orientation: apply EXIF orientation to a PIL Image and return RGB
- get_exif_transform: compute if EXIF would change dimensions, and the sizes
"""

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

EXIF_ORIENTATION_TAG = 274


def apply_exif_orientation(pil_image: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation so pixel data matches intended display, then ensure RGB.
    """
    transformed_image = ImageOps.exif_transpose(pil_image)
    if transformed_image is not None:
        pil_image = transformed_image
    if pil_image.mode == "RGBA":
        # Convert with white background for transparency handling
        from PIL import Image as PILImage

        bg = PILImage.new("RGB", pil_image.size, (255, 255, 255))
        bg.paste(pil_image, mask=pil_image.split()[3])
        return bg
    return pil_image.convert("RGB")


def get_exif_transform(image_path: Path) -> Tuple[bool, int, int, int, int, int]:
    """
    Compute whether EXIF orientation requires a pixel-space transform.

    Returns:
        (has_orientation, original_width, original_height, oriented_width, oriented_height, orientation_tag)
    """
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        orientation_value = img.getexif().get(EXIF_ORIENTATION_TAG, 1)
        try:
            orientation = int(orientation_value)
        except (TypeError, ValueError):
            orientation = 1
        transformed = ImageOps.exif_transpose(img) or img
        new_w, new_h = transformed.size

        has_orientation = orientation not in (0, 1)
        return has_orientation, orig_w, orig_h, new_w, new_h, orientation
