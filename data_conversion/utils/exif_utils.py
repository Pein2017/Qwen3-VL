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


def get_exif_transform(image_path: Path) -> Tuple[bool, int, int, int, int]:
    """
    Compute whether EXIF orientation changes dimensions.

    Returns: (is_transformed, original_width, original_height, new_width, new_height)
    """
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        transformed = ImageOps.exif_transpose(img) or img
        new_w, new_h = transformed.size
        is_changed = (orig_w != new_w) or (orig_h != new_h)
        return is_changed, orig_w, orig_h, new_w, new_h
