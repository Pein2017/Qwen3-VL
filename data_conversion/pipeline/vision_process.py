"""
Vision Processing Module for Data Conversion Pipeline

Core vision processing functionality for the data conversion pipeline:
- Image processing, resizing, and format conversion
- EXIF orientation handling
- Smart resize with factor-based dimension constraints

Includes ImageProcessor class for data conversion pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from data_conversion.utils.exif_utils import apply_exif_orientation
from src.datasets.preprocessors.resize import (
    IMAGE_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    smart_resize,
)


logger = logging.getLogger(__name__)


# Data Conversion Pipeline Image Processor
class ImageProcessor:
    """Unified image processing for the data conversion pipeline."""

    def __init__(self, config):
        """Initialize with configuration."""

        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = config.get_dataset_output_dir()
        self.output_image_dir = config.get_dataset_image_dir()

        logger.info(f"ImageProcessor initialized: resize={config.resize}")

    def to_rgb(self, pil_image: Image.Image) -> Image.Image:
        """
        Convert PIL image to RGB with proper EXIF orientation handling.

        Applies EXIF orientation transformation to ensure image display
        matches annotation space, then converts to RGB with white background
        for transparency handling.
        """
        # Apply EXIF orientation transformation (centralized)
        pil_image = apply_exif_orientation(pil_image)
        return pil_image

    def process_image(
        self,
        image_path: Path,
        width: int,
        height: int,
        output_base_dir: Optional[Path] = None,
        final_width: Optional[int] = None,
        final_height: Optional[int] = None,
    ) -> Tuple[Path, int, int]:
        """Process a single image using the canonical resize + EXIF pipeline.

        Pipeline:
          1. Apply EXIF orientation so pixels match the annotation space.
          2. If ``final_width``/``final_height`` are provided, resize to that size.
          3. Otherwise, if ``self.config.resize`` is True, apply :func:`smart_resize`
             to the EXIF dimensions.
          4. Otherwise, only EXIF orientation is applied (no geometric resize).

        Returns:
            Tuple of (output_image_path, final_width, final_height)
        """
        from data_conversion.utils.file_ops import FileOperations

        # If no output image directory is configured, fall back to returning the
        # original path with the best-effort dimensions from the caller.
        if not self.output_image_dir:
            out_w = final_width if final_width is not None else width
            out_h = final_height if final_height is not None else height
            return image_path, out_w, out_h

        # Calculate output path
        try:
            rel_path = image_path.relative_to(self.input_dir)
        except ValueError:
            # image_path is not relative to input_dir, it might already be in output_dir
            rel_path = image_path.name

        output_path = self.output_image_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if image already exists and has been processed; trust its size.
        if output_path.exists() and output_path != image_path:
            existing_width, existing_height = FileOperations.get_image_dimensions(
                output_path
            )
            logger.debug(
                f"Using existing processed image: {output_path} ({existing_width}x{existing_height})"
            )
            return output_path, existing_width, existing_height

        # Materialise EXIF orientation before any resizing so that geometry and
        # pixels live in the same coordinate frame.
        with Image.open(image_path) as img:
            processed_img = self.to_rgb(img)
            base_width, base_height = processed_img.size

            # Decide target size
            if (final_width is None) != (final_height is None):
                raise ValueError(
                    "process_image received mismatched final dimensions: "
                    f"final_width={final_width}, final_height={final_height}"
                )

            if final_width is not None and final_height is not None:
                target_width, target_height = final_width, final_height
            elif self.config.resize:
                # Smart resize using the canonical implementation on EXIF size
                target_height, target_width = smart_resize(
                    height=base_height,
                    width=base_width,
                    factor=IMAGE_FACTOR,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
            else:
                target_width, target_height = base_width, base_height

            # Resize only if needed
            if target_width != base_width or target_height != base_height:
                resized_img = processed_img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                resized_img.save(output_path)
                logger.debug(
                    f"Resized {image_path.name}: "
                    f"{base_width}x{base_height} → {target_width}x{target_height} "
                    f"(MAX_PIXELS={MAX_PIXELS})"
                )
            else:
                processed_img.save(output_path)
                logger.debug(
                    f"Copied {image_path.name} with EXIF orientation applied "
                    f"(dimensions {base_width}x{base_height})"
                )

        return output_path, target_width, target_height

    def get_relative_image_path(self, absolute_image_path: Path) -> str:
        """Get relative image path for use in JSONL files."""
        try:
            # Make path relative to dataset output directory
            rel_path = absolute_image_path.relative_to(self.output_dir)
            return str(rel_path)
        except ValueError:
            # If path is not relative to output_dir, just return the name under images/
            return f"images/{absolute_image_path.name}"
