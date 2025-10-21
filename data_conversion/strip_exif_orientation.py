"""
DEPRECATED: This standalone utility for EXIF orientation handling is superseded by
the CoordinateManager.get_exif_transform_matrix() method in coordinate_manager.py.

The current pipeline automatically handles EXIF orientation during coordinate transformations.
This file is kept for manual image processing but is not used in the main pipeline.
"""

import argparse
import logging
from pathlib import Path

from PIL import Image, ImageOps


# Configure logging to file
LOG_FILE = Path(__file__).parent / "convert.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=str(LOG_FILE),
    filemode="a",
)
logger = logging.getLogger(__name__)


def process_image(path: Path, dry_run: bool = False) -> bool:
    """Apply EXIF orientation and strip metadata in-place.

    Returns True if the file was modified, False otherwise.
    """
    try:
        with Image.open(path) as img:
            orientation = img.getexif().get(274, 1)
            # If no orientation transform required and the file already has no
            # EXIF block, we can skip to speed things up.
            if orientation == 1 and not img.info.get("exif"):
                return False

            # Apply EXIF orientation and convert to RGB
            img_oriented = ImageOps.exif_transpose(img)
            if img_oriented is not None:
                img_oriented = img_oriented.convert("RGB")
            else:
                # If exif_transpose failed, use original image
                img_oriented = img.convert("RGB")

            if dry_run:
                return orientation != 1 or bool(img.info.get("exif"))

            # Preserve a valid image extension so PIL can infer format.
            tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)
            img_oriented.save(tmp_path)  # intentionally omit exif kwarg
            tmp_path.replace(path)
            return True
    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        return False


def traverse_and_fix(root: Path, recursive: bool = True, dry_run: bool = False) -> None:
    patterns = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern) if recursive else root.glob(pattern))

    modified = 0
    total = 0
    for f in files:
        total += 1
        if process_image(f, dry_run=dry_run):
            modified += 1

    logger.info("Processed %d images, modified %d", total, modified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strip EXIF orientation from images by applying it to the pixel data."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing images to fix (e.g., ds)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search sub-directories recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files that would be modified.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    traverse_and_fix(
        Path(args.input_dir), recursive=not args.no_recursive, dry_run=args.dry_run
    )
