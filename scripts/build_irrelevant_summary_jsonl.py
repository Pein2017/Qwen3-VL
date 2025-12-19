#!/usr/bin/env python3
"""
Build an "irrelevant image" summary JSONL for summary-mode SFT.

This helper emits dense-caption compatible JSONL records (per
`docs/data/DATA_JSONL_CONTRACT.md`) so the samples can be mixed via fusion as a
summary-mode target stream.

Each image becomes one record:
  - images: [<path relative to the JSONL directory>]
  - width/height: Pre-processed dimensions (after smart_resize + pad_to_multiple)
  - objects: one dummy full-frame bbox_2d with non-empty desc
  - summary: "无关图片"

Images are pre-processed offline:
  1. Apply EXIF orientation
  2. Smart resize to fit max_pixels budget (default: 1048576)
  3. Pad to 32x32 multiples
  4. Save processed images

Default layout:
  data/irrelevant_summary/
    images/*.jpg|*.jpeg|*.png
    train.jsonl

Usage:
  python scripts/build_irrelevant_summary_jsonl.py
  python scripts/build_irrelevant_summary_jsonl.py --images-dir data/irrelevant_summary/images --output-jsonl data/irrelevant_summary/train.jsonl --max-pixels 1048576
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

from data_conversion.utils.exif_utils import apply_exif_orientation
from src.datasets.augmentation.ops import _pad_to_multiple
from src.datasets.preprocessors.resize import MIN_PIXELS, smart_resize


def _iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    paths.sort(key=lambda p: p.name)
    return paths


def _process_image(
    image_path: Path,
    output_dir: Path,
    max_pixels: int,
    image_factor: int = 32,
) -> Tuple[Path, int, int]:
    """Process image: EXIF orientation → smart_resize → pad_to_multiple → save.

    Args:
        image_path: Input image path
        output_dir: Directory to save processed images
        max_pixels: Maximum pixel budget for smart_resize
        image_factor: Factor for dimension alignment (default: 32)

    Returns:
        Tuple of (output_image_path, final_width, final_height)
    """
    # Load and apply EXIF orientation
    with Image.open(image_path) as img:
        oriented = apply_exif_orientation(img)
        if oriented.mode != "RGB":
            oriented = oriented.convert("RGB")
        base_width, base_height = oriented.size

        # Create a copy to work with outside the context manager
        # (apply_exif_orientation may return the original or a new image)
        if oriented is img:
            working_img = oriented.copy()
        else:
            working_img = oriented

    # Smart resize to fit max_pixels budget
    target_height, target_width = smart_resize(
        height=base_height,
        width=base_width,
        factor=image_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=max_pixels,
    )

    # Resize if needed
    if target_width != base_width or target_height != base_height:
        resized = working_img.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
    else:
        resized = working_img

    # Pad to 32x32 multiples
    padded = _pad_to_multiple(resized, mult=image_factor)
    final_width, final_height = padded.size

    # Save processed image
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name
    padded.save(output_path, quality=95, optimize=True)

    return output_path, int(final_width), int(final_height)


def _relpath_posix(path: Path, *, base_dir: Path) -> str:
    rel = os.path.relpath(path.resolve(), base_dir.resolve())
    return Path(rel).as_posix()


def _build_record(*, rel_image: str, width: int, height: int, summary: str) -> dict:
    return {
        "images": [rel_image],
        "objects": [{"bbox_2d": [0, 0, width, height], "desc": "irrelevant"}],
        "summary": summary,
        "width": width,
        "height": height,
    }


def build_jsonl(
    *,
    images_dir: Path,
    output_jsonl: Path,
    summary: str,
    max_pixels: int,
    image_factor: int = 32,
    overwrite_images: bool = False,
) -> Tuple[int, int]:
    image_paths = _iter_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No .jpg/.jpeg/.png images found under: {images_dir}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")

    # Determine output image directory
    # If overwrite_images, save to same directory; otherwise save alongside JSONL
    if overwrite_images:
        output_image_dir = images_dir
    else:
        output_image_dir = output_jsonl.parent / "images"
        output_image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        for image_path in image_paths:
            try:
                # Process image: EXIF → resize → pad → save
                processed_path, width, height = _process_image(
                    image_path=image_path,
                    output_dir=output_image_dir,
                    max_pixels=max_pixels,
                    image_factor=image_factor,
                )
            except Exception as exc:
                skipped += 1
                print(
                    f"[WARN] Skipping unreadable image: {image_path} ({exc})",
                    file=sys.stderr,
                )
                continue

            # Use relative path from JSONL directory
            rel_image = _relpath_posix(processed_path, base_dir=output_jsonl.parent)
            record = _build_record(
                rel_image=rel_image, width=width, height=height, summary=summary
            )
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"All images failed to load; no JSONL written (skipped={skipped})."
        )

    tmp_path.replace(output_jsonl)
    return written, skipped


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build irrelevant-image summary JSONL (summary: 无关图片)"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/irrelevant_summary/images"),
        help="Directory containing .jpg/.jpeg/.png images",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/irrelevant_summary/train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="无关图片",
        help="Summary string to write for every record",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1048576,
        help="Maximum pixel budget for smart_resize (default: 1048576 = 1024×1024)",
    )
    parser.add_argument(
        "--image-factor",
        type=int,
        default=32,
        help="Factor for dimension alignment (default: 32)",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite original images instead of saving to output directory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    written, skipped = build_jsonl(
        images_dir=args.images_dir,
        output_jsonl=args.output_jsonl,
        summary=args.summary,
        max_pixels=args.max_pixels,
        image_factor=args.image_factor,
        overwrite_images=args.overwrite_images,
    )
    print(
        f"[OK] Wrote {written} record(s) to {args.output_jsonl} "
        f"(skipped {skipped}, max_pixels={args.max_pixels})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
