from __future__ import annotations

import io
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from ..utils.logger import get_logger


def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _capture_crop_metadata(pipeline: Any) -> Optional[Dict[str, Any]]:
    kept = getattr(pipeline, "last_kept_indices", None)
    if kept is None:
        return None
    width = getattr(pipeline, "last_image_width", None)
    height = getattr(pipeline, "last_image_height", None)
    padding_ratio = getattr(pipeline, "last_padding_ratio", None)
    return {
        "kept_indices": list(kept),
        "coverages": list(getattr(pipeline, "last_object_coverages", []) or []),
        "allows_geometry_drops": bool(getattr(pipeline, "allows_geometry_drops", False)),
        "width": width,
        "height": height,
        "padding_ratio": padding_ratio,
        "skip_reason": getattr(pipeline, "last_crop_skip_reason", None),
        "skip_counts": getattr(pipeline, "last_skip_counters", {}),
    }


def apply_augmentations(
    images: List[str | Image.Image],
    per_object_geoms: List[Dict[str, Any]],
    pipeline: Any,
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Plugin-based augmentation wrapper (back-compatible images-bytes output).

    Args:
        images: paths or PIL images
        per_object_geoms: geometry dicts for objects
        pipeline: instance of Compose (or any ImageAugmenter) with .apply
        rng: optional RNG

    Returns:
        images_bytes, updated_per_obj
    """
    if pipeline is None or not hasattr(pipeline, "apply"):
        raise TypeError("augmentation pipeline must provide an 'apply' method")

    if rng is None:
        rng = random

    pil_images: List[Image.Image] = []
    for p in images:
        if isinstance(p, str):
            path = p
            if not os.path.isabs(path):
                base = os.environ.get('ROOT_IMAGE_DIR')
                if not base:
                    raise FileNotFoundError(
                        f"Relative image path '{p}' cannot be resolved: ROOT_IMAGE_DIR is not set. "
                        f"Set ROOT_IMAGE_DIR to the directory of your JSONL (auto-set by sft.py) or use absolute paths."
                    )
                path = os.path.join(base, path)
            im = Image.open(path)
        else:
            im = p
        if not isinstance(im, Image.Image):
            raise TypeError("images must be file paths or PIL.Image instances")
        if im.mode != "RGB":
            im = im.convert("RGB")
        pil_images.append(im)
    if not pil_images:
        return [], per_object_geoms
    base_w, base_h = pil_images[0].width, pil_images[0].height
    for i, im in enumerate(pil_images[1:], start=1):
        if im.width != base_w or im.height != base_h:
            raise ValueError(
                f"All images in a sample must share identical size; expected {base_w}x{base_h}, found image[{i}]={im.width}x{im.height}"
            )
    out_imgs, geoms = pipeline.apply(pil_images, per_object_geoms, width=base_w, height=base_h, rng=rng)
    if not isinstance(out_imgs, list) or not all(isinstance(i, Image.Image) for i in out_imgs):
        raise TypeError("pipeline.apply must return list[Image.Image] as first element")
    if not isinstance(geoms, list):
        raise TypeError("pipeline.apply must return list[dict] as second element")
    if len(out_imgs) != len(pil_images):
        raise ValueError(f"pipeline.apply returned {len(out_imgs)} images, expected {len(pil_images)}")
    # Check if pipeline allows geometry drops (from crop operations)
    allows_drops = getattr(pipeline, 'allows_geometry_drops', False)
    
    if len(geoms) != len(per_object_geoms):
        if not allows_drops:
            raise ValueError(f"pipeline.apply returned {len(geoms)} geometries, expected {len(per_object_geoms)}")
        # Crop operation: log but allow count change
        logger = get_logger("augmentation.validation")
        logger.debug(f"Crop filtered {len(per_object_geoms)} → {len(geoms)} objects")
    # Validate that all output images share the same size; allow size change (e.g., padding to multiples)
    out_w, out_h = out_imgs[0].width, out_imgs[0].height
    for i, im in enumerate(out_imgs):
        if im.width != out_w or im.height != out_h:
            raise ValueError(
                f"augmentation pipeline must produce images with identical size; image[0]={out_w}x{out_h}, image[{i}]={im.width}x{im.height}"
            )
    images_bytes = [{"bytes": _image_to_bytes(img)} for img in out_imgs]

    # Attach telemetry for downstream debug consumers
    crop_meta = _capture_crop_metadata(pipeline)
    if crop_meta is not None:
        logger = get_logger("augmentation.telemetry")
        logger.debug(
            "Crop telemetry: kept=%s coverages=%s drops=%s size=%s padding=%s skip=%s counts=%s",
            crop_meta["kept_indices"],
            [round(float(c), 4) for c in crop_meta["coverages"]],
            crop_meta["allows_geometry_drops"],
            (crop_meta.get("width"), crop_meta.get("height")),
            crop_meta.get("padding_ratio"),
            crop_meta.get("skip_reason"),
            crop_meta.get("skip_counts"),
        )
        # expose for preprocessors (pipeline metadata already set, but ensure Compose sees latest)
        setattr(pipeline, "last_crop_summary", crop_meta)

    return images_bytes, geoms



