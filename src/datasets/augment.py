from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance

from .geometry import (
    apply_affine,
    compose_affine,
    hflip_matrix,
    rotate_center,
    scale_matrix,
    vflip_matrix,
    points_to_xyxy,
)


@dataclass
class AugmentationConfig:
    prob: float = 0.5
    ops: List[str] = None
    max_rotate_deg: float = 10.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    color_jitter: bool = True
    brightness: Tuple[float, float] = (0.9, 1.1)
    contrast: Tuple[float, float] = (0.9, 1.1)
    saturation: Tuple[float, float] = (0.9, 1.1)

    def __post_init__(self):
        if self.ops is None:
            self.ops = ["hflip", "rotate", "scale", "color_jitter"]


def _rand_uniform(lo: float, hi: float) -> float:
    return lo + (hi - lo) * random.random()


def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_color_jitter(img: Image.Image, cfg: AugmentationConfig) -> Image.Image:
    if not cfg.color_jitter:
        return img
    b = _rand_uniform(*cfg.brightness)
    c = _rand_uniform(*cfg.contrast)
    s = _rand_uniform(*cfg.saturation)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img


def apply_augmentations(
    images: List[str | Image.Image],
    objects: Dict[str, Any],
    per_object_geoms: List[Dict[str, Any]],
    cfg: Optional[AugmentationConfig] = None,
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Apply simple, geometry-aware augmentations to images and update geometries.

    Returns:
        images_bytes: List[{"bytes": bytes}]
        objects: updated objects dict with bbox recomputed (xyxy)
        per_object_geoms: updated input geoms for JSON-lines builder
    """
    if cfg is None:
        return [{"bytes": _image_to_bytes(Image.open(p) if isinstance(p, str) else p)} for p in images], objects, per_object_geoms

    if rng is None:
        rng = random

    if rng.random() > cfg.prob:
        return [{"bytes": _image_to_bytes(Image.open(p) if isinstance(p, str) else p)} for p in images], objects, per_object_geoms

    imgs: List[Image.Image] = []
    for p in images:
        img = Image.open(p) if isinstance(p, str) else p
        if img.mode != "RGB":
            img = img.convert("RGB")
        imgs.append(img)

    width = imgs[0].width
    height = imgs[0].height

    M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for op in cfg.ops:
        if op == "hflip" and rng.random() < 0.5:
            M = compose_affine(hflip_matrix(width), M)
        elif op == "vflip" and rng.random() < 0.1:
            M = compose_affine(vflip_matrix(height), M)
        elif op == "rotate" and cfg.max_rotate_deg > 0 and rng.random() < 0.5:
            deg = _rand_uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
            M = compose_affine(rotate_center(deg, width / 2.0, height / 2.0), M)
        elif op == "scale" and cfg.scale_range != (1.0, 1.0) and rng.random() < 0.5:
            s = _rand_uniform(cfg.scale_range[0], cfg.scale_range[1])
            M = compose_affine(scale_matrix(s, s), M)

    out_imgs: List[Image.Image] = []
    for img in imgs:
        out_imgs.append(img.transform(img.size, Image.AFFINE, data=(M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2]), resample=Image.BICUBIC))

    out_imgs = [_apply_color_jitter(img, cfg) for img in out_imgs]

    # Update geometries
    updated_per_obj: List[Dict[str, Any]] = []
    for g in per_object_geoms:
        if "bbox_2d" in g:
            pts = g["bbox_2d"]
            new_pts = apply_affine(pts, M)
            g2 = {**g, "bbox_2d": new_pts}
        elif "quad" in g:
            pts = g["quad"]
            new_pts = apply_affine(pts, M)
            g2 = {**g, "quad": new_pts}
        elif "line" in g:
            pts = g["line"]
            new_pts = apply_affine(pts, M)
            g2 = {**g, "line": new_pts}
        else:
            g2 = g
        updated_per_obj.append(g2)

    # Recompute objects.bbox as xyxy per updated geometry
    if objects and objects.get("bbox") is not None:
        # rebuild from updated_per_obj rather than transforming the previous xyxy
        bbox_list: List[List[float]] = []
        for g in updated_per_obj:
            if "bbox_2d" in g:
                bbox_list.append(points_to_xyxy(g["bbox_2d"]))
            elif "quad" in g:
                bbox_list.append(points_to_xyxy(g["quad"]))
            elif "line" in g:
                bbox_list.append(points_to_xyxy(g["line"]))
        objects = {**objects, "bbox": bbox_list}

    images_bytes = [{"bytes": _image_to_bytes(img)} for img in out_imgs]
    return images_bytes, objects, updated_per_obj



