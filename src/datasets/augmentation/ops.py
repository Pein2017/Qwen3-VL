from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageOps
import numpy as np

from ...utils.logger import get_logger

from ..geometry import (
    apply_affine,
    compose_affine,
    hflip_matrix,
    rotate_center,
    vflip_matrix,
    clamp_points,
    dedupe_consecutive_points,
    invert_affine,
    scale_center,
    scale_matrix,
    sutherland_hodgman_clip,
    min_area_rect,
    to_clockwise,
    clip_polyline_to_rect,
    points_to_xyxy,
    translate,
    # Coverage and cropping utilities
    get_aabb,
    intersect_aabb,
    compute_coverage,
    translate_geometry,
)
from .base import ImageAugmenter
from .registry import register


def _pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return img


def _pad_to_multiple(img: Image.Image, *, mult: int = 32) -> Image.Image:
    w, h = img.width, img.height
    new_w = ((w + mult - 1) // mult) * mult
    new_h = ((h + mult - 1) // mult) * mult
    if new_w == w and new_h == h:
        return img
    # Use middle gray (128, 128, 128) for padding to achieve zero in normalized space.
    # Qwen3-VL normalization: (pixel/255 - 0.5) / 0.5, so 128 → ~0
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(img, (0, 0))
    return canvas


@register("hflip")
class HFlip(ImageAugmenter):
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)
        self.kind = "affine"

    def affine(self, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return None
        return hflip_matrix(width)


@register("vflip")
class VFlip(ImageAugmenter):
    def __init__(self, prob: float = 0.1):
        self.prob = float(prob)
        self.kind = "affine"

    def affine(self, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return None
        return vflip_matrix(height)


@register("rotate")
class Rotate(ImageAugmenter):
    def __init__(self, max_deg: float = 10.0, prob: float = 0.5):
        self.max_deg = float(max_deg)
        self.prob = float(prob)
        self.kind = "affine"

    def affine(self, width: int, height: int, rng: Any):
        if self.max_deg <= 0 or rng.random() >= self.prob:
            return None
        deg = rng.uniform(-self.max_deg, self.max_deg)
        # Use pixel-center pivot ((W-1)/2, (H-1)/2) to match image warp
        return rotate_center(deg, (width - 1) / 2.0, (height - 1) / 2.0)


@register("scale")
class Scale(ImageAugmenter):
    def __init__(self, lo: float = 0.9, hi: float = 1.1, prob: float = 0.5):
        self.lo = float(lo)
        self.hi = float(hi)
        self.prob = float(prob)
        self.kind = "affine"

    def affine(self, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return None
        s = rng.uniform(self.lo, self.hi)
        return scale_center(s, s, width / 2.0, height / 2.0)


@register("color_jitter")
class ColorJitter(ImageAugmenter):
    def __init__(self, brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), prob: float = 1.0):
        self.brightness = tuple(map(float, brightness))
        self.contrast = tuple(map(float, contrast))
        self.saturation = tuple(map(float, saturation))
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img)
            b = rng.uniform(self.brightness[0], self.brightness[1])
            c = rng.uniform(self.contrast[0], self.contrast[1])
            s = rng.uniform(self.saturation[0], self.saturation[1])
            im = ImageEnhance.Brightness(im).enhance(b)
            im = ImageEnhance.Contrast(im).enhance(c)
            im = ImageEnhance.Color(im).enhance(s)
            out_imgs.append(im)
        return out_imgs, geoms


@register("pad_to_multiple")
class PadToMultiple(ImageAugmenter):
    def __init__(self, multiple: int = 32):
        if multiple <= 0:
            raise ValueError(f"multiple must be > 0, got {multiple}")
        self.multiple = int(multiple)
        self.kind = "barrier"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img)
            out_imgs.append(_pad_to_multiple(im, mult=self.multiple))
        # Geometry remains the same in pixel coordinates; padding appends zeros to the right/bottom
        return out_imgs, geoms


@register("expand_to_fit_affine")
class ExpandToFitAffine(ImageAugmenter):
    """
    Barrier op: expand canvas to enclose the image after applying accumulated affines, then pad to multiple.

    Uses pre_flush_hook to modify the affine matrix and canvas dimensions before warping, ensuring
    rotated/scaled content is fully contained without cropping.

    Params:
      multiple: optional int to pad to multiple after expansion (e.g., 32). None to skip padding.
      max_pixels: maximum pixel count (width × height). If expansion would exceed this, scales down
                  proportionally to fit. Default: 921600 (960×960) to align with Qwen3-VL constraints.
    """

    def __init__(self, multiple: int | None = 32, max_pixels: int = 921600):
        self.multiple = int(multiple) if multiple else None
        self.max_pixels = int(max_pixels)
        self.kind = "barrier"

    def pre_flush_hook(self, M_total: List[List[float]], width: int, height: int, rng: Any) -> Tuple[List[List[float]], int, int]:
        """
        Compute AABB of original corners under M_total, translate to top-left origin, and pad to multiple.
        If resulting dimensions exceed max_pixels, scales down proportionally to fit within limit.
        
        Returns: (M_total_updated, new_width, new_height)
        """
        logger = get_logger("augmentation.expand")
        
        # Check if M_total is identity (skip expansion if no transform)
        is_identity = (
            abs(M_total[0][0] - 1.0) < 1e-9 and 
            abs(M_total[1][1] - 1.0) < 1e-9 and
            abs(M_total[0][1]) < 1e-9 and 
            abs(M_total[1][0]) < 1e-9 and
            abs(M_total[0][2]) < 1e-9 and
            abs(M_total[1][2]) < 1e-9
        )
        if is_identity:
            # Still apply padding to multiple if requested
            if self.multiple and self.multiple > 1:
                m = self.multiple
                new_width = ((width + m - 1) // m) * m
                new_height = ((height + m - 1) // m) * m
                # Check pixel limit even for identity
                if new_width * new_height > self.max_pixels:
                    scale = math.sqrt(self.max_pixels / (new_width * new_height))
                    new_width = int(new_width * scale)
                    new_height = int(new_height * scale)
                    # Re-align to multiple after scaling
                    new_width = ((new_width + m - 1) // m) * m
                    new_height = ((new_height + m - 1) // m) * m
                return M_total, new_width, new_height
            return M_total, width, height
        
        # Compute AABB of original corners under M_total
        corners = [0.0, 0.0, float(width - 1), 0.0, float(width - 1), float(height - 1), 0.0, float(height - 1)]
        transformed = apply_affine(corners, M_total)
        bbox = points_to_xyxy(transformed)
        minX, minY, maxX, maxY = bbox
        
        # Translate to top-left origin (keep non-negative coordinates)
        T = translate(-minX, -minY)
        M_total_updated = compose_affine(T, M_total)
        
        # Compute new dimensions
        new_width = int(math.ceil(maxX - minX + 1))
        new_height = int(math.ceil(maxY - minY + 1))
        
        # Round to multiples if requested
        if self.multiple and self.multiple > 1:
            m = self.multiple
            new_width = ((new_width + m - 1) // m) * m
            new_height = ((new_height + m - 1) // m) * m
        
        # Safety check: enforce max pixel count
        pixel_count = new_width * new_height
        if pixel_count > self.max_pixels:
            # Scale down proportionally to fit within limit
            scale_factor = math.sqrt(self.max_pixels / pixel_count)
            scaled_width = int(new_width * scale_factor)
            scaled_height = int(new_height * scale_factor)
            
            # Re-align to multiple after scaling
            if self.multiple and self.multiple > 1:
                m = self.multiple
                scaled_width = ((scaled_width + m - 1) // m) * m
                scaled_height = ((scaled_height + m - 1) // m) * m
            
            # Update affine matrix to account for the scaling
            S = scale_matrix(scale_factor, scale_factor)
            M_total_updated = compose_affine(S, M_total_updated)
            
            logger.warning(
                f"ExpandToFitAffine: Canvas expansion ({new_width}×{new_height} = {pixel_count} pixels) "
                f"exceeds max_pixels={self.max_pixels}. Scaling down to {scaled_width}×{scaled_height} "
                f"({scaled_width * scaled_height} pixels, factor={scale_factor:.3f}). "
                f"Consider reducing rotation/scale augmentation strength."
            )
            
            new_width = scaled_width
            new_height = scaled_height
        
        return M_total_updated, new_width, new_height

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        No-op: expansion already handled in pre_flush_hook.
        Images have already been warped to the expanded canvas by Compose.
        """
        return images, geoms


@register("resize_by_scale")
class ResizeByScale(ImageAugmenter):
    def __init__(self, lo: float = 0.8, hi: float = 1.2, scales: List[float] | None = None, align_multiple: int | None = 32, prob: float = 1.0):
        self.lo = float(lo)
        self.hi = float(hi)
        self.scales = [float(s) for s in (scales or [])]
        self.align_multiple = int(align_multiple) if align_multiple else None
        self.prob = float(prob)
        self.kind = "barrier"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        # choose scale
        if self.scales:
            s = self.scales[int(rng.randrange(0, len(self.scales)))]
        else:
            s = rng.uniform(self.lo, self.hi)
        new_w = max(1, int(round(width * s)))
        new_h = max(1, int(round(height * s)))
        if self.align_multiple and self.align_multiple > 1:
            m = self.align_multiple
            new_w = ((new_w + m - 1) // m) * m
            new_h = ((new_h + m - 1) // m) * m
        sx = new_w / float(width)
        sy = new_h / float(height)

        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img)
            out_imgs.append(im.resize((new_w, new_h), resample=Image.BICUBIC))

        # scale geometries in pixel space
        out_geoms: List[Dict[str, Any]] = []
        for g in geoms:
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                bb = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                bb = clamp_points(bb, new_w, new_h)
                out_geoms.append({"bbox_2d": bb})
            elif "quad" in g:
                pts = g["quad"]
                scaled: List[float] = []
                for i in range(0, len(pts), 2):
                    scaled.append(float(pts[i]) * sx)
                    scaled.append(float(pts[i + 1]) * sy)
                clipped = sutherland_hodgman_clip(scaled, new_w, new_h)
                if len(clipped) // 2 >= 3:
                    # Valid polygon after clipping
                    if len(clipped) // 2 != 4:
                        rect = min_area_rect(clipped)
                        if rect:
                            clipped = rect
                        # else: keep clipped polygon even if not exactly 4 points
                    clipped = to_clockwise(clipped)
                    q = clamp_points(clipped, new_w, new_h)
                    out_geoms.append({"quad": q})
                else:
                    # Degenerate: preserve by clamping original scaled coords
                    q = clamp_points(to_clockwise(scaled), new_w, new_h)
                    out_geoms.append({"quad": q})
            elif "line" in g:
                pts = g["line"]
                scaled: List[float] = []
                for i in range(0, len(pts), 2):
                    scaled.append(float(pts[i]) * sx)
                    scaled.append(float(pts[i + 1]) * sy)
                clipped = clip_polyline_to_rect(scaled, new_w, new_h)
                clipped = clamp_points(clipped, new_w, new_h)
                clipped = dedupe_consecutive_points(clipped)
                if len(clipped) >= 4:
                    out_geoms.append({"line": clipped})
                else:
                    # Degenerate: preserve by collapsing to minimal 2-point line
                    raw = clamp_points(scaled, new_w, new_h)
                    raw = dedupe_consecutive_points(raw)
                    if len(raw) >= 4:
                        out_geoms.append({"line": raw[:4]})
                    elif len(raw) >= 2:
                        out_geoms.append({"line": [raw[0], raw[1], raw[0], raw[1]]})
                    else:
                        # Extreme fallback: point at (0,0)
                        out_geoms.append({"line": [0.0, 0.0, 0.0, 0.0]})
            else:
                out_geoms.append(g)

        return out_imgs, out_geoms


@register("gamma")
class Gamma(ImageAugmenter):
    def __init__(self, gamma=(0.7, 1.4), gain: float = 1.0, prob: float = 1.0):
        self.gamma = (float(gamma[0]), float(gamma[1]))
        self.gain = float(gain)
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        g = max(1e-6, rng.uniform(self.gamma[0], self.gamma[1]))
        lut = [min(255, max(0, int(round(self.gain * (i / 255.0) ** g * 255.0)))) for i in range(256)]
        lut = lut * 3  # apply to R,G,B equally
        for img in images:
            im = _pil(img).convert("RGB").point(lut)
            out_imgs.append(im)
        return out_imgs, geoms


@register("hsv")
class HueSaturationValue(ImageAugmenter):
    def __init__(self, hue_delta_deg=(-20.0, 20.0), sat=(0.7, 1.4), val=(0.7, 1.4), prob: float = 1.0):
        self.hue_delta_deg = (float(hue_delta_deg[0]), float(hue_delta_deg[1]))
        self.sat = (float(sat[0]), float(sat[1]))
        self.val = (float(val[0]), float(val[1]))
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        hue_deg = rng.uniform(self.hue_delta_deg[0], self.hue_delta_deg[1])
        hue_delta = int(round(hue_deg / 360.0 * 255.0))
        sat_scale = rng.uniform(self.sat[0], self.sat[1])
        val_scale = rng.uniform(self.val[0], self.val[1])
        for img in images:
            im = _pil(img).convert("HSV")
            h, s, v = im.split()
            h_arr = np.asarray(h, dtype=np.int16)
            s_arr = np.asarray(s, dtype=np.float32)
            v_arr = np.asarray(v, dtype=np.float32)
            h_arr = (h_arr + hue_delta) % 256
            s_arr = np.clip(s_arr * sat_scale, 0, 255).astype(np.uint8)
            v_arr = np.clip(v_arr * val_scale, 0, 255).astype(np.uint8)
            h_img = Image.fromarray(h_arr.astype(np.uint8), mode="L")
            s_img = Image.fromarray(s_arr, mode="L")
            v_img = Image.fromarray(v_arr, mode="L")
            out = Image.merge("HSV", (h_img, s_img, v_img)).convert("RGB")
            out_imgs.append(out)
        return out_imgs, geoms


@register("clahe")
class CLAHE(ImageAugmenter):
    def __init__(self, clip_limit: float = 3.0, tile_grid_size=(8, 8), prob: float = 0.5):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError("CLAHE requires opencv-python-headless installed in the 'ms' environment") from e
        out_imgs: List[Any] = []
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        for img in images:
            im = _pil(img).convert("RGB")
            arr = np.asarray(im)
            bgr = arr[:, :, ::-1]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            bgr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            rgb = bgr2[:, :, ::-1]
            out_imgs.append(Image.fromarray(rgb))
        return out_imgs, geoms


@register("auto_contrast")
class AutoContrast(ImageAugmenter):
    def __init__(self, cutoff: int = 0, prob: float = 0.5):
        self.cutoff = int(cutoff)
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageOps.autocontrast(im, cutoff=self.cutoff))
        return out_imgs, geoms


@register("solarize")
class Solarize(ImageAugmenter):
    def __init__(self, threshold: int = 128, prob: float = 0.3):
        self.threshold = int(threshold)
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageOps.solarize(im, threshold=self.threshold))
        return out_imgs, geoms


@register("posterize")
class Posterize(ImageAugmenter):
    def __init__(self, bits: int = 4, prob: float = 0.3):
        self.bits = int(bits)
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageOps.posterize(im, bits=self.bits))
        return out_imgs, geoms


@register("sharpness")
class Sharpness(ImageAugmenter):
    def __init__(self, factor=(0.5, 1.8), prob: float = 0.4):
        self.factor = (float(factor[0]), float(factor[1]))
        self.prob = float(prob)
        self.kind = "color"

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        f = rng.uniform(self.factor[0], self.factor[1])
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageEnhance.Sharpness(im).enhance(f))
        return out_imgs, geoms


@register("albumentations_color")
class AlbumentationsColor(ImageAugmenter):
    def __init__(self, preset: str = "strong", prob: float = 1.0):
        self.preset = str(preset)
        self.prob = float(prob)
        self.kind = "color"

    def _build_pipeline(self):
        try:
            import albumentations as A  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "AlbumentationsColor requires 'albumentations'. Install it with 'opencv-python-headless' in the 'ms' env."
            ) from e
        if self.preset == "strong":
            return A.ReplayCompose([
                A.OneOf([
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
                    A.RandomGamma(gamma_limit=(60, 180), p=1.0),
                ], p=0.9),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.7),
                A.OneOf([
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                    A.Equalize(p=1.0),
                ], p=0.5),
                A.ChannelShuffle(p=0.1),
            ], p=1.0)
        elif self.preset == "extreme":
            return A.ReplayCompose([
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, p=0.9),
                A.RandomGamma(gamma_limit=(40, 220), p=0.7),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=40, p=0.9),
                A.OneOf([
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.Equalize(p=1.0),
                    A.Solarize(threshold=128, p=1.0),
                    A.Posterize(num_bits=4, p=1.0),
                ], p=0.7),
                A.ChannelShuffle(p=0.2),
            ], p=1.0)
        else:
            raise ValueError(f"Unknown albumentations preset: {self.preset}")

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        try:
            import numpy as _np  # type: ignore
            import cv2  # type: ignore
            import albumentations as A  # type: ignore
            import random as _random
        except Exception as e:
            raise RuntimeError(
                "AlbumentationsColor requires 'albumentations' and 'opencv-python-headless' in the 'ms' env."
            ) from e

        pipeline = self._build_pipeline()

        # Deterministic per-call across images
        seed = rng.randint(0, 2**31 - 1)
        py_state = _random.getstate()
        np_state = _np.random.get_state()
        _random.seed(seed)
        _np.random.seed(seed)
        try:
            first = _pil(images[0]).convert("RGB")
            bgr0 = _np.asarray(first)[:, :, ::-1]
            res0 = pipeline(image=bgr0)
            replay = res0["replay"]
            out_imgs: List[Any] = [Image.fromarray(res0["image"][:, :, ::-1])]
            for img in images[1:]:
                im = _pil(img).convert("RGB")
                bgr = _np.asarray(im)[:, :, ::-1]
                res = A.ReplayCompose.replay(replay, image=bgr)
                out_imgs.append(Image.fromarray(res["image"][:, :, ::-1]))
        finally:
            _random.setstate(py_state)
            _np.random.set_state(np_state)
        return out_imgs, geoms


@register("random_crop")
class RandomCrop(ImageAugmenter):
    """
    Random crop with label filtering for dense captioning.
    
    Crops a random region from the image and filters geometries based on visibility.
    If filtered objects < min_objects or any filtered object is a line (when skip_if_line=True),
    the entire crop operation is skipped and original images/geometries are returned unchanged.
    
    Parameters:
        scale: (min, max) tuple for crop size as fraction of original image (default: (0.6, 1.0))
        aspect_ratio: (min, max) tuple for aspect ratio variation (default: (0.8, 1.2))
        min_coverage: minimum coverage ratio to keep object (default: 0.3)
        completeness_threshold: coverage threshold for marking "只显示部分" (default: 0.95)
        min_objects: skip crop if filtered objects < this value (default: 4)
        skip_if_line: skip crop if any filtered object is a line (default: True)
        prob: probability of applying crop (default: 1.0)
    """
    def __init__(
        self,
        scale: Tuple[float, float] = (0.6, 1.0),
        aspect_ratio: Tuple[float, float] = (0.8, 1.2),
        min_coverage: float = 0.3,
        completeness_threshold: float = 0.95,
        min_objects: int = 4,
        skip_if_line: bool = True,
        prob: float = 1.0,
    ):
        self.scale = (float(scale[0]), float(scale[1]))
        self.aspect_ratio = (float(aspect_ratio[0]), float(aspect_ratio[1]))
        self.min_coverage = float(min_coverage)
        self.completeness_threshold = float(completeness_threshold)
        self.min_objects = int(min_objects)
        self.skip_if_line = bool(skip_if_line)
        self.prob = float(prob)
        self.kind = "barrier"
        
        # Metadata storage (set by apply, read by preprocessor)
        self.last_kept_indices: List[int] | None = None
        self.last_object_coverages: List[float] | None = None
        self.allows_geometry_drops = True  # Signal to validation
    
    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        logger = get_logger("augmentation.random_crop")
        
        # Clear metadata from previous call
        self.last_kept_indices = None
        self.last_object_coverages = None
        
        if rng.random() >= self.prob:
            return images, geoms
        
        # Sample crop size
        crop_scale = rng.uniform(self.scale[0], self.scale[1])
        aspect = rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        
        # Compute crop dimensions
        base_size = math.sqrt(width * height * crop_scale)
        crop_w = max(1, int(round(base_size * math.sqrt(aspect))))
        crop_h = max(1, int(round(base_size / math.sqrt(aspect))))
        
        # Clamp to image bounds
        crop_w = min(crop_w, width)
        crop_h = min(crop_h, height)
        
        # Sample random position
        max_x = width - crop_w
        max_y = height - crop_h
        crop_x = int(rng.randrange(0, max_x + 1)) if max_x > 0 else 0
        crop_y = int(rng.randrange(0, max_y + 1)) if max_y > 0 else 0
        
        crop_bbox = [float(crop_x), float(crop_y), float(crop_x + crop_w), float(crop_y + crop_h)]
        
        # Compute coverage and filter geometries
        kept_indices: List[int] = []
        coverages: List[float] = []
        filtered_geoms: List[Dict[str, Any]] = []
        
        for idx, g in enumerate(geoms):
            cov = compute_coverage(g, crop_bbox)
            if cov >= self.min_coverage:
                kept_indices.append(idx)
                coverages.append(cov)
                filtered_geoms.append(g)
        
        # Check skip conditions
        if len(filtered_geoms) < self.min_objects:
            logger.debug(f"Crop would filter to {len(filtered_geoms)} < {self.min_objects} objects. Skipping crop.")
            return images, geoms
        
        if self.skip_if_line:
            has_line = any("line" in g for g in filtered_geoms)
            if has_line:
                logger.debug("Crop region contains line object. Skipping crop to preserve cable/fiber integrity.")
                return images, geoms
        
        # Proceed with crop - truncate and translate geometries
        cropped_geoms: List[Dict[str, Any]] = []
        for g in filtered_geoms:
            # Truncate geometry to crop boundary
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                # Clip to crop region
                clipped_x1 = max(crop_bbox[0], min(crop_bbox[2], x1))
                clipped_y1 = max(crop_bbox[1], min(crop_bbox[3], y1))
                clipped_x2 = max(crop_bbox[0], min(crop_bbox[2], x2))
                clipped_y2 = max(crop_bbox[1], min(crop_bbox[3], y2))
                truncated = {"bbox_2d": [clipped_x1, clipped_y1, clipped_x2, clipped_y2]}
            elif "quad" in g:
                pts = g["quad"]
                # Translate to crop origin for clipping
                pts_translated = []
                for i in range(0, len(pts), 2):
                    pts_translated.append(pts[i] - crop_bbox[0])
                    pts_translated.append(pts[i + 1] - crop_bbox[1])
                clipped = sutherland_hodgman_clip(pts_translated, crop_w, crop_h)
                # Reduce redundant vertices introduced by axis-aligned clipping
                from ..geometry import simplify_polygon, choose_four_corners
                clipped = simplify_polygon(clipped)
                
                if len(clipped) // 2 >= 3:
                    # Prefer true 4-corner representation when possible
                    if len(clipped) // 2 > 4:
                        best4 = choose_four_corners(clipped)
                        if best4:
                            clipped = best4
                    if len(clipped) // 2 != 4:
                        rect = min_area_rect(clipped)
                        if rect:
                            clipped = rect
                    clipped = to_clockwise(clipped)
                    # Translate back to image coords for final translation step
                    final_pts = []
                    for i in range(0, len(clipped), 2):
                        final_pts.append(clipped[i] + crop_bbox[0])
                        final_pts.append(clipped[i + 1] + crop_bbox[1])
                    truncated = {"quad": final_pts}
                else:
                    # Degenerate quad - clamp to crop boundary
                    clamped = []
                    for i in range(0, len(pts), 2):
                        clamped.append(max(crop_bbox[0], min(crop_bbox[2], pts[i])))
                        clamped.append(max(crop_bbox[1], min(crop_bbox[3], pts[i + 1])))
                    truncated = {"quad": clamped}
            elif "line" in g:
                pts = g["line"]
                # Translate to crop origin
                pts_translated = []
                for i in range(0, len(pts), 2):
                    pts_translated.append(pts[i] - crop_bbox[0])
                    pts_translated.append(pts[i + 1] - crop_bbox[1])
                # Clip line to crop boundary
                clipped = clip_polyline_to_rect(pts_translated, crop_w, crop_h)
                # Translate back
                final_pts = []
                for i in range(0, len(clipped), 2):
                    final_pts.append(clipped[i] + crop_bbox[0])
                    final_pts.append(clipped[i + 1] + crop_bbox[1])
                
                if len(final_pts) >= 4:
                    truncated = {"line": final_pts}
                else:
                    # Degenerate line - clamp to crop boundary
                    clamped = []
                    for i in range(0, len(pts), 2):
                        clamped.append(max(crop_bbox[0], min(crop_bbox[2], pts[i])))
                        clamped.append(max(crop_bbox[1], min(crop_bbox[3], pts[i + 1])))
                    if len(clamped) >= 4:
                        truncated = {"line": clamped[:4]}
                    else:
                        truncated = {"line": [crop_bbox[0], crop_bbox[1], crop_bbox[0], crop_bbox[1]]}
            else:
                truncated = g
            
            # Translate to crop coordinates
            translated = translate_geometry(truncated, -crop_bbox[0], -crop_bbox[1])
            cropped_geoms.append(translated)
        
        # Crop images
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img)
            out_imgs.append(im.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)))
        
        # Store metadata for preprocessor
        self.last_kept_indices = kept_indices
        self.last_object_coverages = coverages
        
        logger.debug(f"Crop applied: {len(geoms)} → {len(cropped_geoms)} objects (region: {crop_bbox})")
        return out_imgs, cropped_geoms


__all__ = [
    "HFlip",
    "VFlip",
    "Rotate",
    "Scale",
    "ColorJitter",
    "Gamma",
    "HueSaturationValue",
    "CLAHE",
    "AutoContrast",
    "Solarize",
    "Posterize",
    "Sharpness",
    "AlbumentationsColor",
    "PadToMultiple",
    "RandomCrop",
]


