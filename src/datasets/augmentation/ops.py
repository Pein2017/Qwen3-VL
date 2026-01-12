from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, cast, final, override

from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import numpy as np
from numpy.typing import NDArray

from ...utils.logger import get_logger
from ..contracts import DatasetObject
from ..geometry import (
    apply_affine,
    compose_affine,
    hflip_matrix,
    rotate_center,
    vflip_matrix,
    clamp_points,
    dedupe_consecutive_points,
    scale_center,
    scale_matrix,
    sutherland_hodgman_clip,
    min_area_rect,
    to_clockwise,
    clip_polyline_to_rect,
    points_to_xyxy,
    translate,
    transform_geometry,
    # Coverage and cropping utilities
    get_aabb,
    intersect_aabb,
    aabb_area,
    compute_polygon_coverage,
    translate_geometry,
)
from .base import (
    AffineOp,
    ColorOp,
    PatchOp,
    CurriculumMixin,
    ImageAugmenter,
    RngLike,
)
from .registry import register


class _Resample(Protocol):
    BICUBIC: int


_PIL_RESAMPLE = cast(_Resample, getattr(Image, "Resampling", Image))
_PIL_BICUBIC = _PIL_RESAMPLE.BICUBIC


def _pil(img: Image.Image) -> Image.Image:
    return img


def _pad_to_multiple(img: Image.Image, *, mult: int = 32) -> Image.Image:
    w, h = img.width, img.height
    new_w = ((w + mult - 1) // mult) * mult
    new_h = ((h + mult - 1) // mult) * mult
    # Ensure dimensions are integers (PIL requires int, not float)
    new_w = int(new_w)
    new_h = int(new_h)
    if new_w == w and new_h == h:
        return img
    # Use middle gray (128, 128, 128) for padding to achieve zero in normalized space.
    # Qwen3-VL normalization: (pixel/255 - 0.5) / 0.5, so 128 → ~0
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(img, (0, 0))
    return canvas


@register("hflip")
@final
class HFlip(AffineOp):
    curriculum_param_names = ("prob",)

    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    @override
    def affine(self, width: int, height: int, rng: RngLike):
        if rng.random() >= self.prob:
            return None
        return hflip_matrix(width)


@register("vflip")
@final
class VFlip(AffineOp):
    curriculum_param_names = ("prob",)

    def __init__(self, prob: float = 0.1):
        self.prob = float(prob)

    @override
    def affine(self, width: int, height: int, rng: RngLike):
        if rng.random() >= self.prob:
            return None
        return vflip_matrix(height)


@register("rotate")
@final
class Rotate(AffineOp):
    curriculum_param_names = ("max_deg", "prob")

    def __init__(self, max_deg: float = 10.0, prob: float = 0.5):
        self.max_deg = float(max_deg)
        self.prob = float(prob)

    @override
    def affine(self, width: int, height: int, rng: RngLike):
        if self.max_deg <= 0 or rng.random() >= self.prob:
            return None
        deg = rng.uniform(-self.max_deg, self.max_deg)
        # Use pixel-center pivot ((W-1)/2, (H-1)/2) to match image warp
        return rotate_center(deg, (width - 1) / 2.0, (height - 1) / 2.0)


@register("scale")
@final
class Scale(AffineOp):
    curriculum_param_names = ("lo", "hi", "prob")

    def __init__(self, lo: float = 0.9, hi: float = 1.1, prob: float = 0.5):
        self.lo = float(lo)
        self.hi = float(hi)
        self.prob = float(prob)

    @override
    def affine(self, width: int, height: int, rng: RngLike):
        if rng.random() >= self.prob:
            return None
        s = rng.uniform(self.lo, self.hi)
        return scale_center(s, s, width / 2.0, height / 2.0)


@register("color_jitter")
@final
class ColorJitter(ColorOp):
    curriculum_param_names = ("brightness", "contrast", "saturation", "prob")

    def __init__(
        self,
        brightness: Sequence[float] = (0.7, 1.3),
        contrast: Sequence[float] = (0.7, 1.3),
        saturation: Sequence[float] = (0.7, 1.3),
        prob: float = 1.0,
    ):
        self.brightness = tuple(map(float, brightness))
        self.contrast = tuple(map(float, contrast))
        self.saturation = tuple(map(float, saturation))
        self.prob = float(prob)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: list[Image.Image] = []
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


@register("expand_to_fit_affine")
@final
class ExpandToFitAffine(CurriculumMixin, ImageAugmenter):
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
        self.force_flush_affine = True
        names = ["max_pixels"]
        if self.multiple is not None:
            names.insert(0, "multiple")
        self.curriculum_param_names = tuple(names)

    def pre_flush_hook(
        self, M_total: list[list[float]], width: int, height: int, rng: RngLike
    ) -> tuple[list[list[float]], int, int]:
        """
        Compute AABB of original corners under M_total, translate to top-left origin, and pad to multiple.
        If resulting dimensions exceed max_pixels, scales down proportionally to fit within limit.

        Returns: (M_total_updated, new_width, new_height)
        """
        logger = get_logger("augmentation.expand")

        def _align_and_cap(
            canvas_w: int, canvas_h: int
        ) -> tuple[int, int, int, int, float, float, int, int, bool]:
            if canvas_w <= 0 or canvas_h <= 0:
                raise ValueError(
                    f"ExpandToFitAffine received non-positive dimensions ({canvas_w}, {canvas_h})."
                )

            aligned_w, aligned_h = canvas_w, canvas_h
            if self.multiple and self.multiple > 1:
                m = self.multiple
                aligned_w = ((aligned_w + m - 1) // m) * m
                aligned_h = ((aligned_h + m - 1) // m) * m

            initial_pixels = aligned_w * aligned_h
            scale_x = 1.0
            scale_y = 1.0
            final_w = aligned_w
            final_h = aligned_h
            scaled = False

            if initial_pixels > self.max_pixels:
                scaled = True
                scale = math.sqrt(self.max_pixels / initial_pixels)
                target_w = aligned_w * scale
                target_h = aligned_h * scale

                if self.multiple and self.multiple > 1:
                    m = self.multiple
                    floor_w = int(math.floor(target_w))
                    floor_h = int(math.floor(target_h))
                    floor_w = max(m, (floor_w // m) * m)
                    floor_h = max(m, (floor_h // m) * m)
                    if floor_w == 0:
                        floor_w = m
                    if floor_h == 0:
                        floor_h = m
                    pixel_count = floor_w * floor_h
                    while pixel_count > self.max_pixels and floor_w > m:
                        floor_w -= m
                        pixel_count = floor_w * floor_h
                    while pixel_count > self.max_pixels and floor_h > m:
                        floor_h -= m
                        pixel_count = floor_w * floor_h
                    final_w, final_h = floor_w, floor_h
                else:
                    floor_w = max(1, int(math.floor(target_w)))
                    floor_h = max(1, int(math.floor(target_h)))
                    pixel_count = floor_w * floor_h
                    while pixel_count > self.max_pixels and floor_w > 1:
                        floor_w -= 1
                        pixel_count = floor_w * floor_h
                    while pixel_count > self.max_pixels and floor_h > 1:
                        floor_h -= 1
                        pixel_count = floor_w * floor_h
                    final_w, final_h = floor_w, floor_h

                if final_w <= 0 or final_h <= 0:
                    raise ValueError(
                        "ExpandToFitAffine could not enforce positive canvas dimensions after scaling."
                    )

                final_pixels = final_w * final_h
                if final_pixels > self.max_pixels:
                    raise ValueError(
                        f"ExpandToFitAffine could not enforce max_pixels={self.max_pixels}: "
                        f"final canvas {final_w}×{final_h} ({final_pixels} pixels)."
                    )

                scale_x = final_w / aligned_w
                scale_y = final_h / aligned_h
            else:
                final_pixels = initial_pixels

            return (
                aligned_w,
                aligned_h,
                final_w,
                final_h,
                scale_x,
                scale_y,
                initial_pixels,
                final_pixels,
                scaled,
            )

        # Check if M_total is identity (skip expansion if no transform)
        is_identity = (
            abs(M_total[0][0] - 1.0) < 1e-9
            and abs(M_total[1][1] - 1.0) < 1e-9
            and abs(M_total[0][1]) < 1e-9
            and abs(M_total[1][0]) < 1e-9
            and abs(M_total[0][2]) < 1e-9
            and abs(M_total[1][2]) < 1e-9
        )
        if is_identity:
            (
                aligned_w,
                aligned_h,
                new_width,
                new_height,
                scale_x,
                scale_y,
                original_pixels,
                final_pixels,
                scaled,
            ) = _align_and_cap(width, height)
            if scaled:
                S = scale_matrix(scale_x, scale_y)
                M_total = compose_affine(S, M_total)
                logger.debug(
                    "ExpandToFitAffine: Canvas ({0}×{1} = {2} pixels) exceeds max_pixels={3}. "
                    "Scaled down to {4}×{5} = {6} pixels (scale_x={7:.4f}, scale_y={8:.4f}). "
                    "Consider reducing rotation/scale augmentation strength.".format(
                        aligned_w,
                        aligned_h,
                        original_pixels,
                        self.max_pixels,
                        new_width,
                        new_height,
                        final_pixels,
                        scale_x,
                        scale_y,
                    )
                )
            total_pixels = max(1, new_width * new_height)
            padded_pixels = max(0.0, float(total_pixels) - float(width * height))
            self.padding_ratio = padded_pixels / float(total_pixels)
            # Ensure dimensions are integers (PIL requires int, not float)
            return M_total, int(new_width), int(new_height)

        # Compute AABB of original corners under M_total
        corners = [
            0.0,
            0.0,
            float(width - 1),
            0.0,
            float(width - 1),
            float(height - 1),
            0.0,
            float(height - 1),
        ]
        transformed = apply_affine(corners, M_total)
        bbox = points_to_xyxy(transformed)
        minX, minY, maxX, maxY = bbox

        # Translate to top-left origin (keep non-negative coordinates)
        T = translate(-minX, -minY)
        M_total_updated = compose_affine(T, M_total)

        # Compute new dimensions and enforce alignment/pixel cap
        raw_width = int(math.ceil(maxX - minX + 1))
        raw_height = int(math.ceil(maxY - minY + 1))
        (
            aligned_w,
            aligned_h,
            new_width,
            new_height,
            scale_x,
            scale_y,
            original_pixels,
            final_pixels,
            scaled,
        ) = _align_and_cap(raw_width, raw_height)

        self.padding_ratio = 0.0

        if scaled:
            scale_mat = scale_matrix(scale_x, scale_y)
            M_total_updated = compose_affine(scale_mat, M_total_updated)
            logger.debug(
                "ExpandToFitAffine: Canvas expansion ({0}×{1} = {2} pixels) exceeds max_pixels={3}. "
                "Scaled down to {4}×{5} = {6} pixels (scale_x={7:.4f}, scale_y={8:.4f}). "
                "Consider reducing rotation/scale augmentation strength.".format(
                    aligned_w,
                    aligned_h,
                    original_pixels,
                    self.max_pixels,
                    new_width,
                    new_height,
                    final_pixels,
                    scale_x,
                    scale_y,
                )
            )

        total_pixels = max(1, new_width * new_height)
        padded_pixels = max(0.0, float(total_pixels) - float(raw_width * raw_height))
        self.padding_ratio = padded_pixels / float(total_pixels)

        # Ensure dimensions are integers (PIL requires int, not float)
        return M_total_updated, int(new_width), int(new_height)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ) -> tuple[list[Image.Image], list[DatasetObject]]:
        """
        No-op: expansion already handled in pre_flush_hook.
        Images have already been warped to the expanded canvas by Compose.
        """
        return images, geoms


@register("resize_by_scale")
@final
class ResizeByScale(CurriculumMixin, ImageAugmenter):
    def __init__(
        self,
        lo: float = 0.8,
        hi: float = 1.2,
        scales: list[float] | None = None,
        align_multiple: int | None = 32,
        prob: float = 1.0,
    ):
        self.lo = float(lo)
        self.hi = float(hi)
        self.scales = [float(s) for s in (scales or [])]
        self.align_multiple = int(align_multiple) if align_multiple else None
        self.prob = float(prob)
        self.kind = "barrier"
        self.curriculum_param_names = ("lo", "hi", "prob")

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        # Ensure input dimensions are integers
        width = int(width)
        height = int(height)
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
        # Ensure dimensions are integers (PIL requires int, not float)
        new_w = int(new_w)
        new_h = int(new_h)
        sx = new_w / float(width)
        sy = new_h / float(height)

        out_imgs: list[Image.Image] = []
        for img in images:
            im = _pil(img)
            out_imgs.append(im.resize((new_w, new_h), resample=_PIL_BICUBIC))

        # scale geometries in pixel space
        out_geoms: list[DatasetObject] = []
        for g in geoms:
            meta = {k: v for k, v in g.items() if k not in {"bbox_2d", "poly", "line"}}
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                bb = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                bb = clamp_points(bb, new_w, new_h)
                out_geoms.append(cast(DatasetObject, {**meta, "bbox_2d": bb}))
            elif "poly" in g:
                pts = g["poly"]
                scaled_poly: list[float] = []
                for i in range(0, len(pts), 2):
                    scaled_poly.append(float(pts[i]) * sx)
                    scaled_poly.append(float(pts[i + 1]) * sy)
                clipped = sutherland_hodgman_clip(scaled_poly, new_w, new_h)
                if len(clipped) // 2 >= 3:
                    # Valid polygon after clipping
                    if len(clipped) // 2 != 4:
                        rect = min_area_rect(clipped)
                        if rect:
                            clipped = rect
                        # else: keep clipped polygon even if not exactly 4 points
                    clipped = to_clockwise(clipped)
                    q = clamp_points(clipped, new_w, new_h)
                    out_geoms.append(cast(DatasetObject, {**meta, "poly": q}))
                else:
                    # Degenerate: preserve by clamping original scaled coords
                    q = clamp_points(to_clockwise(scaled_poly), new_w, new_h)
                    out_geoms.append(cast(DatasetObject, {**meta, "poly": q}))
            elif "line" in g:
                pts = g["line"]
                scaled_line: list[float] = []
                for i in range(0, len(pts), 2):
                    scaled_line.append(float(pts[i]) * sx)
                    scaled_line.append(float(pts[i + 1]) * sy)
                clipped = clip_polyline_to_rect(scaled_line, new_w, new_h)
                clipped = clamp_points(clipped, new_w, new_h)
                clipped = dedupe_consecutive_points(clipped)
                if len(clipped) >= 4:
                    out_geoms.append(cast(DatasetObject, {**meta, "line": clipped}))
                else:
                    # Degenerate: preserve by collapsing to minimal 2-point line
                    raw = clamp_points(scaled_line, new_w, new_h)
                    raw = dedupe_consecutive_points(raw)
                    if len(raw) >= 4:
                        out_geoms.append(cast(DatasetObject, {**meta, "line": raw[:4]}))
                    elif len(raw) >= 2:
                        out_geoms.append(
                            cast(
                                DatasetObject,
                                {**meta, "line": [raw[0], raw[1], raw[0], raw[1]]},
                            )
                        )
                    else:
                        # Extreme fallback: point at (0,0)
                        out_geoms.append(
                            cast(DatasetObject, {**meta, "line": [0.0, 0.0, 0.0, 0.0]})
                        )
            else:
                out_geoms.append(g)

        return out_imgs, out_geoms


@register("gamma")
@final
class Gamma(ColorOp):
    curriculum_param_names = ("gamma", "gain", "prob")

    def __init__(
        self,
        gamma: Sequence[float] = (0.7, 1.4),
        gain: float = 1.0,
        prob: float = 1.0,
    ):
        self.gamma = (float(gamma[0]), float(gamma[1]))
        self.gain = float(gain)
        self.prob = float(prob)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: list[Image.Image] = []
        g = max(1e-6, rng.uniform(self.gamma[0], self.gamma[1]))
        lut = [
            min(255, max(0, int(round(self.gain * (i / 255.0) ** g * 255.0))))
            for i in range(256)
        ]
        lut = lut * 3  # apply to R,G,B equally
        for img in images:
            im = _pil(img).convert("RGB").point(lut)
            out_imgs.append(im)
        return out_imgs, geoms


@register("hsv")
@final
class HueSaturationValue(ColorOp):
    curriculum_param_names = ("hue_delta_deg", "sat", "val", "prob")

    def __init__(
        self,
        hue_delta_deg: Sequence[float] = (-20.0, 20.0),
        sat: Sequence[float] = (0.7, 1.4),
        val: Sequence[float] = (0.7, 1.4),
        prob: float = 1.0,
    ):
        self.hue_delta_deg = (float(hue_delta_deg[0]), float(hue_delta_deg[1]))
        self.sat = (float(sat[0]), float(sat[1]))
        self.val = (float(val[0]), float(val[1]))
        self.prob = float(prob)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: list[Image.Image] = []
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
@final
class CLAHE(ColorOp):
    curriculum_param_names = ("clip_limit", "prob")

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_grid_size: Sequence[int] = (8, 8),
        prob: float = 0.5,
    ):
        self.clip_limit = float(clip_limit)
        # Ensure tile_grid_size is a tuple of integers (OpenCV requires tuple, not list)
        if isinstance(tile_grid_size, (list, tuple)):
            self.tile_grid_size = tuple(int(x) for x in tile_grid_size)
        else:
            raise TypeError(
                f"tile_grid_size must be a list or tuple, got {type(tile_grid_size)}"
            )
        if len(self.tile_grid_size) != 2:
            raise ValueError(
                f"tile_grid_size must have 2 elements, got {len(self.tile_grid_size)}"
            )
        self.prob = float(prob)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "CLAHE requires opencv-python-headless installed in the 'ms' environment"
            ) from e
        out_imgs: list[Image.Image] = []
        # Ensure tile_grid_size is a tuple of integers for OpenCV (explicit conversion)
        tile_size = tuple(int(x) for x in self.tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=tile_size)
        for img in images:
            im = _pil(img).convert("RGB")
            arr = np.asarray(im)
            bgr = arr[:, :, ::-1]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge((l_channel, a, b))
            bgr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            rgb = bgr2[:, :, ::-1]
            out_imgs.append(Image.fromarray(rgb))
        return out_imgs, geoms


@register("auto_contrast")
@final
class AutoContrast(ColorOp):
    curriculum_param_names = ("cutoff", "prob")

    def __init__(self, cutoff: int = 0, prob: float = 0.5):
        self.cutoff = int(cutoff)
        self.prob = float(prob)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: list[Image.Image] = []
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageOps.autocontrast(im, cutoff=self.cutoff))
        return out_imgs, geoms


@register("sharpness")
@final
class Sharpness(ColorOp):
    curriculum_param_names = ("factor", "prob")

    def __init__(self, factor: Sequence[float] = (0.5, 1.8), prob: float = 0.4):
        self.factor = (float(factor[0]), float(factor[1]))
        self.prob = float(prob)

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: list[Image.Image] = []
        f = rng.uniform(self.factor[0], self.factor[1])
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageEnhance.Sharpness(im).enhance(f))
        return out_imgs, geoms


def _aabb_iou(a: list[float], b: list[float]) -> float:
    inter = intersect_aabb(a, b)
    inter_area = aabb_area(inter)
    if inter_area <= 0.0:
        return 0.0
    union = aabb_area(a) + aabb_area(b) - inter_area
    return inter_area / union if union > 0 else 0.0


def _buffer_aabb(bbox: list[float], pad: float, width: int, height: int) -> list[float]:
    if pad <= 0:
        return bbox
    return [
        max(0.0, bbox[0] - pad),
        max(0.0, bbox[1] - pad),
        min(float(width), bbox[2] + pad),
        min(float(height), bbox[3] + pad),
    ]


def _geom_mask_iou(
    a: DatasetObject,
    b: DatasetObject,
    *,
    width: int,
    height: int,
    line_buffer: float,
) -> float:
    """Approximate IoU by rasterizing geometry regions into a per-pixel mask.

    This mimics COCO-style mask IoU (but implemented via PIL+NumPy to avoid pycocotools).
    It's used only for overlap rejection in copy/paste-style ops, so a small amount of
    discretization error is acceptable.
    """

    def _geom_aabb(g: DatasetObject) -> list[float]:
        aabb = get_aabb(g)
        pad = line_buffer if "line" in g else 0.0
        return _buffer_aabb(aabb, pad, width, height)

    aabb_a = _geom_aabb(a)
    aabb_b = _geom_aabb(b)
    if aabb_area(intersect_aabb(aabb_a, aabb_b)) <= 0.0:
        return 0.0

    # Rasterize inside a tight union window for speed.
    ux1 = max(0, int(math.floor(min(aabb_a[0], aabb_b[0]))))
    uy1 = max(0, int(math.floor(min(aabb_a[1], aabb_b[1]))))
    ux2 = min(width, int(math.ceil(max(aabb_a[2], aabb_b[2]))))
    uy2 = min(height, int(math.ceil(max(aabb_a[3], aabb_b[3]))))
    w = max(1, ux2 - ux1)
    h = max(1, uy2 - uy1)

    line_width = max(1, int(round(line_buffer * 2.0)))

    def _rasterize(g: DatasetObject) -> NDArray[np.uint8]:
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        if "bbox_2d" in g:
            x1, y1, x2, y2 = map(float, g["bbox_2d"])
            draw.rectangle([x1 - ux1, y1 - uy1, x2 - ux1, y2 - uy1], fill=255)
        elif "poly" in g:
            pts = g.get("poly") or []
            xy = [
                (float(pts[i]) - ux1, float(pts[i + 1]) - uy1)
                for i in range(0, len(pts), 2)
            ]
            if len(xy) >= 3:
                draw.polygon(xy, fill=255)
        elif "line" in g:
            pts = g.get("line") or []
            xy = [
                (float(pts[i]) - ux1, float(pts[i + 1]) - uy1)
                for i in range(0, len(pts), 2)
            ]
            if len(xy) >= 2:
                draw.line(xy, fill=255, width=line_width)
        else:
            # Unknown geometry - treat as empty mask.
            return np.zeros((h, w), dtype=np.uint8)

        return np.asarray(mask, dtype=np.uint8)

    ma = _rasterize(a)
    mb = _rasterize(b)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return (inter / union) if union > 0 else 0.0


@register("small_object_zoom_paste")
@final
class SmallObjectZoomPaste(PatchOp):
    """
    Single-image small-object zoom-and-paste to boost recall.

    Steps:
      1) Select small objects by size/length thresholds (optional class whitelist).
      2) Crop patch with context, scale up, and translate within the same image.
      3) Reject placements overlapping existing annotations beyond threshold; retry a few times.
      4) Apply identical affine (scale+translate) to bbox/poly/line; originals remain.
    """

    def __init__(
        self,
        prob: float = 0.2,
        max_targets: int = 1,
        max_attempts: int = 20,
        scale: tuple[float, float] = (1.4, 1.8),
        max_size: float = 96.0,
        max_line_length: float = 128.0,
        context: float = 4.0,
        overlap_threshold: float = 0.1,
        line_buffer: float = 4.0,
        overlap_mode: str = "aabb",
        class_whitelist: list[str] | None = None,
    ):
        self.curriculum_param_names = (
            "prob",
            "max_targets",
            "max_attempts",
            "scale",
            "max_size",
            "max_line_length",
            "context",
            "overlap_threshold",
            "line_buffer",
        )
        self.prob = float(prob)
        self.max_targets = int(max_targets)
        self.max_attempts = int(max_attempts)
        self.scale = (float(scale[0]), float(scale[1]))
        self.max_size = float(max_size)
        self.max_line_length = float(max_line_length)
        self.context = float(context)
        self.overlap_threshold = float(overlap_threshold)
        self.line_buffer = float(line_buffer)
        self.overlap_mode = str(overlap_mode or "aabb").lower()
        if self.overlap_mode not in {"aabb", "mask"}:
            raise ValueError(
                f"overlap_mode must be one of {{'aabb','mask'}}, got {overlap_mode!r}"
            )
        self.class_whitelist = list(class_whitelist) if class_whitelist else None
        self.allows_geometry_drops = (
            True  # geometry count may increase; allow validation bypass
        )

    def _is_small(self, geom: DatasetObject) -> bool:
        aabb = get_aabb(geom)
        w = aabb[2] - aabb[0]
        h = aabb[3] - aabb[1]
        if w <= 0 or h <= 0:
            return False
        if max(w, h) <= self.max_size:
            return True
        if "line" in geom:
            pts = geom["line"]
            length = 0.0
            for i in range(0, len(pts) - 2, 2):
                dx = pts[i + 2] - pts[i]
                dy = pts[i + 3] - pts[i + 1]
                length += math.hypot(dx, dy)
            return length <= self.max_line_length
        return False

    def _class_allowed(self, desc: str) -> bool:
        if not self.class_whitelist:
            return True

        # Match against the full desc string (substrings), so users can whitelist by:
        # - 类别 name: "类别=BBU安装螺丝" / "BBU安装螺丝"
        # - attribute tokens: "可见性=部分", "标签=有标签", etc.
        # - free text fields: 备注/文本 fragments
        haystack = "".join(str(desc).split()).lower()
        for raw in self.class_whitelist:
            needle = "".join(str(raw).split()).lower()
            if needle and needle in haystack:
                return True
        return False

    def _transform_geom(
        self,
        geom: DatasetObject,
        crop_origin: tuple[float, float],
        scale_factor: float,
        offset: tuple[float, float],
        width: int,
        height: int,
    ) -> DatasetObject | None:
        cx, cy = crop_origin
        tx, ty = offset
        M = compose_affine(
            translate(tx, ty),
            compose_affine(
                scale_matrix(scale_factor, scale_factor), translate(-cx, -cy)
            ),
        )
        transformed = transform_geometry(geom, M, width=width, height=height)
        if "line" in transformed and len(transformed["line"]) < 4:
            return None
        if "poly" in transformed and len(transformed["poly"]) < 6:
            return None
        return transformed

    def _iou_too_high(
        self,
        new_geom: DatasetObject,
        existing: list[DatasetObject],
        width: int,
        height: int,
    ) -> bool:
        new_aabb = get_aabb(new_geom)
        new_aabb = _buffer_aabb(
            new_aabb, self.line_buffer if "line" in new_geom else 0.0, width, height
        )
        for g in existing:
            aabb = get_aabb(g)
            pad = self.line_buffer if "line" in g else 0.0
            aabb = _buffer_aabb(aabb, pad, width, height)

            # Quick reject on bbox overlap first (cheap).
            if aabb_area(intersect_aabb(new_aabb, aabb)) <= 0.0:
                continue

            if self.overlap_mode == "mask":
                iou = _geom_mask_iou(
                    new_geom,
                    g,
                    width=width,
                    height=height,
                    line_buffer=self.line_buffer,
                )
            else:
                iou = _aabb_iou(new_aabb, aabb)
            if iou > self.overlap_threshold:
                return True
        return False

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        if (
            not images
            or not geoms
            or rng.random() >= self.prob
            or self.max_targets <= 0
        ):
            return images, geoms

        logger = get_logger("augmentation.small_object_zoom_paste")

        pil_images = [_pil(img).copy() for img in images]
        working_geoms: list[DatasetObject] = list(geoms)

        # Collect candidate indices
        candidates: list[int] = []
        for idx, g in enumerate(geoms):
            if "desc" in g and not self._class_allowed(str(g["desc"])):
                continue
            if self._is_small(g):
                candidates.append(idx)

        if not candidates:
            return images, geoms

        rng.shuffle(candidates)
        targets: list[int] = candidates[: self.max_targets]

        for idx in targets:
            g = geoms[idx]
            aabb = get_aabb(g)
            x1 = max(0.0, aabb[0] - self.context)
            y1 = max(0.0, aabb[1] - self.context)
            x2 = min(float(width), aabb[2] + self.context)
            y2 = min(float(height), aabb[3] + self.context)
            patch_w = int(round(max(1.0, x2 - x1)))
            patch_h = int(round(max(1.0, y2 - y1)))
            if patch_w <= 1 or patch_h <= 1:
                continue

            placed = False
            for _ in range(self.max_attempts):
                scale_factor = rng.uniform(self.scale[0], self.scale[1])
                new_w = int(round(patch_w * scale_factor))
                new_h = int(round(patch_h * scale_factor))
                if new_w < 1 or new_h < 1 or new_w > width or new_h > height:
                    continue
                max_tx = width - new_w
                max_ty = height - new_h
                if max_tx < 0 or max_ty < 0:
                    continue
                tx = rng.randint(0, max_tx) if max_tx > 0 else 0
                ty = rng.randint(0, max_ty) if max_ty > 0 else 0

                transformed = self._transform_geom(
                    g, (x1, y1), scale_factor, (tx, ty), width, height
                )
                if transformed is None:
                    continue
                # Track which source geometry produced this duplicate (used by the record-level
                # augmentation preprocessor to append the corresponding duplicated object).
                transformed["__src_geom_idx"] = int(idx)
                transformed["__aug_op"] = "small_object_zoom_paste"
                if self._iou_too_high(transformed, working_geoms, width, height):
                    continue

                # Paste patch onto all images
                for i, img in enumerate(pil_images):
                    patch = img.crop(
                        (int(x1), int(y1), int(x1 + patch_w), int(y1 + patch_h))
                    )
                    if new_w != patch_w or new_h != patch_h:
                        patch = patch.resize((new_w, new_h), _PIL_BICUBIC)
                    img.paste(patch, (tx, ty))
                    pil_images[i] = img

                working_geoms.append(transformed)
                placed = True
                break

            if not placed:
                logger.debug(
                    "small_object_zoom_paste: no valid placement for target %d", idx
                )

        return pil_images, working_geoms


@register("roi_crop")
@final
class RoiCrop(PatchOp):
    """
    Device-anchored ROI crop.

    Motivation (BBU/RRU巡检场景):
      - Small targets (e.g. screws/labels) need zoom-in to increase pixel share.
      - Cable/fiber topology needs wide-angle; blind random crops often sever lines.

    This op crops around an "anchor" object (device/cabinet) selected by strict category
    match on the `desc` token `类别=...`, then filters objects by coverage and truncates geometries to the crop
    boundary. Unlike `RandomCrop`, this op does NOT skip when line objects exist; line
    geometries are clipped and retained as long as any segment remains in the crop.

    Parameters:
        anchor_classes: category values to match exactly against the `desc` token `类别=...`
            (default: ("BBU设备", "RRU设备", "机柜"))
        scale_range: expansion factor range applied to anchor AABB (default: (1.2, 2.0))
        min_crop_size: minimum crop side length in pixels (default: 320)
        min_coverage: coverage threshold for keeping bbox/poly objects (default: 0.4)
        completeness_threshold: coverage threshold for marking partial visibility (default: 0.95)
        prob: probability of applying ROI crop (default: 0.5)
    """

    def __init__(
        self,
        anchor_classes: Sequence[str] = ("BBU设备", "RRU设备", "机柜"),
        scale_range: tuple[float, float] = (1.2, 2.0),
        min_crop_size: int = 320,
        prob: float = 0.5,
        *,
        min_coverage: float = 0.4,
        completeness_threshold: float = 0.95,
    ):
        self.curriculum_param_names = (
            "scale_range",
            "min_crop_size",
            "min_coverage",
            "completeness_threshold",
            "prob",
        )
        self.anchor_classes = tuple(
            s_str[len("类别=") :].strip() if s_str.startswith("类别=") else s_str
            for s in anchor_classes
            if (s_str := str(s).strip())
        )
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.min_crop_size = int(min_crop_size)
        self.min_coverage = float(min_coverage)
        self.completeness_threshold = float(completeness_threshold)
        self.prob = float(prob)

        # Metadata storage (set by apply, read by Compose / preprocessor)
        self.last_kept_indices: list[int] | None = None
        self.last_object_coverages: list[float] | None = None
        self.allows_geometry_drops = True  # Signal to validation
        self.last_skip_reason: str | None = None
        # Compose looks for `last_crop_skip_reason`; keep both names in sync.
        self.last_crop_skip_reason: str | None = None
        self.last_skip_counters: dict[str, int] = {}

    def _desc_is_anchor(self, desc: str) -> bool:
        if not desc:
            return False
        # Strict matching: only consider the structured `类别=` token (no substring matching, no fallbacks).
        # Note: dataset strings sometimes use Chinese commas; normalize to ',' before splitting.
        normalized_desc = desc.replace("，", ",")
        for token in normalized_desc.split(","):
            token = token.strip()
            if not token.startswith("类别="):
                continue
            category = token[len("类别=") :].strip()
            if category and category in self.anchor_classes:
                return True
        return False

    def _record_skip(self, reason: str) -> None:
        self.last_skip_reason = reason
        self.last_crop_skip_reason = reason
        self.last_skip_counters[reason] = self.last_skip_counters.get(reason, 0) + 1

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        logger = get_logger("augmentation.roi_crop")

        # Clear metadata from previous call
        self.last_kept_indices = None
        self.last_object_coverages = None
        self.last_skip_reason = None
        self.last_crop_skip_reason = None
        self.last_skip_counters = {}

        if rng.random() >= self.prob:
            return images, geoms

        # 1) Find anchor candidates (prefer bbox/poly devices/cabinets; never choose a line)
        anchor_indices: list[int] = []
        for idx, g in enumerate(geoms):
            desc = g.get("desc")
            if not isinstance(desc, str) or not self._desc_is_anchor(desc):
                continue
            if "line" in g:
                continue
            try:
                aabb = get_aabb(g)
            except Exception:
                continue
            if aabb_area(aabb) <= 0.0:
                continue
            anchor_indices.append(idx)

        # 2) Determine anchor AABB (strict; skip if no anchor is found)
        if not anchor_indices:
            self._record_skip("no_anchor")
            return images, geoms

        chosen = anchor_indices[rng.randrange(0, len(anchor_indices))]
        anchor_aabb = list(get_aabb(geoms[chosen]))

        # 3) Compute crop bbox around anchor (expanded and clamped to image bounds)
        x1, y1, x2, y2 = anchor_aabb
        w = max(1.0, float(x2) - float(x1))
        h = max(1.0, float(y2) - float(y1))
        cx = (float(x1) + float(x2)) * 0.5
        cy = (float(y1) + float(y2)) * 0.5

        s = rng.uniform(self.scale_range[0], self.scale_range[1])
        crop_w = max(int(round(w * s)), self.min_crop_size)
        crop_h = max(int(round(h * s)), self.min_crop_size)
        crop_w = max(1, min(crop_w, width))
        crop_h = max(1, min(crop_h, height))

        # Center crop around anchor; clamp to fit in [0, W] × [0, H]
        crop_x = int(round(cx - crop_w / 2.0))
        crop_y = int(round(cy - crop_h / 2.0))
        max_x = width - crop_w
        max_y = height - crop_h
        crop_x = max(0, min(crop_x, max_x))
        crop_y = max(0, min(crop_y, max_y))

        crop_bbox = [
            float(crop_x),
            float(crop_y),
            float(crop_x + crop_w),
            float(crop_y + crop_h),
        ]

        # 4) Filter geometries by visibility
        kept_indices: list[int] = []
        coverages: list[float] = []
        filtered_geoms: list[DatasetObject] = []

        for idx, g in enumerate(geoms):
            # Lines: keep if any segment remains after clipping (do not filter by coverage).
            if "line" in g:
                pts = g["line"]
                line_pts_translated_filter: list[float] = []
                for i in range(0, len(pts), 2):
                    line_pts_translated_filter.append(pts[i] - crop_bbox[0])
                    line_pts_translated_filter.append(pts[i + 1] - crop_bbox[1])
                clipped = clip_polyline_to_rect(
                    line_pts_translated_filter, crop_w, crop_h
                )
                if len(clipped) < 4:
                    continue
                cov = compute_polygon_coverage(g, crop_bbox, fallback="bbox")
                kept_indices.append(idx)
                coverages.append(cov)
                filtered_geoms.append(g)
                continue

            cov = compute_polygon_coverage(g, crop_bbox, fallback="bbox")
            if cov >= self.min_coverage:
                kept_indices.append(idx)
                coverages.append(cov)
                filtered_geoms.append(g)

        if not filtered_geoms:
            self._record_skip("no_objects")
            return images, geoms

        # 5) Proceed with crop - truncate and translate geometries
        cropped_geoms: list[DatasetObject] = []
        for g in filtered_geoms:
            truncated: DatasetObject
            # Truncate geometry to crop boundary
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                clipped_x1 = max(crop_bbox[0], min(crop_bbox[2], x1))
                clipped_y1 = max(crop_bbox[1], min(crop_bbox[3], y1))
                clipped_x2 = max(crop_bbox[0], min(crop_bbox[2], x2))
                clipped_y2 = max(crop_bbox[1], min(crop_bbox[3], y2))
                truncated = cast(
                    DatasetObject,
                    {"bbox_2d": [clipped_x1, clipped_y1, clipped_x2, clipped_y2]},
                )
            elif "poly" in g:
                pts = g["poly"]
                poly_pts_translated: list[float] = []
                for i in range(0, len(pts), 2):
                    poly_pts_translated.append(pts[i] - crop_bbox[0])
                    poly_pts_translated.append(pts[i + 1] - crop_bbox[1])
                clipped = sutherland_hodgman_clip(poly_pts_translated, crop_w, crop_h)

                # Reduce redundant vertices introduced by axis-aligned clipping.
                from ..geometry import simplify_polygon, choose_four_corners

                clipped = simplify_polygon(clipped)

                if len(clipped) // 2 >= 3:
                    # Prefer true 4-corner representation when possible.
                    if len(clipped) // 2 > 4:
                        best4 = choose_four_corners(clipped)
                        if best4:
                            clipped = best4
                    if len(clipped) // 2 != 4:
                        rect = min_area_rect(clipped)
                        if rect:
                            clipped = rect
                    clipped = to_clockwise(clipped)

                    # Translate back to image coords for final translation step.
                    final_poly_pts: list[float] = []
                    for i in range(0, len(clipped), 2):
                        final_poly_pts.append(clipped[i] + crop_bbox[0])
                        final_poly_pts.append(clipped[i + 1] + crop_bbox[1])
                    truncated = cast(DatasetObject, {"poly": final_poly_pts})
                else:
                    # Degenerate poly - clamp to crop boundary.
                    clamped: list[float] = []
                    for i in range(0, len(pts), 2):
                        clamped.append(max(crop_bbox[0], min(crop_bbox[2], pts[i])))
                        clamped.append(max(crop_bbox[1], min(crop_bbox[3], pts[i + 1])))
                    truncated = cast(DatasetObject, {"poly": clamped})
            elif "line" in g:
                pts = g["line"]
                line_pts_translated_crop: list[float] = []
                for i in range(0, len(pts), 2):
                    line_pts_translated_crop.append(pts[i] - crop_bbox[0])
                    line_pts_translated_crop.append(pts[i + 1] - crop_bbox[1])
                clipped = clip_polyline_to_rect(
                    line_pts_translated_crop, crop_w, crop_h
                )

                final_line_pts: list[float] = []
                for i in range(0, len(clipped), 2):
                    final_line_pts.append(clipped[i] + crop_bbox[0])
                    final_line_pts.append(clipped[i + 1] + crop_bbox[1])

                if len(final_line_pts) >= 4:
                    truncated = cast(DatasetObject, {"line": final_line_pts})
                else:
                    # No segment remains; this should be rare since we filtered for intersection.
                    truncated = cast(
                        DatasetObject,
                        {
                            "line": [
                                crop_bbox[0],
                                crop_bbox[1],
                                crop_bbox[0],
                                crop_bbox[1],
                            ]
                        },
                    )
            else:
                truncated = g

            translated = translate_geometry(truncated, -crop_bbox[0], -crop_bbox[1])
            cropped_geoms.append(translated)

        # 6) Crop images
        out_imgs: list[Image.Image] = []
        for img in images:
            im = _pil(img)
            out_imgs.append(im.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)))

        # Store metadata for preprocessor/telemetry
        self.last_kept_indices = kept_indices
        self.last_object_coverages = coverages

        logger.debug(
            f"ROI crop applied: {len(geoms)} → {len(cropped_geoms)} objects (region: {crop_bbox})"
        )
        return out_imgs, cropped_geoms


@register("random_crop")
@final
class RandomCrop(PatchOp):
    """
    Random crop with label filtering for dense captioning.

    Crops a random region from the image and filters geometries based on visibility.
    If filtered objects < min_objects or any filtered object is a line (when skip_if_line=True),
    the entire crop operation is skipped and original images/geometries are returned unchanged.

    Parameters:
        scale: (min, max) tuple for crop size as fraction of original image (default: (0.6, 1.0))
        aspect_ratio: (min, max) tuple for aspect ratio variation (default: (0.8, 1.2))
        min_coverage: minimum coverage ratio to keep object (default: 0.3)
        completeness_threshold: coverage threshold for marking "部分" (default: 0.95)
        min_objects: skip crop if filtered objects < this value (default: 4)
        skip_if_line: skip crop if any filtered object is a line (default: True)
        prob: probability of applying crop (default: 1.0)
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.6, 1.0),
        aspect_ratio: tuple[float, float] = (0.8, 1.2),
        min_coverage: float = 0.3,
        completeness_threshold: float = 0.95,
        min_objects: int = 4,
        skip_if_line: bool = True,
        prob: float = 1.0,
    ):
        self.curriculum_param_names = (
            "scale",
            "aspect_ratio",
            "min_coverage",
            "completeness_threshold",
            "min_objects",
            "prob",
        )
        self.scale = (float(scale[0]), float(scale[1]))
        self.aspect_ratio = (float(aspect_ratio[0]), float(aspect_ratio[1]))
        self.min_coverage = float(min_coverage)
        self.completeness_threshold = float(completeness_threshold)
        self.min_objects = int(min_objects)
        self.skip_if_line = bool(skip_if_line)
        self.prob = float(prob)

        # Metadata storage (set by apply, read by preprocessor)
        self.last_kept_indices: list[int] | None = None
        self.last_object_coverages: list[float] | None = None
        self.allows_geometry_drops = True  # Signal to validation
        self.last_skip_reason: str | None = None
        # Compose looks for `last_crop_skip_reason`; keep both names in sync.
        self.last_crop_skip_reason: str | None = None
        self.last_skip_counters: dict[str, int] = {}

    @override
    def apply(
        self,
        images: list[Image.Image],
        geoms: list[DatasetObject],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ):
        logger = get_logger("augmentation.random_crop")

        # Clear metadata from previous call
        self.last_kept_indices = None
        self.last_object_coverages = None
        self.last_skip_reason = None
        self.last_crop_skip_reason = None
        self.last_skip_counters = {}

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

        crop_bbox = [
            float(crop_x),
            float(crop_y),
            float(crop_x + crop_w),
            float(crop_y + crop_h),
        ]

        # Compute coverage and filter geometries
        kept_indices: list[int] = []
        coverages: list[float] = []
        filtered_geoms: list[DatasetObject] = []

        for idx, g in enumerate(geoms):
            cov = compute_polygon_coverage(g, crop_bbox, fallback="bbox")
            if cov >= self.min_coverage:
                kept_indices.append(idx)
                coverages.append(cov)
                filtered_geoms.append(g)

        # Check skip conditions
        if len(filtered_geoms) < self.min_objects:
            reason = "min_objects"
            logger.debug(
                f"Crop would filter to {len(filtered_geoms)} < {self.min_objects} objects. Skipping crop."
            )
            self.last_skip_reason = reason
            self.last_crop_skip_reason = reason
            self.last_skip_counters[reason] = self.last_skip_counters.get(reason, 0) + 1
            return images, geoms

        if self.skip_if_line:
            has_line = any("line" in g for g in filtered_geoms)
            if has_line:
                reason = "line_object"
                logger.debug(
                    "Crop region contains line object. Skipping crop to preserve cable/fiber integrity."
                )
                self.last_skip_reason = reason
                self.last_crop_skip_reason = reason
                self.last_skip_counters[reason] = (
                    self.last_skip_counters.get(reason, 0) + 1
                )
                return images, geoms

        # Proceed with crop - truncate and translate geometries
        cropped_geoms: list[DatasetObject] = []
        for g in filtered_geoms:
            truncated: DatasetObject
            # Truncate geometry to crop boundary
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                # Clip to crop region
                clipped_x1 = max(crop_bbox[0], min(crop_bbox[2], x1))
                clipped_y1 = max(crop_bbox[1], min(crop_bbox[3], y1))
                clipped_x2 = max(crop_bbox[0], min(crop_bbox[2], x2))
                clipped_y2 = max(crop_bbox[1], min(crop_bbox[3], y2))
                truncated = cast(
                    DatasetObject,
                    {"bbox_2d": [clipped_x1, clipped_y1, clipped_x2, clipped_y2]},
                )
            elif "poly" in g:
                pts = g["poly"]
                # Translate to crop origin for clipping
                poly_pts_translated: list[float] = []
                for i in range(0, len(pts), 2):
                    poly_pts_translated.append(pts[i] - crop_bbox[0])
                    poly_pts_translated.append(pts[i + 1] - crop_bbox[1])
                clipped = sutherland_hodgman_clip(poly_pts_translated, crop_w, crop_h)
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
                    final_poly_pts: list[float] = []
                    for i in range(0, len(clipped), 2):
                        final_poly_pts.append(clipped[i] + crop_bbox[0])
                        final_poly_pts.append(clipped[i + 1] + crop_bbox[1])
                    truncated = cast(DatasetObject, {"poly": final_poly_pts})
                else:
                    # Degenerate poly - clamp to crop boundary
                    clamped_poly: list[float] = []
                    for i in range(0, len(pts), 2):
                        clamped_poly.append(
                            max(crop_bbox[0], min(crop_bbox[2], pts[i]))
                        )
                        clamped_poly.append(
                            max(crop_bbox[1], min(crop_bbox[3], pts[i + 1]))
                        )
                    truncated = cast(DatasetObject, {"poly": clamped_poly})
            elif "line" in g:
                pts = g["line"]
                # Translate to crop origin
                line_pts_translated: list[float] = []
                for i in range(0, len(pts), 2):
                    line_pts_translated.append(pts[i] - crop_bbox[0])
                    line_pts_translated.append(pts[i + 1] - crop_bbox[1])
                # Clip line to crop boundary
                clipped = clip_polyline_to_rect(line_pts_translated, crop_w, crop_h)
                # Translate back
                final_line_pts: list[float] = []
                for i in range(0, len(clipped), 2):
                    final_line_pts.append(clipped[i] + crop_bbox[0])
                    final_line_pts.append(clipped[i + 1] + crop_bbox[1])

                if len(final_line_pts) >= 4:
                    truncated = cast(DatasetObject, {"line": final_line_pts})
                else:
                    # Degenerate line - clamp to crop boundary
                    clamped_line: list[float] = []
                    for i in range(0, len(pts), 2):
                        clamped_line.append(
                            max(crop_bbox[0], min(crop_bbox[2], pts[i]))
                        )
                        clamped_line.append(
                            max(crop_bbox[1], min(crop_bbox[3], pts[i + 1]))
                        )
                    if len(clamped_line) >= 4:
                        truncated = cast(DatasetObject, {"line": clamped_line[:4]})
                    else:
                        truncated = cast(
                            DatasetObject,
                            {
                                "line": [
                                    crop_bbox[0],
                                    crop_bbox[1],
                                    crop_bbox[0],
                                    crop_bbox[1],
                                ]
                            },
                        )
            else:
                truncated = g

            # Translate to crop coordinates
            translated = translate_geometry(truncated, -crop_bbox[0], -crop_bbox[1])
            cropped_geoms.append(translated)

        # Crop images
        out_imgs: list[Image.Image] = []
        for img in images:
            im = _pil(img)
            out_imgs.append(im.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)))

        # Store metadata for preprocessor
        self.last_kept_indices = kept_indices
        self.last_object_coverages = coverages

        logger.debug(
            f"Crop applied: {len(geoms)} → {len(cropped_geoms)} objects (region: {crop_bbox})"
        )
        return out_imgs, cropped_geoms


__all__ = [
    "HFlip",
    "VFlip",
    "Rotate",
    "Scale",
    "ColorJitter",
    "ResizeByScale",
    "Gamma",
    "HueSaturationValue",
    "CLAHE",
    "AutoContrast",
    "Sharpness",
    "ExpandToFitAffine",
    "SmallObjectZoomPaste",
    "RoiCrop",
    "RandomCrop",
]
