from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageOps
import numpy as np

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
)
from .base import ImageAugmenter
from .registry import register


def _pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return img


def _to_bytes(img: Image.Image) -> Dict[str, bytes]:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def _bbox_from_points(points: List[float], width: int, height: int) -> List[int]:
    xs = points[0::2]
    ys = points[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bb = clamp_points([x1, y1, x2, y2], width, height)
    # degeneracy: if collapsed, caller may fallback
    return bb


@register("hflip")
class HFlip(ImageAugmenter):
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        M = hflip_matrix(width)
        Minv = invert_affine(M)
        coeffs = (Minv[0][0], Minv[0][1], Minv[0][2], Minv[1][0], Minv[1][1], Minv[1][2])
        out_imgs = [_pil(img).transform((width, height), Image.AFFINE, data=coeffs, resample=Image.BICUBIC) for img in images]
        out_geoms: List[Dict[str, Any]] = []
        for g in geoms:
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                t = apply_affine(pts, M)
                bb = _bbox_from_points(t, width, height)
                # fallback if degenerate
                if bb[0] == bb[2] or bb[1] == bb[3]:
                    out_geoms.append({"bbox_2d": [x1, y1, x2, y2]})
                else:
                    out_geoms.append({"bbox_2d": bb})
            elif "quad" in g:
                t = apply_affine(g["quad"], M)
                q = clamp_points(t, width, height)
                if min(q[0::2]) == max(q[0::2]) and min(q[1::2]) == max(q[1::2]):
                    out_geoms.append({"quad": g["quad"]})
                else:
                    out_geoms.append({"quad": q})
            elif "line" in g:
                t = apply_affine(g["line"], M)
                l = clamp_points(t, width, height)
                l = dedupe_consecutive_points(l)
                if len(l) < 4:
                    out_geoms.append({"line": g["line"]})
                else:
                    out_geoms.append({"line": l})
            else:
                out_geoms.append(g)
        return out_imgs, out_geoms


@register("vflip")
class VFlip(ImageAugmenter):
    def __init__(self, prob: float = 0.1):
        self.prob = float(prob)

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        M = vflip_matrix(height)
        Minv = invert_affine(M)
        coeffs = (Minv[0][0], Minv[0][1], Minv[0][2], Minv[1][0], Minv[1][1], Minv[1][2])
        out_imgs = [_pil(img).transform((width, height), Image.AFFINE, data=coeffs, resample=Image.BICUBIC) for img in images]
        out_geoms: List[Dict[str, Any]] = []
        for g in geoms:
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                t = apply_affine(pts, M)
                bb = _bbox_from_points(t, width, height)
                if bb[0] == bb[2] or bb[1] == bb[3]:
                    out_geoms.append({"bbox_2d": [x1, y1, x2, y2]})
                else:
                    out_geoms.append({"bbox_2d": bb})
            elif "quad" in g:
                t = apply_affine(g["quad"], M)
                q = clamp_points(t, width, height)
                if min(q[0::2]) == max(q[0::2]) and min(q[1::2]) == max(q[1::2]):
                    out_geoms.append({"quad": g["quad"]})
                else:
                    out_geoms.append({"quad": q})
            elif "line" in g:
                t = apply_affine(g["line"], M)
                l = clamp_points(t, width, height)
                l = dedupe_consecutive_points(l)
                if len(l) < 4:
                    out_geoms.append({"line": g["line"]})
                else:
                    out_geoms.append({"line": l})
            else:
                out_geoms.append(g)
        return out_imgs, out_geoms


@register("rotate")
class Rotate(ImageAugmenter):
    def __init__(self, max_deg: float = 10.0, prob: float = 0.5):
        self.max_deg = float(max_deg)
        self.prob = float(prob)

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if self.max_deg <= 0 or rng.random() >= self.prob:
            return images, geoms
        deg = rng.uniform(-self.max_deg, self.max_deg)
        M = rotate_center(deg, width / 2.0, height / 2.0)
        Minv = invert_affine(M)
        coeffs = (Minv[0][0], Minv[0][1], Minv[0][2], Minv[1][0], Minv[1][1], Minv[1][2])
        out_imgs = [_pil(img).transform((width, height), Image.AFFINE, data=coeffs, resample=Image.BICUBIC) for img in images]
        out_geoms: List[Dict[str, Any]] = []
        for g in geoms:
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                t = apply_affine(pts, M)
                bb = _bbox_from_points(t, width, height)
                if bb[0] == bb[2] or bb[1] == bb[3]:
                    out_geoms.append({"bbox_2d": [x1, y1, x2, y2]})
                else:
                    out_geoms.append({"bbox_2d": bb})
            elif "quad" in g:
                t = apply_affine(g["quad"], M)
                q = clamp_points(t, width, height)
                if min(q[0::2]) == max(q[0::2]) and min(q[1::2]) == max(q[1::2]):
                    out_geoms.append({"quad": g["quad"]})
                else:
                    out_geoms.append({"quad": q})
            elif "line" in g:
                t = apply_affine(g["line"], M)
                l = clamp_points(t, width, height)
                l = dedupe_consecutive_points(l)
                if len(l) < 4:
                    out_geoms.append({"line": g["line"]})
                else:
                    out_geoms.append({"line": l})
            else:
                out_geoms.append(g)
        return out_imgs, out_geoms


@register("scale")
class Scale(ImageAugmenter):
    def __init__(self, lo: float = 0.9, hi: float = 1.1, prob: float = 0.5):
        self.lo = float(lo)
        self.hi = float(hi)
        self.prob = float(prob)

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        s = rng.uniform(self.lo, self.hi)
        M = scale_center(s, s, width / 2.0, height / 2.0)
        Minv = invert_affine(M)
        coeffs = (Minv[0][0], Minv[0][1], Minv[0][2], Minv[1][0], Minv[1][1], Minv[1][2])
        out_imgs = [_pil(img).transform((width, height), Image.AFFINE, data=coeffs, resample=Image.BICUBIC) for img in images]
        out_geoms: List[Dict[str, Any]] = []
        for g in geoms:
            if "bbox_2d" in g:
                x1, y1, x2, y2 = g["bbox_2d"]
                pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                t = apply_affine(pts, M)
                bb = _bbox_from_points(t, width, height)
                if bb[0] == bb[2] or bb[1] == bb[3]:
                    out_geoms.append({"bbox_2d": [x1, y1, x2, y2]})
                else:
                    out_geoms.append({"bbox_2d": bb})
            elif "quad" in g:
                t = apply_affine(g["quad"], M)
                q = clamp_points(t, width, height)
                if min(q[0::2]) == max(q[0::2]) and min(q[1::2]) == max(q[1::2]):
                    out_geoms.append({"quad": g["quad"]})
                else:
                    out_geoms.append({"quad": q})
            elif "line" in g:
                t = apply_affine(g["line"], M)
                l = clamp_points(t, width, height)
                l = dedupe_consecutive_points(l)
                if len(l) < 4:
                    out_geoms.append({"line": g["line"]})
                else:
                    out_geoms.append({"line": l})
            else:
                out_geoms.append(g)
        return out_imgs, out_geoms


@register("color_jitter")
class ColorJitter(ImageAugmenter):
    def __init__(self, brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), prob: float = 1.0):
        self.brightness = tuple(map(float, brightness))
        self.contrast = tuple(map(float, contrast))
        self.saturation = tuple(map(float, saturation))
        self.prob = float(prob)

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


@register("gamma")
class Gamma(ImageAugmenter):
    def __init__(self, gamma=(0.7, 1.4), gain: float = 1.0, prob: float = 1.0):
        self.gamma = (float(gamma[0]), float(gamma[1]))
        self.gain = float(gain)
        self.prob = float(prob)

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

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageOps.autocontrast(im, cutoff=self.cutoff))
        return out_imgs, geoms


@register("equalize")
class Equalize(ImageAugmenter):
    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    def apply(self, images: List[Any], geoms: List[Dict[str, Any]], *, width: int, height: int, rng: Any):
        if rng.random() >= self.prob:
            return images, geoms
        out_imgs: List[Any] = []
        for img in images:
            im = _pil(img).convert("RGB")
            out_imgs.append(ImageOps.equalize(im))
        return out_imgs, geoms


@register("solarize")
class Solarize(ImageAugmenter):
    def __init__(self, threshold: int = 128, prob: float = 0.3):
        self.threshold = int(threshold)
        self.prob = float(prob)

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
    "Equalize",
    "Solarize",
    "Posterize",
    "Sharpness",
    "AlbumentationsColor",
]


