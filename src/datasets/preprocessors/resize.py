"""Shared smart-resize helpers for detection datasets.

This module centralizes the pixel-budget/grid-alignment logic used during
offline conversion and can be reused as an online guard. The implementation
stays aligned with the historical ``data_conversion.pipeline.vision_process``
behaviour while exposing a lightweight preprocessor wrapper for record-level
geometry scaling.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from PIL import Image

from ..contracts import ConversationRecord
from ..geometry import clamp_points, scale_points
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

# Default hyperparameters mirrored from the data conversion pipeline
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200


def _round_by_factor(number: int, factor: int) -> int:
    """Return the closest integer to ``number`` divisible by ``factor``."""
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest integer >= ``number`` divisible by ``factor``."""
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    """Return the largest integer <= ``number`` divisible by ``factor``."""
    return math.floor(number / factor) * factor


def smart_resize(
    *,
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    max_ratio: int = MAX_RATIO,
) -> tuple[int, int]:
    """Compute resized dimensions that satisfy the detection pixel budget.

    The resized dimensions:
    - Respect the :data:`max_pixels` budget while staying above :data:`min_pixels`
    - Preserve aspect ratio
    - Snap to multiples of :data:`factor`
    - Reject extreme aspect ratios that would break patch grids
    """
    height = int(height)
    width = int(width)
    factor = int(factor)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)

    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, _floor_by_factor(int(width / beta), factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(int(height * beta), factor)
        w_bar = _ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def _scale_and_clamp_geometry(
    objects: Iterable[MutableMapping[str, Any]],
    sx: float,
    sy: float,
    width: int,
    height: int,
) -> List[MutableMapping[str, Any]]:
    """Scale geometries by (sx, sy) and clamp to the new bounds."""
    scaled: List[MutableMapping[str, Any]] = []
    for obj in objects:
        updated = dict(obj)
        if obj.get("bbox_2d") is not None:
            pts = scale_points(obj["bbox_2d"], sx, sy)
            updated["bbox_2d"] = clamp_points(pts, width, height)
            updated.pop("poly", None)
            updated.pop("line", None)
        elif obj.get("poly") is not None:
            pts = scale_points(obj["poly"], sx, sy)
            updated["poly"] = clamp_points(pts, width, height)
            updated.pop("bbox_2d", None)
            updated.pop("line", None)
        elif obj.get("line") is not None:
            pts = scale_points(obj["line"], sx, sy)
            updated["line"] = clamp_points(pts, width, height)
            updated.pop("bbox_2d", None)
            updated.pop("poly", None)
        scaled.append(updated)
    return scaled


@dataclass(frozen=True)
class SmartResizeParams:
    """Configuration for smart resize."""

    max_pixels: int = MAX_PIXELS
    image_factor: int = IMAGE_FACTOR
    min_pixels: int = MIN_PIXELS
    max_ratio: int = MAX_RATIO
    warn_downscale: float = 2.0


class Resizer:
    """Unified image + geometry resize helper."""

    def __init__(
        self,
        *,
        params: SmartResizeParams | None = None,
        jsonl_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        write_images: bool = True,
        images_root_override: str | Path | None = None,
        relative_output_root: str | Path | None = None,
        exif_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.params = params or SmartResizeParams()
        self.jsonl_dir = Path(jsonl_dir) if jsonl_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.write_images = bool(write_images)
        self.images_root_override = (
            Path(images_root_override).resolve() if images_root_override else None
        )
        self.relative_output_root = (
            Path(relative_output_root).resolve() if relative_output_root else None
        )
        self.exif_fn = exif_fn
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def resize_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        images = row.get("images") or []
        if not images:
            return row

        resolved_paths = self._resolve_image_paths(images)
        width = self._safe_int(row.get("width"))
        height = self._safe_int(row.get("height"))
        if width is None or height is None:
            width, height = self._probe_image_size(resolved_paths[0])

        target_h, target_w = smart_resize(
            height=height,
            width=width,
            factor=self.params.image_factor,
            min_pixels=self.params.min_pixels,
            max_pixels=self.params.max_pixels,
            max_ratio=self.params.max_ratio,
        )

        # Idempotent path: still rewrite images to output_dir when requested
        if target_h == height and target_w == width:
            if self.write_images:
                rewritten: List[str] = []
                for path in resolved_paths:
                    rewritten.append(
                        str(self._resize_image(Path(path), target_w, target_h))
                    )
                row["images"] = self._maybe_relativize_paths(rewritten)
            else:
                row["images"] = self._maybe_relativize_paths(resolved_paths)
            row["width"] = width
            row["height"] = height
            return row

        sx = float(target_w) / float(width)
        sy = float(target_h) / float(height)

        row["width"] = target_w
        row["height"] = target_h
        row["objects"] = _scale_and_clamp_geometry(
            row.get("objects") or [], sx, sy, target_w, target_h
        )

        downscale_factor = max(
            float(width) / float(target_w) if target_w else 1.0,
            float(height) / float(target_h) if target_h else 1.0,
        )
        if downscale_factor >= self.params.warn_downscale:
            logger.warning(
                "Smart resize applied >%.2fx downscale: %s -> %s",
                downscale_factor,
                (width, height),
                (target_w, target_h),
            )

        if self.write_images:
            rewritten: List[str] = []
            for path in resolved_paths:
                rewritten.append(
                    str(self._resize_image(Path(path), target_w, target_h))
                )
            row["images"] = self._maybe_relativize_paths(rewritten)
        else:
            row["images"] = self._maybe_relativize_paths(resolved_paths)
        return row

    def _resolve_image_paths(self, images: Sequence[Any]) -> List[str]:
        resolved: List[str] = []
        base = self.jsonl_dir
        for img in images:
            if isinstance(img, str):
                p = Path(img)
                if not p.is_absolute():
                    if self.images_root_override:
                        p = (self.images_root_override / p).resolve()
                    elif base:
                        p = (base / p).resolve()
                    else:
                        p = p.resolve()
                resolved.append(str(p))
            else:
                resolved.append(img)
        return resolved

    def _maybe_relativize_paths(self, paths: Sequence[Any]) -> List[Any]:
        if not self.relative_output_root:
            return list(paths)
        rel_paths: List[Any] = []
        relative_output_root_resolved = self.relative_output_root.resolve()
        output_dir_resolved = self.output_dir.resolve() if self.output_dir else None

        for p in paths:
            if not isinstance(p, str):
                rel_paths.append(p)
                continue

            path_obj = Path(p)
            path_resolved = path_obj.resolve() if path_obj.is_absolute() else path_obj

            # Try direct relativization first
            try:
                rel_paths.append(
                    str(path_resolved.relative_to(relative_output_root_resolved))
                )
                continue
            except (ValueError, RuntimeError):
                pass

            # If path is under output_dir, construct relative path manually
            if output_dir_resolved:
                try:
                    if path_resolved.is_absolute() and str(path_resolved).startswith(
                        str(output_dir_resolved)
                    ):
                        # Path is under output_dir
                        rel_to_output = path_resolved.relative_to(output_dir_resolved)
                        # Compute relative path from output_dir to relative_output_root
                        try:
                            output_to_root = output_dir_resolved.relative_to(
                                relative_output_root_resolved
                            )
                            # Construct: output_to_root / rel_to_output
                            rel_path = output_to_root / rel_to_output
                        except (ValueError, RuntimeError):
                            # output_dir is not under relative_output_root, use just the name
                            rel_path = Path(output_dir_resolved.name) / rel_to_output
                        rel_paths.append(str(rel_path))
                        continue
                except (ValueError, RuntimeError):
                    pass

            # If path is already relative, try to normalize it
            if not path_obj.is_absolute():
                # Already relative, but might not be relative to relative_output_root
                # Try to make it relative by joining with relative_output_root
                try:
                    # If it's a simple relative path, assume it's already correct
                    # Otherwise, construct path relative to relative_output_root
                    if str(path_obj).startswith("images/") or str(path_obj).startswith(
                        "./images/"
                    ):
                        rel_paths.append(str(path_obj).lstrip("./"))
                    else:
                        # Try to resolve it relative to relative_output_root
                        test_path = relative_output_root_resolved / path_obj
                        if test_path.exists():
                            rel_paths.append(str(path_obj))
                        else:
                            # Fallback: use just the filename under images/
                            rel_paths.append(f"images/{path_obj.name}")
                except Exception:
                    # Last resort: use just the filename under images/
                    rel_paths.append(f"images/{path_obj.name}")
            else:
                # Absolute path that we couldn't relativize
                # This shouldn't happen for resized images, but log a warning
                logger.warning(
                    f"Could not relativize absolute path {p} to {relative_output_root_resolved}. "
                    f"Using filename only."
                )
                rel_paths.append(f"images/{path_obj.name}")

        return rel_paths

    def _probe_image_size(self, image_path: str) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return int(img.height), int(img.width)

    def _resize_image(self, image_path: Path, width: int, height: int) -> Path:
        out_path = image_path
        if self.output_dir:
            rel = image_path.name
            base = self.jsonl_dir
            if base:
                try:
                    rel = image_path.relative_to(base)
                except ValueError:
                    rel = image_path.name
            out_path = (self.output_dir / rel).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            try:
                with Image.open(out_path) as existing:
                    if existing.size == (width, height):
                        return out_path
            except Exception:
                pass  # fall through to rewrite if probing fails

        with Image.open(image_path) as img:
            if self.exif_fn:
                try:
                    img = self.exif_fn(img)
                except Exception:
                    img = img
            rgb = img.convert("RGB")
            resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
            resized.save(out_path)
        return out_path

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class SmartResizePreprocessor(BasePreprocessor):
    """Record-level smart resize that can run offline or online."""

    def __init__(
        self,
        *,
        params: SmartResizeParams | None = None,
        jsonl_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        write_images: bool = False,
        images_root_override: str | Path | None = None,
        relative_output_root: str | Path | None = None,
        exif_fn: Callable[[Image.Image], Image.Image] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.resizer = Resizer(
            params=params,
            jsonl_dir=jsonl_dir,
            output_dir=output_dir,
            write_images=write_images,
            images_root_override=images_root_override,
            relative_output_root=relative_output_root,
            exif_fn=exif_fn,
        )

    def preprocess(self, row: ConversationRecord) -> ConversationRecord | None:
        result = self.resizer.resize_record(cast(Dict[str, Any], row))
        return cast(ConversationRecord | None, result)


def smart_resize_params_from_env(
    prefix: str = "SMART_RESIZE_GUARD",
) -> Optional[SmartResizeParams]:
    """Build SmartResizeParams from environment variables when guard is requested.

    Env variables:
      - SMART_RESIZE_GUARD (truthy to enable)
      - SMART_RESIZE_GUARD_MAX_PIXELS
      - SMART_RESIZE_GUARD_IMAGE_FACTOR
      - SMART_RESIZE_GUARD_MIN_PIXELS
    """
    flag = os.getenv(prefix)
    if flag is None:
        return None
    if str(flag).strip().lower() not in {"1", "true", "yes", "on"}:
        return None

    def _read_int(env_name: str, default: int) -> int:
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except Exception:
            logger.warning("Invalid %s=%s; using default %s", env_name, raw, default)
            return default

    defaults = SmartResizeParams()
    return SmartResizeParams(
        max_pixels=_read_int(f"{prefix}_MAX_PIXELS", defaults.max_pixels),
        image_factor=_read_int(f"{prefix}_IMAGE_FACTOR", defaults.image_factor),
        min_pixels=_read_int(f"{prefix}_MIN_PIXELS", defaults.min_pixels),
        max_ratio=defaults.max_ratio,
        warn_downscale=defaults.warn_downscale,
    )


__all__ = [
    "smart_resize",
    "Resizer",
    "SmartResizeParams",
    "SmartResizePreprocessor",
    "smart_resize_params_from_env",
    "IMAGE_FACTOR",
    "MIN_PIXELS",
    "MAX_PIXELS",
    "MAX_RATIO",
]
