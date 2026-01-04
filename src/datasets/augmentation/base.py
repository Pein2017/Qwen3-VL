from __future__ import annotations

import re
from typing import Protocol, cast

from .curriculum import NumericParam

from PIL import Image

_PIL_TRANSFORM = getattr(Image, "Transform", Image)
_PIL_RESAMPLE = getattr(Image, "Resampling", Image)
_PIL_AFFINE = getattr(_PIL_TRANSFORM, "AFFINE")
_PIL_BICUBIC = getattr(_PIL_RESAMPLE, "BICUBIC")

from ..geometry import (
    compose_affine,
    invert_affine,
    transform_geometry,
)
from ..contracts import AugmentationTelemetry


def _is_prob_field(name: str) -> bool:
    lowered = name.lower()
    return lowered == "prob" or lowered.endswith("_prob")


class RngLike(Protocol):
    def random(self) -> float: ...

    def uniform(self, a: float, b: float) -> float: ...

    def randint(self, a: int, b: int) -> int: ...

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int: ...

    def shuffle(self, x: list[object]) -> None: ...


class CurriculumMixin:
    """
    Shared helper for ops that expose curriculum-adjustable numeric parameters.
    Subclasses set `curriculum_param_names` to the list of attribute names that are
    eligible for curriculum overrides. Only scalars or 2-element numeric ranges are
    allowed; invalid values raise ValueError (fail-fast).
    """

    curriculum_param_names: tuple[str, ...] = tuple()

    @property
    def curriculum_params(self) -> dict[str, NumericParam]:
        params: dict[str, NumericParam] = {}
        for name in self.curriculum_param_names:
            if not hasattr(self, name):
                continue
            raw = getattr(self, name)
            numeric = NumericParam.from_raw(raw)
            if _is_prob_field(name):
                vals = numeric.values
                for v in vals:
                    if v < 0.0 or v > 1.0:
                        raise ValueError(
                            f"Curriculum param '{name}' must be within [0, 1]; got {v}"
                        )
            params[name] = numeric
        return params


class AffineOp(CurriculumMixin):
    kind: str = "affine"
    allows_geometry_drops: bool = False
    curriculum_param_names: tuple[str, ...] = ("prob",)

    def affine(
        self, width: int, height: int, rng: RngLike
    ) -> list[list[float]] | None:
        raise NotImplementedError


class ColorOp(CurriculumMixin):
    kind: str = "color"
    allows_geometry_drops: bool = False
    curriculum_param_names: tuple[str, ...] = ("prob",)

    def apply(
        self,
        images: list[Image.Image],
        geoms: list[dict[str, object]],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ) -> tuple[list[Image.Image], list[dict[str, object]]]:
        raise NotImplementedError


class PatchOp(CurriculumMixin):
    """
    Patch-style ops (crop or copy/paste) that may change geometry count or ordering.
    Compose flushes pending affines before invoking a PatchOp and records telemetry
    from crop-style implementations that populate kept/coverage metadata.
    """

    kind: str = "patch"
    allows_geometry_drops: bool = False
    curriculum_param_names: tuple[str, ...] = ("prob",)

    def apply(
        self,
        images: list[Image.Image],
        geoms: list[dict[str, object]],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ) -> tuple[list[Image.Image], list[dict[str, object]]]:
        raise NotImplementedError


class ImageAugmenter(Protocol):
    """
    Protocol for augmentation operators.

    Operators implement one or more methods depending on their type:

    1. **Affine Ops** (kind="affine"): HFlip, VFlip, Rotate, Scale
       - Implement: affine(width, height, rng) -> Optional[Matrix3x3]
       - Compose accumulates these and applies in a single warp for efficiency

    2. **Color Ops** (kind="color"): ColorJitter, Gamma, HSV, etc.
       - Implement: apply(images, geoms, ...) -> (images, geoms)
       - Deferred until after all affines are flushed (geometry unchanged)

    3. **Barrier Ops** (kind="barrier"): ResizeByScale, ExpandToFitAffine
       - Implement: apply(images, geoms, ...) -> (images, geoms)
       - Optionally: pre_flush_hook(M_total, width, height, rng) -> (M_total, width, height)
       - Barrier forces affine flush before and after; may change canvas size

    ## Pre-Flush Hook Protocol (Advanced)

    Barrier ops may implement `pre_flush_hook()` to modify accumulated affines and canvas
    dimensions BEFORE warping occurs. This enables operations like canvas expansion.

    Signature:
        def pre_flush_hook(
            self,
            M_total: list[list[float]],  # Accumulated 3×3 affine matrix
            width: int,                   # Current canvas width
            height: int,                  # Current canvas height
            rng: random.Random            # RNG for deterministic behavior
        ) -> tuple[list[list[float]], int, int]:
            '''Returns: (M_total_modified, new_width, new_height)'''

    Execution Flow:
        1. Accumulate affine ops into M_total
        2. Encounter barrier → call pre_flush_hook(M_total, W, H) if present
        3. Warp images using returned (M, W, H)
        4. Call barrier.apply() on warped images
        5. Update working (width, height) from output images

    Example: ExpandToFitAffine
        - Computes AABB of corners under M_total
        - Translates by (-minX, -minY) to shift to non-negative coords
        - Returns (translated_M, expanded_W, expanded_H)
        - Result: Rotated content fully visible without cropping

    Constraints:
        - Must be deterministic (use provided rng)
        - Must not access/modify images or geometry directly
        - Should validate returned dimensions are reasonable
    """

    def apply(
        self,
        images: list[Image.Image],
        geoms: list[dict[str, object]],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ) -> tuple[list[Image.Image], list[dict[str, object]]]: ...


class AugmentationPipeline(Protocol):
    allows_geometry_drops: bool
    last_summary: AugmentationTelemetry | None

    def apply(
        self,
        images: list[Image.Image],
        geoms: list[dict[str, object]],
        *,
        width: int,
        height: int,
        rng: RngLike,
    ) -> tuple[list[Image.Image], list[dict[str, object]]]: ...


class Compose:
    def __init__(self, ops: list[ImageAugmenter]):
        self.ops = list(ops)

        # Metadata storage for crop operations (propagated from operators)
        self.last_kept_indices: list[int] | None = None
        self.last_object_coverages: list[float] | None = None
        self.last_summary: AugmentationTelemetry | None = None
        self.last_padding_ratio: float | None = None
        self.last_image_width: int | None = None
        self.last_image_height: int | None = None
        self.last_crop_skip_reason: str | None = None
        self.last_skip_counters: dict[str, int] = {}
        self._augmentation_meta: list[dict[str, object]] | None = None
        self._augmentation_name_map: dict[str, list[ImageAugmenter]] | None = None
        self._curriculum_base_ops: dict[str, dict[str, NumericParam]] | None = None

        # Check if any operator allows geometry drops
        self.allows_geometry_drops = any(
            getattr(op, "allows_geometry_drops", False) for op in ops
        )

        # Best-effort name and curriculum metadata for pipelines built without the YAML builder
        name_map: dict[str, list[ImageAugmenter]] = {}
        curriculum_base: dict[str, dict[str, NumericParam]] = {}
        for op in self.ops:
            name = getattr(op, "_aug_name", None)
            if not name:
                name = re.sub(r"(?<!^)(?=[A-Z])", "_", op.__class__.__name__).lower()
                setattr(op, "_aug_name", name)
            name_map.setdefault(name, []).append(op)
            raw_curr = getattr(op, "curriculum_params", None)
            if callable(raw_curr):
                raw_curr = raw_curr()
            if isinstance(raw_curr, dict):
                for param_name, value in raw_curr.items():
                    numeric = (
                        value if isinstance(value, NumericParam) else NumericParam.from_raw(value)
                    )
                    if _is_prob_field(param_name):
                        for v in numeric.values:
                            if v < 0.0 or v > 1.0:
                                raise ValueError(
                                    f"augmentation op '{name}' probability '{param_name}' must be within [0,1]; got {v}"
                                )
                    curriculum_base.setdefault(name, {})[param_name] = numeric
        self._augmentation_name_map = name_map  # type: ignore[attr-defined]
        self._curriculum_base_ops = curriculum_base  # type: ignore[attr-defined]

    def apply(
        self,
        images: list[Image.Image],
        geoms: list[dict[str, object]],
        *,
        width: int,
        height: int,
        rng: object,
    ) -> tuple[list[Image.Image], list[dict[str, object]]]:
        out_images: list[Image.Image] = images
        out_geoms: list[dict[str, object]] = geoms

        # Clear metadata from previous run
        self.last_kept_indices = None
        self.last_object_coverages = None
        self.last_summary = None
        self.last_padding_ratio = None
        self.last_crop_skip_reason = None
        self.last_skip_counters = {}

        # Make width/height mutable for pre-flush hooks
        current_width = width
        current_height = height

        # Accumulate affine transforms, defer color ops, flush on barriers
        def _warp_images_with_matrix(
            imgs: list[Image.Image], M: list[list[float]], w: int, h: int
        ) -> list[Image.Image]:
            Minv = invert_affine(M)
            coeffs = (
                Minv[0][0],
                Minv[0][1],
                Minv[0][2],
                Minv[1][0],
                Minv[1][1],
                Minv[1][2],
            )
            # Use middle gray (128, 128, 128) for fill areas to achieve zero in normalized space
            # after Qwen3-VL's normalization: (pixel/255 - 0.5) / 0.5
            return [
                img.transform(
                    (w, h),
                    _PIL_AFFINE,
                    data=coeffs,
                    resample=_PIL_BICUBIC,
                    fillcolor=(128, 128, 128),
                )
                for img in imgs
            ]

        def _apply_affine_to_geoms(
            gs: list[dict[str, object]], M: list[list[float]], w: int, h: int
        ) -> list[dict[str, object]]:
            new_geoms: list[dict[str, object]] = []
            for g in gs:
                out = transform_geometry(g, M, width=w, height=h)
                new_geoms.append(out)
            return new_geoms

        M_total = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        deferred_color_ops: list[ImageAugmenter] = []

        def _is_identity_matrix(M: list[list[float]]) -> bool:
            return (
                abs(M[0][0] - 1.0) < 1e-9
                and abs(M[1][1] - 1.0) < 1e-9
                and abs(M[0][1]) < 1e-9
                and abs(M[1][0]) < 1e-9
                and abs(M[0][2]) < 1e-9
                and abs(M[1][2]) < 1e-9
            )

        def _flush_affine(force: bool = False):
            nonlocal out_images, out_geoms, M_total
            if not force and _is_identity_matrix(M_total):
                out_geoms = _apply_affine_to_geoms(
                    out_geoms, M_total, current_width, current_height
                )
                self.last_image_width = current_width
                self.last_image_height = current_height
                return
            out_images = _warp_images_with_matrix(
                out_images, M_total, current_width, current_height
            )
            out_geoms = _apply_affine_to_geoms(
                out_geoms, M_total, current_width, current_height
            )
            M_total = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            # Track latest image size for telemetry consumers
            self.last_image_width = current_width
            self.last_image_height = current_height

        def _is_affine_op(op: object) -> bool:
            return isinstance(op, AffineOp) or getattr(op, "kind", None) == "affine"

        def _is_color_op(op: object) -> bool:
            return isinstance(op, ColorOp) or getattr(op, "kind", None) == "color"

        def _is_patch_op(op: object) -> bool:
            return isinstance(op, PatchOp) or getattr(op, "kind", None) == "patch"

        for op in self.ops:
            if _is_affine_op(op):
                affine_fn = getattr(op, "affine", None)
                M_op = None
                if callable(affine_fn):
                    M_op = affine_fn(current_width, current_height, rng)
                if M_op is not None:
                    M_total = compose_affine(cast(list[list[float]], M_op), M_total)
                continue

            if _is_color_op(op):
                deferred_color_ops.append(op)
                continue

            # Barrier or Patch op: flush accumulated affines first
            pre_flush = getattr(op, "pre_flush_hook", None)
            if callable(pre_flush):
                pre_flush_result = pre_flush(M_total, current_width, current_height, rng)
                if not isinstance(pre_flush_result, tuple) or len(pre_flush_result) != 3:
                    raise TypeError("pre_flush_hook must return (M_total, width, height)")
                M_total, current_width, current_height = pre_flush_result
            force_flush = getattr(op, "force_flush_affine", False) or _is_patch_op(op)
            _flush_affine(force=force_flush)

            out_images, out_geoms = op.apply(
                out_images,
                out_geoms,
                width=current_width,
                height=current_height,
                rng=rng,
            )

            # Barrier/Patch may change image size (e.g., padding, crop); update width/height
            if isinstance(out_images, list) and out_images:
                im0 = out_images[0]
                if isinstance(im0, Image.Image):
                    current_width, current_height = im0.width, im0.height
                    self.last_image_width = current_width
                    self.last_image_height = current_height
                    # Update padding ratio if available
                    if hasattr(op, "padding_ratio"):
                        self.last_padding_ratio = getattr(op, "padding_ratio")
                    else:
                        self.last_padding_ratio = None

            # Propagate telemetry from crop-style PatchOps or barriers that emit it
            if _is_patch_op(op):
                has_crop_meta = bool(
                    getattr(op, "last_kept_indices", None)
                    or getattr(op, "last_crop_skip_reason", None)
                    or getattr(op, "last_object_coverages", None)
                )
                if has_crop_meta:
                    self._record_telemetry(op)
            elif hasattr(op, "last_kept_indices") or hasattr(op, "last_crop_skip_reason"):
                self._record_telemetry(op)

        # Final flush for any remaining accumulated affines
        _flush_affine()

        # Apply deferred color ops in order
        for op in deferred_color_ops:
            out_images, out_geoms = op.apply(
                out_images,
                out_geoms,
                width=current_width,
                height=current_height,
                rng=rng,
            )

        return out_images, out_geoms

    def _record_telemetry(self, op: object) -> None:
        kept = getattr(op, "last_kept_indices", None) or []
        coverages = getattr(op, "last_object_coverages", None) or []
        skip_reason = getattr(op, "last_crop_skip_reason", None)
        skip_counts_raw = getattr(op, "last_skip_counters", {}) or {}
        skip_counts: dict[str, int] = {}
        if isinstance(skip_counts_raw, dict):
            for key, value in skip_counts_raw.items():
                try:
                    skip_counts[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue

        telemetry = AugmentationTelemetry(
            kept_indices=tuple(int(idx) for idx in kept),
            coverages=tuple(float(c) for c in coverages),
            allows_geometry_drops=bool(self.allows_geometry_drops),
            width=self.last_image_width,
            height=self.last_image_height,
            padding_ratio=self.last_padding_ratio,
            skip_reason=skip_reason,
            skip_counts=skip_counts,
        )

        self.last_summary = telemetry
        self.last_kept_indices = list(telemetry.kept_indices) or None
        self.last_object_coverages = list(telemetry.coverages) or None
        self.last_crop_skip_reason = telemetry.skip_reason
        self.last_skip_counters = dict(telemetry.skip_counts)


__all__ = [
    "ImageAugmenter",
    "Compose",
    "AugmentationPipeline",
    "AffineOp",
    "ColorOp",
    "PatchOp",
]
