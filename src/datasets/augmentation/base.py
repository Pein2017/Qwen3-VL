from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from PIL import Image

from ..geometry import (
    apply_affine,
    clamp_points,
    compose_affine,
    dedupe_consecutive_points,
    invert_affine,
    sutherland_hodgman_clip,
    to_clockwise,
    classify_affine_kind,
    clip_polyline_to_rect,
    transform_geometry,
)
from ..contracts import AugmentationTelemetry


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

    3. **Barrier Ops** (kind="barrier"): ResizeByScale, PadToMultiple, ExpandToFitAffine
       - Implement: apply(images, geoms, ...) -> (images, geoms)
       - Optionally: pre_flush_hook(M_total, width, height, rng) -> (M_total, width, height)
       - Barrier forces affine flush before and after; may change canvas size

    ## Pre-Flush Hook Protocol (Advanced)

    Barrier ops may implement `pre_flush_hook()` to modify accumulated affines and canvas
    dimensions BEFORE warping occurs. This enables operations like canvas expansion.

    Signature:
        def pre_flush_hook(
            self,
            M_total: List[List[float]],  # Accumulated 3×3 affine matrix
            width: int,                   # Current canvas width
            height: int,                  # Current canvas height
            rng: random.Random            # RNG for deterministic behavior
        ) -> Tuple[List[List[float]], int, int]:
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
        images: List[Any],
        geoms: List[Dict[str, Any]],
        *,
        width: int,
        height: int,
        rng: Any,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]: ...


class AugmentationPipeline(Protocol):
    allows_geometry_drops: bool
    last_summary: AugmentationTelemetry | None

    def apply(
        self,
        images: List[Any],
        geoms: List[Dict[str, Any]],
        *,
        width: int,
        height: int,
        rng: Any,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]: ...


class Compose:
    def __init__(self, ops: List[ImageAugmenter]):
        self.ops = list(ops)

        # Metadata storage for crop operations (propagated from operators)
        self.last_kept_indices: List[int] | None = None
        self.last_object_coverages: List[float] | None = None
        self.last_summary: AugmentationTelemetry | None = None
        self.last_padding_ratio: Optional[float] = None
        self.last_image_width: Optional[int] = None
        self.last_image_height: Optional[int] = None
        self.last_crop_skip_reason: Optional[str] = None
        self.last_skip_counters: Dict[str, int] = {}

        # Check if any operator allows geometry drops
        self.allows_geometry_drops = any(
            getattr(op, "allows_geometry_drops", False) for op in ops
        )

    def apply(
        self,
        images: List[Any],
        geoms: List[Dict[str, Any]],
        *,
        width: int,
        height: int,
        rng: Any,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        out_images: List[Any] = images
        out_geoms: List[Dict[str, Any]] = geoms

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
        def _warp_images_with_matrix(imgs: List[Any], M, w: int, h: int) -> List[Any]:
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
                (img if isinstance(img, Image.Image) else img).transform(
                    (w, h),
                    Image.AFFINE,
                    data=coeffs,
                    resample=Image.BICUBIC,
                    fillcolor=(128, 128, 128),
                )
                for img in imgs
            ]

        def _apply_affine_to_geoms(
            gs: List[Dict[str, Any]], M, w: int, h: int
        ) -> List[Dict[str, Any]]:
            new_geoms: List[Dict[str, Any]] = []
            for g in gs:
                out = transform_geometry(g, M, width=w, height=h)
                new_geoms.append(out)
            return new_geoms

        M_total = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        deferred_color_ops: List[Any] = []

        def _is_identity_matrix(M: List[List[float]]) -> bool:
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

        for op in self.ops:
            kind = getattr(op, "kind", None)
            if kind == "affine":
                M_op = op.affine(
                    current_width, current_height, rng
                )  # may be None on skip
                if M_op is not None:
                    M_total = compose_affine(M_op, M_total)
            elif kind == "color":
                deferred_color_ops.append(op)
            else:
                # Barrier: check for pre-flush hook, then flush, then apply barrier op
                if hasattr(op, "pre_flush_hook"):
                    M_total, current_width, current_height = op.pre_flush_hook(
                        M_total, current_width, current_height, rng
                    )
                force_flush = getattr(op, "force_flush_affine", False)
                _flush_affine(force=force_flush)
                out_images, out_geoms = op.apply(
                    out_images,
                    out_geoms,
                    width=current_width,
                    height=current_height,
                    rng=rng,
                )

                # Propagate crop metadata from operator to Compose
                if hasattr(op, "last_kept_indices") or hasattr(
                    op, "last_crop_skip_reason"
                ):
                    self._record_telemetry(op)

                # Barrier may change image size (e.g., padding); update width/height
                if isinstance(out_images, list) and out_images:
                    im0 = out_images[0]
                    if isinstance(im0, Image.Image):
                        current_width, current_height = im0.width, im0.height
                        # Update padding ratio if available
                        if hasattr(op, "padding_ratio"):
                            self.last_padding_ratio = getattr(op, "padding_ratio")
                        else:
                            self.last_padding_ratio = None

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

    def _record_telemetry(self, op: Any) -> None:
        kept = getattr(op, "last_kept_indices", None) or []
        coverages = getattr(op, "last_object_coverages", None) or []
        skip_reason = getattr(op, "last_crop_skip_reason", None)
        skip_counts_raw = getattr(op, "last_skip_counters", {}) or {}
        skip_counts: Dict[str, int] = {}
        if isinstance(skip_counts_raw, Dict):
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


__all__ = ["ImageAugmenter", "Compose", "AugmentationPipeline"]
