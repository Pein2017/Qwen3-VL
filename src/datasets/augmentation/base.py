from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

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
    min_area_rect,
    clip_polyline_to_rect,
    transform_geometry,
)


class ImageAugmenter(Protocol):
    def apply(
        self,
        images: List[Any],
        geoms: List[Dict[str, Any]],
        *,
        width: int,
        height: int,
        rng: Any,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        ...


class Compose:
    def __init__(self, ops: List[ImageAugmenter]):
        self.ops = list(ops)

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

        # Accumulate affine transforms, defer color ops, flush on barriers
        def _warp_images_with_matrix(imgs: List[Any], M) -> List[Any]:
            Minv = invert_affine(M)
            coeffs = (
                Minv[0][0], Minv[0][1], Minv[0][2],
                Minv[1][0], Minv[1][1], Minv[1][2],
            )
            return [
                (img if isinstance(img, Image.Image) else img).transform(
                    (width, height), Image.AFFINE, data=coeffs, resample=Image.BICUBIC
                ) for img in imgs
            ]

        def _apply_affine_to_geoms(gs: List[Dict[str, Any]], M) -> List[Dict[str, Any]]:
            import logging
            new_geoms: List[Dict[str, Any]] = []
            for g in gs:
                out = transform_geometry(g, M, width=width, height=height)
                new_geoms.append(out)
            return new_geoms

        M_total = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        deferred_color_ops: List[Any] = []

        def _flush_affine():
            nonlocal out_images, out_geoms, M_total
            out_images = _warp_images_with_matrix(out_images, M_total)
            out_geoms = _apply_affine_to_geoms(out_geoms, M_total)
            M_total = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        for op in self.ops:
            kind = getattr(op, "kind", None)
            if kind == "affine":
                M_op = op.affine(width, height, rng)  # may be None on skip
                if M_op is not None:
                    M_total = compose_affine(M_op, M_total)
            elif kind == "color":
                deferred_color_ops.append(op)
            else:
                # Barrier: flush current affines then apply barrier op
                _flush_affine()
                out_images, out_geoms = op.apply(out_images, out_geoms, width=width, height=height, rng=rng)
                # Barrier may change image size (e.g., padding); update width/height
                if isinstance(out_images, list) and out_images:
                    im0 = out_images[0]
                    if isinstance(im0, Image.Image):
                        width, height = im0.width, im0.height

        # Final flush for any remaining accumulated affines
        _flush_affine()

        # Apply deferred color ops in order
        for op in deferred_color_ops:
            out_images, out_geoms = op.apply(out_images, out_geoms, width=width, height=height, rng=rng)

        return out_images, out_geoms


__all__ = ["ImageAugmenter", "Compose"]


