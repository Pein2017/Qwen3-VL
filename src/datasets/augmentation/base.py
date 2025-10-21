from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from PIL import Image

from ..geometry import (
    apply_affine,
    clamp_points,
    compose_affine,
    dedupe_consecutive_points,
    invert_affine,
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
            new_geoms: List[Dict[str, Any]] = []
            for g in gs:
                if "bbox_2d" in g:
                    x1, y1, x2, y2 = g["bbox_2d"]
                    pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                    t = apply_affine(pts, M)
                    xs = t[0::2]; ys = t[1::2]
                    bb = clamp_points([min(xs), min(ys), max(xs), max(ys)], width, height)
                    if bb[0] == bb[2] or bb[1] == bb[3]:
                        new_geoms.append({"bbox_2d": [x1, y1, x2, y2]})
                    else:
                        new_geoms.append({"bbox_2d": bb})
                elif "quad" in g:
                    t = apply_affine(g["quad"], M)
                    q = clamp_points(t, width, height)
                    if min(q[0::2]) == max(q[0::2]) and min(q[1::2]) == max(q[1::2]):
                        new_geoms.append({"quad": g["quad"]})
                    else:
                        new_geoms.append({"quad": q})
                elif "line" in g:
                    t = apply_affine(g["line"], M)
                    l = clamp_points(t, width, height)
                    l = dedupe_consecutive_points(l)
                    if len(l) < 4:
                        new_geoms.append({"line": g["line"]})
                    else:
                        new_geoms.append({"line": l})
                else:
                    new_geoms.append(g)
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


