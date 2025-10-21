from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple


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
        out_images, out_geoms = images, geoms
        for op in self.ops:
            out_images, out_geoms = op.apply(out_images, out_geoms, width=width, height=height, rng=rng)
        return out_images, out_geoms


__all__ = ["ImageAugmenter", "Compose"]


