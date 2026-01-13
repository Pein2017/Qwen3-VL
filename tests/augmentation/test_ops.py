from __future__ import annotations

from random import Random

from PIL import Image

from src.datasets.augment import apply_augmentations
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import (
    ColorJitter,
    HFlip,
    ResizeByScale,
    Rotate,
    Scale,
)
from src.datasets.contracts import DatasetObject
from src.datasets.geometry import canonicalize_polygon


def _mk_img(
    w: int = 64, h: int = 48, color: tuple[int, int, int] = (128, 128, 128)
) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def test_hflip_bbox_and_line_pixel_space():
    img = _mk_img()
    geoms: list[DatasetObject] = [
        {"bbox_2d": [10, 10, 30, 30]},
        {"line": [5, 5, 60, 40]},
    ]
    pipe = Compose([HFlip(1.0)])
    imgs_bytes, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(0))
    assert len(imgs_bytes) == 1
    assert "bbox_2d" in new_geoms[0]
    bb = new_geoms[0]["bbox_2d"]
    assert all(isinstance(v, int) for v in bb)
    assert 0 <= min(bb) and max(bb) <= 63
    assert "line" in new_geoms[1]
    ln = new_geoms[1]["line"]
    assert len(ln) >= 4 and all(isinstance(v, int) for v in ln)


def test_hflip_line_direction_is_endpoint_canonicalized() -> None:
    img = _mk_img(w=64, h=48)
    geoms: list[DatasetObject] = [{"line": [5, 5, 60, 40]}]
    pipe = Compose([HFlip(1.0)])

    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(0))
    assert "line" in new_geoms[0]
    line = new_geoms[0]["line"]

    # After hflip (x -> 63-x), endpoints become (58,5) and (3,40).
    # Canonical direction chooses the lexicographically smaller endpoint (3,40) first.
    assert line == [3, 40, 58, 5]


def test_rotate_poly_preserves_type_and_bounds():
    img = _mk_img()
    geoms: list[DatasetObject] = [{"poly": [10, 10, 40, 10, 40, 20, 10, 20]}]
    pipe = Compose([Rotate(10.0, 1.0)])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(123))
    assert "poly" in new_geoms[0]
    q = new_geoms[0]["poly"]
    assert len(q) == 8 and all(
        0 <= v <= 63 if i % 2 == 0 else 0 <= v <= 47 for i, v in enumerate(q)
    )


def test_scale_bbox_non_degenerate_or_fallback():
    img = _mk_img()
    geoms: list[DatasetObject] = [{"bbox_2d": [10, 10, 11, 11]}]
    pipe = Compose([Scale(0.5, 0.5, 1.0)])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(7))
    assert "bbox_2d" in new_geoms[0]
    bb = new_geoms[0]["bbox_2d"]
    assert isinstance(bb, list) and len(bb) == 4
    # either unchanged (fallback) or valid AABB
    assert (bb == [10, 10, 11, 11]) or (bb[0] <= bb[2] and bb[1] <= bb[3])


def test_color_jitter_geometry_unchanged():
    img = _mk_img()
    geoms: list[DatasetObject] = [{"line": [0, 0, 63, 47]}]
    pipe = Compose([ColorJitter()])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(5))
    assert "line" in new_geoms[0]
    assert new_geoms[0]["line"] == [0, 0, 63, 47]


def test_resize_by_scale_preserves_poly_vertex_count() -> None:
    img = _mk_img(w=64, h=48)
    poly = canonicalize_polygon(
        [
            10,
            10,
            30,
            10,
            35,
            20,
            30,
            30,
            10,
            30,
            5,
            20,
        ]
    )
    geoms: list[DatasetObject] = [{"poly": poly}]
    pipe = Compose([ResizeByScale(lo=1.5, hi=1.5, align_multiple=None, prob=1.0)])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(0))
    assert "poly" in new_geoms[0]
    assert len(new_geoms[0]["poly"]) == len(poly)
