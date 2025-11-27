from __future__ import annotations

from random import Random

from PIL import Image

from src.datasets.augment import apply_augmentations
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import ColorJitter, HFlip, Rotate, Scale


def _mk_img(w=64, h=48, color=(128, 128, 128)):
    return Image.new("RGB", (w, h), color=color)


def test_hflip_bbox_and_line_pixel_space():
    img = _mk_img()
    geoms = [
        {"bbox_2d": [10, 10, 30, 30]},
        {"line": [5, 5, 60, 40]},
    ]
    pipe = Compose([HFlip(1.0)])
    imgs_bytes, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(0))
    assert len(imgs_bytes) == 1
    bb = new_geoms[0]["bbox_2d"]
    assert all(isinstance(v, int) for v in bb)
    assert 0 <= min(bb) and max(bb) <= 63
    ln = new_geoms[1]["line"]
    assert len(ln) >= 4 and all(isinstance(v, int) for v in ln)


def test_rotate_poly_preserves_type_and_bounds():
    img = _mk_img()
    geoms = [{"poly": [10, 10, 40, 10, 40, 20, 10, 20]}]
    pipe = Compose([Rotate(10.0, 1.0)])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(123))
    q = new_geoms[0]["poly"]
    assert len(q) == 8 and all(
        0 <= v <= 63 if i % 2 == 0 else 0 <= v <= 47 for i, v in enumerate(q)
    )


def test_scale_bbox_non_degenerate_or_fallback():
    img = _mk_img()
    geoms = [{"bbox_2d": [10, 10, 11, 11]}]
    pipe = Compose([Scale(0.5, 0.5, 1.0)])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(7))
    bb = new_geoms[0]["bbox_2d"]
    assert isinstance(bb, list) and len(bb) == 4
    # either unchanged (fallback) or valid AABB
    assert (bb == [10, 10, 11, 11]) or (bb[0] <= bb[2] and bb[1] <= bb[3])


def test_color_jitter_geometry_unchanged():
    img = _mk_img()
    geoms = [{"line": [0, 0, 63, 47]}]
    pipe = Compose([ColorJitter()])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(5))
    assert new_geoms[0]["line"] == [0, 0, 63, 47]
