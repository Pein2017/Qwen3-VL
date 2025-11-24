from __future__ import annotations

from random import Random

from PIL import Image

from src.datasets.augment import apply_augmentations
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import SmallObjectZoomPaste


def _mk_img(w=64, h=64, color=(128, 128, 128)):
    return Image.new("RGB", (w, h), color=color)


def test_small_object_zoom_paste_adds_instance():
    img = _mk_img()
    geoms = [{"bbox_2d": [5, 5, 9, 9], "desc": "螺丝"}]
    op = SmallObjectZoomPaste(
        prob=1.0,
        max_targets=1,
        max_attempts=10,
        scale=(2.0, 2.0),
        max_size=32,
        context=0.0,
        overlap_threshold=1.0,  # allow placement anywhere
    )
    pipe = Compose([op])
    imgs_bytes, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(42))
    assert len(imgs_bytes) == 1
    assert len(new_geoms) >= 2  # original + at least one pasted copy
    # New geometry should remain within bounds
    for g in new_geoms:
        if "bbox_2d" in g:
            x1, y1, x2, y2 = g["bbox_2d"]
            assert 0 <= x1 <= x2 <= 63
            assert 0 <= y1 <= y2 <= 63
