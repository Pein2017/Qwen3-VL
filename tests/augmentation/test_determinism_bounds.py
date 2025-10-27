from __future__ import annotations

import sys
from random import Random

from PIL import Image

sys.path.append('/data/Qwen3-VL')
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import HFlip, Rotate, Scale
from src.datasets.augment import apply_augmentations_v2


def _mk_img(w=128, h=96):
    return Image.new('RGB', (w, h), color=(80, 80, 80))


def test_determinism_same_seed():
    img = _mk_img()
    geoms = [{"bbox_2d": [20, 20, 40, 40]}, {"line": [10, 10, 100, 80]}]
    pipe = Compose([HFlip(0.7), Rotate(10.0, 1.0), Scale(0.9, 1.1, 1.0)])
    out1 = apply_augmentations_v2([img], geoms, pipe, rng=Random(123))
    out2 = apply_augmentations_v2([img], geoms, pipe, rng=Random(123))
    assert out1[1] == out2[1]


def test_bounds_clamped():
    img = _mk_img()
    geoms = [{"quad": [-50, -50, 200, -50, 200, 150, -50, 150]}]
    pipe = Compose([Rotate(0.0, 1.0), Scale(1.2, 1.2, 1.0)])
    _, new_geoms = apply_augmentations_v2([img], geoms, pipe, rng=Random(9))
    q = new_geoms[0]["quad"]
    w, h = 128, 96
    assert all(0 <= q[i] <= (w-1) if i%2==0 else 0 <= q[i] <= (h-1) for i in range(8))


