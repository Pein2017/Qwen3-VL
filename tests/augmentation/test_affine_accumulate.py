from __future__ import annotations

from random import Random
from typing import List, Dict, Any

from PIL import Image

from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import (
    ColorJitter,
    ExpandToFitAffine,
    Gamma,
    HFlip,
    ResizeByScale,
    Rotate,
    Scale,
)
from src.datasets.geometry import apply_affine, compose_affine


def _make_img(w: int = 64, h: int = 48) -> Image.Image:
    # simple gradient image
    im = Image.linear_gradient("L").resize((w, h)).convert("RGB")
    return im


def _geom_samples() -> List[Dict[str, Any]]:
    return [
        {"bbox_2d": [5, 6, 25, 20]},
        {"poly": [2, 2, 18, 4, 20, 22, 4, 18]},
        {"line": [1, 1, 10, 2, 15, 8]},
    ]


def test_affine_composition_matches_pointwise():
    rng = Random(123)
    w, h = 64, 48
    # Build ops and sample deterministic params via affine()
    hf = HFlip(prob=1.0)
    rt = Rotate(max_deg=10.0, prob=1.0)
    sc = Scale(lo=0.9, hi=1.1, prob=1.0)

    M_list = []
    for op in (hf, rt, sc):
        M = op.affine(w, h, rng)
        assert M is not None
        M_list.append(M)
    M_total = M_list[0]
    for M in M_list[1:]:
        M_total = compose_affine(M, M_total)

    # Compare point transform vs sequential
    pts = [5.0, 6.0, 25.0, 6.0, 25.0, 20.0, 5.0, 20.0]
    seq = pts
    for M in M_list:
        seq = apply_affine(seq, M)
    one = apply_affine(pts, M_total)
    assert all(abs(a - b) < 1e-4 for a, b in zip(seq, one))


def test_compose_apply_geometry_once():
    rng = Random(123)
    w, h = 64, 48
    img = _make_img(w, h)
    geoms = _geom_samples()
    pipe = Compose([HFlip(1.0), Rotate(10.0, 1.0), Scale(0.9, 1.1, 1.0)])
    out_imgs, out_geoms = pipe.apply([img], geoms, width=w, height=h, rng=rng)
    assert isinstance(out_imgs[0], Image.Image)
    assert len(out_geoms) == len(geoms)


def test_barrier_pad_mid_pipeline():
    rng = Random(1)
    w, h = 63, 50
    img = _make_img(w, h)
    geoms = _geom_samples()
    pipe = Compose([Rotate(10.0, 1.0), ExpandToFitAffine(32), Scale(0.95, 1.05, 1.0)])
    out_imgs, out_geoms = pipe.apply([img], geoms, width=w, height=h, rng=rng)
    w2, h2 = out_imgs[0].width, out_imgs[0].height
    assert w2 % 32 == 0 and h2 % 32 == 0
    assert len(out_geoms) == len(geoms)


def test_resize_by_scale_changes_size_and_scales_geoms():
    rng = Random(7)
    w, h = 60, 40
    img = _make_img(w, h)
    geoms = [{"bbox_2d": [10, 10, 20, 20]}]
    pipe = Compose([ResizeByScale(lo=1.5, hi=1.5, align_multiple=4, prob=1.0)])
    out_imgs, out_geoms = pipe.apply([img], geoms, width=w, height=h, rng=rng)
    nw, nh = out_imgs[0].width, out_imgs[0].height
    assert nw == 92 and nh == 60  # 60*1.5=90 -> align 4 => 92; 40*1.5=60
    bb = out_geoms[0]["bbox_2d"]
    # bbox scaled by 1.5 (with integer clamping tolerance)
    assert abs(bb[0] - 15) <= 1 and abs(bb[1] - 15) <= 1
    assert abs(bb[2] - 30) <= 1 and abs(bb[3] - 30) <= 1


def test_color_after_warp_no_geom_change():
    _rng = Random(2)  # noqa: F841
    w, h = 64, 64
    img = _make_img(w, h)
    geoms = _geom_samples()
    pipe1 = Compose([HFlip(1.0), Rotate(10.0, 1.0), Scale(0.9, 1.1, 1.0)])
    pipe2 = Compose(
        [
            HFlip(1.0),
            Rotate(10.0, 1.0),
            Scale(0.9, 1.1, 1.0),
            ColorJitter(prob=1.0),
            Gamma(prob=1.0),
        ]
    )
    _, g1 = pipe1.apply([img], geoms, width=w, height=h, rng=Random(42))
    _, g2 = pipe2.apply([img], geoms, width=w, height=h, rng=Random(42))
    assert g1 == g2


def test_determinism_same_rng():
    rng = Random(999)
    w, h = 64, 48
    img = _make_img(w, h)
    geoms = _geom_samples()
    pipe = Compose([Rotate(10.0, 1.0), Scale(0.95, 1.05, 1.0)])
    i1, g1 = pipe.apply([img], geoms, width=w, height=h, rng=rng)
    rng2 = Random(999)
    i2, g2 = pipe.apply([img], geoms, width=w, height=h, rng=rng2)
    # Pixel-by-pixel equality may not hold due to floating resample; sizes and geoms should match
    assert i1[0].size == i2[0].size
    assert g1 == g2
