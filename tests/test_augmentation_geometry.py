from __future__ import annotations

import random

from PIL import Image

from src.datasets.augmentation.base import Compose
from src.datasets.augmentation import ops as _ops  # register ops
from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.geometry import BBox, Quad, Polyline, transform_geometry


def _blank(w: int, h: int):
    return Image.new("RGB", (w, h), (0, 0, 0))


def test_rotate_then_resize_bbox_to_quad_and_clip():
    rng = random.Random(123)
    cfg = {"ops": [
        {"name": "rotate", "params": {"max_deg": 10.0, "prob": 1.0}},
        {"name": "resize_by_scale", "params": {"lo": 0.9, "hi": 0.9, "align_multiple": None, "prob": 1.0}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(100, 80)]
    geoms = [{"bbox_2d": [10, 10, 30, 30]}]
    out_imgs, out_geoms = compose.apply(imgs, geoms, width=100, height=80, rng=rng)
    assert out_imgs[0].size[0] == int(round(100 * 0.9))
    assert out_imgs[0].size[1] == int(round(80 * 0.9))
    g = out_geoms[0]
    assert "quad" in g
    q = g["quad"]
    assert len(q) == 8
    # all within bounds
    w, h = out_imgs[0].size
    xs = q[0::2]; ys = q[1::2]
    assert min(xs) >= 0 and max(xs) <= w - 1
    assert min(ys) >= 0 and max(ys) <= h - 1


def test_axis_aligned_hflip_then_resize_keeps_bbox():
    rng = random.Random(42)
    cfg = {"ops": [
        {"name": "hflip", "params": {"prob": 1.0}},
        {"name": "resize_by_scale", "params": {"lo": 2.0, "hi": 2.0, "align_multiple": None, "prob": 1.0}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(64, 32)]
    geoms = [{"bbox_2d": [4, 8, 20, 16]}]
    out_imgs, out_geoms = compose.apply(imgs, geoms, width=64, height=32, rng=rng)
    w, h = out_imgs[0].size
    assert (w, h) == (128, 64)
    g = out_geoms[0]
    assert "bbox_2d" in g and "quad" not in g
    x1, y1, x2, y2 = g["bbox_2d"]
    assert 0 <= x1 < x2 <= w - 1
    assert 0 <= y1 < y2 <= h - 1


def test_line_clipping_and_dedup():
    rng = random.Random(7)
    cfg = {"ops": [
        {"name": "rotate", "params": {"max_deg": 0.0, "prob": 1.0}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(50, 50)]
    # polyline that goes out of bounds
    geoms = [{"line": [-10, 10, 10, 10, 60, 10, 60, 60]}]
    _, out_geoms = compose.apply(imgs, geoms, width=50, height=50, rng=rng)
    g = out_geoms[0]
    assert "line" in g
    l = g["line"]
    assert len(l) >= 4
    xs = l[0::2]; ys = l[1::2]
    assert min(xs) >= 0 and max(xs) <= 49
    assert min(ys) >= 0 and max(ys) <= 49


def test_rotate_with_expansion_and_32_alignment():
    """Rotation with canvas expansion: no cropping, dims are multiples of 32, quads align."""
    rng = random.Random(42)
    cfg = {"ops": [
        {"name": "rotate", "params": {"max_deg": 30.0, "prob": 1.0}},
        {"name": "expand_to_fit_affine", "params": {"multiple": 32}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(100, 80)]
    geoms = [{"bbox_2d": [10, 10, 90, 70]}]
    
    out_imgs, out_geoms = compose.apply(imgs, geoms, width=100, height=80, rng=rng)
    w, h = out_imgs[0].size
    
    # Dimensions are multiples of 32
    assert w % 32 == 0 and h % 32 == 0
    # Expanded beyond original (rotated corners extend)
    assert w >= 100 or h >= 80
    # Quad promoted from bbox due to rotation
    assert "quad" in out_geoms[0]
    q = out_geoms[0]["quad"]
    assert len(q) == 8
    # All quad points within new bounds
    xs, ys = q[0::2], q[1::2]
    assert min(xs) >= 0 and max(xs) <= w - 1
    assert min(ys) >= 0 and max(ys) <= h - 1


def test_mixed_affines_with_expansion():
    """Rotate + scale + flip before expansion: AABB correct, geometry consistent."""
    rng = random.Random(123)
    cfg = {"ops": [
        {"name": "rotate", "params": {"max_deg": 15.0, "prob": 1.0}},
        {"name": "scale", "params": {"lo": 1.2, "hi": 1.2, "prob": 1.0}},
        {"name": "hflip", "params": {"prob": 1.0}},
        {"name": "expand_to_fit_affine", "params": {"multiple": 32}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(64, 64)]
    geoms = [
        {"bbox_2d": [5, 5, 20, 20]},
        {"quad": [30, 10, 50, 12, 48, 30, 28, 28]},
        {"line": [10, 50, 30, 50, 30, 60]}
    ]
    
    out_imgs, out_geoms = compose.apply(imgs, geoms, width=64, height=64, rng=rng)
    w, h = out_imgs[0].size
    
    assert w % 32 == 0 and h % 32 == 0
    assert len(out_geoms) == 3
    # Check bbox promoted to quad, quad still quad, line still line
    assert "quad" in out_geoms[0]  # bbox → quad
    assert "quad" in out_geoms[1]
    assert "line" in out_geoms[2]


def test_pixel_limit_enforcement():
    """Verify expand_to_fit_affine enforces max_pixels limit and scales down when exceeded."""
    rng = random.Random(999)
    # Large rotation on large image should trigger scaling
    max_pixels = 100000
    cfg = {"ops": [
        {"name": "rotate", "params": {"max_deg": 45.0, "prob": 1.0}},
        {"name": "expand_to_fit_affine", "params": {"multiple": 32, "max_pixels": max_pixels}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(400, 400)]
    geoms = [{"bbox_2d": [50, 50, 350, 350]}]
    
    out_imgs, out_geoms = compose.apply(imgs, geoms, width=400, height=400, rng=rng)
    w, h = out_imgs[0].size
    
    # Should be significantly smaller than unconstrained expansion (would be ~566×566 for 45° rotation)
    assert w < 400 and h < 400  # Scaled down from original
    # Pixel budget must be respected after alignment adjustments
    assert w * h <= max_pixels
    # Should still be multiple of 32
    assert w % 32 == 0 and h % 32 == 0
    # Geometry should still be valid
    assert "quad" in out_geoms[0]  # bbox promoted to quad due to rotation
    q = out_geoms[0]["quad"]
    assert len(q) == 8
    # All points within bounds
    xs, ys = q[0::2], q[1::2]
    assert min(xs) >= 0 and max(xs) <= w - 1
    assert min(ys) >= 0 and max(ys) <= h - 1


def test_pixel_limit_identity_alignment_scaling():
    """Identity transform with alignment still respects max_pixels."""
    rng = random.Random(123)
    max_pixels = 6400  # 80×80
    cfg = {"ops": [
        {"name": "expand_to_fit_affine", "params": {"multiple": 32, "max_pixels": max_pixels}},
    ]}
    compose: Compose = build_compose_from_config(cfg)
    imgs = [_blank(80, 120)]  # Alignment to 32 would produce 96×128 (12288 px) without scaling
    geoms = [{"bbox_2d": [10, 20, 60, 100]}]

    out_imgs, out_geoms = compose.apply(imgs, geoms, width=80, height=120, rng=rng)
    w, h = out_imgs[0].size

    assert w * h <= max_pixels
    assert w % 32 == 0 and h % 32 == 0
    geom = out_geoms[0]
    coords = geom.get("quad") or geom.get("bbox_2d")
    xs = coords[0::2]
    ys = coords[1::2]
    assert min(xs) >= 0 and max(xs) <= w - 1
    assert min(ys) >= 0 and max(ys) <= h - 1


