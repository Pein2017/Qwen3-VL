from __future__ import annotations

from random import Random

from PIL import Image

from src.datasets.augment import apply_augmentations
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import RoiCrop
from src.datasets.contracts import DatasetObject
from src.datasets.geometry import canonicalize_polygon, get_aabb


def _mk_img(
    w: int = 64, h: int = 64, color: tuple[int, int, int] = (128, 128, 128)
) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def test_roi_crop_skips_when_no_anchor_category_match():
    img = _mk_img()
    # Contains substring "BBU设备" in 备注, but 类别 is 标签, so this must NOT match as an anchor.
    geoms: list[DatasetObject] = [
        {
            "bbox_2d": [5, 5, 30, 30],
            "desc": "类别=标签,可见性=完整,备注=包含BBU设备字样但不是设备本体",
        }
    ]
    op = RoiCrop(
        anchor_classes=["BBU设备"],
        scale_range=(1.0, 1.0),
        min_crop_size=32,
        min_coverage=0.0,
        completeness_threshold=0.95,
        prob=1.0,
    )
    pipe = Compose([op])
    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(42))
    assert new_geoms == geoms
    assert op.last_crop_skip_reason == "no_anchor"


def test_roi_crop_triggers_with_anchor_category_match():
    img = _mk_img()
    geoms: list[DatasetObject] = [
        {
            "bbox_2d": [5, 5, 30, 30],
            "desc": "类别=BBU设备,可见性=完整,备注=设备本体",
        },
        {
            "bbox_2d": [45, 45, 55, 55],
            "desc": "类别=标签,可见性=完整,备注=角落标签",
        },
    ]
    op = RoiCrop(
        anchor_classes=["BBU设备"],
        scale_range=(1.0, 1.0),
        min_crop_size=32,
        min_coverage=0.0,
        completeness_threshold=0.95,
        prob=1.0,
    )
    pipe = Compose([op])
    imgs_bytes, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(123))
    assert len(imgs_bytes) == 1
    assert op.last_kept_indices is not None
    assert len(new_geoms) >= 1


def test_roi_crop_preserves_poly_shape_when_fully_inside_crop() -> None:
    img = _mk_img(w=100, h=100)
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
    geoms: list[DatasetObject] = [{"poly": poly, "desc": "类别=RRU设备,可见性=完整"}]

    op = RoiCrop(
        anchor_classes=["RRU设备"],
        # Use a >1.0 scale so the crop box has margin around the anchor polygon.
        # This avoids boundary-touch clipping artifacts (extra intersection vertices),
        # making the expected translation exact.
        scale_range=(1.5, 1.5),
        min_crop_size=1,
        min_coverage=0.0,
        completeness_threshold=0.95,
        prob=1.0,
    )
    pipe = Compose([op])

    _, new_geoms = apply_augmentations([img], geoms, pipe, rng=Random(0))
    assert "poly" in new_geoms[0]
    out_poly = new_geoms[0]["poly"]
    assert len(out_poly) == len(poly)

    # Recompute expected crop offset (matches RoiCrop.apply).
    x1, y1, x2, y2 = get_aabb({"poly": poly})
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    s = 1.5
    crop_w = max(int(round(w * s)), 1)
    crop_h = max(int(round(h * s)), 1)
    crop_w = max(1, min(crop_w, img.width))
    crop_h = max(1, min(crop_h, img.height))
    crop_x = int(round(cx - crop_w / 2.0))
    crop_y = int(round(cy - crop_h / 2.0))
    crop_x = max(0, min(crop_x, img.width - crop_w))
    crop_y = max(0, min(crop_y, img.height - crop_h))

    translated = [
        (float(v) - float(crop_x)) if i % 2 == 0 else (float(v) - float(crop_y))
        for i, v in enumerate(poly)
    ]
    assert out_poly == canonicalize_polygon(translated)
