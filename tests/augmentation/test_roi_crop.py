from __future__ import annotations

from random import Random

from PIL import Image

from src.datasets.augment import apply_augmentations
from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import RoiCrop


def _mk_img(w=64, h=64, color=(128, 128, 128)):
    return Image.new("RGB", (w, h), color=color)


def test_roi_crop_skips_when_no_anchor_category_match():
    img = _mk_img()
    # Contains substring "BBU设备" in 备注, but 类别 is 标签, so this must NOT match as an anchor.
    geoms = [
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
    geoms = [
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
