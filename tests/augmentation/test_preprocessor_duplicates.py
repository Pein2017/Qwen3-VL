from __future__ import annotations

from random import Random

from PIL import Image

from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.ops import RandomCrop, ResizeByScale, SmallObjectZoomPaste, _geom_mask_iou
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


def _mk_record():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    # Put the *small* object second so we can verify __src_geom_idx mapping is respected.
    objects = [
        {"bbox_2d": [0, 0, 60, 60], "desc": "大目标"},
        {"bbox_2d": [5, 5, 9, 9], "desc": "小目标"},
    ]
    return {"images": [img], "objects": objects, "width": 64, "height": 64}


def test_preprocessor_appends_duplicate_objects_no_crop():
    op = SmallObjectZoomPaste(
        prob=1.0,
        max_targets=1,
        max_attempts=20,
        scale=(1.0, 1.0),
        max_size=12.0,  # only the second object qualifies
        context=0.0,
        overlap_threshold=1.0,  # never reject on overlap
    )
    # Ensure any provenance keys survive a later geometry rewrite op.
    resize = ResizeByScale(prob=1.0, lo=1.0, hi=1.0, align_multiple=None)
    pipeline = Compose([op, resize])
    pre = AugmentationPreprocessor(augmenter=pipeline, rng=Random(123), bypass_prob=0.0)

    rec = _mk_record()
    out = pre(rec)
    assert out is not None

    objs = out.get("objects") or []
    assert len(objs) == 3  # original 2 + 1 duplicated instance
    assert objs[-1]["desc"] == "小目标"
    assert "bbox_2d" in objs[-1] and objs[-1].get("poly") is None and objs[-1].get("line") is None


def test_preprocessor_appends_duplicate_objects_with_crop():
    crop = RandomCrop(
        scale=(1.0, 1.0),
        aspect_ratio=(1.0, 1.0),
        min_coverage=0.0,
        completeness_threshold=0.95,
        min_objects=1,
        skip_if_line=False,
        prob=1.0,
    )
    op = SmallObjectZoomPaste(
        prob=1.0,
        max_targets=1,
        max_attempts=20,
        scale=(1.0, 1.0),
        max_size=12.0,  # only the second object qualifies
        context=0.0,
        overlap_threshold=1.0,
    )
    resize = ResizeByScale(prob=1.0, lo=1.0, hi=1.0, align_multiple=None)
    pipeline = Compose([crop, op, resize])
    pre = AugmentationPreprocessor(augmenter=pipeline, rng=Random(123), bypass_prob=0.0)

    rec = _mk_record()
    out = pre(rec)
    assert out is not None

    objs = out.get("objects") or []
    assert len(objs) == 3
    assert objs[-1]["desc"] == "小目标"
    assert "bbox_2d" in objs[-1] and objs[-1].get("poly") is None and objs[-1].get("line") is None


def test_mask_iou_smoke():
    a = {"bbox_2d": [0.0, 0.0, 10.0, 10.0]}
    b = {"bbox_2d": [5.0, 5.0, 15.0, 15.0]}
    iou = _geom_mask_iou(a, b, width=32, height=32, line_buffer=4.0)
    assert 0.0 < iou < 1.0
