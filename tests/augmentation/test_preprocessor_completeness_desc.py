from __future__ import annotations

from random import Random
from typing import cast

from PIL import Image

from src.datasets.augmentation.base import Compose, PatchOp
from src.datasets.contracts import ConversationRecord, DatasetObject
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


class _FakeCropTelemetry(PatchOp):
    """PatchOp stub that emits crop telemetry without changing geometry.

    This lets us unit-test the augmentation preprocessor's completeness/visibility
    updates without depending on a particular crop implementation.
    """

    def __init__(self, coverages: list[float], *, completeness_threshold: float = 0.95):
        self.coverages = list(coverages)
        self.completeness_threshold = float(completeness_threshold)
        self.allows_geometry_drops = True
        self.last_kept_indices: list[int] | None = None
        self.last_object_coverages: list[float] | None = None
        self.last_crop_skip_reason: str | None = None
        self.last_skip_counters: dict[str, int] = {}

    def apply(self, images, geoms, *, width: int, height: int, rng):  # type: ignore[override]
        self.last_kept_indices = list(range(len(geoms)))
        # Pad/truncate coverages to match geoms length for deterministic tests.
        covs = list(self.coverages)
        if len(covs) < len(geoms):
            covs.extend([1.0] * (len(geoms) - len(covs)))
        self.last_object_coverages = covs[: len(geoms)]
        self.last_crop_skip_reason = None
        self.last_skip_counters = {}
        return images, geoms


def test_preprocessor_updates_only_visibility_token_not_other_complete_words():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    record = cast(
        ConversationRecord,
        {
            "images": [img],
            "objects": [
                {
                    "bbox_2d": [0, 0, 10, 10],
                    "desc": "类别=BBU设备,品牌=华为,可见性=完整,备注=品牌未显示完整，无法判断是否足够空间安装挡风板",
                },
                {
                    "bbox_2d": [20, 20, 30, 30],
                    "desc": "类别=BBU安装螺丝,可见性=完整,符合性=符合",
                },
            ],
            "width": 64,
            "height": 64,
        },
    )

    # First object is partially visible; second object stays complete.
    op = _FakeCropTelemetry([0.5, 1.0], completeness_threshold=0.95)
    pipeline = Compose([op])
    pre = AugmentationPreprocessor(augmenter=pipeline, rng=Random(123), bypass_prob=0.0)

    out = pre(record)
    assert out is not None
    objs = cast(list[DatasetObject], out.get("objects") or [])

    desc0 = cast(str, objs[0].get("desc"))
    assert desc0.startswith("类别=BBU设备")
    assert "可见性=部分" in desc0
    # Ensure we did NOT rewrite unrelated semantics (e.g., 备注=...未显示完整).
    assert "品牌未显示完整" in desc0

    desc1 = cast(str, objs[1].get("desc"))
    assert "可见性=完整" in desc1


def test_preprocessor_does_not_touch_display_complete_in_remarks_when_visibility_already_partial():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    record = cast(
        ConversationRecord,
        {
            "images": [img],
            "objects": [
                {
                    "bbox_2d": [0, 0, 10, 10],
                    "desc": "类别=BBU设备,可见性=部分,备注=品牌未显示完整",
                }
            ],
            "width": 64,
            "height": 64,
        },
    )

    # Even if coverage is low, this desc already says 部分; we must not mutate 备注.
    op = _FakeCropTelemetry([0.1], completeness_threshold=0.95)
    pipeline = Compose([op])
    pre = AugmentationPreprocessor(augmenter=pipeline, rng=Random(123), bypass_prob=0.0)

    out = pre(record)
    assert out is not None
    objs = cast(list[DatasetObject], out.get("objects") or [])

    desc0 = cast(str, objs[0].get("desc"))
    assert desc0 == "类别=BBU设备,可见性=部分,备注=品牌未显示完整"
