from __future__ import annotations

from random import Random

import pytest
from PIL import Image

from src.datasets.augmentation.base import Compose
from src.datasets.augmentation.curriculum import AugmentationCurriculumScheduler
from src.datasets.augmentation.ops import RoiCrop
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


def _build_op_meta() -> list[dict[str, dict[str, float]]]:
    return [
        {
            "name": "rotate",
            "params": {"prob": 0.1, "max_deg": 8.0, "scale": [0.9, 1.0]},
        }
    ]


def test_linear_phase_ramp():
    curriculum = {
        "phases": [
            {
                "until_step": 5,
                "bypass_prob": 0.2,
                "ops": {"rotate": {"prob": 0.5, "scale": [0.95, 1.05]}},
            },
            {
                "until_step": 10,
                "bypass_prob": 0.4,
                "ops": {"rotate": {"prob": 0.8, "scale": [1.05, 1.15]}},
            },
        ]
    }
    scheduler = AugmentationCurriculumScheduler.from_config(
        base_bypass=0.1,
        op_meta=_build_op_meta(),
        curriculum_raw=curriculum,
    )

    state_start = scheduler.get_state(0)
    assert state_start["bypass_prob"] == pytest.approx(0.1)
    assert state_start["ops"]["rotate"]["prob"] == pytest.approx(0.1)
    assert state_start["ops"]["rotate"]["scale"] == [0.9, 1.0]

    state_mid = scheduler.get_state(2)
    assert state_mid["bypass_prob"] == pytest.approx(0.14)
    assert state_mid["ops"]["rotate"]["prob"] == pytest.approx(0.26)
    assert state_mid["ops"]["rotate"]["scale"] == pytest.approx([0.92, 1.02])

    state_phase_end = scheduler.get_state(5)
    assert state_phase_end["bypass_prob"] == pytest.approx(0.2)
    assert state_phase_end["ops"]["rotate"]["prob"] == pytest.approx(0.5)

    state_next = scheduler.get_state(7)
    assert state_next["bypass_prob"] == pytest.approx(0.28)
    assert state_next["ops"]["rotate"]["prob"] == pytest.approx(0.62)
    assert state_next["ops"]["rotate"]["scale"] == pytest.approx([0.99, 1.09])

    state_last = scheduler.get_state(20)
    assert state_last["bypass_prob"] == pytest.approx(0.4)
    assert state_last["ops"]["rotate"]["prob"] == pytest.approx(0.8)
    assert state_last["ops"]["rotate"]["scale"] == pytest.approx([1.05, 1.15])


def test_invalid_curriculum_raises():
    curriculum = {
        "phases": [
            {"until_step": 5, "ops": {"rotate": {"prob": 0.5}}},
            {"until_step": 3, "ops": {"rotate": {"prob": 0.7}}},
        ]
    }
    with pytest.raises(ValueError):
        AugmentationCurriculumScheduler.from_config(
            base_bypass=0.1, op_meta=_build_op_meta(), curriculum_raw=curriculum
        )


def test_percent_curriculum_resolves_with_total_steps():
    curriculum = {
        "phases": [
            {"until_percent": 50, "ops": {"rotate": {"prob": 0.5}}},
            {"until_percent": 100, "ops": {"rotate": {"prob": 0.8}}},
        ]
    }
    scheduler = AugmentationCurriculumScheduler.from_config(
        base_bypass=0.1, op_meta=_build_op_meta(), curriculum_raw=curriculum
    )
    with pytest.raises(ValueError):
        scheduler.get_state(0)
    scheduler.set_total_steps(10)
    state_mid = scheduler.get_state(7)
    assert state_mid["ops"]["rotate"]["prob"] == pytest.approx(0.62)


def test_override_unknown_param_fail_fast():
    curriculum = {
        "phases": [{"until_step": 5, "ops": {"rotate": {"unknown_param": 0.5}}}]
    }
    with pytest.raises(ValueError):
        AugmentationCurriculumScheduler.from_config(
            base_bypass=0.1, op_meta=_build_op_meta(), curriculum_raw=curriculum
        )


def test_probability_bounds_enforced():
    curriculum = {"phases": [{"until_step": 5, "ops": {"rotate": {"prob": 1.2}}}]}
    with pytest.raises(ValueError):
        AugmentationCurriculumScheduler.from_config(
            base_bypass=0.1, op_meta=_build_op_meta(), curriculum_raw=curriculum
        )


def test_curriculum_override_preserves_types_for_ops():
    op = RoiCrop(
        anchor_classes=["BBU设备"],
        scale_range=(1.2, 1.3),
        min_crop_size=320,
        min_coverage=0.4,
        completeness_threshold=0.95,
        prob=1.0,
    )
    pipeline = Compose([op])
    state = {
        "step": 0,
        "bypass_prob": 0.0,
        "ops": {
            "roi_crop": {
                "min_crop_size": 384.0,
                "scale_range": [1.1, 1.4],
            }
        },
    }
    pre = AugmentationPreprocessor(
        augmenter=pipeline, curriculum_state=state, rng=Random(123)
    )

    pre._sync_curriculum()

    assert isinstance(op.min_crop_size, int)
    assert isinstance(op.scale_range, tuple)

    # Should run without TypeError from float indices
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    geoms = [{"bbox_2d": [2, 2, 30, 30], "desc": "类别=BBU设备,可见性=完整"}]
    imgs = [img]
    out_images, _ = op.apply(imgs, geoms, width=32, height=32, rng=Random(1))
    assert len(out_images) == len(imgs)
