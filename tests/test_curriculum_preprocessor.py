import random

from src.datasets.augmentation import ops as _register_ops  # noqa: F401

import pytest

from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


def test_curriculum_state_applied():
    augmenter = build_compose_from_config(
        {"ops": [{"name": "hflip", "params": {"prob": 0.1}}]}
    )
    curriculum_state = {
        "step": 1,
        "bypass_prob": 0.8,
        "ops": {"hflip": {"prob": 0.65}},
    }
    preprocessor = AugmentationPreprocessor(
        augmenter=augmenter,
        rng=random.Random(42),
        bypass_prob=0.0,
        curriculum_state=curriculum_state,
    )

    preprocessor._sync_curriculum()

    assert preprocessor.bypass_prob == pytest.approx(0.8)
    hflip = augmenter.ops[0]
    assert getattr(hflip, "prob", None) == pytest.approx(0.65)


def test_curriculum_state_invalid_param_raises():
    augmenter = build_compose_from_config(
        {"ops": [{"name": "hflip", "params": {"prob": 0.5}}]}
    )
    curriculum_state = {
        "step": 1,
        "bypass_prob": 0.2,
        "ops": {"hflip": {"unknown": 0.3}},
    }
    preprocessor = AugmentationPreprocessor(
        augmenter=augmenter,
        rng=random.Random(0),
        bypass_prob=0.0,
        curriculum_state=curriculum_state,
    )
    with pytest.raises(ValueError):
        preprocessor._sync_curriculum()


def test_curriculum_state_prob_bounds():
    augmenter = build_compose_from_config(
        {"ops": [{"name": "hflip", "params": {"prob": 0.5}}]}
    )
    curriculum_state = {
        "step": 1,
        "bypass_prob": 1.5,
        "ops": {},
    }
    preprocessor = AugmentationPreprocessor(
        augmenter=augmenter,
        rng=random.Random(0),
        bypass_prob=0.0,
        curriculum_state=curriculum_state,
    )
    with pytest.raises(ValueError):
        preprocessor._sync_curriculum()
