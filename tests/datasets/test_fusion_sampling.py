import random

from src.datasets.fusion import _sample_indices


def test_without_replacement_within_pool_deterministic():
    rng = random.Random(123)
    indices, fallback = _sample_indices(
        pool_len=5, quota=3, rng=rng, sample_without_replacement=True
    )

    assert indices == [3, 1, 4]
    assert len(indices) == len(set(indices))
    assert fallback is False


def test_without_replacement_fallback_when_quota_exceeds_pool():
    rng = random.Random(321)
    indices, fallback = _sample_indices(
        pool_len=2, quota=3, rng=rng, sample_without_replacement=True
    )

    # Falls back to replacement, but remains deterministic.
    assert indices == [1, 1, 0]
    assert fallback is True
