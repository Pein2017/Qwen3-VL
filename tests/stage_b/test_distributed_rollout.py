from __future__ import annotations

import pytest

from src import distributed as stage_dist
from src.stage_b.runner import _shard_bounds


@pytest.mark.parametrize(
    ("total", "world_size"),
    [
        (0, 1),
        (1, 1),
        (7, 1),
        (0, 2),
        (1, 2),
        (7, 2),
        (8, 3),
        (9, 8),
        (16, 8),
    ],
)
def test_shard_bounds_partitions_cover_range(total: int, world_size: int) -> None:
    if world_size < 1:
        pytest.skip("world_size must be >= 1")

    all_indices: list[int] = []
    sizes: list[int] = []
    for rank in range(world_size):
        start, end = _shard_bounds(total, world_size=world_size, rank=rank)
        assert 0 <= start <= end <= total
        sizes.append(end - start)
        all_indices.extend(range(start, end))

    assert sorted(all_indices) == list(range(total))
    if sizes:
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(
    ("world_size", "per_rank", "total"),
    [
        # Final rollout batch smaller than global_batch_size = world_size * per_rank.
        (8, 2, 3),
        (8, 2, 15),
        (8, 2, 16),
        (8, 4, 1),
        (8, 4, 31),
        (8, 4, 32),
        (2, 16, 1),
        (2, 16, 31),
        (2, 16, 32),
    ],
)
def test_shard_bounds_never_exceeds_per_rank_when_total_leq_global(
    world_size: int, per_rank: int, total: int
) -> None:
    global_batch_size = world_size * per_rank
    assert total <= global_batch_size

    shard_sizes: list[int] = []
    all_indices: list[int] = []
    for rank in range(world_size):
        start, end = _shard_bounds(total, world_size=world_size, rank=rank)
        shard_sizes.append(end - start)
        all_indices.extend(range(start, end))

    assert sorted(all_indices) == list(range(total))
    assert all(size <= per_rank for size in shard_sizes)


def test_shard_bounds_invalid_rank_raises() -> None:
    with pytest.raises(ValueError):
        _shard_bounds(10, world_size=2, rank=-1)
    with pytest.raises(ValueError):
        _shard_bounds(10, world_size=2, rank=2)


def test_distributed_helpers_single_process_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure env-based distributed detection does not trigger.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    obj = {"k": "v"}
    assert stage_dist.broadcast_object(obj, src=0) == obj
    assert stage_dist.broadcast_int(7, src=0) == 7
    assert stage_dist.gather_object({"a": 1}, dst=0) == [{"a": 1}]
    assert stage_dist.all_gather_object({"b": 2}) == [{"b": 2}]


def test_broadcast_int_mocked_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_dist, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(stage_dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(stage_dist, "get_rank", lambda: 1)  # non-src

    def _fake_broadcast_object_list(payload, src):  # noqa: ANN001
        assert src == 0
        # Simulate rank0 broadcasting 123
        payload[0] = 123

    monkeypatch.setattr(
        stage_dist.dist, "broadcast_object_list", _fake_broadcast_object_list
    )
    assert stage_dist.broadcast_int(0, src=0) == 123


def test_gather_object_mocked_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_dist, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(stage_dist, "get_world_size", lambda: 2)

    gathered_calls: dict[str, object] = {}

    def _fake_gather_object(obj, object_list, dst):  # noqa: ANN001
        gathered_calls["obj"] = obj
        gathered_calls["dst"] = dst
        if object_list is not None:
            object_list[0] = {"rank0": True}
            object_list[1] = {"rank1": True}

    monkeypatch.setattr(stage_dist.dist, "gather_object", _fake_gather_object)

    monkeypatch.setattr(stage_dist, "get_rank", lambda: 0)
    assert stage_dist.gather_object({"rank0": True}, dst=0) == [
        {"rank0": True},
        {"rank1": True},
    ]

    monkeypatch.setattr(stage_dist, "get_rank", lambda: 1)
    assert stage_dist.gather_object({"rank1": True}, dst=0) is None
    assert gathered_calls["dst"] == 0
