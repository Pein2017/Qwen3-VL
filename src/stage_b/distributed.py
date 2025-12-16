"""Stage-B distributed helpers (single-node torchrun only).

This module is intentionally lightweight:
- No-op in single-process runs.
- Initializes torch.distributed via env:// when WORLD_SIZE>1.
- Provides small helpers used by the Stage-B runner to coordinate rollout workers.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, List, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def is_distributed_available() -> bool:
    return dist.is_available()


def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_distributed_initialized():
        return dist.get_world_size()
    value = os.environ.get("WORLD_SIZE")
    return int(value) if value is not None else 1


def get_rank() -> int:
    if is_distributed_initialized():
        return dist.get_rank()
    value = os.environ.get("RANK")
    return int(value) if value is not None else 0


def get_local_rank() -> int:
    value = os.environ.get("LOCAL_RANK")
    return int(value) if value is not None else 0


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(*, timeout_seconds: int = 1800) -> None:
    """Initialize torch.distributed if launched under torchrun.

    This uses env:// rendezvous (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK).
    """
    if not is_distributed_available() or is_distributed_initialized():
        return

    world_size = get_world_size()
    if world_size <= 1:
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=int(timeout_seconds)),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def barrier() -> None:
    if is_distributed_initialized() and get_world_size() > 1:
        dist.barrier()


def broadcast_object(obj: Optional[T], *, src: int = 0) -> T:
    """Broadcast a Python object from src to all ranks and return it."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        assert obj is not None
        return obj

    payload: List[Optional[T]]
    if get_rank() == src:
        payload = [obj]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=src)
    result = payload[0]
    assert result is not None
    return result


def gather_object(obj: T, *, dst: int = 0) -> Optional[List[T]]:
    """Gather Python objects on dst. Returns list on dst, else None."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        return [obj]

    world_size = get_world_size()
    if get_rank() == dst:
        gathered: List[Optional[T]] = [None for _ in range(world_size)]
        dist.gather_object(obj, gathered, dst=dst)
        # dist.gather_object fills the list in rank order.
        return [item for item in gathered if item is not None]
    dist.gather_object(obj, None, dst=dst)
    return None


def all_gather_object(obj: T) -> List[T]:
    """All-gather Python objects across ranks in rank order."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        return [obj]

    world_size = get_world_size()
    gathered: List[Optional[T]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return [item for item in gathered if item is not None]


def broadcast_int(value: int, *, src: int = 0) -> int:
    if not is_distributed_initialized() or get_world_size() <= 1:
        return int(value)
    return int(broadcast_object(int(value) if get_rank() == src else None, src=src))


def broadcast_list_int(values: Sequence[int], *, src: int = 0) -> List[int]:
    """Broadcast a list of ints from src; returns list on all ranks."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        return [int(v) for v in values]
    return broadcast_object(list(values), src=src)
