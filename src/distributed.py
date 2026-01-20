"""Shared distributed helpers (single-node torchrun only).

This module is intentionally lightweight:
- No-op in single-process runs.
- Initializes torch.distributed via env:// when WORLD_SIZE>1.
- Provides small helpers used by Stage-A and Stage-B runtimes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import timedelta
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")




# PyTorch "object" collectives (broadcast_object_list/gather_object/all_gather_object)
# serialize Python objects into byte tensors. When the default process group backend
# is NCCL, those tensors are moved to the current CUDA device, which can OOM when
# objects are large (e.g. rollout payloads) or GPU memory is already tight.
#
# To keep object collectives off-GPU, we create a dedicated GLOO process group and
# use it for object ops while leaving the default group (often NCCL) untouched.
_OBJECT_COLLECTIVE_GROUP: object | None = None
_OBJECT_COLLECTIVE_GROUP_READY: bool = False
_OBJECT_COLLECTIVE_TIMEOUT_SECONDS: int = 1800


def _ensure_object_collective_group() -> None:
    """Ensure we have a CPU (gloo) group for object collectives when using NCCL."""

    global _OBJECT_COLLECTIVE_GROUP
    global _OBJECT_COLLECTIVE_GROUP_READY

    if _OBJECT_COLLECTIVE_GROUP_READY:
        return

    # Don't mark as ready until the default group exists. This keeps the helper
    # safe if called before init_distributed() in single-process paths.
    if not is_distributed_initialized() or get_world_size() <= 1:
        return

    _OBJECT_COLLECTIVE_GROUP_READY = True

    # Only needed when default backend is NCCL.
    try:
        backend = dist.get_backend()
    except Exception:  # noqa: BLE001
        backend = None
    if backend != "nccl":
        _OBJECT_COLLECTIVE_GROUP = None
        return

    try:
        _OBJECT_COLLECTIVE_GROUP = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=int(_OBJECT_COLLECTIVE_TIMEOUT_SECONDS)),
        )
    except Exception:  # noqa: BLE001
        # Fall back to default group if gloo group creation fails.
        _OBJECT_COLLECTIVE_GROUP = None

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

    # Keep object collectives off-GPU when the default backend is NCCL.
    global _OBJECT_COLLECTIVE_TIMEOUT_SECONDS
    _OBJECT_COLLECTIVE_TIMEOUT_SECONDS = int(timeout_seconds)
    _ensure_object_collective_group()


def barrier() -> None:
    if is_distributed_initialized() and get_world_size() > 1:
        dist.barrier()


def broadcast_object(obj: T | None, *, src: int = 0) -> T:
    """Broadcast a Python object from src to all ranks and return it."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        assert obj is not None
        return obj

    _ensure_object_collective_group()
    group = _OBJECT_COLLECTIVE_GROUP

    payload: list[T | None]
    if get_rank() == src:
        payload = [obj]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=src, group=group)
    result = payload[0]
    assert result is not None
    return result


def gather_object(obj: T, *, dst: int = 0) -> list[T] | None:
    """Gather Python objects on dst. Returns list on dst, else None."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        return [obj]

    _ensure_object_collective_group()

    world_size = get_world_size()
    if get_rank() == dst:
        gathered: list[T | None] = [None for _ in range(world_size)]
        dist.gather_object(obj, gathered, dst=dst, group=_OBJECT_COLLECTIVE_GROUP)
        # dist.gather_object fills the list in rank order.
        return [item for item in gathered if item is not None]
    dist.gather_object(obj, None, dst=dst, group=_OBJECT_COLLECTIVE_GROUP)
    return None


def all_gather_object(obj: T) -> list[T]:
    """All-gather Python objects across ranks in rank order."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        return [obj]

    _ensure_object_collective_group()

    world_size = get_world_size()
    gathered: list[T | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj, group=_OBJECT_COLLECTIVE_GROUP)
    return [item for item in gathered if item is not None]


def broadcast_int(value: int, *, src: int = 0) -> int:
    if not is_distributed_initialized() or get_world_size() <= 1:
        return int(value)
    return int(broadcast_object(int(value) if get_rank() == src else None, src=src))


def broadcast_list_int(values: Sequence[int], *, src: int = 0) -> list[int]:
    """Broadcast a list of ints from src; returns list on all ranks."""
    if not is_distributed_initialized() or get_world_size() <= 1:
        return [int(v) for v in values]
    return broadcast_object(list(values), src=src)


__all__ = [
    "all_gather_object",
    "barrier",
    "broadcast_int",
    "broadcast_list_int",
    "broadcast_object",
    "gather_object",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed_available",
    "is_distributed_initialized",
    "is_main_process",
]
