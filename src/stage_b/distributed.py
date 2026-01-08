"""Stage-B distributed helpers (single-node torchrun only).

This module exists for backward compatibility (tests and legacy imports).

The canonical, stage-neutral implementation lives in `src/distributed.py` and is
used by the Stage-A and Stage-B runtime entrypoints. Keep this module's surface
API stable to avoid churn in downstream tooling.
"""

from __future__ import annotations

import torch.distributed as dist

from ..distributed import (
    all_gather_object,
    barrier,
    broadcast_int,
    broadcast_list_int,
    broadcast_object,
    gather_object,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed_available,
    is_distributed_initialized,
    is_main_process,
)

__all__ = [
    "all_gather_object",
    "barrier",
    "broadcast_int",
    "broadcast_list_int",
    "broadcast_object",
    "dist",
    "gather_object",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed_available",
    "is_distributed_initialized",
    "is_main_process",
]
