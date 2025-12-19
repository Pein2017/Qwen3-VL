#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small performance helpers for Stage-B inference.

Keep these as opt-in toggles so we can debug memory issues without slowing down
normal runs.
"""

from __future__ import annotations

import os

import torch


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def maybe_empty_cache(reason: str = "") -> None:
    """Optionally call torch.cuda.empty_cache().

    torch.cuda.empty_cache() is a synchronization-heavy operation and can
    noticeably slow down token generation when called frequently. We keep it
    behind an env flag for rare OOM/debug scenarios.

    Enable with: STAGE_B_EMPTY_CACHE=1
    """

    if not torch.cuda.is_available():
        return
    if not _truthy_env("STAGE_B_EMPTY_CACHE", "0"):
        return
    # "reason" is intentionally unused (kept for potential logging hook).
    torch.cuda.empty_cache()


def enable_tf32() -> None:
    """Enable TF32 for CUDA matmuls where applicable.

    Note: TF32 primarily affects float32 matmuls; bf16/fp16 paths are unchanged.
    """

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

