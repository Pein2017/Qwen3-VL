#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A per-image inference for GRPO group reasoning.

This module provides lightweight inference for generating Chinese single-line
summaries per image, grouped by mission and work order ID.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Heavyweight runtime entrypoint.
    "run_stage_a_inference": ("src.stage_a.inference", "run_stage_a_inference"),
    # Lightweight prompt helper.
    "build_user_prompt": ("src.stage_a.prompts", "build_user_prompt"),
    # Shared config constant.
    "SUPPORTED_MISSIONS": ("src.config.missions", "SUPPORTED_MISSIONS"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


__all__ = list(_LAZY_ATTRS.keys())
