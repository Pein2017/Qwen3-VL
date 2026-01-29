#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized generation engine package."""

from __future__ import annotations

import importlib
from typing import Any

# Keep `import src.generation` cheap: contracts are lightweight, engines are heavyweight.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Contracts
    "ChatTemplateOptions": ("src.generation.contracts", "ChatTemplateOptions"),
    "DecodeOptions": ("src.generation.contracts", "DecodeOptions"),
    "GenerationOptions": ("src.generation.contracts", "GenerationOptions"),
    "GenerationResult": ("src.generation.contracts", "GenerationResult"),
    "ModelLoadConfig": ("src.generation.contracts", "ModelLoadConfig"),
    "StopOptions": ("src.generation.contracts", "StopOptions"),
    "TextGenerationRequest": ("src.generation.contracts", "TextGenerationRequest"),
    "VlmGenerationRequest": ("src.generation.contracts", "VlmGenerationRequest"),
    "VlmPreprocessOptions": ("src.generation.contracts", "VlmPreprocessOptions"),
    "VllmEngineOptions": ("src.generation.contracts", "VllmEngineOptions"),
    # Engine (heavyweight; imports torch/transformers/etc.)
    "GenerationEngine": ("src.generation.engine", "GenerationEngine"),
    "build_hf_engine": ("src.generation.engine", "build_hf_engine"),
    "build_vllm_engine": ("src.generation.engine", "build_vllm_engine"),
    # Stop-policy constants
    "QWEN_STOP_TOKENS": ("src.generation.stop_policy", "QWEN_STOP_TOKENS"),
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
