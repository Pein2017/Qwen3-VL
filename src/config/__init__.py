"""Configuration management for YAML-based training setup.

This package is an *explicit boundary* between lightweight imports (`import src.config`)
and heavyweight training/runtime dependencies (ms-swift, torch, transformers).

To keep `import src.config` cheap, we lazily expose the public API via `__getattr__`.
`from src.config import ConfigLoader` is considered an explicit boundary import and may
pull in heavier dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Loader (heavyweight boundary)
    "ConfigLoader": ("src.config.loader", "ConfigLoader"),
    # Structured schema
    "TrainingConfig": ("src.config.schema", "TrainingConfig"),
    "CustomConfig": ("src.config.schema", "CustomConfig"),
    "PromptOverrides": ("src.config.schema", "PromptOverrides"),
    "VisualKDTargetConfig": ("src.config.schema", "VisualKDTargetConfig"),
    "VisualKDConfig": ("src.config.schema", "VisualKDConfig"),
    "DeepSpeedConfig": ("src.config.schema", "DeepSpeedConfig"),
    "SaveDelayConfig": ("src.config.schema", "SaveDelayConfig"),
    "GrpoChordConfig": ("src.config.schema", "GrpoChordConfig"),
    "GrpoConfig": ("src.config.schema", "GrpoConfig"),
    # Prompt defaults
    "SYSTEM_PROMPT": ("src.config.prompts", "SYSTEM_PROMPT"),
    "USER_PROMPT": ("src.config.prompts", "USER_PROMPT"),
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
