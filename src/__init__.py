"""Qwen3-VL training package.

This package intentionally keeps `import src` lightweight:
- avoid importing heavyweight training/runtime dependencies at import time
- avoid importing heavyweight internal entrypoints as an indirect side effect

See OpenSpec change: 2026-01-21-refactor-src-architecture.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_LAZY_SUBMODULES: dict[str, str] = {
    # Keep these available for convenience, but lazy to keep imports cheap.
    "config": "src.config",
    "datasets": "src.datasets",
    "sft": "src.sft",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_SUBMODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value: ModuleType = module
    globals()[name] = value  # cache for subsequent attribute access
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES.keys()))


__all__ = list(_LAZY_SUBMODULES.keys())
