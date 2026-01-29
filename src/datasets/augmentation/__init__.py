from __future__ import annotations

import importlib
from typing import Any

from .base import AffineOp, ColorOp, Compose, ImageAugmenter, PatchOp
from .registry import available, get, register

# Keep `import src.datasets.augmentation` cheap and avoid internal import cycles.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "build_compose_from_config": (
        "src.datasets.augmentation.builder",
        "build_compose_from_config",
    ),
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

__all__ = [
    "Compose",
    "ImageAugmenter",
    "AffineOp",
    "ColorOp",
    "PatchOp",
    "register",
    "get",
    "available",
    "build_compose_from_config",
]
