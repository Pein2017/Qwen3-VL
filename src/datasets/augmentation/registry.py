from __future__ import annotations

from typing import Any, Callable, Dict


_REGISTRY: Dict[str, Any] = {}


def register(name: str) -> Callable[[Any], Any]:
    def deco(obj: Any) -> Any:
        _REGISTRY[name] = obj
        return obj

    return deco


def get(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"augmentation op '{name}' is not registered")
    return _REGISTRY[name]


def available() -> Dict[str, Any]:
    return dict(_REGISTRY)


__all__ = ["register", "get", "available"]
