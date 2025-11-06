from typing import Any, Callable


class IdentityCollator:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def __call__(self, features):
        return self.fn(features)


__all__ = ["IdentityCollator"]
