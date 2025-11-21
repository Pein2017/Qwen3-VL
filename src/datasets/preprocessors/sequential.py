from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from .base import BasePreprocessor


class SequentialPreprocessor(BasePreprocessor):
    """Apply multiple preprocessors in order."""

    def __init__(self, preprocessors: Sequence[BasePreprocessor], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not preprocessors:
            raise ValueError("SequentialPreprocessor requires at least one preprocessor")
        self.preprocessors = list(preprocessors)
        self._curriculum_state: Any = None
        self._rng: Any = None

    @property
    def curriculum_state(self) -> Any:
        return self._curriculum_state

    @curriculum_state.setter
    def curriculum_state(self, state: Any) -> None:
        self._curriculum_state = state
        for pre in self.preprocessors:
            if hasattr(pre, "curriculum_state"):
                try:
                    pre.curriculum_state = state
                except Exception:
                    pass

    @property
    def rng(self) -> Any:
        return self._rng

    @rng.setter
    def rng(self, rng_obj: Any) -> None:
        self._rng = rng_obj
        for pre in self.preprocessors:
            if hasattr(pre, "rng"):
                try:
                    pre.rng = rng_obj
                except Exception:
                    pass

    def preprocess(self, row: Any) -> Optional[Any]:
        current = row
        for pre in self.preprocessors:
            current = pre(current)
            if current is None:
                return None
        return current


__all__ = ["SequentialPreprocessor"]
