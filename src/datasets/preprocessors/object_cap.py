from __future__ import annotations

import random
from typing import Any, Optional

from ..contracts import ConversationRecord
from .base import BasePreprocessor


class ObjectCapPreprocessor(BasePreprocessor):
    """Randomly downsample objects to a maximum count per image.

    Keeps images intact while trimming dense annotations to control sequence length.
    Sampling is random per call and can be driven by the `rng` attribute when set
    by the caller (e.g., dataset epoch/worker seed).
    """

    def __init__(self, max_objects: int, min_objects: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if max_objects <= 0:
            raise ValueError("max_objects must be positive")
        if min_objects <= 0:
            raise ValueError("min_objects must be positive")
        self.max_objects = int(max_objects)
        self.min_objects = int(min_objects)
        self.rng: Optional[random.Random] = None

    def preprocess(self, row: ConversationRecord) -> Optional[ConversationRecord]:
        objects = list(row.get("objects") or [])
        if len(objects) <= self.max_objects:
            return row

        keep = max(self.min_objects, self.max_objects)
        rng = self.rng if isinstance(self.rng, random.Random) else random
        # Sample indices, then restore original order for stable geometry ordering.
        keep_indices = sorted(rng.sample(range(len(objects)), keep))
        row_copy = dict(row)
        row_copy["objects"] = [objects[i] for i in keep_indices]
        return row_copy  # type: ignore[return-value]


__all__ = ["ObjectCapPreprocessor"]
