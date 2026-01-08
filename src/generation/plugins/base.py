#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plugin protocol for optional generation-time behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class GenerationPlugin(Protocol):
    """Optional generation plugin interface."""

    name: str
    backends: tuple[str, ...]
    request_types: tuple[str, ...]
    uses_hf_stopping_criteria: bool
    uses_hf_logits_processor: bool

    def hf_stopping_criteria(self, prompt_lengths: Sequence[int]) -> Any: ...

    def hf_logits_processor(self, prompt_lengths: Sequence[int]) -> Any: ...

    def postprocess_text(self, text: str) -> str: ...


__all__ = ["GenerationPlugin"]
