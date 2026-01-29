#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend protocol for the generation engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..contracts import (
    GenerationOptions,
    GenerationResult,
    TextGenerationRequest,
    VlmGenerationRequest,
)


class GenerationBackend(Protocol):
    name: str
    supports_vlm: bool
    tokenizer: object | None
    processor: object | None

    def generate_text_batch(
        self,
        requests: Sequence[TextGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]: ...

    def generate_vlm_batch(
        self,
        requests: Sequence[VlmGenerationRequest],
        options: GenerationOptions,
        *,
        plugins: Sequence[object] | None = None,
    ) -> list[GenerationResult]: ...


__all__ = ["GenerationBackend"]
