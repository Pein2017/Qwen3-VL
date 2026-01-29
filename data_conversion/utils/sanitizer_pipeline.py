#!/usr/bin/env python3
"""
Composable sanitizer pipeline with consistent error handling.

Provides:
- SanitizerStep: describes a single sanitizer callable
- SanitizerPipeline: executes steps sequentially with fail-fast semantics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional


logger = logging.getLogger(__name__)


class SanitizerError(RuntimeError):
    """Raised when a mandatory sanitizer fails."""


SanitizerFunc = Callable[[Optional[str]], Optional[str]]


@dataclass(frozen=True)
class SanitizerStep:
    """Description of a sanitizer in the pipeline."""

    name: str
    func: SanitizerFunc
    mandatory: bool = True


class SanitizerPipeline:
    """Executes sanitizers sequentially with configurable error handling."""

    def __init__(
        self,
        steps: Iterable[SanitizerStep],
        *,
        fail_fast: bool = True,
    ) -> None:
        self.steps: List[SanitizerStep] = list(steps)
        self.fail_fast = fail_fast

    def run(self, description: Optional[str]) -> Optional[str]:
        """Apply each sanitizer in order."""
        result = description
        for step in self.steps:
            try:
                result = step.func(result)
            except Exception as exc:  # pragma: no cover - defensive guard
                message = (
                    f"Sanitizer '{step.name}' failed "
                    f"({'mandatory' if step.mandatory else 'optional'})"
                )
                if step.mandatory and self.fail_fast:
                    raise SanitizerError(message) from exc
                logger.warning("%s: %s", message, exc)
        return result
