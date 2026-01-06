"""Shared context objects for dense GRPO rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..summary.parsing import ensure_list, extract_text, split_lines
from .parsing import (
    DenseObject,
    DenseParsedPayload,
    DenseParsedStrict,
    parse_dense_completion_strict,
    parse_dense_payload_mapping,
)

_UNSET = object()


@dataclass(frozen=True)
class DenseSample:
    """A single GRPO reward sample for dense-mode detection.

    This object is immutable (Schema Constitution: internal state uses frozen dataclasses).
    It still supports cached parsing/matching artifacts via internal sentinel fields.
    """

    completion: Any
    metadata: Any
    assistant_payload: Any

    text: str
    lines: list[str]

    _pred_strict: DenseParsedStrict | object = field(default=_UNSET, repr=False)
    _gt_payload: DenseParsedPayload | object = field(default=_UNSET, repr=False)

    @classmethod
    def from_inputs(
        cls, completion: Any, metadata: Any, assistant_payload: Any
    ) -> "DenseSample":
        text = extract_text(completion)
        lines = split_lines(text)
        return cls(
            completion=completion,
            metadata=metadata,
            assistant_payload=assistant_payload,
            text=text,
            lines=lines,
        )

    def pred_strict(self) -> DenseParsedStrict:
        cached = self._pred_strict
        if isinstance(cached, DenseParsedStrict):
            return cached
        parsed = parse_dense_completion_strict(text=self.text, meta=self.metadata)
        object.__setattr__(self, "_pred_strict", parsed)
        return parsed

    def gt_payload(self) -> DenseParsedPayload:
        cached = self._gt_payload
        if isinstance(cached, DenseParsedPayload):
            return cached
        parsed = parse_dense_payload_mapping(raw=self.assistant_payload, path="assistant_payload")
        object.__setattr__(self, "_gt_payload", parsed)
        return parsed


def build_samples(
    completions: Iterable[Any],
    metadata: Any,
    assistant_payload: Any,
) -> list[DenseSample]:
    completions_list = list(completions)
    metas = ensure_list(metadata, len(completions_list))
    payloads = ensure_list(assistant_payload, len(completions_list))
    return [
        DenseSample.from_inputs(completion, meta, payload)
        for completion, meta, payload in zip(completions_list, metas, payloads)
    ]


__all__ = [
    "DenseObject",
    "DenseSample",
    "DenseParsedPayload",
    "DenseParsedStrict",
    "build_samples",
]
