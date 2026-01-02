"""Shared context objects for summary GRPO rewards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .parsing import (
    ensure_list,
    extract_strict_json_line,
    extract_text,
    get_domain_token,
    get_summary_ref,
    is_irrelevant,
    split_lines,
)

_UNSET = object()


@dataclass(slots=True)
class SummarySample:
    completion: Any
    metadata: Any
    text: str
    lines: list[str]
    is_irrelevant: bool
    domain_token: str | None
    summary_ref: str | None
    strict_json_line: str | None
    _pred_json: Any = field(default=_UNSET, repr=False)
    _ref_json: Any = field(default=_UNSET, repr=False)

    @classmethod
    def from_inputs(cls, completion: Any, metadata: Any) -> "SummarySample":
        text = extract_text(completion)
        lines = split_lines(text)
        irrelevant = is_irrelevant(metadata)
        domain_token = get_domain_token(metadata)
        summary_ref = get_summary_ref(metadata)
        strict_json_line = extract_strict_json_line(
            meta=metadata,
            text=text,
            lines=lines,
            domain_token=domain_token,
        )
        return cls(
            completion=completion,
            metadata=metadata,
            text=text,
            lines=lines,
            is_irrelevant=irrelevant,
            domain_token=domain_token,
            summary_ref=summary_ref,
            strict_json_line=strict_json_line,
        )

    def pred_json(self) -> Any | None:
        if self._pred_json is _UNSET:
            if not self.strict_json_line:
                self._pred_json = None
            else:
                try:
                    self._pred_json = json.loads(self.strict_json_line)
                except Exception:
                    self._pred_json = None
        return None if self._pred_json is _UNSET else self._pred_json

    def ref_json(self) -> Any | None:
        if self._ref_json is _UNSET:
            if not self.summary_ref:
                self._ref_json = None
            else:
                try:
                    self._ref_json = json.loads(self.summary_ref)
                except Exception:
                    self._ref_json = None
        return None if self._ref_json is _UNSET else self._ref_json


def build_samples(completions: Any, metadata: Any) -> list[SummarySample]:
    completions_list = list(completions)
    metas = ensure_list(metadata, len(completions_list))
    return [
        SummarySample.from_inputs(completion, meta)
        for completion, meta in zip(completions_list, metas)
    ]


__all__ = ["SummarySample", "build_samples"]
