#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop-policy helpers shared across backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import StopOptions

QWEN_STOP_TOKENS: tuple[str, ...] = ("<|endoftext|>", "<|im_end|>")


def collect_single_token_stop_ids(
    stop_strings: Sequence[str], tokenizer: Any
) -> list[int]:
    stop_token_ids: list[int] = []
    if not stop_strings:
        return stop_token_ids
    for token in stop_strings:
        if not token:
            continue
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception:
            ids = []
        if len(ids) == 1:
            stop_token_ids.append(int(ids[0]))
    return stop_token_ids


def merge_stop_token_ids(
    base: Sequence[int] | None, extra: Sequence[int]
) -> list[int]:
    merged: list[int] = []
    if base:
        merged.extend(int(item) for item in base if item is not None)
    for item in extra:
        if item not in merged:
            merged.append(int(item))
    return merged


def merge_eos_token_ids(
    eos_token_id: int | Sequence[int] | None, stop_token_ids: Sequence[int]
) -> list[int]:
    eos_ids: list[int] = []
    if isinstance(eos_token_id, Sequence) and not isinstance(
        eos_token_id, (str, bytes)
    ):
        eos_ids.extend(int(item) for item in eos_token_id)
    elif isinstance(eos_token_id, int):
        eos_ids.append(eos_token_id)
    return merge_stop_token_ids(eos_ids, stop_token_ids)


def truncate_text_at_stops(text: str, stop_strings: Sequence[str]) -> str:
    if not text or not stop_strings:
        return text
    earliest: int | None = None
    for token in stop_strings:
        if not token:
            continue
        pos = text.find(token)
        if pos == -1:
            continue
        if earliest is None or pos < earliest:
            earliest = pos
    if earliest is None:
        return text
    return text[:earliest]


def normalize_stop_options(
    options: StopOptions, *, tokenizer: Any
) -> tuple[list[int], list[str]]:
    stop_strings = list(options.stop)
    stop_token_ids = list(options.stop_token_ids)
    stop_token_ids.extend(
        tid
        for tid in collect_single_token_stop_ids(stop_strings, tokenizer)
        if tid not in stop_token_ids
    )
    return stop_token_ids, stop_strings


__all__ = [
    "collect_single_token_stop_ids",
    "QWEN_STOP_TOKENS",
    "merge_eos_token_ids",
    "merge_stop_token_ids",
    "normalize_stop_options",
    "truncate_text_at_stops",
]
