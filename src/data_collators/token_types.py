"""Helpers for token-type telemetry (desc/coord/format) on assistant payloads."""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase
from src.utils import get_logger

logger = get_logger(__name__)


class TokenType:
    IGNORE = -1
    DESC = 1
    COORD = 2
    FORMAT = 3


def _dumps_with_types(payload: Any) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Serialize payload to JSON text and collect typed character spans.

    Span tuple: (start, end, type_id) in character offsets.
    """

    spans: List[Tuple[int, int, int]] = []
    parts: List[str] = []

    def write(text: str, typ: int) -> None:
        start = sum(len(p) for p in parts)
        parts.append(text)
        end = start + len(text)
        if end > start:
            spans.append((start, end, typ))

    def emit_value(value: Any, context: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            text = json.dumps(value, ensure_ascii=False)
            write(text, TokenType.COORD if context == "coord" else TokenType.FORMAT)
        elif isinstance(value, str):
            text = json.dumps(value, ensure_ascii=False)
            write(text, TokenType.DESC if context == "desc" else TokenType.FORMAT)
        elif isinstance(value, list):
            write("[", TokenType.FORMAT)
            for idx, item in enumerate(value):
                if idx > 0:
                    write(", ", TokenType.FORMAT)
                emit_value(item, context)
            write("]", TokenType.FORMAT)
        elif isinstance(value, dict):
            write("{", TokenType.FORMAT)
            for idx, (k, v) in enumerate(value.items()):
                if idx > 0:
                    write(", ", TokenType.FORMAT)
                key_text = json.dumps(k, ensure_ascii=False)
                write(key_text, TokenType.FORMAT)
                write(": ", TokenType.FORMAT)
                next_ctx = "desc" if k == "desc" else "coord" if k in {"bbox_2d", "poly", "line", "line_points"} else "format"
                emit_value(v, next_ctx)
            write("}", TokenType.FORMAT)
        else:
            # Fallback for unexpected types (booleans/null) -> format
            text = json.dumps(value, ensure_ascii=False)
            write(text, TokenType.FORMAT)

    emit_value(payload, "format")
    text_out = "".join(parts)
    return text_out, spans


def _apply_suffix(text: str, suffix: Iterable[str] | None) -> Tuple[str, List[Tuple[int, int, int]]]:
    if not suffix:
        return text, []
    suffix_text = "".join(suffix)
    if not suffix_text:
        return text, []
    start = len(text)
    end = start + len(suffix_text)
    return text + suffix_text, [(start, end, TokenType.FORMAT)]


def _char_types_from_spans(length: int, spans: Sequence[Tuple[int, int, int]]) -> List[int]:
    arr = [TokenType.FORMAT] * length
    for start, end, typ in spans:
        for i in range(start, min(end, length)):
            arr[i] = typ
    return arr


def compute_token_types(
    *,
    tokenizer: PreTrainedTokenizerBase,
    payload: Any,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    suffix_tokens: Iterable[str] | None = None,
) -> torch.Tensor | None:
    """Compute per-token types aligned to labels.

    Returns a 1D tensor (seq_len) with types or None on mismatch.
    """

    if labels.dim() != 1:
        raise ValueError("labels must be 1D for a single sample")

    text, spans = _dumps_with_types(payload)
    text, suffix_spans = _apply_suffix(text, suffix_tokens)
    spans = spans + suffix_spans

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return None

    char_types = _char_types_from_spans(len(text), spans)
    token_types: List[int] = []
    for start, end in offsets:
        if start is None or end is None:
            token_types.append(TokenType.FORMAT)
            continue
        if end <= start or start < 0:
            token_types.append(TokenType.FORMAT)
            continue
        slice_types = char_types[start:end]
        # Majority vote; fallback to first if empty
        if slice_types:
            counts = {t: slice_types.count(t) for t in set(slice_types)}
            token_types.append(max(counts, key=counts.get))
        else:
            token_types.append(TokenType.FORMAT)

    supervised_positions = [
        idx for idx, flag in enumerate(labels.tolist()) if flag != -100
    ]
    supervised_count = len(supervised_positions)
    if supervised_count == 0:
        return torch.full_like(labels, TokenType.IGNORE)

    if len(token_types) != supervised_count:
        logger.debug(
            "Token-type length mismatch (text=%d, supervised=%d); padding/truncating to align.",
            len(token_types),
            supervised_count,
        )

    # Align token_types to the supervised positions (assistant tokens only).
    aligned = list(token_types[:supervised_count])
    if len(aligned) < supervised_count:
        aligned.extend([TokenType.FORMAT] * (supervised_count - len(aligned)))

    full_types = [TokenType.IGNORE] * labels.shape[0]
    for pos, typ in zip(supervised_positions, aligned):
        if 0 <= pos < len(full_types):
            full_types[pos] = typ

    return torch.tensor(full_types, dtype=torch.long)


__all__ = ["TokenType", "compute_token_types"]
