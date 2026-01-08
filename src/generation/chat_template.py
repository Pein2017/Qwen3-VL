#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chat-template rendering helpers with safe compatibility fallbacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.utils import get_logger
from src.utils.unstructured import require_mapping_sequence

from .contracts import ChatTemplateOptions

logger = get_logger(__name__)


def render_chat_template(
    owner: object,
    messages: Sequence[Mapping[str, object]],
    *,
    options: ChatTemplateOptions,
) -> str:
    """Render chat-template messages using tokenizer or processor ownership."""

    require_mapping_sequence(messages, context="chat_template.messages")
    apply = getattr(owner, "apply_chat_template", None)
    if not callable(apply):
        raise AttributeError("chat_template owner has no apply_chat_template()")
    kwargs: dict[str, Any] = {
        "add_generation_prompt": options.add_generation_prompt,
        "tokenize": options.tokenize,
        "continue_final_message": options.continue_final_message,
        "enable_thinking": options.enable_thinking,
    }
    try:
        rendered = apply(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = apply(messages, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template must return a string when tokenize=False")
    return rendered


__all__ = ["render_chat_template"]
