#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rollout sampler for the Stage-B rule-search pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import re
from datetime import datetime, timezone
from typing import cast

import torch
from transformers import PreTrainedTokenizerBase

from src.generation import (
    ChatTemplateOptions,
    DecodeOptions,
    GenerationEngine,
    GenerationOptions,
    QWEN_STOP_TOKENS,
    StopOptions,
    TextGenerationRequest,
)
from src.generation.chat_template import render_chat_template

from .config import SamplerConfig
from .sampling.prompts import build_messages
from .types import (
    DecodeConfig,
    GroupLabel,
    GroupTicket,
    MissionGuidance,
    ParsedTrajectory,
    Trajectory,
)
from .utils.chinese import normalize_spaces, to_simplified
from .utils.perf import maybe_empty_cache

logger = logging.getLogger(__name__)


_ASSISTANT_MARKERS: Sequence[str] = (
    "assistant\n",
    "assistant:",
    "assistant",
    "Assistant\n",
    "Assistant:",
    "Assistant",
    "<im_start>assistant\n",
    "<|im_start|>assistant\n",
)

_DEFAULT_STOP: tuple[str, ...] = QWEN_STOP_TOKENS

_DEFAULT_MAX_PROMPT_TOKENS = 4096


def _trim_assistant_prefix(text: str) -> str:
    last_index = -1
    marker_length = 0
    for marker in _ASSISTANT_MARKERS:
        candidate = text.rfind(marker)
        if candidate != -1 and candidate >= last_index:
            last_index = candidate
            marker_length = len(marker)
    if last_index == -1:
        return text
    return text[last_index + marker_length :]


def _normalize_verdict(text: str) -> GroupLabel | None:
    cleaned = normalize_spaces(to_simplified(text or "")).strip()
    cleaned = cleaned.replace(" ", "").lower()
    if cleaned in {"通过", "pass", "通过。"}:
        return "pass"
    if cleaned in {"不通过", "fail", "未通过", "不通过。"}:
        return "fail"
    return None


def _parse_two_line_response(
    response: str,
) -> tuple[bool, GroupLabel | None, str | None]:
    """Parse strict two-line protocol: Verdict + Reason (binary only)."""

    text = _trim_assistant_prefix(response).strip()
    if not text:
        return False, None, None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 2:
        return False, None, None

    verdict_line, reason_line = lines
    if not verdict_line.lower().startswith("verdict"):
        return False, None, None
    if not reason_line.lower().startswith("reason"):
        return False, None, None

    verdict_parts = re.split(r"[:：]", verdict_line, maxsplit=1)
    if len(verdict_parts) != 2:
        return False, None, None
    verdict_text = verdict_parts[1].strip()
    verdict = _normalize_verdict(verdict_text)
    if verdict is None:
        return False, None, None

    reason_parts = re.split(r"[:：]", reason_line, maxsplit=1)
    if len(reason_parts) != 2:
        return False, None, None
    reason = reason_parts[1].strip()
    if not reason:
        return False, None, None

    return True, verdict, reason


class RolloutSampler:
    """Generate multi-attempt candidates for Stage-B tickets."""

    def __init__(
        self,
        engine: GenerationEngine,
        config: SamplerConfig,
    ) -> None:
        self.engine: GenerationEngine = engine
        self.tokenizer: PreTrainedTokenizerBase = cast(
            PreTrainedTokenizerBase, engine.tokenizer
        )
        self.config: SamplerConfig = config
        self.chat_template = ChatTemplateOptions(
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_messages(
        self, ticket: GroupTicket, guidance: MissionGuidance, *, domain: str
    ) -> Sequence[Mapping[str, object]]:
        return build_messages(ticket, guidance, domain=domain)

    def _build_prompt(
        self, ticket: GroupTicket, guidance: MissionGuidance, *, domain: str
    ) -> str:
        messages = self._build_messages(ticket, guidance, domain=domain)
        return render_chat_template(
            self.tokenizer, messages, options=self.chat_template
        )

    def _count_prompt_tokens(self, prompts: Sequence[str]) -> list[int | None]:
        if not prompts:
            return []
        try:
            encoded = self.tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            if isinstance(encoded, Mapping):
                attention_mask = encoded.get("attention_mask")
                if isinstance(attention_mask, torch.Tensor):
                    lengths = attention_mask.sum(dim=1).tolist()
                    result: list[int | None] = [
                        int(value) if value is not None else None for value in lengths
                    ]
                    return result
                input_ids = encoded.get("input_ids")
                if isinstance(input_ids, torch.Tensor):
                    result: list[int | None] = [
                        int((row != 0).sum().item()) for row in input_ids
                    ]
                    return result
        except Exception:  # noqa: BLE001
            pass

        lengths: list[int | None] = []
        for prompt in prompts:
            try:
                encoded = self.tokenizer(prompt, truncation=False)
                input_ids = getattr(encoded, "input_ids", None)
                if input_ids is None and isinstance(encoded, Mapping):
                    input_ids = encoded.get("input_ids")
                lengths.append(len(input_ids) if input_ids is not None else None)
            except Exception:  # noqa: BLE001
                lengths.append(None)
        return lengths

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _generate_with_messages(
        self,
        messages: Sequence[Sequence[Mapping[str, object]]],
        decode: DecodeConfig,
        sample_offset: int,
    ) -> list[str]:
        if not messages:
            return []

        stop_tokens = decode.stop if decode.stop else _DEFAULT_STOP
        options = GenerationOptions(
            max_new_tokens=decode.max_new_tokens,
            temperature=decode.temperature,
            top_p=decode.top_p,
            repetition_penalty=decode.repetition_penalty,
            no_repeat_ngram_size=decode.no_repeat_ngram_size,
            stop=StopOptions(stop=tuple(stop_tokens)),
            seed=(decode.seed + sample_offset) if decode.seed is not None else None,
            decode=DecodeOptions(
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                strip_whitespace=True,
            ),
        )

        requests = [TextGenerationRequest(messages=msg) for msg in messages]
        results = self.engine.generate_text_batch(requests, options=options)
        maybe_empty_cache("rollout.generate")

        outputs: list[str] = []
        for result in results:
            trimmed = result.text.strip()
            for marker in _ASSISTANT_MARKERS:
                if trimmed.startswith(marker):
                    trimmed = trimmed[len(marker) :]
                    break
            outputs.append(trimmed.strip())
        return outputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_for_batch(
        self,
        tickets: Sequence[GroupTicket],
        guidance_map: Mapping[str, MissionGuidance],
        domain_map: Mapping[str, str],
    ) -> tuple[Mapping[str, list[ParsedTrajectory]], tuple[str, ...]]:
        if not tickets:
            return {}, ()

        messages_list: list[Sequence[Mapping[str, object]]] = []
        prompts: list[str] = []
        for ticket in tickets:
            if ticket.mission not in guidance_map:
                raise KeyError(f"Missing mission guidance for {ticket.mission}")
            if ticket.mission not in domain_map:
                raise ValueError(
                    f"Missing domain mapping for mission '{ticket.mission}'"
                )
            messages = self._build_messages(
                ticket,
                guidance_map[ticket.mission],
                domain=domain_map[ticket.mission],
            )
            messages_list.append(messages)
            prompts.append(
                render_chat_template(
                    self.tokenizer, messages, options=self.chat_template
                )
            )

        max_prompt_tokens = (
            self.config.max_prompt_tokens
            if self.config.max_prompt_tokens is not None
            else _DEFAULT_MAX_PROMPT_TOKENS
        )
        prompt_lengths = self._count_prompt_tokens(prompts)
        kept_tickets: list[GroupTicket] = []
        kept_messages: list[Sequence[Mapping[str, object]]] = []
        dropped_keys: list[str] = []
        for ticket, prompt, length, messages in zip(
            tickets, prompts, prompt_lengths, messages_list
        ):
            if length is None or length > max_prompt_tokens:
                dropped_keys.append(ticket.key)
            else:
                kept_tickets.append(ticket)
                kept_messages.append(messages)

        if not kept_tickets:
            return {}, tuple(dropped_keys)

        per_group: dict[str, list[ParsedTrajectory]] = {
            ticket.key: [] for ticket in kept_tickets
        }
        counters: dict[str, int] = {ticket.key: 0 for ticket in kept_tickets}

        for decode in self.config.grid:
            for sample_index in range(self.config.samples_per_decode):
                responses = self._generate_with_messages(
                    kept_messages, decode, sample_index
                )
                if len(responses) != len(kept_tickets):
                    raise RuntimeError("Sampler returned mismatched response count")

                current_time = datetime.now(timezone.utc)
                for ticket, response_text in zip(kept_tickets, responses):
                    ticket_key = ticket.key
                    candidate_index = counters[ticket_key]
                    counters[ticket_key] += 1

                    base = Trajectory(
                        group_id=ticket.group_id,
                        mission=ticket.mission,
                        candidate_index=candidate_index,
                        decode=decode,
                        response_text=response_text,
                        created_at=current_time,
                    )
                    format_ok, verdict, reason = _parse_two_line_response(response_text)

                    per_group[ticket_key].append(
                        ParsedTrajectory(
                            base=base,
                            verdict=verdict,
                            reason=reason,
                            format_ok=format_ok,
                        )
                    )

        maybe_empty_cache("rollout.batch_end")

        return per_group, tuple(dropped_keys)


__all__ = ["RolloutSampler"]
