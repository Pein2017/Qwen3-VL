#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rollout sampler for the Stage-B rule-search pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
import re
from datetime import datetime, timezone
from typing import cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

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

_DEFAULT_STOP: tuple[str, ...] = (
    "\nassistant",
    "assistant\n",
    "assistant:",
    "Assistant:",
    "<|endoftext|>",
    "</s>",
    "<|im_end|>",
)

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
    cleaned = text.strip().replace(" ", "").lower()
    if cleaned in {"通过", "pass", "通过。"}:
        return "pass"
    if cleaned in {"不通过", "fail", "未通过", "不通过。"}:
        return "fail"
    # Third-state / pending phrases are forbidden in Stage-B inference outputs.
    if any(
        term in cleaned for term in ["复核", "不确定", "无法判断", "无法判定", "待复核"]
    ):
        return None
    if cleaned in {"通过需复核", "通过需要复核", "通过需要复核。", "通过需复核。"}:
        return None
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

    forbidden = (
        "需复核",
        "需人工复核",
        "need-review",
        "needreview",
        "证据不足",
        "待定",
        "通过但需复核",
        "通过但需人工复核",
    )
    simplified_reason = normalize_spaces(to_simplified(reason))
    if any(term in simplified_reason for term in forbidden):
        return False, None, None

    return True, verdict, reason


class RolloutSampler:
    """Generate multi-attempt candidates for Stage-B tickets."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: SamplerConfig,
        *,
        device: str | None = None,
    ) -> None:
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.config: SamplerConfig = config
        self.device: torch.device | str = (
            device or (model.device if hasattr(model, "device") else "cpu")
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_prompt(
        self, ticket: GroupTicket, guidance: MissionGuidance, *, domain: str
    ) -> str:
        messages = build_messages(ticket, guidance, domain=domain)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Disable Qwen3 "thinking" blocks (<think>...</think>) to keep outputs simple
            enable_thinking=False,
        )
        assert isinstance(rendered, str), "apply_chat_template must return string"
        return rendered

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
    def _generate_with_prompts(
        self,
        prompts: Sequence[str],
        decode: DecodeConfig,
        sample_offset: int,
    ) -> list[str]:
        if not prompts:
            return []

        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        input_len = inputs["input_ids"].shape[1]

        do_sample = decode.temperature > 0
        stop_tokens = decode.stop if decode.stop else _DEFAULT_STOP
        # Treat common chat terminators as EOS to hard-stop generation
        stop_token_ids = []
        for token in stop_tokens:
            try:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
            except Exception:
                ids = []
            if len(ids) == 1:
                stop_token_ids.append(ids[0])

        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, int):
                eos_ids.append(eos_token_id)
        for tid in stop_token_ids:
            if tid not in eos_ids:
                eos_ids.append(tid)
        generator_kwargs = {
            "max_new_tokens": decode.max_new_tokens,
            "temperature": decode.temperature if do_sample else None,
            "top_p": decode.top_p,
            "do_sample": do_sample,
            "repetition_penalty": decode.repetition_penalty,
            "no_repeat_ngram_size": decode.no_repeat_ngram_size,
            "pad_token_id": self.tokenizer.pad_token_id
            or (eos_ids[0] if eos_ids else None),
            "eos_token_id": eos_ids or None,
            "return_dict_in_generate": True,
            "use_cache": True,  # Explicitly enable KV cache for faster inference
        }
        generator_kwargs = {k: v for k, v in generator_kwargs.items() if v is not None}

        if decode.seed is not None:
            torch.manual_seed(decode.seed + sample_offset)

        with torch.inference_mode():
            generate_fn = cast(
                Callable[..., torch.Tensor | GenerateOutput],
                getattr(self.model, "generate"),
            )
            generation = generate_fn(**inputs, **generator_kwargs)
            maybe_empty_cache("rollout.generate")

        if isinstance(generation, torch.Tensor):
            sequences = generation
        else:
            sequences = generation.sequences
        sequences = sequences.to("cpu")

        outputs: list[str] = []
        for idx in range(sequences.size(0)):
            generated_ids = sequences[idx, input_len:]

            # Optionally truncate at stop token ids (robust to specials removal)
            if stop_token_ids:
                try:
                    stop_pos = next(
                        pos
                        for pos, tid in enumerate(generated_ids.tolist())
                        if tid in stop_token_ids
                    )
                except StopIteration:
                    stop_pos = None
                if stop_pos is not None:
                    generated_ids = generated_ids[:stop_pos]

            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            trimmed = text.strip()
            for marker in _ASSISTANT_MARKERS:
                if trimmed.startswith(marker):
                    trimmed = trimmed[len(marker) :]
                    break

            if stop_tokens:
                for token in stop_tokens:
                    pos = trimmed.find(token)
                    if pos > 0:
                        trimmed = trimmed[:pos]
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

        prompts: list[str] = []
        for ticket in tickets:
            if ticket.mission not in guidance_map:
                raise KeyError(f"Missing mission guidance for {ticket.mission}")
            if ticket.mission not in domain_map:
                raise ValueError(
                    f"Missing domain mapping for mission '{ticket.mission}'"
                )
            prompts.append(
                self._build_prompt(
                    ticket,
                    guidance_map[ticket.mission],
                    domain=domain_map[ticket.mission],
                )
            )

        max_prompt_tokens = (
            self.config.max_prompt_tokens
            if self.config.max_prompt_tokens is not None
            else _DEFAULT_MAX_PROMPT_TOKENS
        )
        prompt_lengths = self._count_prompt_tokens(prompts)
        kept_tickets: list[GroupTicket] = []
        kept_prompts: list[str] = []
        dropped_keys: list[str] = []
        for ticket, prompt, length in zip(tickets, prompts, prompt_lengths):
            if length is None or length > max_prompt_tokens:
                dropped_keys.append(ticket.key)
            else:
                kept_tickets.append(ticket)
                kept_prompts.append(prompt)

        if not kept_tickets:
            return {}, tuple(dropped_keys)

        per_group: dict[str, list[ParsedTrajectory]] = {
            ticket.key: [] for ticket in kept_tickets
        }
        counters: dict[str, int] = {ticket.key: 0 for ticket in kept_tickets}

        for decode in self.config.grid:
            for sample_index in range(self.config.samples_per_decode):
                responses = self._generate_with_prompts(
                    kept_prompts, decode, sample_index
                )
                if len(responses) != len(kept_tickets):
                    raise RuntimeError("Sampler returned mismatched response count")

                current_time = datetime.now(timezone.utc)
                for ticket, response_text in zip(kept_tickets, responses):
                    ticket_key = ticket.key
                    candidate_index = counters[ticket_key]
                    counters[ticket_key] += 1

                    # Normalize response text (convert to simplified Chinese and normalize spaces)
                    normalized_response_text = to_simplified(response_text)
                    normalized_response_text = normalize_spaces(
                        normalized_response_text
                    )

                    base = Trajectory(
                        group_id=ticket.group_id,
                        mission=ticket.mission,
                        candidate_index=candidate_index,
                        decode=decode,
                        response_text=normalized_response_text,
                        created_at=current_time,
                    )
                    format_ok, verdict, reason = _parse_two_line_response(
                        normalized_response_text
                    )

                    # Convert traditional Chinese to simplified Chinese and normalize spaces
                    if reason:
                        reason = to_simplified(reason)
                        reason = normalize_spaces(reason)

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
