#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rollout sampler for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Dict, List, Mapping, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import SamplerConfig
from .prompts import build_messages
from .types import (
    DecodeConfig,
    GroupLabel,
    GroupTicket,
    MissionGuidance,
    ParsedTrajectory,
    Trajectory,
)

logger = logging.getLogger(__name__)


_ASSISTANT_MARKERS: Sequence[str] = (
    "assistant\n",
    "assistant:",
    "assistant",
    "Assistant\n",
    "Assistant:",
    "Assistant",
)


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
    return None


def _parse_confidence(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    if ":" in stripped:
        _, value = stripped.split(":", 1)
    elif "：" in stripped:
        _, value = stripped.split("：", 1)
    else:
        value = stripped
    value = value.strip().rstrip("。")
    if not value:
        return None
    try:
        confidence = float(value)
    except ValueError:
        return None
    if confidence > 1.0:
        # assume percentage style
        confidence /= 100.0
    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0
    return confidence


def _parse_three_line_response(
    response: str,
) -> tuple[bool, GroupLabel | None, str | None, float | None]:
    stripped = _trim_assistant_prefix(response).strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) < 3:
        return False, None, None, None

    verdict_line, reason_line, confidence_line = lines[0], lines[1], lines[2]

    # Verdict parsing
    if ":" in verdict_line or "：" in verdict_line:
        verdict_text = verdict_line.split(":", 1)[-1]
        verdict_text = verdict_text.split("：", 1)[-1] if "：" in verdict_line else verdict_text
    else:
        verdict_text = verdict_line
    verdict = _normalize_verdict(verdict_text)

    # Reason parsing
    if reason_line.lower().startswith("reason") or reason_line.startswith("理由"):
        reason = reason_line.split(":", 1)[-1]
        reason = reason.split("：", 1)[-1] if "：" in reason_line else reason
        reason = reason.strip()
    else:
        reason = reason_line
    reason = reason or None

    # Confidence parsing
    confidence = _parse_confidence(confidence_line)

    format_ok = all([verdict is not None, reason is not None, confidence is not None])
    return format_ok, verdict, reason, confidence


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
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or (model.device if hasattr(model, "device") else "cpu")

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_prompt(self, ticket: GroupTicket, guidance: MissionGuidance) -> str:
        messages = build_messages(ticket, guidance)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(rendered, str), "apply_chat_template must return string"
        return rendered

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _generate_with_prompts(
        self,
        prompts: Sequence[str],
        decode: DecodeConfig,
        sample_offset: int,
    ) -> List[str]:
        if not prompts:
            return []

        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: value.to(self.device) for key, value in encoded.items()}

        do_sample = decode.temperature is not None and decode.temperature > 0
        generator_kwargs = {
            "max_new_tokens": decode.max_new_tokens,
            "temperature": decode.temperature if do_sample else None,
            "top_p": decode.top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }
        generator_kwargs = {k: v for k, v in generator_kwargs.items() if v is not None}

        if decode.seed is not None:
            torch.manual_seed(decode.seed + sample_offset)

        with torch.no_grad():
            generation = self.model.generate(**inputs, **generator_kwargs)  # type: ignore[operator]

        sequences = generation.sequences if hasattr(generation, "sequences") else generation
        sequences = sequences.to("cpu")

        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            prompt_lengths = attention_mask.sum(dim=1)
        else:
            prompt_lengths = torch.full(
                (sequences.size(0),),
                sequences.size(1),
                dtype=torch.long,
            )

        outputs: List[str] = []
        for idx in range(sequences.size(0)):
            prompt_length = int(prompt_lengths[idx].item())
            generated_ids = sequences[idx, prompt_length:]
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

            if decode.stop:
                for token in decode.stop:
                    pos = trimmed.find(token)
                    if pos != -1:
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
    ) -> Mapping[str, List[ParsedTrajectory]]:
        if not tickets:
            return {}

        prompts: List[str] = []
        for ticket in tickets:
            if ticket.mission not in guidance_map:
                raise KeyError(f"Missing mission guidance for {ticket.mission}")
            prompts.append(self._build_prompt(ticket, guidance_map[ticket.mission]))

        per_group: Dict[str, List[ParsedTrajectory]] = {
            ticket.group_id: [] for ticket in tickets
        }
        counters: Dict[str, int] = {ticket.group_id: 0 for ticket in tickets}

        for decode in self.config.grid:
            for sample_index in range(self.config.samples_per_decode):
                responses = self._generate_with_prompts(prompts, decode, sample_index)
                if len(responses) != len(tickets):
                    raise RuntimeError("Sampler returned mismatched response count")

                current_time = datetime.now(timezone.utc)
                for ticket, response_text in zip(tickets, responses):
                    candidate_index = counters[ticket.group_id]
                    counters[ticket.group_id] += 1
                    base = Trajectory(
                        group_id=ticket.group_id,
                        mission=ticket.mission,
                        candidate_index=candidate_index,
                        decode=decode,
                        response_text=response_text,
                        created_at=current_time,
                    )
                    format_ok, verdict, reason, confidence = _parse_three_line_response(
                        response_text
                    )

                    if not format_ok and self.config.format_filter:
                        logger.debug(
                            "Discarding candidate %s for %s due to format violation",
                            candidate_index,
                            ticket.group_id,
                        )
                        continue

                    per_group[ticket.group_id].append(
                        ParsedTrajectory(
                            base=base,
                            verdict=verdict,
                            reason=reason,
                            confidence=confidence,
                            format_ok=format_ok,
                        )
                    )

        return per_group


__all__ = ["RolloutSampler"]
