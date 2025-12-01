#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rollout sampler for the reflection-centric Stage-B pipeline."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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

_DEFAULT_STOP: Tuple[str, ...] = (
    "\nassistant",
    "assistant\n",
    "assistant:",
    "Assistant:",
    "<|endoftext|>",
    "</s>",
    "<|im_end|>",
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


def _normalize_verdict(text: str) -> Optional[GroupLabel]:
    cleaned = text.strip().replace(" ", "").lower()
    if cleaned in {"通过", "pass", "通过。"}:
        return "pass"
    if cleaned in {"不通过", "fail", "未通过", "不通过。"}:
        return "fail"
    if any(
        term in cleaned for term in ["复核", "不确定", "无法判断", "无法判定", "待复核"]
    ):
        return "fail"
    if cleaned in {"通过需复核", "通过需要复核", "通过需要复核。", "通过需复核。"}:
        return "pass"
    return None


def _parse_two_line_response(
    response: str,
) -> Tuple[bool, Optional[GroupLabel], Optional[str]]:
    """Parse minimal two-line protocol: Verdict + Reason."""

    text = _trim_assistant_prefix(response).strip()
    if not text:
        return False, None, None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, None, None

    verdict = None
    reason = None

    for line in lines:
        lower = line.lower()
        if verdict is None and (
            lower.startswith(("verdict", "判定", "结论"))
            or line.startswith(("通过", "不通过"))
        ):
            parts = re.split(r"[:：]", line, maxsplit=1)
            verdict_text = parts[1].strip() if len(parts) == 2 else line.strip()
            verdict = _normalize_verdict(verdict_text)
            continue
        if reason is None and lower.startswith(("reason", "理由")):
            parts = re.split(r"[:：]", line, maxsplit=1)
            reason = parts[1].strip() if len(parts) == 2 else line.strip()
            continue

    if verdict is None:
        # Try fallback: first token containing 通过/不通过
        for line in lines:
            match = re.search(r"(通过|不通过)", line)
            if match:
                verdict = _normalize_verdict(match.group(1))
                break

    if reason is None:
        # Use first non-verdict line as reason fallback
        for line in lines:
            if "通过" in line or "不通过" in line:
                continue
            reason = line.strip()
            break

    format_ok = verdict is not None and reason is not None
    return format_ok, verdict, reason


def _fallback_coerce(
    response: str,
) -> Tuple[bool, Optional[GroupLabel], Optional[str]]:
    """Best-effort salvage when two-line parsing fails.

    - Infer verdict from first verdict keyword; default to fail.
    - Use the first non-empty line (trimmed to 200 chars) as reason.
    """

    verdict_candidates = re.findall(r"(通过|不通过)", response)
    verdict: Optional[GroupLabel] = (
        _normalize_verdict(verdict_candidates[0]) if verdict_candidates else "fail"
    )

    reason = None
    for line in response.splitlines():
        text = line.strip()
        if text:
            reason = text[:200]
            break
    if reason is None:
        reason = "empty_response"

    return True, verdict, reason


class RolloutSampler:
    """Generate multi-attempt candidates for Stage-B tickets."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: SamplerConfig,
        *,
        device: Optional[str] = None,
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
        input_len = inputs["input_ids"].shape[1]

        do_sample = decode.temperature is not None and decode.temperature > 0
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

        eos_ids: List[int] = []
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
        }
        generator_kwargs = {k: v for k, v in generator_kwargs.items() if v is not None}

        if decode.seed is not None:
            torch.manual_seed(decode.seed + sample_offset)

        with torch.no_grad():
            generation = self.model.generate(**inputs, **generator_kwargs)  # type: ignore[operator]

        sequences = (
            generation.sequences if hasattr(generation, "sequences") else generation
        )
        sequences = sequences.to("cpu")

        outputs: List[str] = []
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

                    if not format_ok:
                        logger.warning(
                            "Rollout parse failed (group=%s, cand=%d), raw response=%.300s",
                            ticket.group_id,
                            candidate_index,
                            normalized_response_text,
                        )
                        format_ok, verdict, reason = _fallback_coerce(
                            normalized_response_text
                        )
                        logger.warning(
                            "Salvaged malformed rollout candidate (group=%s, cand=%d)",
                            ticket.group_id,
                            candidate_index,
                        )

                    # Convert traditional Chinese to simplified Chinese and normalize spaces
                    if reason:
                        reason = to_simplified(reason)
                        reason = normalize_spaces(reason)

                    per_group[ticket.group_id].append(
                        ParsedTrajectory(
                            base=base,
                            verdict=verdict,
                            reason=reason,
                            format_ok=format_ok,
                        )
                    )

        # Safety net: if a group ended with zero candidates (all filtered), synthesize one
        now = datetime.now(timezone.utc)
        for ticket in tickets:
            if per_group[ticket.group_id]:
                continue
            logger.warning(
                "Sampler yielded zero valid candidates after filtering; injecting fallback for %s",
                ticket.group_id,
            )
            decode = self.config.grid[0]
            base = Trajectory(
                group_id=ticket.group_id,
                mission=ticket.mission,
                candidate_index=0,
                decode=decode,
                response_text="sampling_failed",
                created_at=now,
            )
            per_group[ticket.group_id].append(
                ParsedTrajectory(
                    base=base,
                    verdict="fail",
                    reason="sampling_failed",
                    format_ok=True,
                )
            )

        return per_group


__all__ = ["RolloutSampler"]
