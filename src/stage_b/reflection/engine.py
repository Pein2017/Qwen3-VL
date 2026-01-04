#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reflection engine for Stage-B guidance updates using an in-process LLM.

Stage-B uses a **two-pass** reflection design:
1) Decision pass: classify stop-gradient tickets after seeing GT (`no_evidence_group_ids`).
2) Ops pass: propose strict JSON guidance operations using only learnable tickets.

The Stage-B runner is responsible for:
- gradient-candidate selection,
- enforcing learnability closure and bounded retries,
- routing stop-gradient tickets to a quarantine queue,
- applying operations and buffering rule feedback.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config.missions import STAGE_B_MISSION_FOCUS

from ..config import ReflectionConfig
from ..io.guidance import GuidanceRepository
from ..types import (
    DeterministicSignals,
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceOperation,
    ExperienceRecord,
    GroupTicket,
    HypothesisCandidate,
    ReflectionOutcome,
)
from ..utils.chinese import normalize_spaces, to_simplified
from ..utils.perf import maybe_empty_cache

logger = logging.getLogger(__name__)

_SCAFFOLD_KEY_RE = re.compile(r"S\d+")
_GUIDANCE_SIMILARITY_THRESHOLD = 0.9

_FORBIDDEN_THIRD_STATE_PHRASES = (
    "\u590d\u6838",  # review placeholder
    "\u4eba\u5de5\u590d\u6838",  # manual review
    "\u7b2c\u4e09\u6001",  # third state
    "待定",
    "review needed",
    "证据不足",
    "不应直接",
    "不建议直接",
    "不得直接",
    "佐证",
    "需进一步",
    "需要进一步",
    "进一步确认",
    "不写不通过",
    "通过但",
)

_FORBIDDEN_AMBIGUOUS_NEGATION_PHRASES = (
    "不应判定为",
    "不应判断为",
    "不应判为",
    "不建议判定为",
    "不建议判断为",
    "不建议判为",
    "不要判定为",
    "不要判断为",
    "不得判定为",
    "不得判断为",
)

_BRAND_DIMENSION_TOKENS = ("brand", "品牌")


def _is_scaffold_experience_key(key: object) -> bool:
    """Return True for mission-scoped scaffold keys (S*), e.g. 'S1'.

    Scaffold keys are immutable and MUST NOT be modified by reflection ops.
    """

    value = str(key or "").strip()
    return bool(_SCAFFOLD_KEY_RE.fullmatch(value))


def _is_non_removable_experience_key(key: object) -> bool:
    """Return True for experiences that must never be removed."""

    value = str(key or "").strip()
    return value == "G0" or _is_scaffold_experience_key(value)


def _balanced_json_span(text: str, start: int) -> str | None:
    opener = text[start]
    if opener not in "{[":
        return None
    stack: list[str] = [opener]
    in_string = False
    escape = False

    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            expected = "}" if stack[-1] == "{" else "]"
            if ch != expected:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1]

    return None


class PromptTooLongError(RuntimeError):
    """Raised when a reflection prompt exceeds the configured max length."""


class ReflectionEngine:
    """Coordinates reflection prompting and strict JSON parsing/validation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: ReflectionConfig,
        guidance_repo: GuidanceRepository,
        *,
        reflection_log: Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.guidance_repo = guidance_repo
        self.decision_prompt_template = Path(config.decision_prompt_path).read_text(
            encoding="utf-8"
        )
        self.ops_prompt_template = Path(config.ops_prompt_path).read_text(
            encoding="utf-8"
        )
        self.reflection_log = reflection_log
        self.device = model.device if hasattr(model, "device") else "cpu"

        self._last_debug_info: dict[str, Any] | None = None
        self._group_id_mapping: dict[str, str] = {}

        self._validate_template(self.decision_prompt_template)
        self._validate_template(self.ops_prompt_template)

    def _validate_template(self, template: str) -> None:
        if not template.strip():
            raise ValueError("Reflection prompt template must be non-empty")

    @staticmethod
    def _extract_json_candidates(text: str) -> list[str]:
        raw = normalize_spaces(to_simplified(text or "")).strip()
        if not raw:
            return []

        candidates: list[str] = []

        # Code-fenced JSON blocks (common for LLMs).
        for match in re.finditer(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            raw,
            flags=re.IGNORECASE,
        ):
            block = match.group(1).strip()
            if not block:
                continue
            starts = [i for i, ch in enumerate(block) if ch in "{["]
            for start in starts[:3]:
                span = _balanced_json_span(block, start)
                if span:
                    candidates.append(span.strip())
                    break
            else:
                candidates.append(block)

        # Balanced spans from the full response (prefer the earliest).
        starts = [i for i, ch in enumerate(raw) if ch in "{["]
        for start in starts[:6]:
            span = _balanced_json_span(raw, start)
            if span:
                candidates.append(span.strip())

        # Fallback brace/bracket spans.
        if "{" in raw and "}" in raw:
            candidates.append(raw[raw.find("{") : raw.rfind("}") + 1].strip())
        if "[" in raw and "]" in raw:
            candidates.append(raw[raw.find("[") : raw.rfind("]") + 1].strip())

        if (raw.startswith("{") and raw.endswith("}")) or (
            raw.startswith("[") and raw.endswith("]")
        ):
            candidates.append(raw)

        # De-dup while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            unique.append(cand)
        return unique

    @classmethod
    def _loads_first_json(cls, text: str) -> Any:
        for cand in cls._extract_json_candidates(text):
            try:
                return json.loads(cand)
            except Exception:
                continue
        raise ValueError("No valid JSON found in reflection response")

    def _generate_json_payload(
        self,
        *,
        system_template: str,
        user_prompt: str,
    ) -> Any:
        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_prompt},
        ]
        try:
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            chat_prompt = self.tokenizer.apply_chat_template(  # type: ignore[call-arg]
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        assert isinstance(chat_prompt, str)

        encoded_full = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids_full = cast(torch.Tensor, encoded_full["input_ids"])
        full_token_length = int(input_ids_full.size(1))
        model_max_length_raw = getattr(self.tokenizer, "model_max_length", None)
        model_max_length: int | None
        try:
            model_max_length = int(model_max_length_raw) if model_max_length_raw else None
        except Exception:  # noqa: BLE001
            model_max_length = None
        if model_max_length is not None and (model_max_length <= 0 or model_max_length > 1_000_000):
            model_max_length = None
        if model_max_length is not None:
            total_requested = min(full_token_length, self.config.max_reflection_length) + self.config.max_new_tokens
            if self.config.max_reflection_length > model_max_length:
                logger.warning(
                    "reflection max_reflection_length exceeds model_max_length: max_reflection_length=%d model_max_length=%d",
                    self.config.max_reflection_length,
                    model_max_length,
                )
            if full_token_length > model_max_length:
                logger.error(
                    "reflection prompt exceeds model_max_length: prompt=%d model_max_length=%d",
                    full_token_length,
                    model_max_length,
                )
            elif total_requested > model_max_length:
                logger.warning(
                    "reflection prompt+generation exceeds model_max_length: prompt=%d new=%d total=%d model_max_length=%d",
                    min(full_token_length, self.config.max_reflection_length),
                    self.config.max_new_tokens,
                    total_requested,
                    model_max_length,
                )
        if full_token_length > self.config.max_reflection_length:
            logger.warning(
                "Reflection prompt exceeds max_reflection_length: %d tokens > %d; dropping.",
                full_token_length,
                self.config.max_reflection_length,
            )
            raise PromptTooLongError(
                f"Reflection prompt exceeds max_reflection_length: {full_token_length} > {self.config.max_reflection_length}"
            )

        encoded = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": getattr(self.config, "repetition_penalty", 1.0),
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            model = cast(Any, self.model)
            output = model.generate(
                **inputs,
                **generate_kwargs,
            )
            maybe_empty_cache("reflection.generate_json_payload")

        prompt_length = inputs["input_ids"].size(1)
        generated_tokens = output[0, prompt_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = normalize_spaces(to_simplified(response))

        try:
            return self._loads_first_json(response)
        except Exception as exc:
            self._last_debug_info = {
                "raw_response": response[:500],
                "parse_error": str(exc),
            }
            raise

    def _resolve_group_identifier(self, identifier: object) -> str:
        value = str(identifier).strip()
        return self._group_id_mapping.get(value, value)

    @staticmethod
    def _normalize_forbidden_check(text: str) -> str:
        simplified = to_simplified(text or "")
        simplified = normalize_spaces(simplified).lower()
        simplified = simplified.replace("_", " ").replace("-", " ")
        simplified = normalize_spaces(simplified)
        return simplified

    @classmethod
    def _contains_forbidden_phrase(cls, text: str) -> bool:
        normalized = cls._normalize_forbidden_check(text)
        if "第三态" in normalized or "\u590d\u6838" in normalized:
            return True
        return any(term in normalized for term in _FORBIDDEN_THIRD_STATE_PHRASES)

    @classmethod
    def _contains_ambiguous_negation(cls, text: str) -> bool:
        normalized = cls._normalize_forbidden_check(text)
        return any(term in normalized for term in _FORBIDDEN_AMBIGUOUS_NEGATION_PHRASES)

    @staticmethod
    def _normalize_guidance_signature(text: str) -> str:
        simplified = to_simplified(text or "")
        simplified = normalize_spaces(simplified).lower()
        simplified = re.sub(
            r"[，,。.!！？;；:：\-—_()（）\[\]{}<>《》“”\"'·~]",
            " ",
            simplified,
        )
        simplified = normalize_spaces(simplified)
        return simplified.strip()

    @classmethod
    def _find_similar_guidance_key(
        cls, text: str, experiences: Mapping[str, str]
    ) -> str | None:
        if not text or not experiences:
            return None
        target = cls._normalize_guidance_signature(text)
        if not target:
            return None
        best_key = None
        best_ratio = 0.0
        for key, value in experiences.items():
            if _is_scaffold_experience_key(key):
                continue
            candidate = cls._normalize_guidance_signature(value)
            if not candidate:
                continue
            if candidate == target:
                return key
            ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
            if (
                target in candidate or candidate in target
            ) and min(len(target), len(candidate)) / max(len(target), len(candidate)) >= 0.85:
                return key
        if best_ratio >= _GUIDANCE_SIMILARITY_THRESHOLD:
            return best_key
        return None

    @staticmethod
    def _is_brand_dimension(value: str | None) -> bool:
        if not value:
            return False
        lowered = normalize_spaces(to_simplified(value)).lower()
        return any(token in lowered for token in _BRAND_DIMENSION_TOKENS)

    @staticmethod
    def _has_fail_verdict(text: str) -> bool:
        lowered = normalize_spaces(to_simplified(text)).lower()
        return any(term in lowered for term in ("不通过", "判不通过", "不能通过"))

    @staticmethod
    def _has_pass_verdict(text: str) -> bool:
        lowered = normalize_spaces(to_simplified(text)).lower()
        for term in ("不通过", "判不通过", "不能通过"):
            lowered = lowered.replace(term, "")
        return "通过" in lowered

    @staticmethod
    def _extract_conflict_anchors(text: str) -> set[str]:
        anchors = set()
        for token in (
            "全局图",
            "局部图",
            "证据",
            "覆盖",
            "缺失",
            "必须",
            "不得",
            "禁止",
            "无法确认",
            "只显示部分",
            "显示完整",
            "可见性=部分",
            "可见性=完整",
            "关键点",
        ):
            if token in text:
                anchors.add(token)
        return anchors

    def _hypothesis_conflicts_scaffold(
        self, text: str, scaffold_texts: Sequence[str]
    ) -> bool:
        if not scaffold_texts:
            return False
        hypo_text = normalize_spaces(to_simplified(text))
        hypo_anchors = self._extract_conflict_anchors(hypo_text)
        hypo_fail = self._has_fail_verdict(hypo_text)
        hypo_pass = self._has_pass_verdict(hypo_text)
        for scaffold in scaffold_texts:
            scaffold_norm = normalize_spaces(to_simplified(scaffold))
            scaffold_anchors = self._extract_conflict_anchors(scaffold_norm)
            if not (hypo_anchors & scaffold_anchors):
                continue
            scaffold_fail = self._has_fail_verdict(scaffold_norm)
            scaffold_pass = self._has_pass_verdict(scaffold_norm)
            if scaffold_fail and hypo_pass and not hypo_fail:
                return True
            if scaffold_pass and hypo_fail and not hypo_pass:
                return True
        return False

    @classmethod
    def _is_binary_hypothesis(cls, text: str) -> bool:
        normalized = cls._normalize_forbidden_check(text)
        return ("通过" in normalized) or ("不通过" in normalized)

    def run_decision_pass(self, bundle: ExperienceBundle) -> tuple[tuple[str, ...], str]:
        """Run decision pass and return (no_evidence_ticket_keys, decision_analysis)."""

        user_prompt = self._build_reflection_prompt(
            bundle,
            system_template=self.decision_prompt_template,
        )
        try:
            payload = self._generate_json_payload(
                system_template=self.decision_prompt_template,
                user_prompt=user_prompt,
            )
        except PromptTooLongError as exc:
            logger.warning("Reflection decision pass skipped: %s", exc)
            return (), "prompt_too_long"
        if not isinstance(payload, Mapping):
            raise ValueError("Decision pass must return a JSON object")

        ticket_keys = {rec.ticket.key for rec in bundle.records}
        raw_ids = payload.get("no_evidence_group_ids", [])
        no_evidence: list[str] = []
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
            for item in raw_ids:
                resolved = self._resolve_group_identifier(item)
                if resolved in ticket_keys:
                    no_evidence.append(resolved)

        decision_analysis = str(payload.get("decision_analysis") or "").strip()
        return tuple(dict.fromkeys(no_evidence)), decision_analysis

    def run_ops_pass(
        self,
        bundle: ExperienceBundle,
    ) -> tuple[
        tuple[ExperienceOperation, ...],
        tuple[HypothesisCandidate, ...],
        tuple[str, ...],
        str,
    ]:
        """Run ops pass and return (operations, hypotheses, evidence_ticket_keys_union, evidence_analysis)."""

        user_prompt = self._build_reflection_prompt(
            bundle,
            system_template=self.ops_prompt_template,
        )
        try:
            payload = self._generate_json_payload(
                system_template=self.ops_prompt_template,
                user_prompt=user_prompt,
            )
        except PromptTooLongError as exc:
            logger.warning("Reflection ops pass skipped: %s", exc)
            return (), (), (), "prompt_too_long"
        if not isinstance(payload, Mapping):
            raise ValueError("Ops pass must return a JSON object")

        learnable_ticket_keys = {rec.ticket.key for rec in bundle.records}
        ops_raw = payload.get("operations", [])
        if not isinstance(ops_raw, Sequence) or isinstance(ops_raw, (str, bytes)):
            ops_raw = []

        evidence_analysis = str(payload.get("evidence_analysis") or "").strip()

        max_ops = self.config.max_operations
        max_ops = int(max_ops) if max_ops is not None else None

        operations: list[ExperienceOperation] = []
        hypotheses: list[HypothesisCandidate] = []
        evidence_union: list[str] = []
        seen_op_norms: set[str] = set()

        guidance_map = self.guidance_repo.load()
        current_guidance = guidance_map.get(bundle.mission)
        existing_experiences: Mapping[str, str] = (
            current_guidance.experiences if current_guidance else {}
        )

        def _norm_text(txt: str) -> str:
            return " ".join(str(txt).strip().split())

        for entry in ops_raw:
            if max_ops is not None and len(operations) >= max_ops:
                break
            if not isinstance(entry, Mapping):
                continue

            op_raw = str(entry.get("op", "")).strip().lower()
            if op_raw == "none":
                continue
            if op_raw not in {"add", "update", "delete", "merge"}:
                continue

            key_raw = entry.get("key")
            key = str(key_raw).strip() if key_raw is not None else None
            if _is_scaffold_experience_key(key):
                continue

            rationale_raw = entry.get("rationale")
            rationale = (
                str(rationale_raw).strip()
                if isinstance(rationale_raw, str) and rationale_raw.strip()
                else None
            )

            evidence_raw = entry.get("evidence")
            if not (
                isinstance(evidence_raw, Sequence)
                and not isinstance(evidence_raw, (str, bytes))
            ):
                continue
            evidence_items = [
                self._resolve_group_identifier(x)
                for x in evidence_raw
                if str(x).strip()
            ]
            evidence = tuple(dict.fromkeys(evidence_items))
            if not evidence:
                continue
            if any(tk not in learnable_ticket_keys for tk in evidence):
                continue

            merged_from_raw = entry.get("merged_from")
            merged_from: tuple[str, ...] = tuple()
            if isinstance(merged_from_raw, Sequence) and not isinstance(
                merged_from_raw, (str, bytes)
            ):
                merged_from = tuple(
                    str(x).strip() for x in merged_from_raw if str(x).strip()
                )

            if op_raw == "delete":
                if not key:
                    continue
                if _is_non_removable_experience_key(key):
                    continue
                norm_key = f"del::{key}::{evidence}"
                if norm_key in seen_op_norms:
                    continue
                seen_op_norms.add(norm_key)
                operations.append(
                    ExperienceOperation(
                        op="remove",
                        key=key,
                        text=None,
                        rationale=rationale,
                        evidence=evidence,
                        merged_from=None,
                    )
                )
                evidence_union.extend(list(evidence))
                continue

            text_raw = entry.get("text")
            text = str(text_raw).strip() if text_raw is not None else None
            if not text or self._reject_experience_text(text):
                continue
            if self._contains_forbidden_phrase(text):
                continue
            if self._contains_ambiguous_negation(text):
                continue
            if rationale:
                if self._contains_forbidden_phrase(rationale):
                    continue
                if self._contains_ambiguous_negation(rationale):
                    continue
            if len(text) > 160:
                text = text[:160]

            if op_raw == "add":
                similar_key = self._find_similar_guidance_key(
                    text, existing_experiences
                )
                if similar_key:
                    op_raw = "update"
                    key = similar_key

            norm = _norm_text(text)
            norm_key = f"{op_raw}::{key or ''}::{norm}::{evidence}"
            if norm_key in seen_op_norms:
                continue
            seen_op_norms.add(norm_key)

            if op_raw == "add":
                operations.append(
                    ExperienceOperation(
                        op="upsert",
                        key=None,
                        text=text,
                        rationale=rationale,
                        evidence=evidence,
                        merged_from=None,
                    )
                )
            elif op_raw == "update":
                if not key:
                    continue
                operations.append(
                    ExperienceOperation(
                        op="upsert",
                        key=key,
                        text=text,
                        rationale=rationale,
                        evidence=evidence,
                        merged_from=None,
                    )
                )
            elif op_raw == "merge":
                if not key or not merged_from:
                    continue
                if any(_is_non_removable_experience_key(mkey) for mkey in merged_from):
                    continue
                operations.append(
                    ExperienceOperation(
                        op="merge",
                        key=key,
                        text=text,
                        rationale=rationale,
                        evidence=evidence,
                        merged_from=merged_from,
                    )
                )

            evidence_union.extend(list(evidence))

        hypotheses_raw = payload.get("hypotheses")
        if hypotheses_raw is None:
            hypotheses_raw = []
        if not isinstance(hypotheses_raw, Sequence) or isinstance(
            hypotheses_raw, (str, bytes)
        ):
            raise ValueError("hypotheses must be a list when provided")

        scaffold_texts: list[str] = []
        if current_guidance:
            scaffold_texts = [
                text
                for key, text in current_guidance.experiences.items()
                if _is_scaffold_experience_key(key)
            ]

        for entry in hypotheses_raw:
            if not isinstance(entry, Mapping):
                raise ValueError("hypotheses entries must be JSON objects")
            text_raw = entry.get("text")
            text = str(text_raw).strip() if text_raw is not None else ""
            if not text:
                raise ValueError("hypotheses.text must be non-empty")
            if "\u7b2c\u4e09\u6001" in text or "\u590d\u6838" in text:
                raise ValueError("hypotheses.text contains forbidden third-state wording")
            if self._reject_experience_text(text):
                raise ValueError("hypotheses.text appears to copy summary chains")
            if self._contains_forbidden_phrase(text):
                raise ValueError("hypotheses.text contains forbidden third-state wording")
            if self._contains_ambiguous_negation(text):
                raise ValueError("hypotheses.text must use affirmative verdict phrasing")
            if not self._is_binary_hypothesis(text):
                raise ValueError("hypotheses.text must contain a binary verdict")
            if self._hypothesis_conflicts_scaffold(text, scaffold_texts):
                raise ValueError("hypotheses.text conflicts with scaffold rules")

            falsifier_raw = entry.get("falsifier")
            falsifier = (
                str(falsifier_raw).strip() if falsifier_raw is not None else ""
            )
            if not falsifier:
                raise ValueError("hypotheses.falsifier must be non-empty")
            if self._contains_forbidden_phrase(falsifier):
                raise ValueError("hypotheses.falsifier contains forbidden wording")
            if self._contains_ambiguous_negation(falsifier):
                raise ValueError("hypotheses.falsifier must use affirmative verdict phrasing")

            dimension_raw = entry.get("dimension")
            dimension = (
                str(dimension_raw).strip()
                if dimension_raw is not None
                else None
            )
            if dimension is not None and not dimension:
                dimension = None
            if self._is_brand_dimension(dimension):
                raise ValueError("hypotheses.dimension must not be brand")

            evidence_raw = entry.get("evidence")
            if not (
                isinstance(evidence_raw, Sequence)
                and not isinstance(evidence_raw, (str, bytes))
            ):
                raise ValueError("hypotheses.evidence must be a non-empty list")
            evidence_items = [
                self._resolve_group_identifier(x)
                for x in evidence_raw
                if str(x).strip()
            ]
            evidence = tuple(dict.fromkeys(evidence_items))
            if not evidence:
                raise ValueError("hypotheses.evidence must be non-empty")
            if any(tk not in learnable_ticket_keys for tk in evidence):
                raise ValueError("hypotheses.evidence must be learnable ticket_keys")

            hypotheses.append(
                HypothesisCandidate(
                    text=text,
                    evidence=evidence,
                    falsifier=falsifier,
                    dimension=dimension,
                )
            )
            evidence_union.extend(list(evidence))

        evidence_ticket_keys = tuple(dict.fromkeys(evidence_union))
        return tuple(operations), tuple(hypotheses), evidence_ticket_keys, evidence_analysis

    def build_record(
        self,
        ticket: GroupTicket,
        candidates: Sequence[object],
        winning_candidate: int | None,
        guidance_step: int,
        *,
        epoch_step: int | None = None,
        global_step: int | None = None,
    ) -> ExperienceRecord:
        if not ticket.summaries.per_image:
            raise ValueError(
                f"Stage-A summaries missing for ticket {ticket.group_id}; reflection requires evidence"
            )
        experience_candidates: list[ExperienceCandidate] = []
        for item in candidates:
            parsed = getattr(item, "parsed", None)
            if parsed is None:
                continue
            signals = getattr(item, "signals", None)
            if signals is None:
                signals = DeterministicSignals(
                    label_match=(
                        parsed.verdict == ticket.label if parsed.verdict else None
                    ),
                    self_consistency=None,
                    conflict_flag=False,
                    needs_manual_review=False,
                )
            experience_candidates.append(
                ExperienceCandidate(
                    candidate_index=parsed.base.candidate_index,
                    verdict=parsed.verdict,
                    reason=parsed.reason,
                    signals=signals,
                    summary=None,
                    critique=None,
                    raw_text=parsed.base.response_text,
                )
            )
        return ExperienceRecord(
            ticket=ticket,
            candidates=tuple(experience_candidates),
            winning_candidate=winning_candidate,
            guidance_step=guidance_step,
            epoch_step=epoch_step,
            global_step=global_step,
        )

    @staticmethod
    def _parse_stage_a_summary_json(text: str) -> dict[str, object] | None:
        stripped = (text or "").strip()
        if not stripped or stripped.startswith("无关图片"):
            return None

        required = {"dataset", "统计"}

        def _maybe_parse_obj(candidate: str) -> dict[str, object] | None:
            c = candidate.strip()
            if not (c.startswith("{") and c.endswith("}")):
                return None
            try:
                parsed = json.loads(c)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        obj = _maybe_parse_obj(stripped)
        if obj is not None and required.issubset(obj.keys()):
            return obj

        # Legacy Stage-A may contain a header line "<DOMAIN=...>, <TASK=...>".
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        for line in reversed(lines):
            obj = _maybe_parse_obj(line)
            if obj is not None and required.issubset(obj.keys()):
                return obj

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = _maybe_parse_obj(stripped[start : end + 1])
            if obj is not None and required.issubset(obj.keys()):
                return obj

        return None

    @staticmethod
    def _sanitize_stage_a_summary_for_prompt(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return stripped
        if stripped.startswith("无关图片"):
            return "无关图片"

        obj = ReflectionEngine._parse_stage_a_summary_json(stripped)
        if obj is not None:
            ordered: dict[str, object] = {}
            for key in ("dataset", "统计", "备注", "分组统计", "异常"):
                if key in obj:
                    ordered[key] = obj[key]
            return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))

        # Fallback: normalize to a single line and scrub forbidden markers.
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        # Drop the assistant prefix line when present.
        lines = [
            line
            for line in lines
            if not (line.startswith("<DOMAIN=") and "<TASK=" in line and line.endswith(">"))
        ]
        simplified = to_simplified(" ".join(lines))
        simplified = normalize_spaces(simplified)
        if "\u590d\u6838" in simplified:
            raise ValueError("Stage-A summary contains review marker")
        simplified = normalize_spaces(simplified).strip()
        return simplified

    @staticmethod
    def _estimate_obj_count(text: str) -> int:
        payload = ReflectionEngine._sanitize_stage_a_summary_for_prompt(text)
        stripped = (payload or "").strip()
        simplified = to_simplified(payload or "")
        simplified = normalize_spaces(simplified).strip()
        if simplified == "无关图片":
            return 0

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
            except Exception:  # pragma: no cover - defensive
                obj = None
            if isinstance(obj, dict) and {"dataset", "统计"}.issubset(obj.keys()):
                entries = obj.get("统计")
                if not isinstance(entries, list):
                    entries = []

                def _to_int(value: object) -> int | None:
                    if isinstance(value, bool) or value is None:
                        return None
                    if isinstance(value, int):
                        return value
                    if isinstance(value, float) and value.is_integer():
                        return int(value)
                    if isinstance(value, str):
                        stripped_value = value.strip()
                        if stripped_value.isdigit():
                            try:
                                return int(stripped_value)
                            except Exception:
                                return None
                    return None

                def _sum_counts(value: object) -> int:
                    if isinstance(value, dict):
                        total = 0
                        for count_raw in value.values():
                            count = _to_int(count_raw)
                            if count is None or count <= 0:
                                continue
                            total += int(count)
                        return total
                    if isinstance(value, list):
                        return len([v for v in value if v is not None])
                    if value is None:
                        return 0
                    return 1

                total_estimate = 0
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    category = entry.get("类别")
                    cat = category.strip() if isinstance(category, str) else ""

                    max_attr_total = 0
                    label_text_total = 0
                    label_readability_total = 0
                    for key, val in entry.items():
                        if key in {"类别", "异常"}:
                            continue
                        attr_total = _sum_counts(val)
                        max_attr_total = max(max_attr_total, attr_total)
                        if key == "文本":
                            label_text_total = attr_total
                        elif key == "可读性":
                            label_readability_total = attr_total

                    if cat == "标签":
                        combined = label_text_total + label_readability_total
                        total_estimate += max(1, max_attr_total, combined)
                    else:
                        total_estimate += max(1, max_attr_total)
                return int(total_estimate)

        entries = [seg.strip() for seg in simplified.split("，") if seg.strip()]
        return len(entries) if entries else (1 if simplified else 0)

    @staticmethod
    def _sorted_stage_a_summaries(
        stage_a_summaries: Mapping[str, str],
    ) -> list[tuple[str, str]]:
        def _index(key: str) -> int:
            match = re.search(r"(\d+)$", key)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:  # pragma: no cover - defensive
                    return 0
            return 0

        return sorted(stage_a_summaries.items(), key=lambda item: _index(item[0]))

    @staticmethod
    def _reject_experience_text(text: str) -> bool:
        """Heuristic to block Stage-A style summaries leaking into guidance."""
        stripped = (text or "").strip()
        if ReflectionEngine._parse_stage_a_summary_json(stripped) is not None:
            return True

        lowered = (text or "").lower()
        slash_count = text.count("/")
        has_summary_markers = any(
            marker in text
            for marker in (
                "image_",
                "备注:",
                "标签/",
                "类别=",
            )
        )
        has_object_chain_vocab = any(
            k in text
            for k in (
                "显示完整",
                "只显示部分",
                "完整",
                "部分",
                "符合要求",
                "不符合要求",
                "符合",
                "不符合",
                "合格",
                "不合格",
                "无法判断",
                "\\u590d\\u6838",
            )
        )

        if "标签/" in text or "image_" in lowered or re.search(r"image\\d", lowered):
            return True
        if slash_count >= 2 and has_object_chain_vocab:
            return True
        if has_summary_markers and slash_count >= 1 and has_object_chain_vocab:
            return True
        return False

    def _build_reflection_prompt(
        self,
        bundle: ExperienceBundle,
        *,
        system_template: str,
    ) -> str:
        """Build reflection user prompt with token-budgeted packing and prioritization."""

        guidance_map = self.guidance_repo.load()
        current_guidance = guidance_map.get(bundle.mission)

        experiences_text = ""
        if current_guidance and current_guidance.experiences:
            scaffold_lines = [
                f"[{key}]. {value}"
                for key, value in sorted(current_guidance.experiences.items())
                if _is_scaffold_experience_key(key)
            ]
            guidance_lines = [
                f"[{key}]. {value}"
                for key, value in sorted(current_guidance.experiences.items())
                if not _is_scaffold_experience_key(key)
            ]
            formatted: list[str] = []
            if scaffold_lines:
                formatted.extend(["SCAFFOLD (S*):", *scaffold_lines, ""])
            if guidance_lines:
                formatted.extend(["GUIDANCE (G0+):", *guidance_lines])
            experiences_text = "\n".join(formatted).strip()

        task_focus = None
        if current_guidance:
            task_focus = current_guidance.experiences.get("G0")
        if not task_focus:
            task_focus = STAGE_B_MISSION_FOCUS.get(bundle.mission, "未定义任务侧重点")

        preamble_lines = [
            f"任务: {bundle.mission}",
            f"检查清单: {task_focus}",
            f"反思周期: {bundle.reflection_cycle}",
            f"指导步骤: {bundle.guidance_step}",
            "",
        ]
        if experiences_text:
            preamble_lines += ["EXPERIENCES:", experiences_text, ""]

        def _count_tokens_local(text: str) -> int:
            try:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    ids = cast(torch.Tensor, encoded["input_ids"])
                    return int(ids.size(1))
                if hasattr(encoded, "input_ids"):
                    ids = cast(torch.Tensor, encoded.input_ids)
                    return int(ids.size(1))
            except Exception:
                pass
            return max(1, len(text) // 6)

        def _priority(rec: ExperienceRecord) -> int:
            has_true = any(c.signals.label_match is True for c in rec.candidates)
            has_false = any(c.signals.label_match is False for c in rec.candidates)
            selected_mismatch = False
            if rec.winning_candidate is not None:
                for c in rec.candidates:
                    if (
                        c.candidate_index == rec.winning_candidate
                        and c.signals.label_match is False
                    ):
                        selected_mismatch = True
                        break
            if has_true and has_false:
                return 3
            all_wrong = all(c.signals.label_match is False for c in rec.candidates)
            if all_wrong:
                return 2
            if selected_mismatch:
                return 1
            return 0

        def _record_block(idx: int, rec: ExperienceRecord) -> str:
            lines: list[str] = []
            ticket = rec.ticket
            short_group_id = f"第{idx}组"
            lines.extend(
                [
                    f"{short_group_id} (ticket_key={ticket.key}):",
                    f"  group_id: {ticket.group_id}",
                    f"  gt_label: {ticket.label}",
                    f"  winning_candidate: {rec.winning_candidate}",
                    "",
                ]
            )

            vote_strength = next(
                (
                    cand.signals.vote_strength
                    for cand in rec.candidates
                    if cand.signals.vote_strength is not None
                ),
                None,
            )
            low_agreement = any(cand.signals.low_agreement for cand in rec.candidates)
            if vote_strength is not None or low_agreement:
                lines.append(
                    f"  vote_strength: {vote_strength}; low_agreement: {low_agreement}"
                )

            summaries = ticket.summaries.as_dict()
            if summaries:
                ordered_raw = self._sorted_stage_a_summaries(summaries)
                ordered = [
                    (key, self._sanitize_stage_a_summary_for_prompt(value))
                    for key, value in ordered_raw
                ]
                counts = [
                    (key, self._estimate_obj_count(value)) for key, value in ordered
                ]
                if counts:
                    global_key, global_count = max(counts, key=lambda item: item[1])
                    stats_inline = ", ".join(
                        f"{key}(obj={count})" for key, count in counts
                    )
                    lines.append(f"  IMAGE_STATS: {stats_inline}")
                    lines.append(
                        f"  GLOBAL_CANDIDATE: {global_key}(obj={global_count})"
                    )
                    lines.append(f"  IMAGE_COUNT: {len(ordered)}")
                lines.append("  STAGE-A 摘要:")
                for key, value in ordered:
                    lines.append(
                        f"    {key}(obj={self._estimate_obj_count(value)}): {value}"
                    )
                lines.append("")

            all_wrong = all(c.signals.label_match is False for c in rec.candidates)
            if all_wrong:
                lines.append("  特殊: 全部候选与标签不一致（all-wrong）")

            lines.append("  CANDIDATES:")
            for cand in rec.candidates:
                signals = cand.signals
                label_match_str = (
                    "✓"
                    if signals.label_match
                    else "✗"
                    if signals.label_match is False
                    else "?"
                )
                lines.append(
                    f"  {cand.candidate_index}: {cand.verdict} | {cand.reason} | 匹配{label_match_str}"
                )
            lines.append("")
            return "\n".join(lines)

        indexed_records = list(enumerate(bundle.records, start=1))
        sorted_records = sorted(indexed_records, key=lambda t: (-_priority(t[1]), t[0]))

        preamble_text = "\n".join(preamble_lines)

        included: list[tuple[int, ExperienceRecord, str]] = []
        running_text = preamble_text
        for idx, rec in sorted_records:
            block = _record_block(len(included) + 1, rec)
            trial_text = running_text + f"批次: {len(included) + 1} 组\n\n" + block
            if (
                _count_tokens_local(f"{system_template}\n\n{trial_text}")
                <= self.config.token_budget
            ):
                included.append((idx, rec, block))
                running_text = running_text + f"批次: {len(included)} 组\n\n" + block
            else:
                if not included:
                    included.append((idx, rec, block))
                break

        kept_records = [rec for _, rec, _ in included]
        label_counts = Counter(rec.ticket.label for rec in kept_records)
        label_match_counts = Counter()
        verdict_counts = Counter()
        for rec in kept_records:
            for cand in rec.candidates:
                if cand.signals.label_match is True:
                    label_match_counts[cand.verdict] += 1
                verdict_counts[cand.verdict] += 1

        stats_lines = [
            "统计:",
            f"  标签: {dict(label_counts)}",
            f"  判定: {dict(verdict_counts)}",
            f"  匹配: {dict(label_match_counts)}",
            "",
        ]

        kept_blocks = [blk for _, _, blk in included]
        bundle_lines: list[str] = (
            preamble_lines
            + [f"批次: {len(kept_blocks)} 组", ""]
            + stats_lines
            + ["CASES:", ""]
            + kept_blocks
        )
        bundle_summary = "\n".join(bundle_lines)

        full_prompt = f"{system_template}\n\n{bundle_summary}"
        while len(kept_blocks) > 1 and _count_tokens_local(full_prompt) > self.config.token_budget:
            kept_blocks.pop()
            bundle_lines = (
                preamble_lines
                + [f"批次: {len(kept_blocks)} 组", ""]
                + stats_lines
                + ["CASES:", ""]
                + kept_blocks
            )
            bundle_summary = "\n".join(bundle_lines)
            full_prompt = f"{system_template}\n\n{bundle_summary}"

        # Update mapping for "第N组" shorthand resolution.
        self._group_id_mapping.clear()
        for new_idx, (_, rec, _) in enumerate(included[: len(kept_blocks)], start=1):
            self._group_id_mapping[f"第{new_idx}组"] = rec.ticket.key

        trimmed = len(bundle.records) - len(kept_blocks)
        if trimmed > 0:
            final_tokens = _count_tokens_local(full_prompt)
            logger.info(
                "reflection_token_budget_trim: kept=%d dropped=%d budget=%d tokens=%d",
                len(kept_blocks),
                trimmed,
                self.config.token_budget,
                final_tokens,
            )

        return bundle_summary

    def _append_log(
        self,
        outcome: ReflectionOutcome,
        *,
        epoch: int,
        trace: Mapping[str, object] | None = None,
    ) -> None:
        if self.reflection_log is None:
            return
        self.reflection_log.parent.mkdir(parents=True, exist_ok=True)
        operations_payload = [
            {
                "op": op.op,
                "key": op.key,
                "text": op.text,
                "rationale": op.rationale,
                "evidence": list(op.evidence),
                "merged_from": (list(op.merged_from) if op.merged_from else None),
            }
            for op in outcome.operations
        ]
        hypotheses_payload = [
            {
                "text": hyp.text,
                "falsifier": hyp.falsifier,
                "dimension": hyp.dimension,
                "evidence": list(hyp.evidence),
            }
            for hyp in outcome.proposal.hypotheses
        ]
        payload: dict[str, Any] = {
            "epoch": epoch,
            "reflection": {
                "reflection_id": outcome.reflection_id,
                "mission": outcome.mission,
                "proposal": {
                    "action": outcome.proposal.action,
                    "summary": outcome.proposal.summary,
                    "critique": outcome.proposal.critique,
                    "text": outcome.proposal.text,
                    "operations": operations_payload,
                    "hypotheses": hypotheses_payload,
                    "evidence_group_ids": list(outcome.proposal.evidence_group_ids),
                    "uncertainty_note": outcome.proposal.uncertainty_note,
                    "no_evidence_group_ids": list(outcome.proposal.no_evidence_group_ids),
                },
                "eligible": outcome.eligible,
                "applied": outcome.applied,
                "guidance_step_before": outcome.guidance_step_before,
                "guidance_step_after": outcome.guidance_step_after,
                "ineligible_reason": outcome.ineligible_reason,
                "warnings": list(outcome.warnings),
            },
        }
        reflection_payload = payload.get("reflection")
        if isinstance(reflection_payload, dict):
            if self._last_debug_info is not None:
                reflection_payload["debug_info"] = self._last_debug_info
                self._last_debug_info = None
            if trace:
                reflection_payload["trace"] = trace
        with self.reflection_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")


__all__ = ["ReflectionEngine"]
