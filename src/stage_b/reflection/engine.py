#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reflection engine for Stage-B guidance updates using an in-process LLM.

Stage-B uses a **two-pass** reflection design:
1) Decision pass: classify stop-gradient tickets after seeing GT (`no_evidence_group_ids`).
2) Ops pass: propose strict JSON guidance operations using only learnable tickets.

The Stage-B runner is responsible for:
- gradient-candidate selection,
- enforcing learnability closure and bounded retries,
- routing stop-gradient tickets to need-review artifacts,
- applying operations and buffering rule feedback.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    "复核",
    "需复核",
    "需要复核",
    "需人工复核",
    "人工复核",
    "need-review",
    "needreview",
    "need review",
    "need_review",
    "needs review",
    "needs_review",
    "证据不足",
    "待定",
    "不应直接",
    "不建议直接",
    "不得直接",
    "佐证",
    "需进一步",
    "需要进一步",
    "进一步确认",
    "不写不通过",
    "通过但需复核",
    "通过但需人工复核",
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


def _balanced_json_span(text: str, start: int) -> Optional[str]:
    opener = text[start]
    if opener not in "{[":
        return None
    stack: List[str] = [opener]
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


class ReflectionEngine:
    """Coordinates reflection prompting and strict JSON parsing/validation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: ReflectionConfig,
        guidance_repo: GuidanceRepository,
        *,
        reflection_log: Optional[Path] = None,
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

        self._last_debug_info: Optional[Dict[str, Any]] = None
        self._group_id_mapping: Dict[str, str] = {}

        self._validate_template(self.decision_prompt_template)
        self._validate_template(self.ops_prompt_template)

    def _validate_template(self, template: str) -> None:
        if not template.strip():
            raise ValueError("Reflection prompt template must be non-empty")

    @staticmethod
    def _extract_json_candidates(text: str) -> List[str]:
        raw = normalize_spaces(to_simplified(text or "")).strip()
        if not raw:
            return []

        candidates: List[str] = []

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
        unique: List[str] = []
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
        input_ids_full = encoded_full["input_ids"]  # type: ignore[assignment]
        full_token_length = int(input_ids_full.size(1))  # type: ignore[attr-defined]
        if full_token_length > self.config.max_reflection_length:
            logger.warning(
                "Reflection prompt will be truncated: %d tokens > %d (max_reflection_length).",
                full_token_length,
                self.config.max_reflection_length,
            )

        encoded = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_reflection_length,
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
            output = self.model.generate(  # type: ignore[call-overload]
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
        return any(term in normalized for term in _FORBIDDEN_THIRD_STATE_PHRASES)

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
    ) -> Optional[str]:
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
    def _is_brand_dimension(value: Optional[str]) -> bool:
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

    def run_decision_pass(self, bundle: ExperienceBundle) -> Tuple[Tuple[str, ...], str]:
        """Run decision pass and return (no_evidence_ticket_keys, decision_analysis)."""

        user_prompt = self._build_reflection_prompt(
            bundle,
            system_template=self.decision_prompt_template,
        )
        payload = self._generate_json_payload(
            system_template=self.decision_prompt_template,
            user_prompt=user_prompt,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("Decision pass must return a JSON object")

        ticket_keys = {rec.ticket.key for rec in bundle.records}
        raw_ids = payload.get("no_evidence_group_ids", [])
        no_evidence: List[str] = []
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
    ) -> Tuple[
        Tuple[ExperienceOperation, ...],
        Tuple[HypothesisCandidate, ...],
        Tuple[str, ...],
        str,
    ]:
        """Run ops pass and return (operations, hypotheses, evidence_ticket_keys_union, evidence_analysis)."""

        user_prompt = self._build_reflection_prompt(
            bundle,
            system_template=self.ops_prompt_template,
        )
        payload = self._generate_json_payload(
            system_template=self.ops_prompt_template,
            user_prompt=user_prompt,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("Ops pass must return a JSON object")

        learnable_ticket_keys = {rec.ticket.key for rec in bundle.records}
        ops_raw = payload.get("operations", [])
        if not isinstance(ops_raw, Sequence) or isinstance(ops_raw, (str, bytes)):
            ops_raw = []

        evidence_analysis = str(payload.get("evidence_analysis") or "").strip()

        max_ops = self.config.max_operations
        max_ops = int(max_ops) if max_ops is not None else None

        operations: List[ExperienceOperation] = []
        hypotheses: List[HypothesisCandidate] = []
        evidence_union: List[str] = []
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
            merged_from: Tuple[str, ...] = tuple()
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
            if rationale:
                if self._contains_forbidden_phrase(rationale):
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

        scaffold_texts: List[str] = []
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
            if self._reject_experience_text(text):
                raise ValueError("hypotheses.text appears to copy summary chains")
            if self._contains_forbidden_phrase(text):
                raise ValueError("hypotheses.text contains forbidden third-state wording")
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
        candidates: Sequence,
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
        experience_candidates: List[ExperienceCandidate] = []
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
    def _estimate_obj_count(text: str) -> int:
        simplified = to_simplified(text or "")
        simplified = normalize_spaces(simplified)
        matches = re.findall(r"×(\d+)", simplified)
        if matches:
            total = 0
            for m in matches:
                try:
                    total += int(m)
                except ValueError:  # pragma: no cover - defensive
                    continue
            return total
        entries = [seg.strip() for seg in simplified.split("，") if seg.strip()]
        return len(entries) if entries else (1 if simplified else 0)

    @staticmethod
    def _sorted_stage_a_summaries(
        stage_a_summaries: Mapping[str, str],
    ) -> List[Tuple[str, str]]:
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
        lowered = (text or "").lower()
        slash_count = text.count("/")
        has_summary_markers = any(
            marker in text
            for marker in (
                "×",
                "image_",
                "备注:",
                "标签/",
            )
        )
        has_object_chain_vocab = any(
            k in text
            for k in (
                "显示完整",
                "只显示部分",
                "符合要求",
                "不符合要求",
                "无法判断",
                "需复核",
            )
        )

        if "标签/" in text or "image_" in lowered or re.search(r"image\\d", lowered):
            return True
        if slash_count >= 2 and has_object_chain_vocab:
            return True
        if slash_count >= 1 and "×" in text:
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
            formatted: List[str] = []
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
                    ids = encoded["input_ids"]
                    if hasattr(ids, "size"):
                        return int(ids.size(1))  # type: ignore[attr-defined]
                if hasattr(encoded, "input_ids") and hasattr(encoded.input_ids, "size"):
                    return int(encoded.input_ids.size(1))  # type: ignore[attr-defined]
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
            lines: List[str] = []
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
                ordered = self._sorted_stage_a_summaries(summaries)
                counts = [(key, self._estimate_obj_count(value)) for key, value in ordered]
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

    def _append_log(self, outcome: ReflectionOutcome, *, epoch: int) -> None:
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
        payload = {
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
        if self._last_debug_info is not None:
            payload["reflection"]["debug_info"] = self._last_debug_info
            self._last_debug_info = None
        with self.reflection_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")


__all__ = ["ReflectionEngine"]
