"""Reflection engine for Stage-B guidance updates using in-process LLM."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config.missions import STAGE_B_MISSION_FOCUS

from ..config import ReflectionConfig
from ..io.guidance import GuidanceRepository
from ..sampling.prompts import _render_summaries
from ..types import (
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceOperation,
    ExperienceRecord,
    GroupTicket,
    ReflectionAction,
    ReflectionOutcome,
    ReflectionProposal,
    TrajectoryWithSignals,
)

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Coordinates reflection prompting and logging using in-process model."""

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
        self.prompt_template = Path(config.prompt_path).read_text(encoding="utf-8")
        self.reflection_log = reflection_log
        self.device = model.device if hasattr(model, "device") else "cpu"
        self._last_debug_info: Optional[Dict[str, Any]] = None
        self._group_id_mapping: Dict[str, str] = {}

    @staticmethod
    def _check_eligibility(bundle: ExperienceBundle) -> tuple[bool, str | None]:
        """
        Check if bundle is eligible for reflection.
        
        Eligibility criteria:
        1. Selected (winning) candidate for any group has label_match=False
        2. OR all candidates in any group have label_match=False (all-wrong shortcut)
        
        Returns:
            (eligible: bool, ineligible_reason: str | None)
        """
        has_selected_mismatch = False
        has_all_wrong_group = False
        
        for record in bundle.records:
            # Check if selected candidate has label_match=False
            if record.winning_candidate is not None:
                candidate_index = record.winning_candidate
                for cand in record.candidates:
                    if cand.candidate_index == candidate_index:
                        if cand.signals.label_match is False:
                            has_selected_mismatch = True
                            break
            
            # Check if all candidates in this group are wrong (all-wrong shortcut)
            all_wrong = True
            has_any_candidate = False
            for cand in record.candidates:
                has_any_candidate = True
                if cand.signals.label_match is True:
                    all_wrong = False
                    break
            if has_any_candidate and all_wrong:
                has_all_wrong_group = True
        
        if has_selected_mismatch or has_all_wrong_group:
            return True, None
        
        # Ineligible: no selected mismatches and no all-wrong groups
        return False, "No selected candidate mismatches and no all-wrong groups"

    def _resolve_group_identifier(self, identifier: object) -> str:
        value = str(identifier)
        mapping = getattr(self, "_group_id_mapping", {})
        return mapping.get(value, value)

    def build_record(
        self,
        ticket: GroupTicket,
        candidates: Sequence[TrajectoryWithSignals],
        winning_candidate: int | None,
        guidance_step: int,
    ) -> ExperienceRecord:
        experience_candidates = []
        for item in candidates:
            experience_candidates.append(
                ExperienceCandidate(
                    candidate_index=item.parsed.base.candidate_index,
                    verdict=item.parsed.verdict,
                    reason=item.parsed.reason,
                    confidence=item.signals.confidence,
                    signals=item.signals,
                )
            )
        return ExperienceRecord(
            ticket=ticket,
            candidates=tuple(experience_candidates),
            winning_candidate=winning_candidate,
            guidance_step=guidance_step,
        )

    def build_bundle(
        self,
        records: Sequence[ExperienceRecord],
        *,
        reflection_cycle: int,
    ) -> ExperienceBundle:
        if not records:
            raise ValueError("Reflection bundle requires at least one record")
        mission = records[0].ticket.mission
        guidance_step = records[0].guidance_step
        return ExperienceBundle(
            mission=mission,
            records=tuple(records),
            reflection_cycle=reflection_cycle,
            guidance_step=guidance_step,
        )

    def reflect(
        self,
        bundle: ExperienceBundle,
        *,
        epoch: int,
        pre_uplift: float | None = None,
        log: bool = True,
    ) -> ReflectionOutcome:
        reflection_id = uuid.uuid4().hex[:12]
        guidance_step_before = bundle.guidance_step
        guidance_step_after = bundle.guidance_step
        operations: Tuple[ExperienceOperation, ...] = ()
        ineligible_reason: str | None = None

        # Check eligibility using selected candidate + all-wrong shortcut
        eligible_for_reflection, eligibility_reason = self._check_eligibility(bundle)
        
        if not eligible_for_reflection:
            evidence_ids = tuple(record.ticket.group_id for record in bundle.records)
            proposal = ReflectionProposal(
                action="noop",
                summary="Skipped ineligible batch",
                critique=f"Bundle not eligible for reflection: {eligibility_reason}",
                operations=(),
                evidence_group_ids=evidence_ids,
                uncertainty_note="ineligible_batch",
                text=None,
            )  # type: ignore[call-arg]
            logger.debug(
                "Skipping reflection for mission %s: %s",
                bundle.mission,
                eligibility_reason,
            )
            eligible = False
            ineligible_reason = eligibility_reason
        else:
            proposal = self._generate_reflection(bundle)
            operations = proposal.operations
            eligible = (
                proposal.action == "refine"
                and bool(operations)
                and (self.config.allow_uncertain or not proposal.uncertainty_note)
            )
            if not eligible and proposal.action == "refine":
                if not operations:
                    ineligible_reason = "No operations in proposal"
                elif proposal.uncertainty_note and not self.config.allow_uncertain:
                    ineligible_reason = f"Uncertainty gating: {proposal.uncertainty_note}"
                else:
                    ineligible_reason = "Proposal validation failed"

            if proposal.action == "refine" and not operations:
                logger.warning(
                    "Reflection for mission %s produced no operations; treating as noop",
                    bundle.mission,
                )
                eligible = False

            if (
                proposal.action == "refine"
                and proposal.uncertainty_note
                and not self.config.allow_uncertain
            ):
                logger.info(
                    "Skipping reflection for mission %s due to uncertainty gating",
                    bundle.mission,
                )

        outcome = ReflectionOutcome(
            reflection_id=reflection_id,
            mission=bundle.mission,
            proposal=proposal,
            applied=False,
            pre_uplift=pre_uplift if pre_uplift is not None else 0.0,
            post_uplift=0.0,
            guidance_step_before=guidance_step_before,
            guidance_step_after=guidance_step_after,
            operations=operations,
            eligible=eligible,
            ineligible_reason=ineligible_reason,
        )

        if proposal.action != "noop":
            logger.info(
                "Reflection proposal for mission %s: action=%s, eligible=%s, ops=%d",
                bundle.mission,
                proposal.action,
                eligible,
                len(operations),
            )

        if log:
            self._append_log(outcome, epoch=epoch)
        return outcome

    def finalize_outcome(
        self,
        outcome: ReflectionOutcome,
        *,
        epoch: int,
        pre_uplift: float | None = None,
        post_uplift: float | None = None,
    ) -> ReflectionOutcome:
        """Record reflection outcome with optional metric overrides."""
        pre_value = pre_uplift if pre_uplift is not None else outcome.pre_uplift
        post_value = post_uplift if post_uplift is not None else outcome.post_uplift

        delta_value: Optional[float] = None
        if pre_uplift is not None and post_uplift is not None:
            delta_value = post_uplift - pre_uplift

        should_apply = outcome.eligible and bool(outcome.operations)
        if should_apply and delta_value is not None:
            if delta_value < self.config.apply_if_delta:
                logger.info(
                    (
                        "Skipping reflection %s for mission %s due to insufficient "
                        "holdout uplift (delta=%.4f, threshold=%.4f)"
                    ),
                    outcome.reflection_id,
                    outcome.mission,
                    delta_value,
                    self.config.apply_if_delta,
                )
                should_apply = False

        applied = outcome.applied
        guidance_step_after = outcome.guidance_step_after

        if should_apply:
            try:
                updated_guidance = self.guidance_repo.apply_reflection(
                    mission=outcome.mission,
                    proposal=outcome.proposal,
                    reflection_id=outcome.reflection_id,
                    source_group_ids=list(outcome.proposal.evidence_group_ids),
                    applied_epoch=epoch,
                    operations=outcome.operations,
                )
                guidance_step_after = updated_guidance.step
                applied = True
                logger.info(
                    "Applied reflection for mission %s: step %d -> %d",
                    outcome.mission,
                    outcome.guidance_step_before,
                    guidance_step_after,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to apply reflection for mission %s: %s",
                    outcome.mission,
                    exc,
                    exc_info=True,
                )

        updated = replace(
            outcome,
            applied=applied,
            pre_uplift=pre_value,
            post_uplift=post_value,
            guidance_step_after=guidance_step_after,
            ineligible_reason=outcome.ineligible_reason,  # Preserve existing reason
        )
        self._append_log(updated, epoch=epoch)
        return updated

    def _generate_reflection(self, bundle: ExperienceBundle) -> ReflectionProposal:
        reflection_prompt = self._build_reflection_prompt(bundle)

        prompt_length_chars = len(reflection_prompt)
        logger.debug(
            "Reflection prompt length: %d chars, max_reflection_length: %d",
            prompt_length_chars,
            self.config.max_reflection_length,
        )

        encoded_full = self.tokenizer(
            reflection_prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids_full = encoded_full["input_ids"]  # type: ignore[assignment]
        full_token_length = int(input_ids_full.size(1))  # type: ignore[attr-defined]

        if full_token_length > self.config.max_reflection_length:
            logger.warning(
                "Reflection prompt will be truncated: %d tokens > %d (max_reflection_length). "
                "This may cause the model to generate incomplete or incorrect responses.",
                full_token_length,
                self.config.max_reflection_length,
            )

        encoded = self.tokenizer(
            reflection_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_reflection_length,
        )
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        prompt_token_length = inputs["input_ids"].size(1)
        logger.debug(
            "Reflection prompt token length: %d tokens (max: %d, full: %d)",
            prompt_token_length,
            self.config.max_reflection_length,
            full_token_length,
        )

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            output = self.model.generate(  # type: ignore[call-overload]
                **inputs,
                **generate_kwargs,
            )

        prompt_length = inputs["input_ids"].size(1)
        generated_tokens = output[0, prompt_length:]
        num_generated_tokens = generated_tokens.size(0)

        logger.debug(
            "Generated %d tokens (max_new_tokens: %d)",
            num_generated_tokens,
            self.config.max_new_tokens,
        )

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response_with_special = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=False
        )

        logger.info(
            "Reflection response: %d chars, %d tokens",
            len(response),
            num_generated_tokens,
        )

        if len(response) < 100:
            logger.warning(
                "Short reflection response detected (%d chars):\n"
                "Response (skip_special_tokens=True): %r\n"
                "Response (skip_special_tokens=False): %r\n"
                "First 10 token IDs: %s",
                len(response),
                response,
                response_with_special,
                generated_tokens[:10].tolist() if num_generated_tokens > 0 else "[]",
            )
        elif len(response) > 2000:
            logger.debug(
                "Long reflection response (%d chars), first 500 chars: %s",
                len(response),
                response[:500],
            )

        return self._parse_reflection_response(response, bundle)

    def _parse_experiences_from_text(self, text: str) -> Dict[str, str]:
        experiences: Dict[str, str] = {}
        pattern = (
            r"\[G(\d+)\]\.\s*((?:(?!\[G\d+\]\.)[^\n])*(?:\n(?:(?!\[G\d+\]\.)[^\n])*)*)"
        )
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            key = f"G{match.group(1)}"
            value = match.group(2).strip()
            if value:
                experiences[key] = value
        return experiences

    def _build_reflection_prompt(self, bundle: ExperienceBundle) -> str:
        guidance_map = self.guidance_repo.load()
        current_guidance = guidance_map.get(bundle.mission)
        experiences_text = ""
        if current_guidance and current_guidance.experiences:
            experiences_lines = [
                f"[{key}]. {value}"
                for key, value in sorted(current_guidance.experiences.items())
            ]
            experiences_text = "\n".join(experiences_lines)

        task_focus = None
        if current_guidance and current_guidance.focus:
            task_focus = current_guidance.focus
        else:
            task_focus = STAGE_B_MISSION_FOCUS.get(bundle.mission, "未定义任务侧重点")

        bundle_lines = [
            f"任务: {bundle.mission}",
            f"检查清单: {task_focus}",
            f"反思周期: {bundle.reflection_cycle}",
            f"指导步骤: {bundle.guidance_step}",
            "",
        ]

        if experiences_text:
            bundle_lines.extend(
                [
                    "当前指导经验:",
                    experiences_text,
                    "",
                ]
            )

        bundle_lines.append(f"批次: {len(bundle.records)} 组")
        bundle_lines.append("")

        label_counts = Counter(record.ticket.label for record in bundle.records)
        label_match_counts = Counter()
        verdict_counts = Counter()
        for record in bundle.records:
            for cand in record.candidates:
                if cand.signals.label_match is True:
                    label_match_counts[cand.verdict] += 1
                verdict_counts[cand.verdict] += 1

        bundle_lines.extend(
            [
                "统计:",
                f"  标签: {dict(label_counts)}",
                f"  判定: {dict(verdict_counts)}",
                f"  匹配: {dict(label_match_counts)}",
                "",
            ]
        )

        self._group_id_mapping.clear()
        for idx, record in enumerate(bundle.records, start=1):
            short_form = f"第{idx}组"
            self._group_id_mapping[short_form] = record.ticket.group_id

        for idx, record in enumerate(bundle.records, start=1):
            ticket = record.ticket
            short_group_id = f"第{idx}组"
            bundle_lines.extend(
                [
                    f"{short_group_id}:",
                    f"  标签: {ticket.label}",
                    f"  获胜: {record.winning_candidate}",
                    "",
                ]
            )
            summaries_dict = ticket.summaries.as_dict()
            if summaries_dict:
                summaries_text = _render_summaries(summaries_dict)
                for line in summaries_text.split("\n"):
                    bundle_lines.append(f"    {line}")
            else:
                bundle_lines.append("    （无）")
            bundle_lines.append("")
            for cand in record.candidates:
                signals = cand.signals
                label_match_str = (
                    "✓"
                    if signals.label_match
                    else "✗"
                    if signals.label_match is False
                    else "?"
                )
                confidence_str = (
                    f"{cand.confidence:.2f}" if cand.confidence is not None else "-"
                )
                bundle_lines.append(
                    f"  {cand.candidate_index}: "
                    f"{cand.verdict} | "
                    f"{cand.reason} | "
                    f"匹配{label_match_str} 置信{confidence_str}"
                )
            bundle_lines.append("")

        bundle_summary = "\n".join(bundle_lines)

        full_prompt = f"{self.prompt_template}\n\n{bundle_summary}"
        return full_prompt

    def _parse_reflection_response(
        self, response: str, bundle: ExperienceBundle
    ) -> ReflectionProposal:
        def _extract_all_json_objects(text: str) -> list[tuple[str, int]]:
            json_objects = []
            i = 0
            while i < len(text):
                start_idx = text.find("{", i)
                if start_idx == -1:
                    break

                brace_count = 0
                json_start = start_idx
                for j in range(start_idx, len(text)):
                    if text[j] == "{":
                        brace_count += 1
                    elif text[j] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_text = text[json_start : j + 1]
                            json_objects.append((json_text, json_start))
                            i = j + 1
                            break
                else:
                    i = start_idx + 1

            return json_objects

        json_text = None
        json_candidates = []

        code_block_match = re.search(r"```(?:json)?\s*", response)
        if code_block_match:
            start_pos = code_block_match.end()
            end_pos = response.find("```", start_pos)
            if end_pos != -1:
                code_content = response[start_pos:end_pos].strip()
                json_candidates.extend(_extract_all_json_objects(code_content))

        json_candidates.extend(_extract_all_json_objects(response))

        valid_json_objects = []
        for candidate_text, _ in json_candidates:
            try:
                parsed = json.loads(candidate_text)
                if "error" in parsed:
                    logger.debug("Skipping error JSON object: %s", candidate_text[:200])
                    continue
                if "action" in parsed:
                    valid_json_objects.append((candidate_text, parsed, True))
                else:
                    valid_json_objects.append((candidate_text, parsed, False))
            except json.JSONDecodeError:
                continue

        reflection_json = None
        for candidate_text, parsed, has_action in valid_json_objects:
            if has_action:
                json_text = candidate_text
                reflection_json = parsed
                break

        if json_text is None and valid_json_objects:
            json_text, reflection_json, _ = valid_json_objects[0]

        if not json_text or reflection_json is None:
            response_preview = response[:500] if len(response) > 500 else response
            response_suffix = response[-200:] if len(response) > 200 else response
            
            # Check for truncation (missing closing brace)
            truncated = False
            if json_candidates:
                # Check if any candidate appears truncated
                for cand_text, _ in json_candidates:
                    if cand_text.count("{") > cand_text.count("}"):
                        truncated = True
                        break
            
            error_msg = (
                "Could not parse valid reflection JSON from response. "
                "Found %d JSON candidates, but none had 'action' field. "
                "Response length: %d chars"
            ) % (len(json_candidates), len(response))
            
            if truncated:
                error_msg += " (TRUNCATED - missing closing braces)"
            
            logger.error(
                "%s\n"
                "Response start (first 500 chars):\n%s\n"
                "Response end (last 200 chars):\n%s",
                error_msg,
                response_preview,
                response_suffix,
            )
            
            if json_candidates:
                logger.debug("JSON candidates found: %d", len(json_candidates))
                for idx, (cand_text, _) in enumerate(json_candidates[:5]):
                    logger.debug(
                        "  Candidate %d (first 300 chars): %s",
                        idx,
                        cand_text[:300],
                    )
            if valid_json_objects:
                logger.debug(
                    "Found %d valid JSON objects without 'action' field:",
                    len(valid_json_objects),
                )
                for idx, (cand_text, parsed, _) in enumerate(valid_json_objects[:3]):
                    logger.debug(
                        "  Valid JSON %d (first 300 chars): %s\n  Parsed keys: %s",
                        idx,
                        cand_text[:300],
                        list(parsed.keys()),
                    )

            if self.reflection_log:
                self._last_debug_info = {
                    "timestamp": bundle.records[0].ticket.group_id
                    if bundle.records
                    else "unknown",
                    "mission": bundle.mission,
                    "response_length": len(response),
                    "json_candidates_count": len(json_candidates),
                    "truncated": truncated,
                    "full_response": response,
                    "json_candidates": [cand[0][:500] for cand in json_candidates[:5]],
                    "parse_error": "No valid JSON with 'action' field found",
                }

            # Treat as fatal error (no fallback to text parsing)
            raise ValueError(
                f"Failed to parse reflection response as structured JSON: {error_msg}"
            )

        parsed = reflection_json

        action_raw = str(parsed.get("action", "noop")).lower()
        uncertainty_note_misc = None
        if action_raw not in {"refine", "noop"}:
            logger.warning(
                "Invalid action %s, using noop (only 'refine' and 'noop' are supported)",
                action_raw,
            )
            uncertainty_note_misc = f"Invalid action '{action_raw}', defaulting to noop"
            action_raw = "noop"
        action: ReflectionAction = action_raw  # type: ignore[assignment]

        summary_raw = parsed.get("summary")
        summary_value = (
            str(summary_raw).strip()
            if isinstance(summary_raw, str) and summary_raw.strip()
            else None
        )

        critique_raw = parsed.get("critique")
        critique_value = (
            str(critique_raw).strip()
            if isinstance(critique_raw, str) and critique_raw.strip()
            else None
        )

        text = parsed.get("text")
        text_value = str(text).strip() if text is not None else None
        if text_value == "" or text_value == "null":
            text_value = None

        evidence_raw = parsed.get("evidence_group_ids")
        if evidence_raw is None:
            evidence_group_ids = tuple(
                record.ticket.group_id for record in bundle.records
            )
        elif isinstance(evidence_raw, Sequence) and not isinstance(
            evidence_raw, (str, bytes)
        ):
            mapped_ids = [self._resolve_group_identifier(item) for item in evidence_raw]
            evidence_group_ids = tuple(mapped_ids)
        else:
            evidence_group_ids = tuple(
                record.ticket.group_id for record in bundle.records
            )

        operations_payload = parsed.get("operations")
        operations: list[ExperienceOperation] = []
        if isinstance(operations_payload, Sequence) and not isinstance(
            operations_payload, (str, bytes)
        ):
            for entry in operations_payload:
                if not isinstance(entry, Mapping):
                    continue
                op_type_raw = str(entry.get("op", "upsert")).lower()
                op_type: str
                if op_type_raw in {"remove", "delete"}:
                    op_type = "remove"
                elif op_type_raw in {"upsert", "update", "add"}:
                    op_type = "upsert"
                else:
                    logger.debug("Ignoring unsupported operation type: %s", op_type_raw)
                    continue

                key_raw = entry.get("key")
                key_value = (
                    str(key_raw).strip()
                    if isinstance(key_raw, (str, int)) and str(key_raw).strip()
                    else None
                )

                text_raw = entry.get("text")
                text_value_op = (
                    str(text_raw).strip()
                    if text_raw is not None and str(text_raw).strip()
                    else None
                )

                rationale_raw = entry.get("rationale")
                rationale_value = (
                    str(rationale_raw).strip()
                    if isinstance(rationale_raw, str) and rationale_raw.strip()
                    else None
                )

                evidence_entry = entry.get("evidence")
                if isinstance(evidence_entry, Sequence) and not isinstance(
                    evidence_entry, (str, bytes)
                ):
                    evidence_ids = tuple(
                        self._resolve_group_identifier(item) for item in evidence_entry
                    )
                else:
                    evidence_ids = ()

                operations.append(
                    ExperienceOperation(
                        op=op_type,  # type: ignore[arg-type]
                        key=key_value,
                        text=text_value_op if op_type == "upsert" else None,
                        rationale=rationale_value,
                        evidence=evidence_ids,
                    )
                )

        # Require structured JSON operations; no fallback to text parsing
        if not operations:
            logger.warning(
                "Reflection proposal has no operations; treating as noop. "
                "Structured JSON operations list is required."
            )

        uncertainty_note = parsed.get("uncertainty_note")
        uncertainty_value = (
            str(uncertainty_note).strip()
            if uncertainty_note is not None and str(uncertainty_note).strip()
            else None
        )

        final_uncertainty = uncertainty_note_misc or uncertainty_value

        self._last_debug_info = None

        return ReflectionProposal(
            action=action,
            summary=summary_value,
            critique=critique_value,
            operations=tuple(operations),
            evidence_group_ids=evidence_group_ids,
            uncertainty_note=final_uncertainty,
            text=text_value,
        )  # type: ignore[call-arg]

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
            }
            for op in outcome.operations
        ]
        delta_value = outcome.post_uplift - outcome.pre_uplift
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
                    "evidence_group_ids": list(outcome.proposal.evidence_group_ids),
                    "uncertainty_note": outcome.proposal.uncertainty_note,
                },
                "eligible": outcome.eligible,
                "applied": outcome.applied,
                "pre_uplift": outcome.pre_uplift,
                "post_uplift": outcome.post_uplift,
                "delta": delta_value,
                "guidance_step_before": outcome.guidance_step_before,
                "guidance_step_after": outcome.guidance_step_after,
                "ineligible_reason": outcome.ineligible_reason,
            },
        }
        if self._last_debug_info is not None:
            payload["reflection"]["debug_info"] = self._last_debug_info
            self._last_debug_info = None
        with self.reflection_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")


__all__ = ["ReflectionEngine"]
