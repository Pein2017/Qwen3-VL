#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reflection engine for Stage-B guidance updates using in-process LLM."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
        reflection_log: Optional[Path] = None,
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
        self._epoch_change_counts: Dict[Tuple[str, int], int] = {}
        self._validate_template(self.prompt_template)

    def _validate_template(self, template: str) -> None:
        """Validate that the reflection prompt template satisfies required hints.

        - Must contain required placeholders: {mission}, {focus}, {experiences}, {bundle}
        - Must mention strict JSON output requirement
        - Must mention merge in allowed ops (upsert|remove|merge)
        - If max_operations is set, must mention the budget symbol 'K'
        """
        # Check for required placeholders (warn-only to allow stub prompts in tests)
        required_placeholders = ["{mission}", "{focus}", "{experiences}", "{bundle}"]
        missing = [p for p in required_placeholders if p not in template]
        if missing:
            logger.warning(
                f"Reflection prompt template missing placeholders: {', '.join(missing)}"
            )

        # Check for strict JSON output requirement
        if "JSON" not in template and "json" not in template:
            logger.warning("Reflection prompt template does not mention JSON requirement")

        # Require mention of merge operation in allowed ops
        if "upsert|remove|merge" not in template:
            name = Path(self.config.prompt_path).name
            if name == "stage_b_reflection_prompt.txt":
                raise ValueError("Reflection prompt template must include allowed ops 'upsert|remove|merge'")
            logger.warning("Reflection prompt template missing allowed ops 'upsert|remove|merge' hint")

        # If budgets are configured, require a 'K' mention to signal cap to the model
        if self.config.max_operations is not None and "K" not in template:
            # For canonical template name, treat as error; otherwise warn
            name = Path(self.config.prompt_path).name
            if name == "stage_b_reflection_prompt.txt":
                raise ValueError("Reflection prompt template must mention budget symbol 'K' when max_operations is set")
            logger.warning("Reflection template missing budget symbol 'K' while max_operations is set")

    def _check_eligibility(self, bundle: ExperienceBundle) -> Tuple[bool, Optional[str]]:
        """
        Check if bundle is eligible for reflection according to configured policy.

        Policies:
        - selected_mismatch_or_all_wrong (default):
          Eligible if any record's winning candidate mismatches the label OR
          all candidates in any record mismatch the label.
        - contradictions_only: Eligible if any record contains contradictions across
          candidates (mixed label_match True/False or mixed verdicts pass/fail).
        """
        policy = getattr(
            self.config, "eligibility_policy", "selected_mismatch_or_all_wrong"
        )

        # New guardrails: conflicts or manual-review flags should trigger reflection
        for record in bundle.records:
            for cand in record.candidates:
                sig = cand.signals
                if sig is not None and (sig.conflict_flag or sig.needs_manual_review):
                    return True, None

        if policy == "contradictions_or_all_wrong":
            for record in bundle.records:
                has_true = any(c.signals.label_match is True for c in record.candidates)
                has_false = any(c.signals.label_match is False for c in record.candidates)
                all_wrong = all(c.signals.label_match is False for c in record.candidates)
                if (has_true and has_false) or all_wrong:
                    return True, None
            return False, "No contradictions and no all-wrong groups"
        if policy == "contradictions_only":
            for record in bundle.records:
                has_true = any(c.signals.label_match is True for c in record.candidates)
                has_false = any(
                    c.signals.label_match is False for c in record.candidates
                )
                verdicts = {
                    c.verdict for c in record.candidates if c.verdict is not None
                }
                verdict_contradiction = "pass" in verdicts and "fail" in verdicts
                if (has_true and has_false) or verdict_contradiction:
                    return True, None
            return False, "No contradictions across candidates"

        # Default policy: selected_mismatch_or_all_wrong
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
            # Extract critic insights if available
            summary_text = None
            critique_text = None
            if item.critic is not None:
                summary_text = item.critic.summary
                critique_text = item.critic.critique

            if item.parsed is not None and item.signals is not None:
                experience_candidates.append(
                    ExperienceCandidate(
                        candidate_index=item.parsed.base.candidate_index,
                        verdict=item.parsed.verdict,
                        reason=item.parsed.reason,
                        confidence=item.signals.confidence,
                        signals=item.signals,
                        summary=summary_text,
                        critique=critique_text,
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
        log: bool = True,
    ) -> ReflectionOutcome:
        reflection_id = uuid.uuid4().hex[:12]
        guidance_step_before = bundle.guidance_step
        guidance_step_after = bundle.guidance_step
        operations: Tuple[ExperienceOperation, ...] = ()
        ineligible_reason: Optional[str] = None
        warnings: List[str] = []
        proposal: Optional[ReflectionProposal] = None

        # Check eligibility using configured policy
        eligible_for_reflection, eligibility_reason = self._check_eligibility(bundle)

        if not eligible_for_reflection:
            evidence_ids = tuple(record.ticket.group_id for record in bundle.records)
            proposal = ReflectionProposal(
                action=ReflectionAction("noop"),
                summary="Skipped ineligible batch",
                critique=f"Bundle not eligible for reflection: {eligibility_reason}",
                operations=(),
                evidence_group_ids=evidence_ids,
                uncertainty_note="ineligible_batch",
                text=None,
            )  # type: ignore[call-arg]
            logger.debug(
                f"Skipping reflection for mission {bundle.mission}: {eligibility_reason}"
            )
            eligible = False
            ineligible_reason = eligibility_reason
        else:
            # Initialize eligibility for this branch
            eligible = True
            # If epoch cap is already exhausted, skip generation and mark ineligible
            if self.config.change_cap_per_epoch is not None:
                used_so_far = self._epoch_change_counts.get((bundle.mission, epoch), 0)
                if used_so_far >= self.config.change_cap_per_epoch:
                    ineligible_reason = "Epoch change cap reached"
                    eligible = False
                    operations = tuple()
                    proposal = ReflectionProposal(
                        action=ReflectionAction("noop"),
                        summary="Epoch cap exhausted",
                        critique=None,
                        operations=tuple(),
                        evidence_group_ids=tuple(
                            r.ticket.group_id for r in bundle.records
                        ),
                        uncertainty_note=None,
                        text=None,
                    )  # type: ignore

            if eligible:
                # Manual review strategy for all-wrong groups
                if getattr(self.config, "all_wrong_strategy", "reflect_diagnose") == "manual_review":
                    any_all_wrong = any(all(c.signals.label_match is False for c in r.candidates) for r in bundle.records)
                    if any_all_wrong:
                        ineligible_reason = "all_wrong_manual_review"
                        eligible = False
                        operations = tuple()
                        proposal = ReflectionProposal(  # type: ignore[call-arg]
                            action=ReflectionAction("noop"),
                            summary="All-wrong manual review",
                            critique="Flagged for 人工复核",
                            operations=tuple(),
                            evidence_group_ids=tuple(r.ticket.group_id for r in bundle.records),
                            uncertainty_note="all_wrong_manual_review",
                            text=None,
                        )
                # Otherwise, run reflection to propose operations
                if eligible:
                    gen_error: Optional[str] = None
                    try:
                        proposal = self._generate_reflection(bundle)
                        operations = proposal.operations
                        eligible = (
                            proposal.action == "refine"
                            and bool(operations)
                            and (
                                self.config.allow_uncertain or not proposal.uncertainty_note
                            )
                        )
                        if not eligible and proposal.action == "refine":
                            if not operations:
                                ineligible_reason = "No operations in proposal"
                            elif (
                                proposal.uncertainty_note
                                and not self.config.allow_uncertain
                            ):
                                ineligible_reason = (
                                    f"Uncertainty gating: {proposal.uncertainty_note}"
                                )
                            else:
                                ineligible_reason = "Proposal validation failed"

                        if proposal.action == "refine" and not operations:
                            logger.warning(
                                f"Reflection for mission {bundle.mission} produced no operations; treating as noop"
                            )
                            eligible = False

                        if (
                            proposal.action == "refine"
                            and proposal.uncertainty_note
                            and not self.config.allow_uncertain
                        ):
                            logger.info(
                                f"Skipping reflection for mission {bundle.mission} due to uncertainty gating"
                            )

                        # Budget enforcement
                        if proposal.action == "refine" and eligible and operations:
                            # Per-cycle max operations
                            if (
                                self.config.max_operations is not None
                                and len(operations) > self.config.max_operations
                            ):
                                warnings.append(
                                    f"truncated_by_max_operations: {len(operations)} -> {self.config.max_operations}"
                                )
                                operations = tuple(operations[: self.config.max_operations])
                            # Per-epoch change cap per mission
                            if self.config.change_cap_per_epoch is not None:
                                used_so_far = self._epoch_change_counts.get(
                                    (bundle.mission, epoch), 0
                                )
                                remaining = self.config.change_cap_per_epoch - used_so_far
                                if remaining <= 0:
                                    eligible = False
                                    ineligible_reason = "Epoch change cap reached"
                                    operations = ()
                                    warnings.append(
                                        f"epoch_cap_exhausted: used={used_so_far} cap={self.config.change_cap_per_epoch}"
                                    )
                                elif len(operations) > remaining:
                                    warnings.append(
                                        f"truncated_by_epoch_cap: {len(operations)} -> {remaining} (used={used_so_far}, cap={self.config.change_cap_per_epoch})"
                                    )
                                    operations = tuple(operations[:remaining])
                    except Exception as exc:
                            # Generation failed (likely due to test-time mocks). Create a noop proposal and mark ineligible.
                            gen_error = str(exc)
                            evidence_ids = tuple(r.ticket.group_id for r in bundle.records)
                            proposal = ReflectionProposal(
                                action=ReflectionAction("noop"),
                                summary="Generation error",
                                critique=gen_error,
                                operations=tuple(),
                                evidence_group_ids=evidence_ids,
                                uncertainty_note="generation_error",
                                text=None,
                            )  # type: ignore
                            operations = tuple()
                            eligible = False
                            ineligible_reason = "Generation error"
                            warnings.append("generation_error")

        assert proposal is not None, "proposal should be set by this point"
        if proposal.action != "noop":
                logger.info(
                    f"Reflection proposal for mission {bundle.mission}: action={proposal.action}, eligible={eligible}, ops={len(operations)}"
                )

        # Apply operations if eligible
        applied = False
        if eligible and operations:
            try:
                updated_guidance = self.guidance_repo.apply_reflection(
                    mission=bundle.mission,
                    proposal=proposal,
                    reflection_id=reflection_id,
                    source_group_ids=list(proposal.evidence_group_ids),
                    applied_epoch=epoch,
                    operations=operations,
                )
                guidance_step_after = updated_guidance.step
                applied = True
                logger.info(
                    f"Applied reflection for mission {bundle.mission}: step {guidance_step_before} -> {guidance_step_after}"
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    f"Failed to apply reflection for mission {bundle.mission}: {exc}",
                    exc_info=True,
                )

        # Update per-epoch change counters if applied
        if applied and self.config.change_cap_per_epoch is not None:
            key = (bundle.mission, epoch)
            prev = self._epoch_change_counts.get(key, 0)
            self._epoch_change_counts[key] = prev + len(operations)

        assert proposal is not None, "proposal should always be set"
        outcome = ReflectionOutcome(
            reflection_id=reflection_id,
            mission=bundle.mission,
            proposal=proposal,
            applied=applied,
            guidance_step_before=guidance_step_before,
            guidance_step_after=guidance_step_after,
            operations=operations,
            eligible=eligible,
            applied_epoch=epoch if applied else None,
            ineligible_reason=ineligible_reason,
            warnings=tuple(warnings),
        )

        if log:
            # Attach candidate summaries/critique for provenance
            try:
                self._last_debug_info = {
                    "records": [
                        {
                            "group_id": rec.ticket.group_id,
                            "winning_candidate": rec.winning_candidate,
                            "candidates": [
                                {
                                    "candidate_index": c.candidate_index,
                                    "label_match": c.signals.label_match,
                                }
                                for c in rec.candidates
                            ],
                        }
                        for rec in bundle.records
                    ]
                }
            except Exception:
                pass
            self._append_log(outcome, epoch=epoch)
        return outcome


    def _generate_reflection(self, bundle: ExperienceBundle) -> ReflectionProposal:
        reflection_prompt = self._build_reflection_prompt(bundle)

        prompt_length_chars = len(reflection_prompt)
        logger.debug(
            f"Reflection prompt length: {prompt_length_chars} chars, max_reflection_length: {self.config.max_reflection_length}"
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
                f"Reflection prompt will be truncated: {full_token_length} tokens > {self.config.max_reflection_length} (max_reflection_length). "
                "This may cause the model to generate incomplete or incorrect responses."
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
            f"Reflection prompt token length: {prompt_token_length} tokens (max: {self.config.max_reflection_length}, full: {full_token_length})"
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
            f"Generated {num_generated_tokens} tokens (max_new_tokens: {self.config.max_new_tokens})"
        )

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response_with_special = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=False
        )

        logger.info(
            f"Reflection response: {len(response)} chars, {num_generated_tokens} tokens"
        )

        if len(response) < 100:
            logger.warning(
                f"Short reflection response detected ({len(response)} chars):\n"
                f"Response (skip_special_tokens=True): {response!r}\n"
                f"Response (skip_special_tokens=False): {response_with_special!r}\n"
                f"First 10 token IDs: {generated_tokens[:10].tolist() if num_generated_tokens > 0 else '[]'}"
            )
        elif len(response) > 2000:
            logger.debug(
                f"Long reflection response ({len(response)} chars), first 500 chars: {response[:500]}"
            )

        return self._parse_reflection_response(response, bundle)


    def _build_reflection_prompt(self, bundle: ExperienceBundle) -> str:
        """Build reflection prompt with token-budgeted packing and prioritization.

        Strategy (P2.13):
        - Sort records by priority: contradictions > selected-mismatch > others
        - Pack record blocks until token_budget is reached (including template+preamble)
        - Recompute stats on the kept subset; if still over budget, trim tail
        - Log trimming decisions
        """
        guidance_map = self.guidance_repo.load()
        current_guidance = guidance_map.get(bundle.mission)

        # Experiences text (existing guidance snapshot)
        experiences_text = ""
        if current_guidance and current_guidance.experiences:
            experiences_lines = [
                f"[{key}]. {value}"
                for key, value in sorted(current_guidance.experiences.items())
            ]
            experiences_text = "\n".join(experiences_lines)

        # Focus text (mission-specific)
        if current_guidance and current_guidance.focus:
            task_focus = current_guidance.focus
        else:
            task_focus = STAGE_B_MISSION_FOCUS.get(bundle.mission, "未定义任务侧重点")

        # Preamble (does not depend on chosen records)
        preamble_lines = [
            f"任务: {bundle.mission}",
            f"检查清单: {task_focus}",
            f"反思周期: {bundle.reflection_cycle}",
            f"指导步骤: {bundle.guidance_step}",
            "",
        ]
        if experiences_text:
            preamble_lines += [
                "当前指导经验:",
                experiences_text,
                "",
            ]

        # Local tokenizer helper (exact count)
        def _count_tokens_local(text: str) -> int:
            try:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
                # Dict-style
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    ids = encoded["input_ids"]
                    if hasattr(ids, "size"):
                        return int(ids.size(1))  # type: ignore[attr-defined]
                # Attr-style
                if hasattr(encoded, "input_ids") and hasattr(encoded.input_ids, "size"):
                    return int(encoded.input_ids.size(1))  # type: ignore[attr-defined]
            except Exception:
                pass
            # Fallback heuristic
            return max(1, len(text) // 6)

        # Priority function
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
                return 3  # Mixed contradiction has highest priority
            all_wrong = all(c.signals.label_match is False for c in rec.candidates)
            if all_wrong:
                return 2  # All-wrong has lower priority than mixed contradiction
            if selected_mismatch:
                return 1
            return 0

        # Build per-record text blocks
        def _record_block(idx: int, rec: ExperienceRecord) -> str:
            lines: List[str] = []
            ticket = rec.ticket
            short_group_id = f"第{idx}组"
            lines.extend(
                [
                    f"{short_group_id}:",
                    f"  标签: {ticket.label}",
                    f"  获胜: {rec.winning_candidate}",
                    "",
                ]
            )
            all_wrong = all(c.signals.label_match is False for c in rec.candidates)
            if all_wrong:
                lines.append("  特殊: 全部候选与标签不一致（all-wrong）")

            summaries_dict = ticket.summaries.as_dict()
            if summaries_dict:
                summaries_text = _render_summaries(summaries_dict)
                for line in summaries_text.split("\n"):
                    lines.append(f"    {line}")
            else:
                lines.append("    （无）")
            lines.append("")
            for cand in rec.candidates:
                signals = cand.signals
                label_match_str = (
                    "✓" if signals.label_match else "✗" if signals.label_match is False else "?"
                )
                confidence_str = f"{cand.confidence:.2f}" if cand.confidence is not None else "-"
                lines.append(
                    f"  {cand.candidate_index}: {cand.verdict} | {cand.reason} | 匹配{label_match_str} 置信{confidence_str}"
                )
                # Critic insights are now properly wired from ExperienceCandidate
                summary_text = cand.summary
                critique_text = cand.critique
                if not summary_text or not critique_text:
                    logger.debug(
                        f"No critic output for candidate {cand.candidate_index} in group {rec.ticket.group_id} (critic may be disabled)"
                    )
                lines.append(f"    摘要: {summary_text if summary_text else '（无）'}")
                lines.append(f"    评述: {critique_text if critique_text else '（无）'}")
            lines.append("")
            return "\n".join(lines)

        # Sort by priority desc then keep stable original order
        indexed_records = list(enumerate(bundle.records, start=1))
        sorted_records = sorted(indexed_records, key=lambda t: (-_priority(t[1]), t[0]))

        # Prepare preamble text used in token counting
        preamble_text = f"{self.prompt_template}\n\n" + "\n".join(preamble_lines)

        # Greedy packing under token budget (initial, without stats header)
        included: list[tuple[int, ExperienceRecord, str]] = []
        running_text = preamble_text
        for idx, rec in sorted_records:
            block = _record_block(len(included) + 1, rec)
            trial_text = running_text + f"批次: {len(included) + 1} 组\n\n" + block
            if _count_tokens_local(trial_text) <= self.config.token_budget:
                included.append((idx, rec, block))
                running_text = running_text + f"批次: {len(included)} 组\n\n" + block
            else:
                # Try to include at least one record
                if not included:
                    included.append((idx, rec, block))
                break

        # Compute stats for included subset
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

        # Assemble final text and trim tail until within budget
        kept_blocks = [blk for _, _, blk in included]
        bundle_lines: list[str] = preamble_lines + [f"批次: {len(kept_blocks)} 组", ""] + stats_lines + kept_blocks
        bundle_summary = "\n".join(bundle_lines)
        full_prompt = f"{self.prompt_template}\n\n{bundle_summary}"

        # Trim if still exceeding budget (stats/preamble may push over)
        while len(kept_blocks) > 1 and _count_tokens_local(full_prompt) > self.config.token_budget:
            kept_blocks.pop()
            bundle_lines = preamble_lines + [f"批次: {len(kept_blocks)} 组", ""] + stats_lines + kept_blocks
            bundle_summary = "\n".join(bundle_lines)
            full_prompt = f"{self.prompt_template}\n\n{bundle_summary}"

        # Update group-id mapping for parser (use kept order 1..K)
        self._group_id_mapping.clear()
        for new_idx, (_, rec, _) in enumerate(included[: len(kept_blocks)], start=1):
            short_form = f"第{new_idx}组"
            self._group_id_mapping[short_form] = rec.ticket.group_id

        # Log trimming
        trimmed = len(bundle.records) - len(kept_blocks)
        if trimmed > 0:
            final_tokens = _count_tokens_local(full_prompt)
            logger.info(
                f"reflection_token_budget_trim: kept={len(kept_blocks)} dropped={trimmed} budget={self.config.token_budget} tokens={final_tokens}"
            )

        return full_prompt

    def _parse_reflection_response(
        self, response: str, bundle: ExperienceBundle
    ) -> ReflectionProposal:
        def _extract_all_json_objects(text: str) -> List[Tuple[str, int]]:
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
                    logger.debug(f"Skipping error JSON object: {candidate_text[:200]}")
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
                f"Could not parse valid reflection JSON from response. "
                f"Found {len(json_candidates)} JSON candidates, but none had 'action' field. "
                f"Response length: {len(response)} chars"
            )

            if truncated:
                error_msg += " (TRUNCATED - missing closing braces)"

            logger.error(
                f"{error_msg}\n"
                f"Response start (first 500 chars):\n{response_preview}\n"
                f"Response end (last 200 chars):\n{response_suffix}"
            )

            if json_candidates:
                logger.debug(f"JSON candidates found: {len(json_candidates)}")
                for idx, (cand_text, _) in enumerate(json_candidates[:5]):
                    logger.debug(
                        f"  Candidate {idx} (first 300 chars): {cand_text[:300]}"
                    )
            if valid_json_objects:
                logger.debug(
                    f"Found {len(valid_json_objects)} valid JSON objects without 'action' field:"
                )
                for idx, (cand_text, parsed, _) in enumerate(valid_json_objects[:3]):
                    logger.debug(
                        f"  Valid JSON {idx} (first 300 chars): {cand_text[:300]}\n  Parsed keys: {list(parsed.keys())}"
                    )

            # Store debug info for callers/tests to inspect
            self._last_debug_info = {
                "timestamp": bundle.records[0].ticket.group_id
                if bundle.records
                else "unknown",
                "mission": bundle.mission,
                "response_length": len(response),
                "json_candidates_count": len(json_candidates),
                "truncated": truncated,
                "raw_response": response,
                "json_candidates": [cand[0][:500] for cand in json_candidates[:5]],
                "parse_error": "No valid JSON with 'action' field found",
            }

            # Treat as fatal error (no fallback to text parsing)
            raise ValueError(
                f"No valid JSON: {error_msg}"
            )

        parsed = reflection_json

        action_raw = str(parsed.get("action", "noop")).lower()
        uncertainty_note_misc = None
        if action_raw not in {"refine", "noop"}:
            logger.warning(
                f"Invalid action {action_raw}, using noop (only 'refine' and 'noop' are supported)"
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
        operations: List[ExperienceOperation] = []
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
                elif op_type_raw in {"merge"}:
                    op_type = "merge"
                else:
                    logger.debug(f"Ignoring unsupported operation type: {op_type_raw}")
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

                merged_from_raw = entry.get("merged_from")
                if isinstance(merged_from_raw, Sequence) and not isinstance(
                    merged_from_raw, (str, bytes)
                ):
                    merged_from_val: Tuple[str, ...] | None = tuple(
                        str(m).strip() for m in merged_from_raw if str(m).strip()
                    )
                elif isinstance(merged_from_raw, str) and merged_from_raw.strip():
                    merged_from_val = (merged_from_raw.strip(),)
                else:
                    merged_from_val = None

                operations.append(
                    ExperienceOperation(
                        op=op_type,  # type: ignore[arg-type]
                        key=key_value,
                        text=text_value_op if op_type in {"upsert", "merge"} else None,
                        rationale=rationale_value,
                        evidence=evidence_ids,
                        merged_from=merged_from_val,
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
                "merged_from": (list(op.merged_from) if op.merged_from else None),
            }
            for op in outcome.operations
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
                    "evidence_group_ids": list(outcome.proposal.evidence_group_ids),
                    "uncertainty_note": outcome.proposal.uncertainty_note,
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
