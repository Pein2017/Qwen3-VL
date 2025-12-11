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
from ..io.guidance import GuidanceRepository, MissionGuidanceError
from ..types import (
    DeterministicSignals,
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceMetadata,
    ExperienceOperation,
    ExperienceRecord,
    GroupTicket,
    MissionGuidance,
    ReflectionAction,
    ReflectionOutcome,
    ReflectionProposal,
    TrajectoryWithSignals,
)
from ..utils.chinese import normalize_spaces, to_simplified

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
        # Auxiliary queues
        base_dir = reflection_log.parent if reflection_log else None
        self._wishlist_path = base_dir / "prompt_wishlist.jsonl" if base_dir else None
        self._noise_path = (
            base_dir / "label_or_stageA_noise.jsonl" if base_dir else None
        )
        self._validate_template(self.prompt_template)

    def _validate_template(self, template: str) -> None:
        """Basic sanity check for reflection system prompt."""
        if not template.strip():
            raise ValueError("Reflection prompt template must be non-empty")

    # ------------------------------------------------------------------
    # Three-stage reflection (summary -> critique -> batch update)
    # ------------------------------------------------------------------
    def _cache_dir(self) -> Path:
        if self.reflection_log is None:
            raise ValueError("reflection_log path is required for three_stage pipeline")
        cache_dir = self.reflection_log.parent / "reflection_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _cycle_paths(self, epoch: int, reflection_cycle: int) -> Dict[str, Path]:
        base = self._cache_dir()
        prefix = f"epoch{epoch}_cycle{reflection_cycle}"
        return {
            "summary": base / f"{prefix}_summary.json",
            "critique": base / f"{prefix}_critique.json",
            "plan": base / f"{prefix}_plan.json",
        }

    def _write_json(
        self, path: Path, payload: Mapping[str, Any] | Sequence[Any]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")

    def _record_aux_logs(
        self,
        wishlist_entries: Sequence[Mapping[str, Any]],
        noise_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._wishlist_path:
            for entry in wishlist_entries:
                self._append_jsonl(self._wishlist_path, entry)
        if self._noise_path:
            for entry in noise_entries:
                self._append_jsonl(self._noise_path, entry)

    def _read_json(self, path: Path) -> Mapping[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _bundle_conflict_eligible(
        self, bundle: ExperienceBundle
    ) -> Tuple[bool, Optional[str]]:
        """Eligibility: selected mismatch OR mixed label_match OR conflict/needs_review."""
        has_selected_mismatch = False
        has_mixed = False
        for rec in bundle.records:
            label_vals = []
            for cand in rec.candidates:
                sig = cand.signals
                if sig is not None:
                    if (
                        sig.conflict_flag
                        or sig.needs_manual_review
                        or sig.low_agreement
                    ):
                        return True, None
                    label_vals.append(sig.label_match)
                if (
                    rec.winning_candidate is not None
                    and cand.candidate_index == rec.winning_candidate
                ):
                    if sig is not None and sig.label_match is False:
                        has_selected_mismatch = True
            if any(v is True for v in label_vals) and any(
                v is False for v in label_vals
            ):
                has_mixed = True
        if has_selected_mismatch or has_mixed:
            return True, None
        return False, "non_conflict_bundle"

    def _build_summary_payload(self, bundle: ExperienceBundle) -> Dict[str, Any]:
        records = []
        for rec in bundle.records:
            rec_entry = {
                "group_id": rec.ticket.group_id,
                "label": rec.ticket.label,
                "winning_candidate": rec.winning_candidate,
                "stage_a_summaries": rec.ticket.summaries.as_dict(),
            }
            cand_entries = []
            for cand in rec.candidates:
                sig = cand.signals
                cand_entries.append(
                    {
                        "candidate_index": cand.candidate_index,
                        "verdict": cand.verdict,
                        "reason": cand.reason,
                        "label_match": sig.label_match if sig else None,
                        "conflict_flag": sig.conflict_flag if sig else False,
                        "needs_manual_review": sig.needs_manual_review
                        if sig
                        else False,
                        "low_agreement": sig.low_agreement if sig else False,
                        "vote_strength": sig.vote_strength if sig else None,
                        "summary": cand.summary,
                        "critique": cand.critique,
                        "raw_text": cand.raw_text,
                    }
                )
            rec_entry["candidates"] = cand_entries
            records.append(rec_entry)
        return {
            "mission": bundle.mission,
            "guidance_step": bundle.guidance_step,
            "reflection_cycle": bundle.reflection_cycle,
            "records": records,
        }

    def _load_or_build_summary(
        self, summary_path: Path, bundle: ExperienceBundle
    ) -> Mapping[str, Any]:
        if summary_path.exists():
            return self._read_json(summary_path)
        payload = self._build_summary_payload(bundle)
        self._write_json(summary_path, payload)
        return payload

    @staticmethod
    def _reject_experience_text(text: str) -> bool:
        """Heuristic to block Stage-A style summaries leaking into guidance.

        Relaxed filter: requires multiple Stage-A format markers to trigger.
        """
        # More precise: require multiple Stage-A format markers simultaneously
        is_stage_a_format = (
            "×" in text
            and "/" in text
            and any(k in text for k in ["设备", "显示完整", "只显示部分"])
        )
        return is_stage_a_format or "标签/" in text

    def _compact_guidance(self, mission: str) -> Optional[MissionGuidance]:
        """Dedup, normalize, reindex experiences G0..Gn."""
        guidance_map = self.guidance_repo.load()
        if mission not in guidance_map:
            return None
        current = guidance_map[mission]
        seen_texts = set()
        new_experiences: Dict[str, str] = {}
        new_metadata: Dict[str, ExperienceMetadata] = {}

        def _normalize(text: str) -> str:
            return " ".join(text.strip().split())

        for _, (key, text) in enumerate(current.experiences.items()):
            norm = _normalize(text)
            if self._reject_experience_text(norm):
                raise MissionGuidanceError(
                    f"Experience text appears to be Stage-A summary: {text!r}"
                )
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            new_key = f"G{len(new_experiences)}"
            new_experiences[new_key] = norm
            if key in current.metadata:
                meta = current.metadata[key]
                new_metadata[new_key] = meta
        if not new_experiences:
            raise MissionGuidanceError("Compaction would remove all experiences")
        compacted = MissionGuidance(
            mission=current.mission,
            focus=current.focus,
            experiences=new_experiences,
            step=current.step,
            updated_at=current.updated_at,
            metadata=new_metadata,
        )
        # Snapshot then write
        self.guidance_repo._create_snapshot()
        guidance_map = dict(guidance_map)
        guidance_map[mission] = compacted
        self.guidance_repo._write(guidance_map)
        self.guidance_repo.invalidate()
        return compacted

    def _generate_three_stage_ops(
        self, summary_payload: Mapping[str, Any], *, bundle: ExperienceBundle
    ) -> List[dict]:
        guidance_map = self.guidance_repo.load()
        current = guidance_map.get(bundle.mission)
        if current is None:
            raise MissionGuidanceError(f"Mission {bundle.mission} guidance not found")
        experiences_lines = [
            f"[{key}] {text}" for key, text in sorted(current.experiences.items())
        ]
        case_lines: List[str] = []
        for rec in bundle.records:
            gid = rec.ticket.group_id
            label = rec.ticket.label
            win = rec.winning_candidate
            case_lines.append(f"- group_id: {gid}; 人工标签: {label}; 选中候选: {win}")
            summaries = rec.ticket.summaries.as_dict()
            if summaries:
                summary_lines = [
                    f"    {key}: {value}" for key, value in sorted(summaries.items())
                ]
                case_lines.append("  STAGE-A 摘要:")
                case_lines.extend(summary_lines)
            for cand in rec.candidates:
                sig = cand.signals
                match = sig.label_match if sig else None
                case_lines.append(
                    f"  cand#{cand.candidate_index}: verdict={cand.verdict} match={match} reason={cand.reason}"
                )

        # Evidence-aware reflection prompt (First Principle: Pure LLM, no hardcoded rules)
        prompt = (
            "你是质检规则学习助手。任务：分析候选结果，判断是否有证据支持人工标签，然后决定是否学习规则。\n\n"
            "## 任务步骤\n\n"
            "### Step 1: 证据判断\n"
            "分析所有候选的 Verdict 和 Reason，判断是否有证据支持人工标签：\n\n"
            "1. 如果有候选的 Verdict == 人工标签 → 有证据\n"
            "2. 如果所有候选 Verdict != 人工标签，但 Reason 中包含支持标签的信息：\n"
            "   - 标签是 fail 时：Reason 中有负面信号（缺失/不合规/无法确认/未配备等）→ 有证据\n"
            "   - 标签是 pass 时：Reason 中所有必需项都存在且合规 → 有证据\n"
            "3. 如果 Reason 完全与标签矛盾，无任何支持信息 → 无证据\n\n"
            "### Step 2: 决策\n"
            "- 如果有证据：分析错误原因，提取可学习的规则，输出 operations\n"
            '- 如果无证据：输出空数组 []，并在 evidence_analysis 中标注 "no_evidence_for_label"\n\n'
            "## 输出格式（严格 JSON 对象，不要输出其他文本或 Markdown）\n"
            "{\n"
            '  "evidence_analysis": "简述证据判断过程（<=60字）",\n'
            '  "has_evidence": true | false,\n'
            '  "operations": [...]  // 如果 has_evidence=false，输出空数组\n'
            "}\n\n"
            "operations 数组元素字段:\n"
            "- op: add|update|delete\n"
            "- key: 仅 update/delete 必填，add 可留空\n"
            "- text: 仅 add/update 必填，<=120 字，写通用规则句并标明通过/不通过\n"
            "- rationale: 可空，<=40 字\n"
            "- evidence: list，可空\n\n"
            f'最多 {self.config.max_operations or 3} 条；禁止样本细节、品牌/型号、斜杠/"标签/"。\n'
            "语义重复或只改措辞的规则要合并。\n\n"
            "EXPERIENCES:\n"
            + "\n".join(experiences_lines)
            + "\n\nCASES:\n"
            + "\n".join(case_lines)
        )

        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # Disable Qwen3 thinking blocks in reflection prompts
            enable_thinking=False,
        )
        assert isinstance(chat_prompt, str), (
            "apply_chat_template must return string when tokenize=False"
        )

        encoded = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_reflection_length,
        )
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

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
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        # Normalize response immediately after generation
        response = to_simplified(response)
        response = normalize_spaces(response)
        # Strict JSON only - now expecting object with evidence_analysis, has_evidence, operations
        try:
            parsed = json.loads(response)
            # Handle both old format (list) and new format (object with operations)
            if isinstance(parsed, list):
                # Legacy format: return as-is
                return parsed
            if isinstance(parsed, dict):
                # New format: extract operations and inject evidence metadata
                ops = parsed.get("operations", [])
                if not isinstance(ops, list):
                    ops = []
                # Inject evidence metadata into the result for downstream processing
                evidence_analysis = parsed.get("evidence_analysis", "")
                has_evidence = parsed.get("has_evidence", True)
                # If no evidence, mark with special metadata
                if not has_evidence:
                    # Return a special marker that _ops_from_json can recognize
                    return [{"_no_evidence": True, "_analysis": evidence_analysis}]
                return ops
            raise ValueError("critique JSON is not a list or object")
        except Exception as exc:
            self._last_debug_info = {
                "raw_response": response[:500],
                "parse_error": str(exc),
            }
            raise

    def _ops_from_json(
        self, ops_json: Sequence[Mapping[str, Any]], *, bundle: ExperienceBundle
    ) -> Tuple[
        Tuple[ExperienceOperation, ...],
        List[Mapping[str, Any]],
        List[Mapping[str, Any]],
        Optional[str],  # no_evidence_note: "no_evidence_for_label" if no evidence
    ]:
        """Parse reflection JSON into ExperienceOperation list (add/update/delete/none).

        Guardrails:
        - forbid third-state phrases (e.g., 需复核/需人工复核/不写不通过/通过但需复核)
          to avoid leaking review states into guidance;
        - deduplicate by normalized text+key;
        - truncate long text.

        Returns:
            Tuple of (operations, wishlist_entries, noise_entries, no_evidence_note)
            - no_evidence_note is "no_evidence_for_label" if LLM determined no evidence
        """

        # Check for no-evidence marker from _generate_three_stage_ops
        if ops_json and len(ops_json) == 1:
            first_entry = ops_json[0]
            if isinstance(first_entry, Mapping) and first_entry.get("_no_evidence"):
                analysis = first_entry.get("_analysis", "")
                logger.info(f"Reflection found no evidence for label: {analysis}")
                return tuple(), [], [], "no_evidence_for_label"

        max_ops = self.config.max_operations or len(ops_json)
        ops: List[ExperienceOperation] = []
        forbidden_hit = False
        seen_norms: set[str] = set()

        evidence_default = tuple(rec.ticket.group_id for rec in bundle.records)

        def _norm(txt: str) -> str:
            return " ".join(txt.strip().split())

        forbidden_phrases = (
            "需复核",
            "需人工复核",
            "不写不通过",
            "通过但需复核",
            "通过但需人工复核",
        )

        for entry in ops_json:
            if len(ops) >= max_ops:
                break
            if not isinstance(entry, Mapping):
                continue

            op_raw = str(entry.get("op", "")).strip().lower()
            if op_raw == "none":
                # explicit no-op from the model
                continue
            if op_raw not in {"add", "update", "delete", "merge"}:
                continue

            key_raw = entry.get("key")
            key = str(key_raw).strip() if key_raw is not None else None
            text_raw = entry.get("text")
            text = str(text_raw).strip() if text_raw is not None else None
            rationale_raw = entry.get("rationale")
            rationale = (
                str(rationale_raw).strip()
                if isinstance(rationale_raw, str) and rationale_raw.strip()
                else None
            )

            evidence_list = entry.get("evidence")
            if isinstance(evidence_list, Sequence) and not isinstance(
                evidence_list, (str, bytes)
            ):
                evid = tuple(str(x).strip() for x in evidence_list if str(x).strip())
            else:
                evid = evidence_default

            merged_from_raw = entry.get("merged_from")
            merged_from: Tuple[str, ...] = tuple(
                str(x).strip()
                for x in merged_from_raw
                if isinstance(merged_from_raw, Sequence)
                and not isinstance(merged_from_raw, (str, bytes))
                and str(x).strip()
            )

            if op_raw == "delete":
                if not key:
                    continue
                norm_key = f"del::{key}"
                if norm_key in seen_norms:
                    continue
                seen_norms.add(norm_key)
                ops.append(
                    ExperienceOperation(
                        op="remove",
                        key=key,
                        text=None,
                        rationale=rationale,
                        evidence=evid,
                        merged_from=None,
                    )
                )
                continue

            # add / update
            if not text or self._reject_experience_text(text):
                continue
            lowered = text.lower()
            if any(p.lower() in lowered for p in forbidden_phrases):
                forbidden_hit = True
                continue
            if len(text) > 160:
                text = text[:160]

            norm = _norm(text)
            norm_key = f"{op_raw}::{key or ''}::{norm}"
            if norm_key in seen_norms:
                continue
            seen_norms.add(norm_key)

            if op_raw == "add":
                ops.append(
                    ExperienceOperation(
                        op="upsert",
                        key=None,
                        text=text,
                        rationale=rationale,
                        evidence=evid,
                        merged_from=None,
                    )
                )
            elif op_raw == "update":
                if not key:
                    continue
                ops.append(
                    ExperienceOperation(
                        op="upsert",
                        key=key,
                        text=text,
                        rationale=rationale,
                        evidence=evid,
                        merged_from=None,
                    )
                )
            elif op_raw == "merge":
                if not key or not merged_from:
                    continue
                ops.append(
                    ExperienceOperation(
                        op="merge",
                        key=key,
                        text=text,
                        rationale=rationale,
                        evidence=evid,
                        merged_from=merged_from,
                    )
                )

        if forbidden_hit:
            raise ValueError("forbidden_phrase_in_reflection_ops")

        return tuple(ops), [], [], None

    def _build_plan_payload(
        self,
        critique_json: Sequence[Mapping[str, Any]],
        *,
        bundle: ExperienceBundle,
        reflection_id: str,
    ) -> Dict[str, Any]:
        operations, wishlist_entries, noise_entries, no_evidence_note = (
            self._ops_from_json(critique_json, bundle=bundle)
        )
        evidence_ids = tuple(rec.ticket.group_id for rec in bundle.records)

        # Determine action and uncertainty_note based on evidence analysis
        if no_evidence_note:
            action = "noop"
            uncertainty_note = no_evidence_note
        else:
            action = "refine" if operations else "noop"
            uncertainty_note = None

        proposal = ReflectionProposal(
            action=ReflectionAction(action),  # type: ignore[arg-type]
            summary="three_stage critique",
            critique="json_ops",
            operations=operations,
            evidence_group_ids=evidence_ids,
            uncertainty_note=uncertainty_note,
            text=None,
        )
        ops_serializable = [
            {
                "op": op.op,
                "key": op.key,
                "text": op.text,
                "rationale": op.rationale,
                "evidence": list(op.evidence),
                "merged_from": list(op.merged_from) if op.merged_from else None,
            }
            for op in operations
        ]
        payload = {
            "reflection_id": reflection_id,
            "proposal": proposal,
            "operations": operations,
            "raw_ops": critique_json,
            "ops_serializable": ops_serializable,
            "wishlist_entries": wishlist_entries,
            "noise_entries": noise_entries,
        }
        return payload

    @staticmethod
    def _plan_to_json(plan_payload: Mapping[str, Any]) -> Mapping[str, Any]:
        proposal = plan_payload["proposal"]
        return {
            "reflection_id": plan_payload.get("reflection_id"),
            "raw_ops": plan_payload.get("raw_ops"),
            "wishlist_entries": plan_payload.get("wishlist_entries"),
            "noise_entries": plan_payload.get("noise_entries"),
            "proposal": {
                "action": proposal.action,
                "summary": proposal.summary,
                "critique": proposal.critique,
                "operations": plan_payload.get("ops_serializable"),
                "evidence_group_ids": list(proposal.evidence_group_ids),
                "uncertainty_note": proposal.uncertainty_note,
            },
        }

    def _check_eligibility(
        self, bundle: ExperienceBundle
    ) -> Tuple[bool, Optional[str]]:
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
                if sig is not None and (
                    sig.conflict_flag or sig.needs_manual_review or sig.low_agreement
                ):
                    return True, None

        if policy == "contradictions_or_all_wrong":
            for record in bundle.records:
                has_true = any(c.signals.label_match is True for c in record.candidates)
                has_false = any(
                    c.signals.label_match is False for c in record.candidates
                )
                all_wrong = all(
                    c.signals.label_match is False for c in record.candidates
                )
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
        if not ticket.summaries.per_image:
            raise ValueError(
                f"Stage-A summaries missing for ticket {ticket.group_id}; reflection requires evidence"
            )
        experience_candidates = []
        for item in candidates:
            if item.parsed is None:
                continue
            # fallback signals
            signals = item.signals
            if signals is None:
                signals = DeterministicSignals(
                    label_match=(
                        item.parsed.verdict == ticket.label
                        if item.parsed.verdict
                        else None
                    ),
                    self_consistency=None,
                    conflict_flag=False,
                    needs_manual_review=False,
                )

            experience_candidates.append(
                ExperienceCandidate(
                    candidate_index=item.parsed.base.candidate_index,
                    verdict=item.parsed.verdict,
                    reason=item.parsed.reason,
                    signals=signals,
                    summary=None,
                    critique=None,
                    raw_text=item.parsed.base.response_text,
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
        return self._reflect_three_stage(bundle, epoch=epoch, log=log)

    # ------------------------------------------------------------------
    # Deterministic reflection (no LLM, rule-based experience update)
    # ------------------------------------------------------------------
    def _reflect_three_stage(
        self, bundle: ExperienceBundle, *, epoch: int, log: bool
    ) -> ReflectionOutcome:
        reflection_id = uuid.uuid4().hex[:12]
        guidance_step_before = bundle.guidance_step
        guidance_step_after = bundle.guidance_step
        ineligible_reason: Optional[str] = None
        warnings: List[str] = []
        operations: Tuple[ExperienceOperation, ...] = ()
        applied = False

        eligible, reason = self._bundle_conflict_eligible(bundle)
        if not eligible:
            ineligible_reason = reason
            proposal = ReflectionProposal(
                action=ReflectionAction("noop"),
                summary="Skipped non-conflict bundle",
                critique=reason,
                operations=tuple(),
                evidence_group_ids=tuple(rec.ticket.group_id for rec in bundle.records),
                uncertainty_note=reason,
                text=None,
            )  # type: ignore
            outcome = ReflectionOutcome(
                reflection_id=reflection_id,
                mission=bundle.mission,
                proposal=proposal,
                applied=False,
                guidance_step_before=guidance_step_before,
                guidance_step_after=guidance_step_after,
                operations=tuple(),
                eligible=False,
                applied_epoch=None,
                ineligible_reason=ineligible_reason,
                warnings=tuple(warnings),
            )
            if log:
                self._append_log(outcome, epoch=epoch)
            return outcome

        paths = self._cycle_paths(epoch, bundle.reflection_cycle)

        # Stage 1: summary (deterministic)
        summary_payload = self._load_or_build_summary(paths["summary"], bundle)

        # Stage 2: critique (LLM JSON)
        critique_json: Optional[List[dict]] = None
        if paths["critique"].exists():
            try:
                stored = self._read_json(paths["critique"])
                if isinstance(stored, list):
                    critique_json = stored
            except Exception:
                critique_json = None
        if critique_json is None:
            try:
                critique_json = self._generate_three_stage_ops(
                    summary_payload, bundle=bundle
                )
                self._write_json(paths["critique"], critique_json)
            except Exception as exc:
                ineligible_reason = f"generation_error: {exc}"
                warnings.append("generation_error")
        if critique_json is None:
            proposal = ReflectionProposal(
                action=ReflectionAction("noop"),
                summary="Generation error",
                critique=ineligible_reason,
                operations=tuple(),
                evidence_group_ids=tuple(rec.ticket.group_id for rec in bundle.records),
                uncertainty_note="generation_error",
                text=None,
            )  # type: ignore
            outcome = ReflectionOutcome(
                reflection_id=reflection_id,
                mission=bundle.mission,
                proposal=proposal,
                applied=False,
                guidance_step_before=guidance_step_before,
                guidance_step_after=guidance_step_after,
                operations=tuple(),
                eligible=False,
                applied_epoch=None,
                ineligible_reason=ineligible_reason,
                warnings=tuple(warnings),
            )
            if log:
                self._append_log(outcome, epoch=epoch)
            return outcome

        # Stage 3: batch update plan
        try:
            plan_payload = self._build_plan_payload(
                critique_json, bundle=bundle, reflection_id=reflection_id
            )
            self._write_json(paths["plan"], self._plan_to_json(plan_payload))
            proposal = plan_payload["proposal"]  # type: ignore[index]
            operations = plan_payload["operations"]  # type: ignore[index]
        except Exception as exc:
            ineligible_reason = f"generation_error: {exc}"
            warnings.append("generation_error")
            proposal = ReflectionProposal(
                action=ReflectionAction("noop"),
                summary="Generation error",
                critique=str(exc),
                operations=tuple(),
                evidence_group_ids=tuple(rec.ticket.group_id for rec in bundle.records),
                uncertainty_note="generation_error",
                text=None,
            )  # type: ignore[call-arg]
            outcome = ReflectionOutcome(
                reflection_id=reflection_id,
                mission=bundle.mission,
                proposal=proposal,
                applied=False,
                guidance_step_before=guidance_step_before,
                guidance_step_after=guidance_step_after,
                operations=tuple(),
                eligible=False,
                applied_epoch=None,
                ineligible_reason=ineligible_reason,
                warnings=tuple(warnings),
            )
            if log:
                self._append_log(outcome, epoch=epoch)
            return outcome

        evidence_ids = tuple(rec.ticket.group_id for rec in bundle.records)

        if proposal.action == "refine" and operations:
            try:
                updated_guidance = self.guidance_repo.apply_reflection(
                    mission=bundle.mission,
                    proposal=proposal,
                    reflection_id=reflection_id,
                    source_group_ids=evidence_ids,
                    applied_epoch=epoch,
                    operations=operations,
                )
                guidance_step_after = updated_guidance.step
                # Compact after apply
                self._compact_guidance(bundle.mission)
                applied = True
            except Exception as exc:  # pragma: no cover - defensive
                ineligible_reason = str(exc)
                warnings.append("apply_failed")

        outcome = ReflectionOutcome(
            reflection_id=reflection_id,
            mission=bundle.mission,
            proposal=proposal,
            applied=applied,
            guidance_step_before=guidance_step_before,
            guidance_step_after=guidance_step_after,
            operations=operations,
            eligible=True,
            applied_epoch=epoch if applied else None,
            ineligible_reason=ineligible_reason,
            warnings=tuple(warnings),
        )

        if log:
            self._last_debug_info = {
                "cache_paths": {k: str(v) for k, v in paths.items()},
                "summary_cached": paths["summary"].exists(),
                "critique_cached": paths["critique"].exists(),
            }
            self._append_log(outcome, epoch=epoch)
        return outcome

    def _reflect_deterministic(
        self, bundle: ExperienceBundle, *, epoch: int, log: bool
    ) -> ReflectionOutcome:
        # Deterministic path currently only used when config.engine == "deterministic";
        # keep a defensive definition of auto_text before any usage.
        auto_text = (
            "[AUTO] 矛盾/全错/冲突样本：关键挡风板/BBU要素缺失或存在不确定时，一律视为不通过，"
            "不要在 Reason 中使用“证据不足”之类的措辞，而是直接指出缺失或不确定的具体要素。"
        )
        guidance_map = self.guidance_repo.load()
        current_guidance = guidance_map.get(bundle.mission)
        if current_guidance is None:
            raise ValueError(f"Guidance for mission {bundle.mission} not found")

        reflection_id = uuid.uuid4().hex[:12]
        guidance_step_before = current_guidance.step
        guidance_step_after = current_guidance.step
        warnings: List[str] = []

        eligible, ineligible_reason = self._check_eligibility(bundle)

        # No per-epoch cap in simplified pipeline

        evidence_ids = tuple(rec.ticket.group_id for rec in bundle.records)

        if not eligible:
            proposal = ReflectionProposal(
                action=ReflectionAction("noop"),
                summary="Skipped ineligible batch",
                critique=ineligible_reason or "No contradictions",
                operations=tuple(),
                evidence_group_ids=evidence_ids,
                uncertainty_note="ineligible_batch",
                text=None,
            )  # type: ignore[call-arg]
            outcome = ReflectionOutcome(
                reflection_id=reflection_id,
                mission=bundle.mission,
                proposal=proposal,
                applied=False,
                guidance_step_before=guidance_step_before,
                guidance_step_after=guidance_step_after,
                operations=tuple(),
                eligible=False,
                applied_epoch=None,
                ineligible_reason=ineligible_reason,
                warnings=tuple(warnings),
            )
            if log:
                self._append_log(outcome, epoch=epoch)
            return outcome

        # Deduplicate deterministic rule: if identical text already exists, skip
        if (
            current_guidance.experiences
            and auto_text in current_guidance.experiences.values()
        ):
            proposal = ReflectionProposal(
                action=ReflectionAction("noop"),
                summary="Duplicate auto guidance detected; skipping",
                critique="deterministic_conflict_rule already present",
                operations=tuple(),
                evidence_group_ids=evidence_ids,
                uncertainty_note="duplicate_auto_text",
                text=None,
            )  # type: ignore[call-arg]
            outcome = ReflectionOutcome(
                reflection_id=reflection_id,
                mission=bundle.mission,
                proposal=proposal,
                applied=False,
                guidance_step_before=guidance_step_before,
                guidance_step_after=guidance_step_after,
                operations=tuple(),
                eligible=False,
                applied_epoch=None,
                ineligible_reason="duplicate_auto_text",
                warnings=tuple(warnings),
            )
            if log:
                self._append_log(outcome, epoch=epoch)
            return outcome

        # Auto-upsert conservative guidance
        operations: Tuple[ExperienceOperation, ...] = (
            ExperienceOperation(
                op="upsert",  # type: ignore[arg-type]
                key=None,
                text=auto_text,
                rationale="deterministic_conflict_rule",
                evidence=evidence_ids,
                merged_from=None,
            ),
        )

        proposal = ReflectionProposal(
            action=ReflectionAction("refine"),
            summary="deterministic guidance update",
            critique="auto-generated from contradictions/conflicts",
            operations=operations,
            evidence_group_ids=evidence_ids,
            uncertainty_note=None,
            text=None,
        )  # type: ignore[call-arg]

        applied = False
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
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                f"Failed to apply deterministic reflection for mission {bundle.mission}: {exc}",
                exc_info=True,
            )
            warnings.append("apply_failed")

        outcome = ReflectionOutcome(
            reflection_id=reflection_id,
            mission=bundle.mission,
            proposal=proposal,
            applied=applied,
            guidance_step_before=guidance_step_before,
            guidance_step_after=guidance_step_after,
            operations=operations,
            eligible=True,
            applied_epoch=epoch if applied else None,
            ineligible_reason=None,
            warnings=tuple(warnings),
        )

        if log:
            self._append_log(outcome, epoch=epoch)
        return outcome

    def _generate_reflection(self, bundle: ExperienceBundle) -> ReflectionProposal:
        reflection_prompt = self._build_reflection_prompt(bundle)
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": reflection_prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # Disable Qwen3 thinking blocks in reflection prompts
            enable_thinking=False,
        )
        assert isinstance(chat_prompt, str), (
            "apply_chat_template must return string when tokenize=False"
        )

        prompt_length_chars = len(chat_prompt)
        logger.debug(
            f"Reflection prompt length: {prompt_length_chars} chars, max_reflection_length: {self.config.max_reflection_length}"
        )

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
                f"Reflection prompt will be truncated: {full_token_length} tokens > {self.config.max_reflection_length} (max_reflection_length). "
                "This may cause the model to generate incomplete or incorrect responses."
            )

        encoded = self.tokenizer(
            chat_prompt,
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
        # Normalize response immediately after generation
        response = to_simplified(response)
        response = normalize_spaces(response)
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

        try:
            return self._parse_reflection_response(response, bundle)
        except Exception as exc:
            logger.warning(
                "Reflection parse failed (mission=%s): %s; raw=%.500s",
                bundle.mission,
                exc,
                response,
            )
            raise

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

            # 不再引入逐图摘要，改由候选 Reason 承担证据
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
                # Critic insights are now properly wired from ExperienceCandidate
                summary_text = cand.summary
                critique_text = cand.critique
                lines.append(f"    摘要: {summary_text if summary_text else '（无）'}")
                lines.append(
                    f"    评述: {critique_text if critique_text else '（无）'}"
                )
            lines.append("")
            return "\n".join(lines)

        # Sort by priority desc then keep stable original order
        indexed_records = list(enumerate(bundle.records, start=1))
        sorted_records = sorted(indexed_records, key=lambda t: (-_priority(t[1]), t[0]))

        # Prepare preamble text used in token counting (user content only; system prompt counted separately)
        preamble_text = "\n".join(preamble_lines)

        # Greedy packing under token budget (initial, without stats header)
        included: list[tuple[int, ExperienceRecord, str]] = []
        running_text = preamble_text
        for idx, rec in sorted_records:
            block = _record_block(len(included) + 1, rec)
            trial_text = running_text + f"批次: {len(included) + 1} 组\n\n" + block
            if (
                _count_tokens_local(f"{self.prompt_template}\n\n{trial_text}")
                <= self.config.token_budget
            ):
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
        bundle_lines: list[str] = (
            preamble_lines
            + [f"批次: {len(kept_blocks)} 组", ""]
            + stats_lines
            + kept_blocks
        )
        bundle_summary = "\n".join(bundle_lines)
        system_plus = f"{self.prompt_template}\n\n{bundle_summary}"
        full_prompt = system_plus

        # Trim if still exceeding budget (stats/preamble may push over)
        while (
            len(kept_blocks) > 1
            and _count_tokens_local(full_prompt) > self.config.token_budget
        ):
            kept_blocks.pop()
            bundle_lines = (
                preamble_lines
                + [f"批次: {len(kept_blocks)} 组", ""]
                + stats_lines
                + kept_blocks
            )
            bundle_summary = "\n".join(bundle_lines)
            system_plus = f"{self.prompt_template}\n\n{bundle_summary}"
            full_prompt = system_plus

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

        return bundle_summary

    def _parse_reflection_response(
        self, response: str, bundle: ExperienceBundle
    ) -> ReflectionProposal:
        def _clean_line(line: str) -> str:
            return line.strip().strip("`").strip()

        def _parse_csv(value: str) -> Tuple[str, ...]:
            parts = re.split(r"[，,]\s*|\s+", value.strip())
            cleaned = [
                self._resolve_group_identifier(p) for p in parts if p and p.strip()
            ]
            return tuple(cleaned)

        def _strip_quotes(value: str) -> str:
            v = value.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
                return v[1:-1]
            return v

        def _is_valid_key(key: Optional[str]) -> bool:
            if key is None:
                return True
            return bool(re.fullmatch(r"G\d+", key))

        def _parse_operation_line(line: str) -> ExperienceOperation:
            text = line.lstrip("-").strip().lstrip("[").rstrip("]")
            if not text:
                raise ValueError("Empty operation line")
            tokens = text.split(None, 1)
            op_raw = tokens[0].lower()
            if op_raw not in {"upsert", "remove", "merge"}:
                raise ValueError(f"Unsupported operation type: {op_raw}")
            remainder = tokens[1] if len(tokens) > 1 else ""
            kv_pairs = dict(
                (m.group(1).lower(), _strip_quotes(m.group(2)))
                for m in re.finditer(r"(\w+)=([^ ]+|\"[^\"]+\")", remainder)
            )

            key_val = kv_pairs.get("key")
            if not key_val:
                raise ValueError(f"Operation {op_raw} missing key")
            if not _is_valid_key(key_val):
                raise ValueError(f"Invalid experience key '{key_val}', only G* allowed")

            evidence_raw = kv_pairs.get("evidence")
            evidence_ids = _parse_csv(evidence_raw) if evidence_raw else ()

            merged_from_raw = kv_pairs.get("merged_from")
            if merged_from_raw:
                merged_from_candidates = tuple(
                    s for s in _parse_csv(merged_from_raw) if s
                )
                for mk in merged_from_candidates:
                    if not _is_valid_key(mk):
                        raise ValueError(
                            f"Invalid merged_from key '{mk}', only G* allowed"
                        )
                merged_from_val: Optional[Tuple[str, ...]] = merged_from_candidates
            else:
                merged_from_val = None

            text_val = kv_pairs.get("text")
            rationale_val = kv_pairs.get("rationale")

            # Convert traditional Chinese to simplified Chinese and normalize spaces
            if text_val:
                text_val = to_simplified(text_val)
                text_val = normalize_spaces(text_val)
            if rationale_val:
                rationale_val = to_simplified(rationale_val)
                rationale_val = normalize_spaces(rationale_val)

            if op_raw == "remove":
                text_val = None

            return ExperienceOperation(
                op=op_raw,  # type: ignore[arg-type]
                key=key_val,
                text=text_val,
                rationale=rationale_val,
                evidence=evidence_ids,
                merged_from=merged_from_val,
            )

        # Normalize: split by newline, then by semicolon into atomic segments
        segments: List[str] = []
        for line in response.splitlines():
            if not line.strip():
                continue
            for seg in re.split(r"[;；]\s*", line):
                seg_clean = _clean_line(seg)
                if seg_clean:
                    segments.append(seg_clean)

        action_raw: Optional[str] = None
        summary_raw: Optional[str] = None
        critique_raw: Optional[str] = None
        uncertainty_raw: Optional[str] = None
        evidence_ids: Optional[Tuple[str, ...]] = None
        operations: List[ExperienceOperation] = []

        in_operations = False
        for line in segments:
            upper = line.upper()
            if upper.startswith("OPERATIONS"):
                in_operations = True
                # Parse inline ops if present after '='
                if "=" in line:
                    _, inline = line.split("=", 1)
                    inline = inline.strip()
                    if inline.startswith("["):
                        inline = inline[1:]
                    if inline.endswith("]"):
                        inline = inline[:-1]
                    for op_text in re.split(r"[;；]\s*", inline):
                        op_text = op_text.strip()
                        if not op_text:
                            continue
                        try:
                            operations.append(_parse_operation_line(op_text))
                        except Exception as exc:
                            self._last_debug_info = {
                                "mission": bundle.mission,
                                "raw_response": response,
                                "parse_error": f"operation_error: {exc}",
                            }
                            raise
                continue

            if in_operations:
                # stop if closing bracket
                if line.strip() in {"]", "[", "END"}:
                    in_operations = False
                    continue
                # accept lines with or without leading dash
                if re.match(
                    r"^(?:-?\s*)?(UPSERT|REMOVE|MERGE)\b", line, flags=re.IGNORECASE
                ):
                    try:
                        operations.append(_parse_operation_line(line))
                    except Exception as exc:
                        self._last_debug_info = {
                            "mission": bundle.mission,
                            "raw_response": response,
                            "parse_error": f"operation_error: {exc}",
                        }
                        raise
                    continue

            parts = re.split(r"\s*[:：=]\s*", line, maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip().upper(), parts[1].strip()

            if key == "ACTION":
                action_raw = value.lower()
            elif key == "SUMMARY":
                summary_raw = value
            elif key == "CRITIQUE":
                critique_raw = value
            elif key == "UNCERTAINTY":
                uncertainty_raw = value
            elif key == "EVIDENCE_GROUP_IDS":
                evidence_ids = _parse_csv(value)
            elif key == "OPERATIONS":
                in_operations = True
                continue

        if action_raw:
            # tolerate accidental "refine|noop" by taking the first valid token
            tokens = re.split(r"[|;/,\s]+", action_raw)
            for token in tokens:
                token_clean = token.strip()
                if token_clean in {"refine", "noop"}:
                    if token_clean != action_raw:
                        logger.warning(
                            "Reflection ACTION contained options (%s); coercing to %s",
                            action_raw,
                            token_clean,
                        )
                    action_raw = token_clean
                    break
        if not action_raw or action_raw not in {"refine", "noop"}:
            self._last_debug_info = {
                "mission": bundle.mission,
                "raw_response": response,
                "parse_error": f"missing or invalid ACTION ({action_raw})",
            }
            raise ValueError("Reflection response missing ACTION line")

        if action_raw not in {"refine", "noop"}:
            self._last_debug_info = {
                "mission": bundle.mission,
                "raw_response": response,
                "parse_error": f"invalid action {action_raw}",
            }
            raise ValueError(f"Invalid reflection action: {action_raw}")

        if action_raw == "refine" and not operations:
            self._last_debug_info = {
                "mission": bundle.mission,
                "raw_response": response,
                "parse_error": "refine without operations",
            }
            raise ValueError("Refine action requires at least one operation")

        if evidence_ids is None:
            evidence_group_ids = tuple(
                record.ticket.group_id for record in bundle.records
            )
        else:
            evidence_group_ids = evidence_ids

        uncertainty_value = (
            uncertainty_raw.strip()
            if uncertainty_raw and uncertainty_raw.strip()
            else None
        )

        # Convert traditional Chinese to simplified Chinese and normalize spaces
        summary_final = None
        if summary_raw and summary_raw.strip():
            summary_final = to_simplified(summary_raw.strip())
            summary_final = normalize_spaces(summary_final)
        critique_final = None
        if critique_raw and critique_raw.strip():
            critique_final = to_simplified(critique_raw.strip())
            critique_final = normalize_spaces(critique_final)
        uncertainty_final = None
        if uncertainty_value:
            uncertainty_final = to_simplified(uncertainty_value)
            uncertainty_final = normalize_spaces(uncertainty_final)

        self._last_debug_info = None

        return ReflectionProposal(
            action=action_raw,  # type: ignore[arg-type]
            summary=summary_final,
            critique=critique_final,
            operations=tuple(operations),
            evidence_group_ids=evidence_group_ids,
            uncertainty_note=uncertainty_final,
            text=None,
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
