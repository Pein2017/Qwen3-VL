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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Set

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config.missions import STAGE_B_MISSION_FOCUS

from ..config import ReflectionConfig
from ..io.guidance import GuidanceRepository, MissionGuidanceError
from ..sampling.prompts import _render_summaries
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
                    if sig.conflict_flag or sig.needs_manual_review:
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
                        "summary": cand.summary,
                        "critique": cand.critique,
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
        """Heuristic to block Stage-A style summaries leaking into guidance."""
        if "×" in text:
            return True
        if "标签/" in text:
            return True
        return False

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
            case_lines.append(f"- group_id: {gid}; gt: {label}; winning: {win}")
            for cand in rec.candidates:
                sig = cand.signals
                match = sig.label_match if sig else None
                case_lines.append(
                    f"  cand#{cand.candidate_index}: verdict={cand.verdict} match={match} reason={cand.reason}"
                )

        prompt = (
            "你是规则反思助手。请仅输出 JSON 数组，不要输出其他任何文本或 Markdown。\n"
            "每个元素字段: group_id, action(add_rule|ask_more_info|abstain_noise|keep), why, missing_evidence(list,可空), new_rule(仅 add_rule)。\n"
            "当且仅当现有证据已部分支持 GT 且缺少明确要点时，才提出 add_rule+new_rule；若需要更多材料，用 ask_more_info 并写 missing_evidence；疑似噪声用 abstain_noise；否则 keep。\n"
            f"最多 {self.config.max_operations or 3} 条；禁止使用样本细节或包含“×/标签/”。\n\n"
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
            messages, add_generation_prompt=True, tokenize=False
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
        # Strict JSON only
        try:
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                raise ValueError("critique JSON is not a list")
            return parsed
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
    ]:
        """Parse counterfactual reflection JSON into ops + auxiliary queues."""

        record_by_gid = {rec.ticket.group_id: rec for rec in bundle.records}
        ops: List[ExperienceOperation] = []
        wishlist: List[Mapping[str, Any]] = []
        noise: List[Mapping[str, Any]] = []
        max_ops = self.config.max_operations or len(ops_json)

        # Pre-compute conflict status per group so we can reason independently of
        # what the reflection LLM decides to do for each entry.
        conflict_gids: List[str] = []
        conflict_status: Dict[str, Dict[str, bool]] = {}
        for rec in bundle.records:
            signals = [
                cand.signals for cand in rec.candidates if cand.signals is not None
            ]
            has_true = any(sig.label_match is True for sig in signals)
            has_false = any(sig.label_match is False for sig in signals)
            selected_mismatch = False
            if rec.winning_candidate is not None:
                for cand in rec.candidates:
                    sig = cand.signals
                    if (
                        sig is not None
                        and cand.candidate_index == rec.winning_candidate
                    ):
                        selected_mismatch = sig.label_match is False
                        break
            all_wrong = has_false and not has_true
            if has_false:
                conflict_gids.append(rec.ticket.group_id)
            conflict_status[rec.ticket.group_id] = {
                "has_true": has_true,
                "has_false": has_false,
                "selected_mismatch": selected_mismatch,
                "all_wrong": all_wrong,
            }

        # Track which actions the LLM proposed per group so we can identify
        # conflicts where it effectively did "nothing" (keep) and force
        # a conservative rule instead of labelling them as pure noise.
        per_gid_actions: Dict[str, Set[str]] = {}

        def _as_list(obj: object) -> List[str]:
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                return [str(x).strip() for x in obj if str(x).strip()]
            if obj is None:
                return []
            return [str(obj).strip()] if str(obj).strip() else []

        for entry in ops_json:
            if not isinstance(entry, Mapping):
                continue
            action = str(entry.get("action", "")).strip().lower()
            gid = str(entry.get("group_id", "")).strip()
            why = str(entry.get("why", "")).strip()
            missing = _as_list(entry.get("missing_evidence"))[:3]
            new_rule = str(entry.get("new_rule", "")).strip()
            if not gid or gid not in record_by_gid:
                continue
            rec = record_by_gid[gid]
            per_gid_actions.setdefault(gid, set()).add(action or "keep")

            status = conflict_status.get(gid, {})
            has_true = status.get("has_true", False)
            has_false = status.get("has_false", False)
            selected_mismatch = status.get("selected_mismatch", False)
            all_wrong = status.get("all_wrong", False)
            has_contradiction = has_true and has_false

            if action == "add_rule":
                if len(ops) >= max_ops:
                    continue
                if not new_rule:
                    continue
                if self._reject_experience_text(new_rule):
                    continue
                # Allow longer guidance sentences; soft-cap to avoid runaway generations
                if len(new_rule) > 120:
                    new_rule = new_rule[:120]
                # 允许在标签不符或存在矛盾时也添加规则，用于对冲全错/选错的场景
                allow_add = (
                    has_true or all_wrong or has_contradiction or selected_mismatch
                )
                if not allow_add:
                    noise.append(
                        {
                            "mission": bundle.mission,
                            "group_id": gid,
                            "label": rec.ticket.label,
                            "reason": "no_supporting_or_conflict_signal",
                            "why": why,
                        }
                    )
                    continue
                ops.append(
                    ExperienceOperation(
                        op="upsert",
                        key=None,
                        text=new_rule,
                        rationale="add_rule",
                        evidence=(gid,),
                        merged_from=None,
                    )
                )
            elif action == "ask_more_info":
                wishlist.append(
                    {
                        "mission": bundle.mission,
                        "group_id": gid,
                        "label": rec.ticket.label,
                        "missing_evidence": missing,
                        "why": why,
                    }
                )
            elif action == "abstain_noise":
                noise.append(
                    {
                        "mission": bundle.mission,
                        "group_id": gid,
                        "label": rec.ticket.label,
                        "reason": "abstain_noise",
                        "why": why,
                    }
                )
            else:  # keep or unknown
                # Legacy behaviour: optionally treat keep-on-conflict as noise/wishlist.
                # When treat_keep_conflict_as_noise is False (default), these samples
                # are left for the conflict policy below to handle (no auto-noise).
                if self.config.treat_keep_conflict_as_noise:
                    if all_wrong or (selected_mismatch and not has_true):
                        noise.append(
                            {
                                "mission": bundle.mission,
                                "group_id": gid,
                                "label": rec.ticket.label,
                                "reason": "auto_keep_mismatch",
                                "why": why or "标签与候选全不符，视为噪声或需人工复核",
                            }
                        )
                    elif selected_mismatch and has_true:
                        wishlist.append(
                            {
                                "mission": bundle.mission,
                                "group_id": gid,
                                "label": rec.ticket.label,
                                "missing_evidence": missing
                                or [
                                    f"补充能支撑标签为 {rec.ticket.label} 的关键部件近景/安装状态"
                                ],
                                "why": why or "标签与当前判定不一致，需要补充证据澄清",
                            }
                        )

        # Conflict guardrail: for conflicting groups, always ensure a rule
        # exists (unless LLM already proposed add_rule). This prevents
        # conflicts from being silently marked as noise/keep.
        if self.config.require_rule_for_conflicts and max_ops != 0:
            for gid, status in conflict_status.items():
                is_conflict = status.get("all_wrong", False) or status.get(
                    "selected_mismatch", False
                )
                if not is_conflict:
                    continue
                actions = per_gid_actions.get(gid, set())
                # Only skip when add_rule 已出现；ask_more_info/abstain_noise/keep
                # 仍会触发保守规则注入，以覆盖 GT 与模型冲突场景。
                if "add_rule" in actions:
                    continue
                if len(ops) >= max_ops:
                    break
                # Reuse the generic conservative rule text; duplicates will be
                # merged via GuidanceRepository._build_updated_guidance.
                rule_text = (
                    "当关键部件缺失、未按要求安装或不可见/模糊/遮挡时，判定不通过或需人工复核；证据不足时不放行"
                )
                ops.append(
                    ExperienceOperation(
                        op="upsert",
                        key=None,
                        text=rule_text,
                        rationale="forced_conflict_rule",
                        evidence=(gid,),
                        merged_from=None,
                    )
                )

        # Backwards-compatible safety net when conflict forcing is disabled:
        # if no operations were produced but conflicts/mismatches exist, inject
        # a single conservative rule using the first conflicting group as seed.
        if (
            not self.config.require_rule_for_conflicts
            and not ops
            and conflict_gids
            and max_ops != 0
        ):
            rule_text = (
                "当关键部件缺失、未按要求安装或不可见/模糊/遮挡时，判定不通过或需人工复核；证据不足时不放行"
            )
            ops.append(
                ExperienceOperation(
                    op="upsert",
                    key=None,
                    text=rule_text,
                    rationale="auto_conflict_fail",
                    evidence=tuple(conflict_gids[:1]),
                    merged_from=None,
                )
            )

        return tuple(ops), wishlist, noise

    def _build_plan_payload(
        self,
        critique_json: Sequence[Mapping[str, Any]],
        *,
        bundle: ExperienceBundle,
        reflection_id: str,
    ) -> Dict[str, Any]:
        operations, wishlist_entries, noise_entries = self._ops_from_json(
            critique_json, bundle=bundle
        )
        evidence_ids = tuple(rec.ticket.group_id for rec in bundle.records)
        action = "refine" if operations else "noop"
        proposal = ReflectionProposal(
            action=ReflectionAction(action),  # type: ignore[arg-type]
            summary="three_stage critique",
            critique="json_ops",
            operations=operations,
            evidence_group_ids=evidence_ids,
            uncertainty_note=None,
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
                if sig is not None and (sig.conflict_flag or sig.needs_manual_review):
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
            wishlist_entries = plan_payload.get("wishlist_entries", [])  # type: ignore[arg-type]
            noise_entries = plan_payload.get("noise_entries", [])  # type: ignore[arg-type]
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

        # Record wishlist/noise suggestions even if no guidance is applied
        self._record_aux_logs(wishlist_entries or [], noise_entries or [])

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
            "[AUTO] 矛盾/全错/冲突样本：优先标记人工复核；关键挡风板/BBU要素缺失或不确定时，"
            "倾向不通过或降低置信，并在 Reason 中写明证据不足。"
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
            messages, add_generation_prompt=True, tokenize=False
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
